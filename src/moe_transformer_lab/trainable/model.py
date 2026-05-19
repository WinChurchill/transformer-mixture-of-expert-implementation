from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_transformer_lab.config import ModelConfig


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor


class DenseMLP(nn.Module):
    """Standard token-wise transformer feed-forward network.

    Contract:
        y, aux_loss = feed_forward(x)

    Dense MLP has no router loss, so aux_loss is exactly zero.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, *, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        del train
        y = self.fc_out(F.gelu(self.fc_in(x), approximate="tanh"))
        y = self.dropout(y)
        return y, x.new_zeros(())


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of expert MLPs.

    The module has the same public contract as DenseMLP and can replace it inside
    TransformerBlock without touching attention, norms, residuals, logits, or the
    training loop.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.aux_loss_coef
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                DenseMLP(
                    ModelConfig(
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                        d_model=config.d_model,
                        n_layers=config.n_layers,
                        n_heads=config.n_heads,
                        d_ff=config.d_ff,
                        dropout=config.dropout,
                    )
                )
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: torch.Tensor, *, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        del train
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        flat_y = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            token_positions, top_slots = torch.where(top_indices == expert_id)
            if token_positions.numel() == 0:
                continue
            expert_in = flat_x[token_positions]
            expert_out, _ = expert(expert_in)
            weights = top_probs[token_positions, top_slots].unsqueeze(-1)
            flat_y.index_add_(0, token_positions, expert_out * weights)

        aux_loss = self._load_balance_loss(router_probs, top_indices)
        return flat_y.reshape(original_shape), aux_loss

    def _load_balance_loss(self, router_probs: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        # Switch-style load balancing: match average router probability to actual
        # selected expert frequency. This is simple, differentiable through
        # router_probs, and enough for the assignment.
        selected = F.one_hot(top_indices, num_classes=self.n_experts).float()
        selected_fraction = selected.mean(dim=(0, 1))
        prob_fraction = router_probs.mean(dim=0)
        return self.aux_loss_coef * self.n_experts * torch.sum(selected_fraction * prob_fraction)


def build_feed_forward(config: ModelConfig) -> nn.Module:
    if config.ffn_type == "dense":
        return DenseMLP(config)
    if config.ffn_type == "moe":
        return MoEFeedForward(config)
    raise ValueError(f"unknown ffn_type: {config.ffn_type}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, -torch.inf)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm block with a swappable feed-forward sublayer."""

    def __init__(self, config: ModelConfig, feed_forward: nn.Module | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.feed_forward = feed_forward if feed_forward is not None else build_feed_forward(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.norm1(x))
        ffn_out, aux_loss = self.feed_forward(self.norm2(x), train=self.training)
        x = x + ffn_out
        return x, aux_loss


class DecoderOnlyTransformer(nn.Module):
    """Small GPT-style language model with dense or MoE feed-forward blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> LMOutput:
        bsz, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)

        aux_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux

        logits = self.lm_head(self.norm(x))
        ce_loss = None
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss = ce_loss + aux_loss
        return LMOutput(logits=logits, loss=loss, ce_loss=ce_loss, aux_loss=aux_loss)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits = self(idx_cond).logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def replace_feed_forward_modules(model: DecoderOnlyTransformer, ffn_type: str) -> None:
    """Replace only feed-forward modules while preserving attention/norm modules."""

    if ffn_type not in {"dense", "moe"}:
        raise ValueError("ffn_type must be 'dense' or 'moe'")
    model.config.ffn_type = ffn_type
    model.config.validate()
    reference = next(model.parameters())
    for block in model.blocks:
        block.feed_forward = build_feed_forward(model.config).to(
            device=reference.device,
            dtype=reference.dtype,
        )
