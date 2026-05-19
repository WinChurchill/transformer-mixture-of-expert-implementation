from __future__ import annotations

import torch

from .attention import MultiHeadSelfAttention, causal_mask
from .layers import Embedding, GELU, LayerNorm, Linear, ManualParameter, SoftmaxCrossEntropy


class ManualMLP:
    """Two-layer token-wise MLP used before introducing MoE."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.fc_in = Linear(d_model, d_ff, generator=generator, device=device, dtype=dtype)
        self.act = GELU()
        self.fc_out = Linear(d_ff, d_model, generator=generator, device=device, dtype=dtype)

    def parameters(self) -> list[ManualParameter]:
        return self.fc_in.parameters() + self.fc_out.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out.forward(self.act.forward(self.fc_in.forward(x)))

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        return self.fc_in.backward(self.act.backward(self.fc_out.backward(dout)))


class ManualTransformerBlock:
    """Pre-norm decoder block: x + Attention(LN(x)); x + MLP(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.norm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            n_heads,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        self.norm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.mlp = ManualMLP(d_model, d_ff, generator=generator, device=device, dtype=dtype)

    def parameters(self) -> list[ManualParameter]:
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.mlp.parameters()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn.forward(self.norm1.forward(x), mask)
        x_after_attn = x + attn_out
        self._cache_x_after_attn = x_after_attn
        mlp_out = self.mlp.forward(self.norm2.forward(x_after_attn))
        return x_after_attn + mlp_out

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        d_after_attn = dout + self.norm2.backward(self.mlp.backward(dout))
        return d_after_attn + self.norm1.backward(self.attn.backward(d_after_attn))


class ManualDecoderOnlyTransformer:
    """Small educational language model assembled from manual components."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, d_model, generator=generator, device=device, dtype=dtype)
        self.position_embedding = Embedding(max_seq_len, d_model, generator=generator, device=device, dtype=dtype)
        self.blocks = [
            ManualTransformerBlock(
                d_model,
                n_heads,
                d_ff,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, bias=False, generator=generator, device=device, dtype=dtype)
        self.loss_fn = SoftmaxCrossEntropy()
        self._cache_positions: torch.Tensor | None = None
        self._cache_shape: tuple[int, int] | None = None

    def parameters(self) -> list[ManualParameter]:
        params = self.token_embedding.parameters() + self.position_embedding.parameters()
        for block in self.blocks:
            params += block.parameters()
        return params + self.norm.parameters() + self.lm_head.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(bsz, seq_len)
        self._cache_positions = positions
        self._cache_shape = (bsz, seq_len)
        x = self.token_embedding.forward(token_ids) + self.position_embedding.forward(positions)
        mask = causal_mask(seq_len, device=token_ids.device)
        for block in self.blocks:
            x = block.forward(x, mask)
        return self.lm_head.forward(self.norm.forward(x))

    def loss(self, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(token_ids)
        return self.loss_fn.forward(logits, targets)

    def backward(self) -> None:
        dlogits = self.loss_fn.backward()
        if self._cache_shape is None:
            raise RuntimeError("ManualDecoderOnlyTransformer.backward called before loss")
        bsz, seq_len = self._cache_shape
        dx = self.lm_head.backward(dlogits.reshape(bsz, seq_len, -1))
        dx = self.norm.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.position_embedding.backward(dx)
        self.token_embedding.backward(dx)
