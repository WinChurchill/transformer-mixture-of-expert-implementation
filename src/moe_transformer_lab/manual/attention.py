from __future__ import annotations

import math

import torch

from .layers import Linear, ManualParameter


def causal_mask(seq_len: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a lower-triangular boolean mask of shape (N, N)."""

    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def masked_row_softmax(scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax each row of the last dimension, optionally masking disallowed entries."""

    if mask is not None:
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, -torch.inf)
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    if mask is not None:
        exp = exp.masked_fill(~mask, 0.0)
    return exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class ScaledDotProductAttention:
    """Attention(Q, K, V) = softmax_row(QK^T / sqrt(D_k)) V."""

    def __init__(self) -> None:
        self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor | None] | None = None

    def parameters(self) -> list[ManualParameter]:
        return []

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = masked_row_softmax(scores, mask)
        self._cache = (q, k, v, attn, scale, mask)
        return attn @ v

    def backward(self, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache is None:
            raise RuntimeError("ScaledDotProductAttention.backward called before forward")
        q, k, v, attn, scale, mask = self._cache

        datt = dout @ v.transpose(-2, -1)
        dv = attn.transpose(-2, -1) @ dout

        dscores = attn * (datt - (datt * attn).sum(dim=-1, keepdim=True))
        if mask is not None:
            while mask.ndim < dscores.ndim:
                mask = mask.unsqueeze(0)
            dscores = dscores.masked_fill(~mask, 0.0)

        dq = (dscores @ k) * scale
        dk = (dscores.transpose(-2, -1) @ q) * scale
        return dq, dk, dv


class MultiHeadSelfAttention:
    """Causal multi-head self-attention with handwritten backward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, generator=generator, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, generator=generator, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, generator=generator, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, generator=generator, device=device, dtype=dtype)
        self.attention = ScaledDotProductAttention()

    def parameters(self) -> list[ManualParameter]:
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.reshape(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.q_proj.forward(x))
        k = self._split_heads(self.k_proj.forward(x))
        v = self._split_heads(self.v_proj.forward(x))
        heads = self.attention.forward(q, k, v, mask)
        return self.out_proj.forward(self._merge_heads(heads))

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        dmerged = self.out_proj.backward(dout)
        bsz, seq_len, _ = dmerged.shape
        dheads = dmerged.reshape(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        dq, dk, dv = self.attention.backward(dheads)
        dx = self.q_proj.backward(self._merge_heads(dq))
        dx = dx + self.k_proj.backward(self._merge_heads(dk))
        dx = dx + self.v_proj.backward(self._merge_heads(dv))
        return dx
