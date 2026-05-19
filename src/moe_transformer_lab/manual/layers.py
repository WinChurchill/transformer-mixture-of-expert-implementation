from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class ManualParameter:
    """Small parameter wrapper used by the handwritten backward exercises."""

    data: torch.Tensor
    grad: torch.Tensor

    @classmethod
    def from_data(cls, data: torch.Tensor) -> "ManualParameter":
        return cls(data=data, grad=torch.zeros_like(data))

    def zero_grad(self) -> None:
        self.grad.zero_()


def zero_grad(parameters: Iterable[ManualParameter]) -> None:
    for parameter in parameters:
        parameter.zero_grad()


class Linear:
    """Affine map y = xW + b with a handwritten backward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        scale = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(in_features, out_features, device=device, dtype=dtype)
        weight.normal_(mean=0.0, std=scale, generator=generator)
        self.weight = ManualParameter.from_data(weight)
        self.bias = (
            ManualParameter.from_data(torch.zeros(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self._cache_x: torch.Tensor | None = None

    def parameters(self) -> list[ManualParameter]:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._cache_x = x
        y = x @ self.weight.data
        if self.bias is not None:
            y = y + self.bias.data
        return y

    def backward(self, dy: torch.Tensor) -> torch.Tensor:
        if self._cache_x is None:
            raise RuntimeError("Linear.backward called before forward")
        x = self._cache_x
        x_flat = x.reshape(-1, x.shape[-1])
        dy_flat = dy.reshape(-1, dy.shape[-1])
        self.weight.grad += x_flat.t() @ dy_flat
        if self.bias is not None:
            self.bias.grad += dy_flat.sum(dim=0)
        return dy @ self.weight.data.t()


class Embedding:
    """Lookup table for token or position embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        table = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        table.normal_(mean=0.0, std=0.02, generator=generator)
        self.weight = ManualParameter.from_data(table)
        self._cache_indices: torch.Tensor | None = None

    def parameters(self) -> list[ManualParameter]:
        return [self.weight]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        self._cache_indices = indices
        return self.weight.data[indices]

    def backward(self, dout: torch.Tensor) -> None:
        if self._cache_indices is None:
            raise RuntimeError("Embedding.backward called before forward")
        flat_indices = self._cache_indices.reshape(-1)
        flat_grad = dout.reshape(-1, dout.shape[-1])
        self.weight.grad.index_add_(0, flat_indices, flat_grad)
        return None


class LayerNorm:
    """Layer normalization over the final dimension."""

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.eps = eps
        self.gamma = ManualParameter.from_data(torch.ones(d_model, device=device, dtype=dtype))
        self.beta = ManualParameter.from_data(torch.zeros(d_model, device=device, dtype=dtype))
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def parameters(self) -> list[ManualParameter]:
        return [self.gamma, self.beta]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        x_hat = centered * inv_std
        self._cache = (x_hat, inv_std)
        return self.gamma.data * x_hat + self.beta.data

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("LayerNorm.backward called before forward")
        x_hat, inv_std = self._cache
        reduce_dims = tuple(range(dout.ndim - 1))
        self.gamma.grad += (dout * x_hat).sum(dim=reduce_dims)
        self.beta.grad += dout.sum(dim=reduce_dims)

        dx_hat = dout * self.gamma.data
        d = x_hat.shape[-1]
        return inv_std * (
            dx_hat
            - dx_hat.mean(dim=-1, keepdim=True)
            - x_hat * (dx_hat * x_hat).mean(dim=-1, keepdim=True)
        )


class GELU:
    """Approximate GELU with its analytic derivative."""

    def __init__(self) -> None:
        self._cache_x: torch.Tensor | None = None

    def parameters(self) -> list[ManualParameter]:
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._cache_x = x
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(u))

    def backward(self, dout: torch.Tensor) -> torch.Tensor:
        if self._cache_x is None:
            raise RuntimeError("GELU.backward called before forward")
        x = self._cache_x
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x.pow(3))
        tanh_u = torch.tanh(u)
        du_dx = c * (1.0 + 3.0 * 0.044715 * x.pow(2))
        gelu_prime = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u.pow(2)) * du_dx
        return dout * gelu_prime


class SoftmaxCrossEntropy:
    """Mean next-token cross entropy with a stable row-softmax backward."""

    def __init__(self, ignore_index: int | None = None) -> None:
        self.ignore_index = ignore_index
        self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        if self.ignore_index is None:
            valid = torch.ones_like(targets_flat, dtype=torch.bool)
        else:
            valid = targets_flat != self.ignore_index

        shifted = logits_flat - logits_flat.max(dim=-1, keepdim=True).values
        exp = torch.exp(shifted)
        probs = exp / exp.sum(dim=-1, keepdim=True)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            raise ValueError("no valid targets in cross entropy")

        target_probs = probs[torch.arange(targets_flat.numel(), device=logits.device), targets_flat.clamp_min(0)]
        loss = -torch.log(target_probs[valid] + 1e-12).mean()
        self._cache = (probs, targets_flat, valid, n_valid)
        return loss

    def backward(self) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("SoftmaxCrossEntropy.backward called before forward")
        probs, targets_flat, valid, n_valid = self._cache
        grad = probs.clone()
        rows = torch.arange(targets_flat.numel(), device=targets_flat.device)
        grad[rows[valid], targets_flat[valid]] -= 1.0
        grad[~valid] = 0.0
        grad /= n_valid
        return grad
