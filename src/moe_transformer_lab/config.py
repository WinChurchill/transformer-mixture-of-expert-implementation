from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Shared configuration for dense and MoE decoder-only transformers."""

    vocab_size: int = 257
    max_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    norm_eps: float = 1e-5
    ffn_type: str = "dense"
    n_experts: int = 4
    top_k: int = 1
    aux_loss_coef: float = 0.01

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_type not in {"dense", "moe"}:
            raise ValueError("ffn_type must be 'dense' or 'moe'")
        if self.n_experts < 1:
            raise ValueError("n_experts must be positive")
        if self.top_k < 1 or self.top_k > self.n_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts")

    @classmethod
    def tiny_dense(cls, vocab_size: int = 257, max_seq_len: int = 128) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            ffn_type="dense",
        )

    @classmethod
    def tiny_moe(cls, vocab_size: int = 257, max_seq_len: int = 128) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            ffn_type="moe",
            n_experts=4,
            top_k=1,
        )
