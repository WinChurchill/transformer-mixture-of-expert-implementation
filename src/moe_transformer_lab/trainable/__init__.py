"""Practical PyTorch modules for local dense/MoE language-model training."""

from .model import (
    CausalSelfAttention,
    DecoderOnlyTransformer,
    DenseMLP,
    LMOutput,
    MoEFeedForward,
    TransformerBlock,
    build_feed_forward,
    replace_feed_forward_modules,
)

__all__ = [
    "DenseMLP",
    "MoEFeedForward",
    "CausalSelfAttention",
    "TransformerBlock",
    "DecoderOnlyTransformer",
    "LMOutput",
    "build_feed_forward",
    "replace_feed_forward_modules",
]
