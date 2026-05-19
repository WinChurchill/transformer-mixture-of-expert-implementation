from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import TokenizerProtocol


def read_text(path: str | Path, *, max_chars: int | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]
    return text


def encode_text_file(path: str | Path, tokenizer: TokenizerProtocol, *, max_chars: int | None = None) -> torch.Tensor:
    text = read_text(path, max_chars=max_chars)
    return torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)


def get_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= block_size + 1:
        raise ValueError("not enough tokens to sample a batch")
    starts = torch.randint(0, tokens.numel() - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
