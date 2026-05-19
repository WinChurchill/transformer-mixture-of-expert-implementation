from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class TokenizerProtocol(Protocol):
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


class ByteTokenizer:
    """Dependency-free byte-level tokenizer.

    Token 0 is end-of-text. UTF-8 bytes are shifted by +1, giving vocab size 257.
    It is not as efficient as BPE, but it keeps the assignment runnable before
    optional tokenizer dependencies are installed.
    """

    eos_token_id = 0
    vocab_size = 257

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = [byte + 1 for byte in text.encode("utf-8")]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes([idx - 1 for idx in ids if idx > 0])
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "byte", "vocab_size": self.vocab_size}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte":
            raise ValueError("this loader only supports ByteTokenizer JSON files")
        return cls()


class TokenizersTokenizer:
    """Adapter for tokenizers.Tokenizer BPE files."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.token_to_id("<eot>")
        if self.eos_token_id is None:
            self.eos_token_id = 0
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


def load_tokenizer(path: str | Path | None) -> TokenizerProtocol:
    if path is None or not Path(path).exists():
        return ByteTokenizer()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "byte":
        return ByteTokenizer.load(path)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install tokenizers to load this tokenizer JSON, or use --kind byte.") from exc
    return TokenizersTokenizer(Tokenizer.from_file(str(path)))
