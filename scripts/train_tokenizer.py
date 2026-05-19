from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_transformer_lab.trainable.tokenizer import ByteTokenizer


def train_bpe(raw_dir: Path, out: Path, vocab_size: int) -> None:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise SystemExit("Install tokenizers or use --kind byte.") from exc

    files = [
        str(raw_dir / "TinyStoriesV2-GPT4-train.txt"),
        str(raw_dir / "TinyStoriesV2-GPT4-valid.txt"),
    ]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<eot>", "<unk>"],
        min_frequency=2,
    )
    tokenizer.train(files=[path for path in files if Path(path).exists()], trainer=trainer)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tokenizer for the lab.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/tinystories/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/tinystories/tokenizer.json"))
    parser.add_argument("--kind", choices=["byte", "bpe"], default="byte")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    if args.kind == "byte":
        ByteTokenizer().save(args.out)
        print(f"wrote byte tokenizer to {args.out}")
    else:
        train_bpe(args.raw_dir, args.out, args.vocab_size)
        print(f"wrote BPE tokenizer to {args.out}")


if __name__ == "__main__":
    main()
