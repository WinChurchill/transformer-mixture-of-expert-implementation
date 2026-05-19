from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError as exc:
    raise SystemExit("Install PyTorch before running generation.") from exc

from moe_transformer_lab.config import ModelConfig
from moe_transformer_lab.trainable.model import DecoderOnlyTransformer
from moe_transformer_lab.trainable.tokenizer import load_tokenizer
from moe_transformer_lab.trainable.train_utils import get_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a lab checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = get_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    payload = load_checkpoint(args.checkpoint, map_location=device)
    config = ModelConfig(**payload["config"])
    model = DecoderOnlyTransformer(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
