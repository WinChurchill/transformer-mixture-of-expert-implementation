from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError as exc:
    raise SystemExit("Install PyTorch before running training. See README.md setup notes.") from exc

from moe_transformer_lab.config import ModelConfig
from moe_transformer_lab.trainable.data import encode_text_file, get_batch
from moe_transformer_lab.trainable.model import DecoderOnlyTransformer
from moe_transformer_lab.trainable.tokenizer import load_tokenizer
from moe_transformer_lab.trainable.train_utils import (
    configure_optimizer,
    count_parameters,
    get_device,
    save_checkpoint,
    set_seed,
)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext(), None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype), dtype


@torch.no_grad()
def estimate_loss(model, train_tokens, valid_tokens, args, device) -> tuple[float, float]:
    model.eval()
    losses = {}
    for name, tokens in [("train", train_tokens), ("valid", valid_tokens)]:
        split_losses = []
        for _ in range(args.eval_batches):
            xb, yb = get_batch(tokens, batch_size=args.batch_size, block_size=args.block_size, device=device)
            out = model(xb, yb)
            split_losses.append(float(out.loss.item()))
        losses[name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses["train"], losses["valid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny dense or MoE transformer on TinyStories.")
    parser.add_argument("--ffn", choices=["dense", "moe"], default="moe")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/tinystories/raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tinystories/tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/tinystories_moe"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-chars", type=int, default=5_000_000)
    parser.add_argument("--max-valid-chars", type=int, default=500_000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--aux-loss-coef", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    train_path = args.raw_dir / "TinyStoriesV2-GPT4-train.txt"
    valid_path = args.raw_dir / "TinyStoriesV2-GPT4-valid.txt"
    if not train_path.exists() or not valid_path.exists():
        raise SystemExit("Run scripts/download_tinystories.py first.")

    print("encoding text")
    train_tokens = encode_text_file(train_path, tokenizer, max_chars=args.max_train_chars)
    valid_tokens = encode_text_file(valid_path, tokenizer, max_chars=args.max_valid_chars)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        ffn_type=args.ffn,
        n_experts=args.n_experts,
        top_k=args.top_k,
        aux_loss_coef=args.aux_loss_coef,
    )
    config.validate()
    model = DecoderOnlyTransformer(config).to(device)
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )

    print(f"device={device} ffn={args.ffn} parameters={count_parameters(model):,}")
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            xb, yb = get_batch(train_tokens, batch_size=args.batch_size, block_size=args.block_size, device=device)
            ctx, _ = autocast_context(device, args.amp)
            with ctx:
                out = model(xb, yb)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()

        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_interval == 0 or step == 1:
            train_loss, valid_loss = estimate_loss(model, train_tokens, valid_tokens, args, device)
            print(f"step {step:5d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
            save_checkpoint(args.out_dir / "ckpt_last.pt", model=model, optimizer=optimizer, step=step)

    save_checkpoint(args.out_dir / "ckpt_last.pt", model=model, optimizer=optimizer, step=args.steps)


if __name__ == "__main__":
    main()
