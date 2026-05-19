from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)


def configure_optimizer(model: torch.nn.Module, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.endswith("bias"):
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, step: int, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": vars(model.config),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
