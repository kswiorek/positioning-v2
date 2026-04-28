"""Checkpoint helpers for the v2 training engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload