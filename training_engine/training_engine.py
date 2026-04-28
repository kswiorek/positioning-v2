"""Modular training engine for the v2 project."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .dataset import build_dataloaders
from .geometry import coerce_pose_output
from .losses import PoseLossWeights, pose_loss
from .model import build_model


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    translation: float
    rotation: float
    bbox_corner: float
    rotation_error_deg: float
    confidence: float = 0.0


def _metrics_to_row(prefix: str, epoch: int, metrics: EpochMetrics, lr: float) -> dict[str, Any]:
    return {
        "epoch": epoch + 1,
        "phase": prefix,
        "lr": lr,
        "loss": metrics.loss,
        "translation": metrics.translation,
        "rotation": metrics.rotation,
        "bbox_corner": metrics.bbox_corner,
        "rotation_error_deg": metrics.rotation_error_deg,
        "confidence": metrics.confidence,
    }


class TrainingEngine:
    """Owns dataset loading, optimization, validation, and checkpointing."""

    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module | None = None,
        train_loader: torch.utils.data.DataLoader | None = None,
        val_loader: torch.utils.data.DataLoader | None = None,
    ) -> None:
        self.config = config
        self.model = model or build_model(config)
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "training_log.csv"
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device != "cuda" else "cpu")

        self.model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        if self.train_loader is None or self.val_loader is None:
            self.train_loader, self.val_loader = build_dataloaders(config)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.loss_weights = PoseLossWeights(
            translation_weight=config.loss.translation_weight,
            rotation_weight=config.loss.rotation_weight,
            bbox_corner_weight=config.loss.bbox_corner_weight,
        )
        self.best_val_loss = math.inf
        self.start_epoch = 0

        self._write_config_snapshot()
        self._ensure_log_header()

        if self.config.resume_best:
            best_checkpoint = self.checkpoint_dir / "best.pth"
            if best_checkpoint.exists():
                load_and_resume(self, best_checkpoint)

    def _write_config_snapshot(self) -> None:
        snapshot = {
            "training": asdict(self.config),
            "device": str(self.device),
        }
        with (self.run_dir / "training_config.json").open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, default=str)

    def _ensure_log_header(self) -> None:
        if self.log_path.exists():
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "phase",
                    "lr",
                    "loss",
                    "translation",
                    "rotation",
                    "bbox_corner",
                    "rotation_error_deg",
                ],
            )
            writer.writeheader()

    def _append_log_row(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        scheduler_type = self.config.scheduler.type.lower()
        if scheduler_type in {"", "none"}:
            return None
        min_lr = self.config.scheduler.min_learning_rate
        if scheduler_type == "cosine_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(self.config.scheduler.restart_period, 1),
                T_mult=1,
                eta_min=min_lr,
            )
        if scheduler_type == "cosine":
            effective_epochs = max(self.config.max_epochs - self.config.scheduler.warmup_epochs, 1)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=effective_epochs,
                eta_min=min_lr,
            )
        raise ValueError(f"Unsupported scheduler type: {self.config.scheduler.type!r}")

    def _set_warmup_lr(self, epoch: int) -> None:
        warmup_epochs = max(self.config.scheduler.warmup_epochs, 0)
        if warmup_epochs <= 0 or epoch >= warmup_epochs:
            return
        scale = float(epoch + 1) / float(max(warmup_epochs, 1))
        lr = self.config.optimizer.learning_rate * scale
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "depth": batch["depth"].to(self.device, non_blocking=True),
            "model_points": batch["model_points"].to(self.device, non_blocking=True),
            "gt_transform": batch["gt_transform"].to(self.device, non_blocking=True),
            "bbox_corners": batch["bbox_corners"].to(self.device, non_blocking=True),
        }

    def _run_epoch(self, loader: torch.utils.data.DataLoader, train: bool) -> EpochMetrics:
        self.model.train(mode=train)

        totals = {
            "loss": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "bbox_corner": 0.0,
            "rotation_error_deg": 0.0,
        }
        n_batches = 0
        use_amp = self.scaler.is_enabled()
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in loader:
                batch = self._move_batch(batch)
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    model_output = self.model(batch["depth"], batch["model_points"])
                    pred_transform = coerce_pose_output(model_output)
                    # Extract confidence predictions if available
                    pred_conf_t = model_output.get("confidence_t")
                    pred_conf_r = model_output.get("confidence_r")
                    loss_dict = pose_loss(
                        pred_transform=pred_transform,
                        gt_transform=batch["gt_transform"],
                        bbox_corners=batch["bbox_corners"],
                        pred_conf_t=pred_conf_t,
                        pred_conf_r=pred_conf_r,
                        weights=self.loss_weights,
                    )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        self.scaler.scale(loss_dict["loss"]).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss_dict["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                        self.optimizer.step()

                for key in totals:
                    if key in loss_dict:
                        totals[key] += float(loss_dict[key].detach().cpu())
                # Track confidence loss if present
                if "confidence" in loss_dict and "confidence" not in totals:
                    totals["confidence"] = 0.0
                if "confidence" in loss_dict:
                    totals["confidence"] += float(loss_dict["confidence"].detach().cpu())
                n_batches += 1

        divisor = max(n_batches, 1)
        return EpochMetrics(
            loss=totals["loss"] / divisor,
            translation=totals["translation"] / divisor,
            rotation=totals["rotation"] / divisor,
            bbox_corner=totals["bbox_corner"] / divisor,
            rotation_error_deg=totals["rotation_error_deg"] / divisor,
                    confidence=totals.get("confidence", 0.0) / divisor,
        )

    def _step_scheduler(self, epoch: int) -> None:
        if self.scheduler is None:
            return
        if self.config.scheduler.warmup_epochs > 0 and epoch < self.config.scheduler.warmup_epochs:
            self._set_warmup_lr(epoch)
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            if self.config.scheduler.warmup_epochs > 0:
                adjusted_epoch = epoch - self.config.scheduler.warmup_epochs
                if adjusted_epoch >= 0:
                    self.scheduler.step(adjusted_epoch)
                return
            self.scheduler.step(epoch)
            return
        self.scheduler.step()

    def train(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        for epoch in range(self.start_epoch, self.config.max_epochs):
            if self.config.scheduler.warmup_epochs > 0 and epoch < self.config.scheduler.warmup_epochs:
                self._set_warmup_lr(epoch)

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            current_lr = self._current_lr()
            train_row = _metrics_to_row("train", epoch, train_metrics, current_lr)
            val_row = _metrics_to_row("val", epoch, val_metrics, current_lr)
            self._append_log_row(train_row)
            self._append_log_row(val_row)

            latest_path = self.checkpoint_dir / "latest.pth"
            save_checkpoint(
                latest_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_val_loss=self.best_val_loss,
                extra={"train": train_row, "val": val_row},
            )

            if val_metrics.loss < self.best_val_loss:
                self.best_val_loss = val_metrics.loss
                save_checkpoint(
                    self.checkpoint_dir / "best.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_val_loss=self.best_val_loss,
                    extra={"train": train_row, "val": val_row},
                )

            self._step_scheduler(epoch)

            summary = {
                "epoch": epoch + 1,
                "train": train_row,
                "val": val_row,
                "best_val_loss": self.best_val_loss,
            }

        return summary


def load_and_resume(
    engine: TrainingEngine,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = load_checkpoint(
        checkpoint_path,
        model=engine.model,
        optimizer=engine.optimizer,
        scheduler=engine.scheduler,
        scaler=engine.scaler,
        map_location=engine.device,
    )
    engine.start_epoch = int(payload.get("epoch", -1)) + 1
    engine.best_val_loss = float(payload.get("best_val_loss", math.inf))
    return payload
