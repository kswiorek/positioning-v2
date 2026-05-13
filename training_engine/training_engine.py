"""Modular training engine for the v2 project."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


def _check_finite(name: str, tensor: torch.Tensor, *, epoch: int, batch_idx: int, phase: str) -> None:
    if not torch.isfinite(tensor).all():
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(
            f"Non-finite tensor at {name} during {phase} epoch={epoch + 1} batch={batch_idx}: "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}, bad_count={bad_count}"
        )

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

# Local imports: prefer package-relative imports, but allow running the file
# directly as a script by falling back to absolute imports after adjusting
# `sys.path`.
try:
    from .checkpoints import load_checkpoint, save_checkpoint
    from .config import TrainingConfig
    from .dataset import build_dataloaders
    from .geometry import build_transform_from_Rt, coerce_pose_output
    from .losses import pose_loss, scene_mask_bce_loss
    from .model import build_model
except Exception:  # pragma: no cover - fallback for direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training_engine.checkpoints import load_checkpoint, save_checkpoint
    from training_engine.config import TrainingConfig
    from training_engine.dataset import build_dataloaders
    from training_engine.geometry import build_transform_from_Rt, coerce_pose_output
    from training_engine.losses import pose_loss, scene_mask_bce_loss
    from training_engine.model import build_model


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    translation: float
    rotation: float
    bbox_corner: float
    rotation_error_deg: float
    confidence: float = 0.0
    segmentation: float = 0.0
    val_pose_teacher_loss: float = 0.0


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
        "segmentation": metrics.segmentation,
        "val_pose_teacher_loss": metrics.val_pose_teacher_loss,
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
        self.high_loss_samples_path = self.run_dir / "high_loss_samples.txt"
        self.monitoring = config.monitoring
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device != "cuda" else "cpu")

        self.model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        if self.train_loader is None or self.val_loader is None:
            self.train_loader, self.val_loader = build_dataloaders(config)

        td = self.train_loader.dataset
        self._train_dataset_for_epoch_hook = td if hasattr(td, "set_epoch") else None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.loss_weights = config.loss
        self.best_val_loss = math.inf
        self.start_epoch = 0
        self.global_step = 0
        self.writer = None
        if self.monitoring.tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        elif self.monitoring.tensorboard:
            print("TensorBoard is not available; continuing without scalar event logs.")

        self._write_config_snapshot()
        self._ensure_log_header()

        if self.config.resume_best:
            best_checkpoint = self.checkpoint_dir / "best.pth"
            if best_checkpoint.exists():
                load_and_resume(self, best_checkpoint)
        elif self.config.resume_latest:
            latest_checkpoint = self.checkpoint_dir / "latest.pth"
            if latest_checkpoint.exists():
                load_and_resume(self, latest_checkpoint)

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
                    "confidence",
                    "segmentation",
                    "val_pose_teacher_loss",
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

    def _log_batch_progress(
        self,
        phase: str,
        epoch: int,
        batch_idx: int,
        total_batches: int | None,
        elapsed_s: float,
        metrics: dict[str, float],
    ) -> None:
        if self.monitoring.log_every_n_batches <= 0:
            return
        if batch_idx != 1 and batch_idx % self.monitoring.log_every_n_batches != 0:
            if total_batches is None or batch_idx != total_batches:
                return

        batch_rate = batch_idx / max(elapsed_s, 1e-6)
        parts = [
            f"{phase} epoch {epoch + 1}",
            f"batch {batch_idx}" if total_batches is None else f"batch {batch_idx}/{total_batches}",
            f"loss={metrics['loss']:.4f}",
            f"trans={metrics['translation']:.4f}",
            f"rot={metrics['rotation']:.4f}",
            f"bbox={metrics['bbox_corner']:.4f}",
            f"{batch_rate:.2f} batch/s",
        ]
        if "confidence" in metrics:
            parts.append(f"conf={metrics['confidence']:.4f}")
        if metrics.get("segmentation", 0.0) != 0.0:
            parts.append(f"seg={metrics['segmentation']:.4f}")
        print(" | ".join(parts), flush=True)

    def _write_tensorboard_scalars(
        self,
        phase: str,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, step)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "depth": batch["depth"].to(self.device, non_blocking=True),
            "model_points": batch["model_points"].to(self.device, non_blocking=True),
            "gt_transform": batch["gt_transform"].to(self.device, non_blocking=True),
            "bbox_corners": batch["bbox_corners"].to(self.device, non_blocking=True),
            "sample_id": batch.get("sample_id"),
        }
        if "scene_mask" in batch:
            out["scene_mask"] = batch["scene_mask"].to(self.device, non_blocking=True)
        return out

    def _run_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        train: bool,
        *,
        epoch: int,
        phase: str,
        max_batches: int | None = None,
    ) -> EpochMetrics:
        self.model.train(mode=train)

        totals = {
            "loss": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "bbox_corner": 0.0,
            "rotation_error_deg": 0.0,
            "confidence": 0.0,
            "segmentation": 0.0,
            "val_pose_teacher_loss": 0.0,
        }
        n_batches = 0
        use_amp = self.scaler.is_enabled()
        context = torch.enable_grad() if train else torch.no_grad()
        # use_amp = False
        start_time = time.perf_counter()
        total_batches = len(loader) if hasattr(loader, "__len__") else None

        seg_cfg = self.config.segmentation
        with context:
            for batch_idx, batch in enumerate(loader, start=1):
                batch = self._move_batch(batch)
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    if seg_cfg.enabled:
                        if "scene_mask" not in batch:
                            raise ValueError("segmentation.enabled requires batches to include 'scene_mask'")
                        model_output = self.model(
                            batch["depth"],
                            batch["model_points"],
                            batch["scene_mask"],
                            train=train,
                        )
                    else:
                        model_output = self.model(batch["depth"], batch["model_points"])

                    pred_transform = coerce_pose_output(model_output)
                    _check_finite("pred_transform", pred_transform, epoch=epoch, batch_idx=batch_idx, phase=phase)
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

                    if seg_cfg.enabled and "mask_logits" in model_output:
                        seg_loss = scene_mask_bce_loss(
                            model_output["mask_logits"],
                            model_output["mask_gt_tokens"],
                        )
                        loss_dict["loss"] = loss_dict["loss"] + seg_cfg.loss_weight * seg_loss
                        loss_dict["segmentation"] = float(seg_loss.detach().cpu())

                    if (
                        not train
                        and seg_cfg.enabled
                        and model_output.get("pred_R_teacher") is not None
                        and model_output.get("pred_t_teacher") is not None
                    ):
                        pred_tfm_teacher = build_transform_from_Rt(
                            model_output["pred_R_teacher"],
                            model_output["pred_t_teacher"],
                        )
                        loss_teacher = pose_loss(
                            pred_transform=pred_tfm_teacher,
                            gt_transform=batch["gt_transform"],
                            bbox_corners=batch["bbox_corners"],
                            pred_conf_t=model_output.get("confidence_t_teacher"),
                            pred_conf_r=model_output.get("confidence_r_teacher"),
                            weights=self.loss_weights,
                        )
                        loss_dict["val_pose_teacher_loss"] = float(loss_teacher["loss"].detach().cpu())

                    batch_loss = float(loss_dict["loss"].detach().cpu())
                    if batch_loss > 100:
                        sample_ids = batch.get("sample_id")
                        with self.high_loss_samples_path.open("a", encoding="utf-8") as f:
                            f.write(
                                f"epoch={epoch + 1} batch={batch_idx} phase={phase} loss={batch_loss:.2f} samples={sample_ids}\n"
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
                n_batches += 1
                if train:
                    self.global_step += 1

                running = {key: totals[key] / n_batches for key in totals}
                elapsed_s = time.perf_counter() - start_time
                self._log_batch_progress(phase, epoch, batch_idx, total_batches, elapsed_s, running)

                if train:
                    batch_metrics = dict(running)
                    batch_metrics["lr"] = self._current_lr()
                    self._write_tensorboard_scalars("batch", self.global_step, batch_metrics)

                    interval = self.monitoring.quick_validation_every_n_train_batches
                    if interval > 0 and self.monitoring.quick_validation_batches > 0 and self.global_step % interval == 0:
                        quick_metrics = self._run_epoch(
                            self.val_loader,
                            train=False,
                            epoch=epoch,
                            phase="quick_val",
                            max_batches=self.monitoring.quick_validation_batches,
                        )
                        quick_row = _metrics_to_row("quick_val", epoch, quick_metrics, self._current_lr())
                        self._write_tensorboard_scalars(
                            "quick_val",
                            self.global_step,
                            {
                                "loss": quick_metrics.loss,
                                "translation": quick_metrics.translation,
                                "rotation": quick_metrics.rotation,
                                "bbox_corner": quick_metrics.bbox_corner,
                                "rotation_error_deg": quick_metrics.rotation_error_deg,
                                "confidence": quick_metrics.confidence,
                                "segmentation": quick_metrics.segmentation,
                                "pose_teacher_loss": quick_metrics.val_pose_teacher_loss,
                            },
                        )
                        print(
                            f"quick_val step {self.global_step}: loss={quick_metrics.loss:.4f}, "
                            f"trans={quick_metrics.translation:.4f}, rot={quick_metrics.rotation:.4f}",
                            flush=True,
                        )
                        self._append_log_row(quick_row)
                        self.model.train(mode=True)

                if max_batches is not None and batch_idx >= max_batches:
                    break

        divisor = max(n_batches, 1)
        return EpochMetrics(
            loss=totals["loss"] / divisor,
            translation=totals["translation"] / divisor,
            rotation=totals["rotation"] / divisor,
            bbox_corner=totals["bbox_corner"] / divisor,
            rotation_error_deg=totals["rotation_error_deg"] / divisor,
            confidence=totals["confidence"] / divisor,
            segmentation=totals["segmentation"] / divisor,
            val_pose_teacher_loss=totals["val_pose_teacher_loss"] / divisor,
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

            if self._train_dataset_for_epoch_hook is not None:
                self._train_dataset_for_epoch_hook.set_epoch(epoch)

            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch, phase="train")
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch, phase="val")

            current_lr = self._current_lr()
            train_row = _metrics_to_row("train", epoch, train_metrics, current_lr)
            val_row = _metrics_to_row("val", epoch, val_metrics, current_lr)
            self._append_log_row(train_row)
            self._append_log_row(val_row)

            self._write_tensorboard_scalars(
                "epoch/train",
                epoch + 1,
                {
                    "loss": train_metrics.loss,
                    "translation": train_metrics.translation,
                    "rotation": train_metrics.rotation,
                    "bbox_corner": train_metrics.bbox_corner,
                    "rotation_error_deg": train_metrics.rotation_error_deg,
                    "confidence": train_metrics.confidence,
                    "segmentation": train_metrics.segmentation,
                },
            )
            self._write_tensorboard_scalars(
                "epoch/val",
                epoch + 1,
                {
                    "loss": val_metrics.loss,
                    "translation": val_metrics.translation,
                    "rotation": val_metrics.rotation,
                    "bbox_corner": val_metrics.bbox_corner,
                    "rotation_error_deg": val_metrics.rotation_error_deg,
                    "confidence": val_metrics.confidence,
                    "segmentation": val_metrics.segmentation,
                    "pose_teacher_loss": val_metrics.val_pose_teacher_loss,
                },
            )
            if self.writer is not None:
                self.writer.add_scalar("epoch/lr", current_lr, epoch + 1)

            latest_path = self.checkpoint_dir / "latest.pth"
            save_checkpoint(
                latest_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                best_val_loss=self.best_val_loss,
                global_step=self.global_step,
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
                    global_step=self.global_step,
                    extra={"train": train_row, "val": val_row},
                )

            self._step_scheduler(epoch)

            summary = {
                "epoch": epoch + 1,
                "train": train_row,
                "val": val_row,
                "best_val_loss": self.best_val_loss,
            }

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

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
    engine.global_step = int(payload.get("global_step", 0))
    engine.best_val_loss = float(payload.get("best_val_loss", math.inf))

    if engine.writer is not None and SummaryWriter is not None:
        engine.writer.flush()
        engine.writer.close()
        engine.writer = SummaryWriter(
            log_dir=str(engine.run_dir / "tensorboard"),
            purge_step=engine.global_step,
        )

    return payload
