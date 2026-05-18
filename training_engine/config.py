"""Training configuration objects for the v2 training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


@dataclass(frozen=True)
class SceneEncoderConfig:
    base_channels: int = 32
    num_blocks: int = 4
    res_blocks_per_stride: int = 1
    feature_dim: int = 128

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SceneEncoderConfig":
        data = data or {}
        return cls(
            base_channels=int(data.get("base_channels", 32)),
            num_blocks=int(data.get("num_blocks", 4)),
            res_blocks_per_stride=int(data.get("res_blocks_per_stride", 1)),
            feature_dim=int(data.get("feature_dim", 128)),
        )


@dataclass(frozen=True)
class PointEncoderConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 128, 256])
    feature_dim: int = 128
    k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PointEncoderConfig":
        data = data or {}
        hidden_dims = data.get("hidden_dims", [64, 64, 128, 256])
        return cls(
            hidden_dims=[int(value) for value in hidden_dims],
            feature_dim=int(data.get("feature_dim", 128)),
            k=int(data.get("k", 20)),
        )


@dataclass(frozen=True)
class CrossAttentionConfig:
    num_heads: int = 4
    num_layers: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CrossAttentionConfig":
        data = data or {}
        return cls(
            num_heads=int(data.get("num_heads", 4)),
            num_layers=int(data.get("num_layers", 2)),
        )


@dataclass(frozen=True)
class FusionConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FusionConfig":
        data = data or {}
        hidden_dims = data.get("hidden_dims", [256, 128])
        return cls(
            hidden_dims=[int(value) for value in hidden_dims],
            dropout=float(data.get("dropout", 0.3)),
        )


@dataclass(frozen=True)
class SegmentationConfig:
    """When enabled, multiply depth by the GT object mask before the scene encoder."""

    enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SegmentationConfig":
        data = data or {}
        return cls(enabled=bool(data.get("enabled", False)))


@dataclass(frozen=True)
class ModelConfig:
    scene_encoder: SceneEncoderConfig = field(default_factory=SceneEncoderConfig)
    point_encoder: PointEncoderConfig = field(default_factory=PointEncoderConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        data = data or {}
        return cls(
            scene_encoder=SceneEncoderConfig.from_dict(data.get("scene_encoder")),
            point_encoder=PointEncoderConfig.from_dict(data.get("point_encoder")),
            cross_attention=CrossAttentionConfig.from_dict(data.get("cross_attention")),
            fusion=FusionConfig.from_dict(data.get("fusion")),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OptimizerConfig":
        data = data or {}
        return cls(
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "cosine_warm_restarts"
    min_learning_rate: float = 1e-5
    restart_period: int = 50
    warmup_epochs: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SchedulerConfig":
        data = data or {}
        return cls(
            type=str(data.get("type", "cosine_warm_restarts")),
            min_learning_rate=float(data.get("min_learning_rate", 1e-5)),
            restart_period=int(data.get("restart_period", 50)),
            warmup_epochs=int(data.get("warmup_epochs", 0)),
        )


@dataclass(frozen=True)
class LossConfig:
    translation_weight: float = 1.0
    rotation_weight: float = 0.5
    bbox_corner_weight: float = 2.0
    confidence_weight: float = 0.1
    confidence_temperature: float = 1.0
    translation_axis_weights: list[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LossConfig":
        data = data or {}
        axis_weights = data.get("translation_axis_weights", [10.0, 10.0, 1.0])
        return cls(
            translation_weight=float(data.get("translation_weight", 1.0)),
            rotation_weight=float(data.get("rotation_weight", 0.5)),
            bbox_corner_weight=float(data.get("bbox_corner_weight", 2.0)),
            confidence_weight=float(data.get("confidence_weight", 0.1)),
            confidence_temperature=float(data.get("confidence_temperature", 1.0)),
            translation_axis_weights=[float(w) for w in axis_weights],
        )


@dataclass(frozen=True)
class MonitoringConfig:
    log_every_n_batches: int = 10
    tensorboard: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MonitoringConfig":
        data = data or {}
        return cls(
            log_every_n_batches=int(data.get("log_every_n_batches", 10)),
            tensorboard=bool(data.get("tensorboard", True)),
        )


@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: Path
    train_split: str = "train"
    val_split: str = "val"
    storage: str = "ram"
    cache_file: Path | None = None
    num_points: int = 0
    depth_max_m: float = 5.0
    # "random": uniform subset per __getitem__. "fps": farthest-point greedy subsampling.
    model_points_sampling: str = "random"
    # When FPS is used and raw N exceeds this cap, uniformly pre-subsample (then FPS → num_points).
    model_points_fps_cap: int = 8192
    # Key in each sample `.npz` for the object mask: [H,W] or [1,H,W], same grid as depth.
    mask_npz_key: str = "object_mask"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DatasetConfig":
        data = data or {}
        dataset_dir = _coerce_path(data.get("dataset_dir", "data/generated/dataset_v2"))
        storage = str(data.get("storage", "ram")).lower()
        if storage not in {"ram", "disk"}:
            raise ValueError("dataset.storage must be either 'ram' or 'disk'")
        sampling = str(data.get("model_points_sampling", "random")).lower()
        if sampling not in {"random", "fps"}:
            raise ValueError("dataset.model_points_sampling must be 'random' or 'fps'")
        return cls(
            dataset_dir=dataset_dir,
            train_split=str(data.get("train_split", "train")),
            val_split=str(data.get("val_split", "val")),
            storage=storage,
            cache_file=None if data.get("cache_file") in (None, "") else _coerce_path(data.get("cache_file")),
            num_points=int(data.get("num_points", 0)),
            depth_max_m=float(data.get("depth_max_m", 5.0)),
            model_points_sampling=sampling,
            model_points_fps_cap=int(data.get("model_points_fps_cap", 8192)),
            mask_npz_key=str(data.get("mask_npz_key", "object_mask")),
        )


@dataclass(frozen=True)
class TrainingConfig:
    run_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(dataset_dir=Path("data/generated/dataset_v2")))
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    max_epochs: int = 1
    seed: int = 0
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    grad_clip_norm: float = 1.0
    resume_from: Path | None = None
    resume_best: bool = False
    resume_latest: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        resume_from = data.get("resume_from")
        return cls(
            run_dir=_coerce_path(data.get("run_dir", "runs/default")),
            model=ModelConfig.from_dict(data.get("model")),
            dataset=DatasetConfig.from_dict(data.get("dataset")),
            segmentation=SegmentationConfig.from_dict(data.get("segmentation")),
            optimizer=OptimizerConfig.from_dict(data.get("optimizer")),
            scheduler=SchedulerConfig.from_dict(data.get("scheduler")),
            loss=LossConfig.from_dict(data.get("loss")),
            monitoring=MonitoringConfig.from_dict(data.get("monitoring")),
            max_epochs=int(data.get("max_epochs", 1)),
            seed=int(data.get("seed", 0)),
            device=str(data.get("device", "cuda")),
            batch_size=int(data.get("batch_size", 8)),
            num_workers=int(data.get("num_workers", 0)),
            pin_memory=bool(data.get("pin_memory", True)),
            grad_clip_norm=float(data.get("grad_clip_norm", 1.0)),
            resume_from=None if resume_from in (None, "") else _coerce_path(resume_from),
            resume_best=bool(data.get("resume_best", False)),
            resume_latest=bool(data.get("resume_latest", False)),
        )