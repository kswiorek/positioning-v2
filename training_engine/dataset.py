"""Dataset loaders for the v2 training loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, TrainingConfig


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    sample_seed: int
    depth_npz: Path
    gt_transform_camera_from_object: np.ndarray
    bbox_corners_m: np.ndarray
    object_source: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_npz_path(split_dir: Path, row: dict[str, Any]) -> Path:
    raw_path = row.get("depth_npz")
    if raw_path:
        candidate = Path(str(raw_path))
        if candidate.is_absolute() and candidate.exists():
            return candidate

        search_paths = [
            candidate,
            split_dir / candidate,
            split_dir.parent / candidate,
            split_dir / "samples" / candidate.name,
        ]
        for search_path in search_paths:
            if search_path.exists():
                return search_path

    fallback = split_dir / "samples" / f"{row['sample_id']}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not resolve sample file for {row.get('sample_id')!r}")


def load_split_records(split_dir: Path) -> list[SampleRecord]:
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    records: list[SampleRecord] = []
    for row in _read_jsonl(metadata_path):
        bbox_corners = row.get("bbox_corners_m")
        if bbox_corners is None:
            raise KeyError(f"Sample {row.get('sample_id')} is missing bbox_corners_m")

        records.append(
            SampleRecord(
                sample_id=str(row["sample_id"]),
                split=str(row.get("split", split_dir.name)),
                sample_seed=int(row.get("sample_seed", 0)),
                depth_npz=_resolve_npz_path(split_dir, row),
                gt_transform_camera_from_object=np.asarray(
                    row["gt_transform_camera_from_object"], dtype=np.float32
                ),
                bbox_corners_m=np.asarray(bbox_corners, dtype=np.float32),
                object_source=str(row.get("object_source", "unknown")),
            )
        )
    return records


def _sample_model_points(model_points: np.ndarray, sample_seed: int, num_points: int) -> np.ndarray:
    if num_points <= 0 or len(model_points) == num_points:
        return model_points.astype(np.float32, copy=False)

    rng = np.random.default_rng(sample_seed)
    replace = len(model_points) < num_points
    indices = rng.choice(len(model_points), size=num_points, replace=replace)
    return model_points[indices].astype(np.float32, copy=False)


class PoseDataset(Dataset):
    """Load one split of the v2 dataset into training-ready tensors."""

    def __init__(self, split_dir: Path, dataset_cfg: DatasetConfig):
        self.split_dir = Path(split_dir)
        self.dataset_cfg = dataset_cfg
        self.records = load_split_records(self.split_dir)
        self.storage = dataset_cfg.storage
        self._depths: np.ndarray | None = None
        self._model_points: np.ndarray | None = None

        if self.storage == "ram":
            self._preload_split()

    def _preload_split(self) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        n_samples = len(self.records)
        
        for i, record in enumerate(tqdm(self.records, desc=f"Loading {self.split_dir.name} into RAM")):
            with np.load(record.depth_npz, allow_pickle=False) as sample:
                depth_key = None
                for candidate in ("composite_depth_m", "depth_image", "depth_m"):
                    if candidate in sample.files:
                        depth_key = candidate
                        break
                if depth_key is None:
                    raise KeyError(f"Sample {record.sample_id} does not contain a depth array")

                depth = np.asarray(sample[depth_key], dtype=np.float32)
                if self.dataset_cfg.depth_max_m > 0:
                    depth = np.clip(depth / self.dataset_cfg.depth_max_m, 0.0, 1.0)
                depth = depth[None, ...] # (1, H, W)

                if "model_points" not in sample.files:
                    raise KeyError(f"Sample {record.sample_id} does not contain model_points")
                model_points = _sample_model_points(
                    np.asarray(sample["model_points"], dtype=np.float32),
                    sample_seed=record.sample_seed,
                    num_points=self.dataset_cfg.num_points,
                )
                
                # Preallocate on first iteration
                if self._depths is None or self._model_points is None:
                    self._depths = np.empty((n_samples, *depth.shape), dtype=np.float32)
                    self._model_points = np.empty((n_samples, *model_points.shape), dtype=np.float32)
                
                self._depths[i] = depth
                self._model_points[i] = model_points

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        if self.storage == "ram":
            assert self._depths is not None and self._model_points is not None
            depth = self._depths[index]
            model_points = self._model_points[index]
        else:
            with np.load(record.depth_npz, allow_pickle=False) as sample:
                depth_key = None
                for candidate in ("composite_depth_m", "depth_image", "depth_m"):
                    if candidate in sample.files:
                        depth_key = candidate
                        break
                if depth_key is None:
                    raise KeyError(f"Sample {record.sample_id} does not contain a depth array")

                depth = np.asarray(sample[depth_key], dtype=np.float32)
                if self.dataset_cfg.depth_max_m > 0:
                    depth = np.clip(depth / self.dataset_cfg.depth_max_m, 0.0, 1.0)
                depth = depth[None, ...]

                if "model_points" not in sample.files:
                    raise KeyError(f"Sample {record.sample_id} does not contain model_points")
                model_points = _sample_model_points(
                    np.asarray(sample["model_points"], dtype=np.float32),
                    sample_seed=record.sample_seed,
                    num_points=self.dataset_cfg.num_points,
                )

        return {
            "depth": torch.from_numpy(depth),
            "model_points": torch.from_numpy(model_points),
            "gt_transform": torch.from_numpy(record.gt_transform_camera_from_object),
            "bbox_corners": torch.from_numpy(record.bbox_corners_m),
            "sample_id": record.sample_id,
            "split": record.split,
            "object_source": record.object_source,
        }


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders from the generated dataset splits."""
    train_dataset = PoseDataset(config.dataset.dataset_dir / config.dataset.train_split, config.dataset)
    val_dataset = PoseDataset(config.dataset.dataset_dir / config.dataset.val_split, config.dataset)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader