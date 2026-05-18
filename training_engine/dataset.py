"""Dataset loaders for the v2 training loop."""

from __future__ import annotations

import hashlib
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
    object_name: str


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


def _load_object_mask_array(
    sample: np.lib.npyio.NpzFile, mask_key: str, sample_id: str, depth_shape: tuple[int, ...]
) -> np.ndarray:
    """Return float32 mask [1, H, W] in [0, 1], same shape as normalized depth."""
    if mask_key not in sample.files:
        raise KeyError(f"Sample {sample_id} is missing mask array {mask_key!r}")
    m = np.asarray(sample[mask_key], dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3:
        if m.shape[0] != 1:
            raise ValueError(f"Sample {sample_id}: mask must be [H,W] or [1,H,W], got {m.shape}")
    else:
        raise ValueError(f"Sample {sample_id}: mask must be 2d or 3d, got {m.shape}")
    if tuple(m.shape) != tuple(depth_shape):
        raise ValueError(f"Sample {sample_id}: mask shape {m.shape} must match depth {depth_shape}")
    return np.clip(m, 0.0, 1.0)


def _ram_split_cache_valid(arc: Any, n_records: int) -> bool:
    """Packed RAM cache must contain depths, scene_masks (same shape as depths), and model points."""
    required = ("depths", "scene_masks", "model_points_packed", "model_point_offsets")
    if not all(k in arc.files for k in required):
        return False
    depths = arc["depths"]
    if int(depths.shape[0]) != n_records:
        return False
    if tuple(arc["scene_masks"].shape) != tuple(depths.shape):
        return False
    offsets = np.asarray(arc["model_point_offsets"], dtype=np.int64)
    if offsets.shape != (n_records + 1,):
        return False
    packed = arc["model_points_packed"]
    if int(offsets[-1]) != int(packed.shape[0]):
        return False
    return True


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
                object_name=str(
                    row.get("object_id")
                    or Path(str(row.get("object_asset_path", ""))).stem
                    or row.get("object_source", "unknown")
                ),
            )
        )
    return records


def _mix_seed_ints(*values: int) -> int:
    h = hashlib.sha256()
    for v in values:
        h.update(int(v).to_bytes(8, byteorder="little", signed=False))
    return int.from_bytes(h.digest()[:8], byteorder="little", signed=False) & 0x7FFFFFFF


def _rng_deterministic(training_seed: int, object_name: str, sample_id: str) -> np.random.Generator:
    h = hashlib.sha256()
    h.update(object_name.encode("utf-8"))
    h.update(b"\0")
    h.update(sample_id.encode("utf-8"))
    h.update(b"\0")
    h.update(int(training_seed).to_bytes(8, byteorder="little", signed=False))
    seed = int.from_bytes(h.digest()[:8], byteorder="little", signed=False) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def _rng_training(training_seed: int, epoch: int, index: int, sample_seed: int) -> np.random.Generator:
    seed = _mix_seed_ints(training_seed, epoch, index, sample_seed)
    return np.random.default_rng(seed)


def _farthest_point_indices(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy FPS: returns `num_points` indices into `points` (shape [N, 3])."""
    n = int(points.shape[0])
    if num_points <= 0:
        raise ValueError("num_points must be > 0 for FPS")
    if n <= num_points:
        return np.asarray(rng.choice(n, size=num_points, replace=n < num_points), dtype=np.int64)

    min_sq = np.empty(n, dtype=np.float64)
    i0 = int(rng.integers(0, n))
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = i0
    delta = points - points[i0]
    np.sum(delta * delta, axis=1, out=min_sq)

    d_new = np.empty(n, dtype=np.float64)
    for t in range(1, num_points):
        farthest = int(np.argmax(min_sq))
        selected[t] = farthest
        delta = points - points[farthest]
        np.sum(delta * delta, axis=1, out=d_new)
        np.minimum(min_sq, d_new, out=min_sq)
    return selected


def _subsample_model_points(
    points: np.ndarray,
    num_points: int,
    sampling: str,
    fps_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return `num_points` rows from `points` [N,3] using random or FPS selection."""
    pts = np.asarray(points, dtype=np.float32, copy=False)
    n = pts.shape[0]
    if num_points <= 0 or n == num_points:
        return pts.astype(np.float32, copy=False)

    remap = np.arange(n, dtype=np.int64)
    working = pts
    if sampling == "fps" and n > fps_cap:
        remap = rng.choice(n, size=fps_cap, replace=False)
        working = pts[remap]

    n_w = working.shape[0]
    if sampling == "fps":
        ix_local = _farthest_point_indices(working, num_points, rng)
    else:
        replace = n_w < num_points
        ix_local = np.asarray(rng.choice(n_w, size=num_points, replace=replace), dtype=np.int64)

    return np.asarray(pts[remap[ix_local]], dtype=np.float32, copy=False)


def _cache_paths_for_split(configured: Path, split_dir: Path) -> tuple[Path, list[Path]]:
    """Return `(preferred_write_path, read_paths_in_order)` for split-aware cache files.

    Relative config paths resolve under ``split_dir`` (keeps train/val caches separate).

    Absolute paths use a split suffix for the preferred file so train/val do not overwrite
    each other. The unsuffixed legacy path is tried on read so older single-file caches
    still load when the sample count matches this split.
    """
    split_dir = Path(split_dir)
    configured = Path(configured)

    if not configured.is_absolute():
        primary = split_dir / configured
        return primary, [primary]

    stem, suf = configured.stem, configured.suffix
    token = split_dir.name.replace(" ", "_")
    suffixed = configured.with_name(f"{stem}_{token}{suf}")
    if suffixed.resolve() == configured.resolve():
        return suffixed, [suffixed]
    return suffixed, [suffixed, configured]


class PoseDataset(Dataset):
    """Load one split of the v2 dataset into training-ready tensors.

    Model points are stored at full resolution in RAM (packed array + offsets). Each
    ``__getitem__`` subsamples to ``num_points``; training uses a different random/FPS
    subset each epoch (see ``set_epoch``). Validation uses a deterministic subset per sample.
    """

    def __init__(
        self,
        split_dir: Path,
        dataset_cfg: DatasetConfig,
        *,
        random_model_points: bool = False,
        training_seed: int = 0,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.dataset_cfg = dataset_cfg
        self.records = load_split_records(self.split_dir)
        self.storage = dataset_cfg.storage
        self.random_model_points = random_model_points
        self._training_seed = int(training_seed)
        self._epoch = 0

        self._depths: np.ndarray | None = None
        self._scene_masks: np.ndarray | None = None
        self._model_points_packed: np.ndarray | None = None
        self._model_point_offsets: np.ndarray | None = None
        self._cache_path: Path | None = None

        preferred_cache_path: Path | None = None
        cache_read_paths: list[Path] = []
        if getattr(self.dataset_cfg, "cache_file", None):
            configured = Path(self.dataset_cfg.cache_file)
            preferred_cache_path, cache_read_paths = _cache_paths_for_split(configured, self.split_dir)

        if self.storage == "ram":
            n_records = len(self.records)
            loaded = False

            if cache_read_paths:
                for cand in cache_read_paths:
                    if not cand.exists():
                        continue
                    try:
                        with np.load(cand, allow_pickle=False) as arc:
                            if not _ram_split_cache_valid(arc, n_records):
                                continue
                            self._depths = np.asarray(arc["depths"], dtype=np.float32, copy=False)
                            self._scene_masks = np.asarray(arc["scene_masks"], dtype=np.float32, copy=False)
                            self._model_points_packed = np.asarray(
                                arc["model_points_packed"], dtype=np.float32, copy=False
                            )
                            self._model_point_offsets = np.asarray(
                                arc["model_point_offsets"], dtype=np.int64, copy=False
                            )
                            self._cache_path = cand
                            loaded = True
                            break
                    except Exception as exc:
                        try:
                            print(f"Warning: could not read dataset cache {cand}: {exc!r}", flush=True)
                        except Exception:
                            pass

                if not loaded and preferred_cache_path is not None:
                    # Expected file missing vs present-but-invalid helps debugging.
                    if not any(p.exists() for p in cache_read_paths):
                        try:
                            print(
                                "Note: dataset cache not found yet (will rebuild). "
                                f"Tried: {', '.join(str(p) for p in cache_read_paths)}",
                                flush=True,
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            print(
                                "Warning: dataset cache files exist but none match this split "
                                f"(samples={n_records}, expected depths+scene_masks+model_points). "
                                "Rebuilding from .npz sources.",
                                flush=True,
                            )
                        except Exception:
                            pass

            if not loaded:
                self._preload_split()
                self._cache_path = preferred_cache_path
                self._save_cache()

    def set_epoch(self, epoch: int) -> None:
        """Training only: changes RNG stream for per-item model point subsampling."""
        self._epoch = int(epoch)

    def _preload_split(self) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kwargs: x

        n_samples = len(self.records)
        packed_chunks: list[np.ndarray] = []
        offsets = np.zeros(n_samples + 1, dtype=np.int64)

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
                depth = depth[None, ...]

                if "model_points" not in sample.files:
                    raise KeyError(f"Sample {record.sample_id} does not contain model_points")
                model_points = np.asarray(sample["model_points"], dtype=np.float32)
                if model_points.ndim != 2 or model_points.shape[1] != 3:
                    raise ValueError(
                        f"Sample {record.sample_id}: model_points must be [N, 3], got {model_points.shape}"
                    )

                if self._depths is None:
                    self._depths = np.empty((n_samples, *depth.shape), dtype=np.float32)
                    self._scene_masks = np.empty((n_samples, *depth.shape), dtype=np.float32)

                mask = _load_object_mask_array(
                    sample, self.dataset_cfg.mask_npz_key, record.sample_id, depth.shape
                )
                self._depths[i] = depth
                self._scene_masks[i] = mask
                packed_chunks.append(model_points)
                offsets[i + 1] = offsets[i] + model_points.shape[0]

        self._model_points_packed = (
            np.concatenate(packed_chunks, axis=0) if packed_chunks else np.zeros((0, 3), dtype=np.float32)
        )
        self._model_point_offsets = offsets

    def _save_cache(self) -> None:
        if self._cache_path is None or self._depths is None:
            return
        if self._model_points_packed is None or self._model_point_offsets is None:
            return
        if self._scene_masks is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(self._cache_path),
                depths=self._depths,
                scene_masks=self._scene_masks,
                model_points_packed=self._model_points_packed,
                model_point_offsets=self._model_point_offsets,
            )
        except Exception:
            try:
                print(f"Warning: failed to write dataset cache to {self._cache_path}")
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.records)

    def _rng_for_item(self, record: SampleRecord, index: int) -> np.random.Generator:
        if self.random_model_points:
            return _rng_training(self._training_seed, self._epoch, index, record.sample_seed)
        return _rng_deterministic(self._training_seed, record.object_name, record.sample_id)

    def _model_points_from_ram(self, index: int) -> np.ndarray:
        assert self._model_points_packed is not None and self._model_point_offsets is not None
        s = int(self._model_point_offsets[index])
        e = int(self._model_point_offsets[index + 1])
        return self._model_points_packed[s:e]

    def _live_model_points(self, index: int) -> np.ndarray:
        record = self.records[index]
        rng = self._rng_for_item(record, index)
        if self.storage == "ram":
            full = self._model_points_from_ram(index)
        else:
            with np.load(record.depth_npz, allow_pickle=False) as sample:
                if "model_points" not in sample.files:
                    raise KeyError(f"Sample {record.sample_id} does not contain model_points")
                full = np.asarray(sample["model_points"], dtype=np.float32)

        return _subsample_model_points(
            full,
            num_points=self.dataset_cfg.num_points,
            sampling=self.dataset_cfg.model_points_sampling,
            fps_cap=self.dataset_cfg.model_points_fps_cap,
            rng=rng,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        if self.storage == "ram":
            assert self._depths is not None and self._scene_masks is not None
            depth = self._depths[index]
            scene_mask = self._scene_masks[index]
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
                scene_mask = _load_object_mask_array(
                    sample, self.dataset_cfg.mask_npz_key, record.sample_id, depth.shape
                )

        model_points = self._live_model_points(index)

        return {
            "depth": torch.from_numpy(depth),
            "scene_mask": torch.from_numpy(scene_mask),
            "model_points": torch.from_numpy(model_points),
            "gt_transform": torch.from_numpy(record.gt_transform_camera_from_object),
            "bbox_corners": torch.from_numpy(record.bbox_corners_m),
            "sample_id": record.sample_id,
            "split": record.split,
            "object_source": record.object_source,
        }


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders from the generated dataset splits."""
    train_dataset = PoseDataset(
        config.dataset.dataset_dir / config.dataset.train_split,
        config.dataset,
        random_model_points=True,
        training_seed=config.seed,
    )
    val_dataset = PoseDataset(
        config.dataset.dataset_dir / config.dataset.val_split,
        config.dataset,
        random_model_points=False,
        training_seed=config.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    from torch.utils.data import RandomSampler

    train_sampler = RandomSampler(train_dataset, replacement=False, generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
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
