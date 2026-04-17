"""Data engine module for synthetic/real-composited dataset generation."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data_engine.camera import create_camera_backend
from data_engine.composition import PlacementConstraints, fit_plane_from_depth, sample_pose_on_plane
from data_engine.capture import BackgroundCaptureSession, CaptureConfig
from data_engine.generators import generate_mixed_canonical_model


@dataclass
class DataEngineConfig:
    output_dir: Path
    num_samples: int = 0
    seed: int = 0


class DataEngine:
    """Entry point for data generation workflows."""

    def __init__(self, config: DataEngineConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json_config(path: Path) -> Dict[str, Any]:
        """Load a JSON config file."""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def capture_backgrounds(
        self,
        camera_config: Dict[str, Any],
        capture_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Run an interactive background capture session."""
        capture_config = capture_config or {}

        camera = create_camera_backend(camera_config)
        cap_cfg = CaptureConfig(
            output_root=Path(capture_config.get("output_root", self.config.output_dir / "backgrounds" / "raw")),
            session_name=capture_config.get("session_name"),
            max_frames=int(capture_config.get("max_frames", 0)),
            preview_max_depth_m=float(capture_config.get("preview_max_depth_m", 5.0)),
            save_preview_png=bool(capture_config.get("save_preview_png", True)),
        )
        session = BackgroundCaptureSession(camera=camera, config=cap_cfg)
        return session.run()

    def build_dataset(
        self,
        scene_config: Dict[str, Any],
        dataset_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate dataset assets and metadata with multiprocessing.

        Dataset run authority (seed, counts/splits, output path, workers, IO flags)
        is taken from dataset_config, aligned with the CLI entrypoint.
        """
        dataset_cfg = dataset_config or {}

        background_glob = str(dataset_cfg.get("background_glob", "data/backgrounds/raw/**/depth/*.npz"))
        background_paths = sorted(str(p) for p in Path(".").glob(background_glob) if p.is_file())
        if not background_paths:
            raise RuntimeError(f"No background files matched: {background_glob}")

        from data_engine.generate_dataset import (
            _split_counts_from_config,
            build_dataset as build_dataset_bulk,
        )

        workers = int(dataset_cfg.get("workers", 0))
        if workers <= 0:
            workers = max((os.cpu_count() or 4) - 1, 1)

        samples_per_task = int(dataset_cfg.get("samples_per_task", 8))
        max_backgrounds_in_ram = int(dataset_cfg.get("max_backgrounds_in_ram", 0))
        if max_backgrounds_in_ram < 0:
            max_backgrounds_in_ram = 0

        object_source = str(dataset_cfg.get("object_source", "random"))
        debug_metadata = bool(dataset_cfg.get("debug_metadata", False))
        save_components = bool(dataset_cfg.get("save_components", False))
        compressed_npz = bool(dataset_cfg.get("compressed_npz", False))
        max_attempts_per_sample = max(int(dataset_cfg.get("max_attempts_per_sample", 20)), 1)
        base_seed = int(dataset_cfg.get("seed", self.config.seed))

        output_root = Path(dataset_cfg.get("output_dir", str(self.config.output_dir)))
        output_root.mkdir(parents=True, exist_ok=True)

        if "split" in dataset_cfg and "train_validation_split" not in dataset_cfg:
            split_name = str(dataset_cfg.get("split", "train"))
            split_count = int(dataset_cfg.get("num_samples", self.config.num_samples))
            if split_count <= 0:
                raise ValueError("dataset_config.num_samples must be > 0")
            split_counts = {split_name: split_count}
        else:
            split_counts = _split_counts_from_config(dataset_cfg)

        split_summaries: Dict[str, Any] = {}
        total_requested = 0
        total_success = 0
        total_failed = 0
        total_elapsed = 0.0

        for split_index, (split_name, split_n) in enumerate(split_counts.items()):
            if split_n <= 0:
                continue

            split_out = output_root / split_name
            split_seed = base_seed + split_index * 10000019

            summary = build_dataset_bulk(
                scene_cfg=scene_config,
                background_paths=background_paths,
                output_dir=split_out,
                num_samples=int(split_n),
                base_seed=int(split_seed),
                split=str(split_name),
                workers=int(workers),
                samples_per_task=int(samples_per_task),
                max_backgrounds_in_ram=int(max_backgrounds_in_ram),
                object_source=object_source,
                debug_metadata=bool(debug_metadata),
                save_components=bool(save_components),
                compressed_npz=bool(compressed_npz),
                max_attempts_per_sample=int(max_attempts_per_sample),
            )
            split_summaries[split_name] = summary

            total_requested += int(summary["num_requested"])
            total_success += int(summary["num_success"])
            total_failed += int(summary["num_failed"])
            total_elapsed += float(summary["elapsed_sec"])

        return {
            "num_requested": total_requested,
            "num_success": total_success,
            "num_failed": total_failed,
            "elapsed_sec_sum_splits": total_elapsed,
            "throughput_samples_per_sec": 0.0
            if total_elapsed <= 0
            else float(total_success / total_elapsed),
            "workers": workers,
            "backgrounds_available": len(background_paths),
            "max_backgrounds_in_ram": max_backgrounds_in_ram,
            "object_source": object_source,
            "debug_metadata": debug_metadata,
            "save_components": save_components,
            "compressed_npz": compressed_npz,
            "seed": base_seed,
            "train_validation_split": split_counts,
            "splits": split_summaries,
        }

    def fit_plane_and_sample_object(
        self,
        depth_m: np.ndarray,
        scene_config: Dict[str, Any],
        seed: Optional[int] = None,
        object_source: str = "random",
    ) -> Dict[str, Any]:
        """Fit a background plane and sample a valid object pose.

        This is a debug helper for the upcoming dataset pipeline.
        """
        seed_value = int(self.config.seed if seed is None else seed)

        plane_cfg = scene_config.get("plane_fit", {})
        plane = fit_plane_from_depth(
            depth_m=depth_m,
            camera_cfg=scene_config["camera"],
            stride=int(plane_cfg.get("stride", 2)),
            threshold_m=float(plane_cfg.get("ransac_threshold_m", 0.01)),
            max_iterations=int(plane_cfg.get("ransac_iterations", 300)),
            seed=seed_value,
        )

        _, _, bbox_corners, shape_params = generate_mixed_canonical_model(
            scene_config,
            seed=seed_value,
            source_override=object_source,
        )
        bbox_extent = (bbox_corners.max(axis=0) - bbox_corners.min(axis=0)).astype(np.float64)

        placement_cfg = scene_config["placement"]
        constraints = PlacementConstraints(
            min_plane_distance_m=float(placement_cfg["min_plane_distance_m"]),
            max_plane_distance_m=float(placement_cfg["max_plane_distance_m"]),
        )

        rng = np.random.default_rng(seed_value)
        placement = sample_pose_on_plane(
            plane=plane,
            camera_cfg=scene_config["camera"],
            bbox_extent_m=bbox_extent,
            constraints=constraints,
            rng=rng,
            max_tries=int(placement_cfg.get("max_attempts", 400)),
        )

        return {
            "seed": seed_value,
            "plane": {
                "normal": plane.normal.tolist(),
                "offset": float(plane.offset),
                "inlier_ratio": float(plane.inlier_ratio),
            },
            "shape_params": shape_params,
            "bbox_extent_m": bbox_extent.tolist(),
            "placement": {
                "position_xyz": placement.position_xyz.tolist(),
                "orientation_euler_deg_xyz": placement.orientation_euler_deg_xyz.tolist(),
                "orientation_quat_xyzw": placement.orientation_quat_xyzw.tolist(),
                "plane_offset_m": float(placement.plane_offset_m),
                "center_pixel_uv": placement.center_pixel_uv.tolist(),
            },
        }

    def fit_plane_and_sample_superquadric(
        self,
        depth_m: np.ndarray,
        scene_config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper that forces superquadric source."""
        return self.fit_plane_and_sample_object(
            depth_m=depth_m,
            scene_config=scene_config,
            seed=seed,
            object_source="superquadric",
        )
