"""Data engine module for synthetic/real-composited dataset generation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data_engine.camera import create_camera_backend
from data_engine.composition import PlacementConstraints, fit_plane_from_depth, sample_pose_on_plane
from data_engine.capture import BackgroundCaptureSession, CaptureConfig
from data_engine.generators import generate_superquadric_canonical_model


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

    def build_dataset(self) -> None:
        """Generate dataset assets and metadata."""
        raise NotImplementedError("Data generation is not implemented yet.")

    def fit_plane_and_sample_superquadric(
        self,
        depth_m: np.ndarray,
        scene_config: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit a background plane and sample a valid superquadric pose.

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

        _, _, bbox_corners, shape_params = generate_superquadric_canonical_model(
            scene_config, seed=seed_value
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
            max_tries=int(placement_cfg.get("max_tries", 400)),
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
