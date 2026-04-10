"""Factory for camera backends."""

from __future__ import annotations

from typing import Any, Dict

from .base import DepthCameraBackend
from .opencv_backend import OpenCVDepthBackend
from .realsense_backend import RealSenseDepthBackend
from .synthetic_perlin_backend import SyntheticPerlinDepthBackend


def create_camera_backend(config: Dict[str, Any]) -> DepthCameraBackend:
    """Create camera backend from config dictionary."""
    backend = str(config.get("backend", "realsense")).strip().lower()

    if backend == "realsense":
        return RealSenseDepthBackend(
            width=int(config.get("width", 640)),
            height=int(config.get("height", 480)),
            fps=int(config.get("fps", 30)),
            enable_color=bool(config.get("enable_color", False)),
        )

    if backend == "opencv":
        source = config.get("source", 0)
        return OpenCVDepthBackend(
            source=source,
            width=config.get("width"),
            height=config.get("height"),
            fps=config.get("fps"),
            depth_scale=float(config.get("depth_scale", 1.0)),
        )

    if backend == "synthetic_perlin":
        return SyntheticPerlinDepthBackend(
            width=int(config.get("width", 640)),
            height=int(config.get("height", 480)),
            fps=int(config.get("fps", 30)),
            seed=int(config.get("seed", 1234)),
            base_depth_m=float(config.get("base_depth_m", 2.0)),
            amplitude_m=float(config.get("amplitude_m", 0.15)),
            noise_scale=float(config.get("noise_scale", 2.5)),
            octaves=int(config.get("octaves", 4)),
            temporal_speed=float(config.get("temporal_speed", 0.35)),
            dropout_prob=float(config.get("dropout_prob", 0.0)),
        )

    raise ValueError(f"Unsupported camera backend: {backend}")
