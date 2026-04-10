"""Plane-constrained placement helpers for object compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PlacementConstraints:
    min_plane_distance_m: float
    max_plane_distance_m: float


def is_camera_inside_aabb(camera_xyz: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    """Return True when camera point lies inside axis-aligned bounding box."""
    return bool(np.all(camera_xyz >= aabb_min) and np.all(camera_xyz <= aabb_max))


def center_projects_inside_fov(center_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> bool:
    """Check if 3D center projects into image bounds."""
    z = float(center_cam[2])
    if z <= 1e-6:
        return False

    u = fx * float(center_cam[0]) / z + cx
    v = fy * float(center_cam[1]) / z + cy
    return 0.0 <= u < float(width) and 0.0 <= v < float(height)


def sample_plane_offset_distance(rng: np.random.Generator, constraints: PlacementConstraints) -> float:
    """Sample object offset distance in front of fitted plane."""
    return float(rng.uniform(constraints.min_plane_distance_m, constraints.max_plane_distance_m))
