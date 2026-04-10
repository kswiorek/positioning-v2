"""Placement sampling constrained by fitted plane and camera geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from data_engine.composition.plane_fit import PlaneModel
from data_engine.composition.plane_placement import (
    PlacementConstraints,
    center_projects_inside_fov,
    is_camera_inside_aabb,
    sample_plane_offset_distance,
)
from data_engine.geometry.camera import intrinsics_from_camera_config


@dataclass
class PlacementSample:
    position_xyz: np.ndarray
    orientation_euler_deg_xyz: np.ndarray
    orientation_quat_xyzw: np.ndarray
    plane_offset_m: float
    center_pixel_uv: np.ndarray


def _ray_from_pixel(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def _intersect_ray_plane(ray_dir: np.ndarray, plane: PlaneModel) -> np.ndarray | None:
    denom = float(np.dot(plane.normal, ray_dir))
    if abs(denom) < 1e-9:
        return None
    t = -float(plane.offset) / denom
    if t <= 1e-6:
        return None
    return ray_dir * t


def sample_pose_on_plane(
    plane: PlaneModel,
    camera_cfg: dict,
    bbox_extent_m: np.ndarray,
    constraints: PlacementConstraints,
    rng: np.random.Generator,
    max_tries: int = 400,
) -> PlacementSample:
    """Sample a valid object pose that satisfies camera/FOV/plane constraints."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)
    bbox_extent_m = np.asarray(bbox_extent_m, dtype=np.float64)

    for _ in range(max_tries):
        u = float(rng.uniform(0.0, width - 1.0))
        v = float(rng.uniform(0.0, height - 1.0))
        ray = _ray_from_pixel(u, v, fx, fy, cx, cy)

        hit = _intersect_ray_plane(ray, plane)
        if hit is None:
            continue

        offset_m = sample_plane_offset_distance(rng, constraints)
        center = hit + plane.normal * offset_m

        if not center_projects_inside_fov(center, fx, fy, cx, cy, width, height):
            continue

        euler = np.array([
            rng.uniform(-180.0, 180.0),
            rng.uniform(-180.0, 180.0),
            rng.uniform(-180.0, 180.0),
        ])
        rot = Rotation.from_euler("xyz", euler, degrees=True)
        rot_mat = rot.as_matrix()

        # Check if camera origin is inside oriented bbox in object local frame.
        camera_local = rot_mat.T @ (-center)
        if is_camera_inside_aabb(
            camera_local,
            -0.5 * bbox_extent_m,
            0.5 * bbox_extent_m,
        ):
            continue

        quat = rot.as_quat()  # [x, y, z, w]
        return PlacementSample(
            position_xyz=center,
            orientation_euler_deg_xyz=euler,
            orientation_quat_xyzw=quat,
            plane_offset_m=offset_m,
            center_pixel_uv=np.array([u, v], dtype=np.float64),
        )

    raise RuntimeError("Failed to sample a valid placement under current constraints.")
