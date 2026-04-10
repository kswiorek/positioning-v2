"""Plane fitting utilities for captured background depth images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_engine.geometry.camera import depth_to_camera_points, intrinsics_from_camera_config


@dataclass
class PlaneModel:
    """Plane model in camera frame: n.x + d = 0."""

    normal: np.ndarray
    offset: float
    inlier_ratio: float


def _fit_plane_from_three_points(points: np.ndarray) -> tuple[np.ndarray, float] | tuple[None, None]:
    p0, p1, p2 = points
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None, None
    n = n / norm
    d = -float(np.dot(n, p0))
    return n, d


def fit_plane_ransac(
    points: np.ndarray,
    threshold_m: float = 0.01,
    max_iterations: int = 300,
    seed: int = 0,
) -> PlaneModel:
    """Robust plane fit using a simple RANSAC loop."""
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for plane fitting.")

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_n = None
    best_d = None

    for _ in range(max_iterations):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        n, d = _fit_plane_from_three_points(points[idx])
        if n is None:
            continue

        dist = np.abs(points @ n + d)
        inliers = dist < threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_n = n
            best_d = d

    if best_inliers is None or best_inliers.sum() < 3:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    # Refine with SVD on inliers.
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)

    # Orient normal to face the camera (origin).
    to_camera = -centroid
    if float(np.dot(n, to_camera)) < 0.0:
        n = -n

    d = -float(np.dot(n, centroid))
    inlier_ratio = float(best_inliers.mean())
    return PlaneModel(normal=n.astype(np.float64), offset=d, inlier_ratio=inlier_ratio)


def fit_plane_from_depth(
    depth_m: np.ndarray,
    camera_cfg: dict,
    stride: int = 2,
    threshold_m: float = 0.01,
    max_iterations: int = 300,
    seed: int = 0,
) -> PlaneModel:
    """Fit dominant background plane from depth image and camera config."""
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    points = depth_to_camera_points(depth_m, fx, fy, cx, cy, stride=stride)
    return fit_plane_ransac(
        points=points,
        threshold_m=threshold_m,
        max_iterations=max_iterations,
        seed=seed,
    )
