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
    middle_percentile: float = 0.90,
) -> PlaneModel:
    """Fit dominant background plane using a trimmed robust estimator.

    RANSAC parameters are kept in the signature for backward compatibility,
    but the default path uses percentile trimming + SVD plane fit.
    """
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    points = depth_to_camera_points(depth_m, fx, fy, cx, cy, stride=stride)

    # Fallback for sparse/invalid frames: fronto-parallel plane at median depth.
    if points.shape[0] < 3:
        valid_depth = np.asarray(depth_m, dtype=np.float64)
        valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 1e-6)]
        if valid_depth.size == 0:
            # Safe default if frame has no valid depth at all.
            return PlaneModel(
                normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                offset=1.5,
                inlier_ratio=0.0,
            )

        z_med = float(np.median(valid_depth))
        return PlaneModel(
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            offset=z_med,
            inlier_ratio=1.0,
        )

    mid = np.clip(float(middle_percentile), 0.1, 0.999)
    lo_p = 50.0 * (1.0 - mid)
    hi_p = 100.0 - lo_p

    z = points[:, 2]
    z_lo = float(np.percentile(z, lo_p))
    z_hi = float(np.percentile(z, hi_p))
    keep = (z >= z_lo) & (z <= z_hi)
    trimmed = points[keep]
    if trimmed.shape[0] < 3:
        trimmed = points

    centroid = trimmed.mean(axis=0)
    centered = trimmed - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / max(np.linalg.norm(n), 1e-12)

    # Orient normal to face camera (origin).
    to_camera = -centroid
    if float(np.dot(n, to_camera)) < 0.0:
        n = -n

    d = -float(np.dot(n, centroid))
    inlier_ratio = float(trimmed.shape[0]) / float(points.shape[0])
    return PlaneModel(normal=n.astype(np.float64), offset=d, inlier_ratio=inlier_ratio)
