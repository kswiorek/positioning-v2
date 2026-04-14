"""Plane fitting utilities for captured background depth images."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from data_engine.geometry.camera import intrinsics_from_camera_config


@dataclass
class PlaneModel:
    """Plane model in camera frame: n.x + d = 0."""

    normal: np.ndarray
    offset: float
    inlier_ratio: float


def select_plane_support_mask(depth_m: np.ndarray, middle_percentile: float = 0.90) -> np.ndarray:
    """Select robust far-depth support mask for plane estimation."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-6)
    valid_depth = depth[valid]
    if valid_depth.size == 0:
        return np.zeros_like(valid, dtype=bool)

    mid = np.clip(float(middle_percentile), 0.5, 0.98)
    p_low = 100.0 * mid
    z_low = float(np.percentile(valid_depth, p_low))
    z_high = float(np.percentile(valid_depth, 99.0))
    far_band = valid & (depth >= z_low) & (depth <= z_high)

    if not np.any(far_band):
        return np.zeros_like(valid, dtype=bool)

    return _largest_connected_component(far_band)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return mask of the largest 8-connected component."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels: list[tuple[int, int]] = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue

        q = deque([(int(y0), int(x0))])
        visited[y0, x0] = True
        component: list[tuple[int, int]] = []

        while q:
            y, x = q.popleft()
            component.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))

        if len(component) > len(best_pixels):
            best_pixels = component

    out = np.zeros_like(mask, dtype=bool)
    for y, x in best_pixels:
        out[y, x] = True
    return out


def fit_plane_from_depth(
    depth_m: np.ndarray,
    camera_cfg: dict,
    stride: int = 2,
    seed: int = 0,
    middle_percentile: float = 0.90,
    allow_tilt: bool = True,
) -> PlaneModel:
    """Fit a frontal reference plane using far-depth connected-component heuristics.

    The plane normal is fixed to camera axis (0, 0, -1), and depth is estimated
    robustly from the far-depth band and its largest connected component.
    """
    _ = seed  # deterministic heuristic, seed kept for compatibility
    _ = stride  # full-resolution mask-based estimation is used
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)

    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-6)
    valid_depth = depth[valid]

    frontal_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    if valid_depth.size == 0:
        return PlaneModel(normal=frontal_normal, offset=1.5, inlier_ratio=0.0)

    support = select_plane_support_mask(depth, middle_percentile=middle_percentile)
    if not np.any(support):
        z_ref = float(np.percentile(valid_depth, 90.0))
        return PlaneModel(normal=frontal_normal, offset=z_ref, inlier_ratio=0.0)

    yy, xx = np.where(support)
    z = depth[yy, xx]
    x = (xx.astype(np.float64) - cx) * z / fx
    y = (yy.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    inlier_ratio = float(np.count_nonzero(support)) / float(np.count_nonzero(valid))
    if pts.shape[0] < 3:
        z_ref = float(np.median(z)) if z.size > 0 else float(np.percentile(valid_depth, 90.0))
        return PlaneModel(normal=frontal_normal, offset=z_ref, inlier_ratio=inlier_ratio)

    if not allow_tilt:
        z_ref = float(np.median(z))
        return PlaneModel(normal=frontal_normal, offset=z_ref, inlier_ratio=inlier_ratio)

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = vh[-1]
    n = n / max(np.linalg.norm(n), 1e-12)

    # Orient normal toward camera.
    to_camera = -centroid
    if float(np.dot(n, to_camera)) < 0.0:
        n = -n

    d = -float(np.dot(n, centroid))
    return PlaneModel(normal=n.astype(np.float64), offset=d, inlier_ratio=inlier_ratio)
