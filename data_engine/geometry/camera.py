"""Camera geometry helpers used by dataset generation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def intrinsics_from_camera_config(camera_cfg: Dict) -> Tuple[float, float, float, float, int, int]:
    """Return fx, fy, cx, cy, width, height from camera config.

    Supports either explicit intrinsics or FOV-based derivation.
    """
    width = int(camera_cfg["resolution"]["width"])
    height = int(camera_cfg["resolution"]["height"])

    intr = camera_cfg.get("intrinsics")
    if intr is not None:
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr.get("cx", width / 2.0))
        cy = float(intr.get("cy", height / 2.0))
        return fx, fy, cx, cy, width, height

    fov = camera_cfg.get("fov", {})
    fov_h = np.radians(float(fov["horizontal"]))
    fov_v = np.radians(float(fov["vertical"]))
    fx = width / (2.0 * np.tan(fov_h / 2.0))
    fy = height / (2.0 * np.tan(fov_v / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return float(fx), float(fy), float(cx), float(cy), width, height


def depth_to_camera_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 2,
) -> np.ndarray:
    """Back-project depth image into camera-frame 3D points."""
    h, w = depth_m.shape
    v = np.arange(0, h, stride)
    u = np.arange(0, w, stride)
    uu, vv = np.meshgrid(u, v)

    z = depth_m[vv, uu].astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-6)

    uu = uu[valid].astype(np.float64)
    vv = vv[valid].astype(np.float64)
    z = z[valid]

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def camera_points_to_depth(
    points_xyz: np.ndarray,
    camera_cfg: Dict,
    min_depth_m: float = 1e-6,
    splat_radius_px: int = 0,
) -> np.ndarray:
    """Project camera-frame points to depth image using z-buffering."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)
    depth = np.zeros((height, width), dtype=np.float32)

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return depth

    z = pts[:, 2]
    valid = np.isfinite(z) & (z > float(min_depth_m))
    if not np.any(valid):
        return depth

    pts = pts[valid]
    z = pts[:, 2]

    u = (fx * pts[:, 0] / z + cx)
    v = (fy * pts[:, 1] / z + cy)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        return depth

    ui = ui[inside]
    vi = vi[inside]
    z = z[inside].astype(np.float32)

    splat_radius_px = max(int(splat_radius_px), 0)

    if splat_radius_px == 0:
        for px, py, pz in zip(ui, vi, z):
            cur = depth[py, px]
            if cur <= 1e-6 or pz < cur:
                depth[py, px] = pz
        return depth

    for px, py, pz in zip(ui, vi, z):
        x0 = max(px - splat_radius_px, 0)
        x1 = min(px + splat_radius_px + 1, width)
        y0 = max(py - splat_radius_px, 0)
        y1 = min(py + splat_radius_px + 1, height)

        patch = depth[y0:y1, x0:x1]
        replace = (patch <= 1e-6) | (pz < patch)
        patch[replace] = pz

    return depth
