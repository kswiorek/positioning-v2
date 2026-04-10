"""Background plane normalization and re-randomization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from data_engine.composition.plane_fit import PlaneModel
from data_engine.geometry import camera_points_to_depth, depth_to_camera_points, intrinsics_from_camera_config


@dataclass
class BackgroundTransformParams:
    pitch_deg: float
    yaw_deg: float
    distance_m: float
    inplane_scale_xy: float = 1.0
    projected_fill_u: float = 0.0
    projected_fill_v: float = 0.0


def _rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return rotation matrix that aligns src direction to dst direction."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)

    c = float(np.dot(a, b))
    c = np.clip(c, -1.0, 1.0)

    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / max(s * s, 1e-12))


def _frustum_corners_intersection_spans_on_plane(
    camera_cfg: dict,
    r_rand: np.ndarray,
    distance_m: float,
) -> tuple[float, float] | tuple[None, None]:
    """Return required x/y spans on canonical plane from camera frustum intersection."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)

    # Randomized plane in camera frame: n_rand . x + d_rand = 0
    n_rand = r_rand @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_point = np.array([0.0, 0.0, distance_m], dtype=np.float64)
    d_rand = -float(np.dot(n_rand, plane_point))

    corners_uv = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )

    pts_norm = []
    for u, v in corners_uv:
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        ray /= max(np.linalg.norm(ray), 1e-12)

        denom = float(np.dot(n_rand, ray))
        if abs(denom) < 1e-9:
            return None, None
        t = -d_rand / denom
        if t <= 1e-6:
            return None, None

        p_cam = t * ray
        # Inverse randomized transform to canonical plane frame.
        p_norm = r_rand.T @ (p_cam - plane_point)
        pts_norm.append(p_norm)

    pts_norm = np.asarray(pts_norm, dtype=np.float64)
    span_x = float(np.max(pts_norm[:, 0]) - np.min(pts_norm[:, 0]))
    span_y = float(np.max(pts_norm[:, 1]) - np.min(pts_norm[:, 1]))
    return span_x, span_y


def normalize_and_randomize_background_depth(
    depth_m: np.ndarray,
    camera_cfg: dict,
    fitted_plane: PlaneModel,
    rng: np.random.Generator,
    distance_range_m: tuple[float, float],
    pitch_deg_range: tuple[float, float],
    yaw_deg_range: tuple[float, float],
    backproject_stride: int = 1,
    fill_fov: bool = True,
    target_fill_ratio: float = 0.98,
    max_inplane_scale: float = 3.0,
    depth_splat_radius_px: int = 1,
) -> tuple[np.ndarray, BackgroundTransformParams]:
    """Normalize captured background to canonical plane, then randomize pose.

    Steps:
    1) Align fitted plane normal with +Z and translate so plane passes through origin.
    2) Apply random pitch/yaw and random +Z distance.
    3) Re-project transformed points to a depth image.
    """
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    points = depth_to_camera_points(depth_m, fx, fy, cx, cy, stride=backproject_stride)

    # Canonical normalization: plane -> z=0, normal -> +Z.
    n = np.asarray(fitted_plane.normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    p0 = -float(fitted_plane.offset) * n

    r_align = _rotation_align_vectors(n, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    points_norm = (r_align @ (points - p0).T).T

    # Randomized canonical background pose.
    pitch_deg = float(rng.uniform(*pitch_deg_range))
    yaw_deg = float(rng.uniform(*yaw_deg_range))
    distance_m = float(rng.uniform(*distance_range_m))

    r_rand = Rotation.from_euler("xy", [pitch_deg, yaw_deg], degrees=True).as_matrix()

    def _projected_fill_ratio(points_xyz: np.ndarray) -> tuple[float, float]:
        _, _, _, _, width, height = intrinsics_from_camera_config(camera_cfg)
        z = points_xyz[:, 2]
        valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid):
            return 0.0, 0.0

        pts = points_xyz[valid]
        zz = pts[:, 2]
        u = fx * pts[:, 0] / zz + cx
        v = fy * pts[:, 1] / zz + cy

        span_u = max(float(np.max(u) - np.min(u)), 1e-6)
        span_v = max(float(np.max(v) - np.min(v)), 1e-6)
        return span_u / float(width), span_v / float(height)

    inplane_scale_xy = 1.0
    points_norm_scaled = points_norm
    points_rand = (r_rand @ points_norm_scaled.T).T + np.array([0.0, 0.0, distance_m], dtype=np.float64)
    fill_u, fill_v = _projected_fill_ratio(points_rand)

    if fill_fov:
        required_span_x, required_span_y = _frustum_corners_intersection_spans_on_plane(
            camera_cfg=camera_cfg,
            r_rand=r_rand,
            distance_m=distance_m,
        )

        if required_span_x is not None and required_span_y is not None:
            observed_span_x = float(np.max(points_norm[:, 0]) - np.min(points_norm[:, 0]))
            observed_span_y = float(np.max(points_norm[:, 1]) - np.min(points_norm[:, 1]))

            scale_x = float(target_fill_ratio) * required_span_x / max(observed_span_x, 1e-6)
            scale_y = float(target_fill_ratio) * required_span_y / max(observed_span_y, 1e-6)
            inplane_scale_xy = np.clip(max(scale_x, scale_y), 1.0, max(float(max_inplane_scale), 1.0))

            if inplane_scale_xy > 1.0 + 1e-6:
                points_norm_scaled = points_norm.copy()
                center_xy = points_norm_scaled[:, :2].mean(axis=0, keepdims=True)
                points_norm_scaled[:, :2] = (
                    (points_norm_scaled[:, :2] - center_xy) * inplane_scale_xy + center_xy
                )
                points_rand = (r_rand @ points_norm_scaled.T).T + np.array([0.0, 0.0, distance_m], dtype=np.float64)
                fill_u, fill_v = _projected_fill_ratio(points_rand)

    depth_rand = camera_points_to_depth(
        points_rand,
        camera_cfg=camera_cfg,
        splat_radius_px=depth_splat_radius_px,
    )
    params = BackgroundTransformParams(
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        distance_m=distance_m,
        inplane_scale_xy=float(inplane_scale_xy),
        projected_fill_u=float(fill_u),
        projected_fill_v=float(fill_v),
    )
    return depth_rand, params
