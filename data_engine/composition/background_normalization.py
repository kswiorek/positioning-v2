"""Background plane normalization and re-randomization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from data_engine.composition.plane_fit import PlaneModel, select_plane_support_mask
from data_engine.geometry import intrinsics_from_camera_config


@dataclass
class BackgroundTransformParams:
    pitch_deg: float
    yaw_deg: float
    distance_m: float
    inplane_scale_xy: float = 1.0
    projected_fill_u: float = 0.0
    projected_fill_v: float = 0.0
    out_of_plane_scale_m: float = 0.0


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


def _frustum_corners_on_canonical_plane(
    camera_cfg: dict,
    r_rand: np.ndarray,
    distance_m: float,
) -> np.ndarray | None:
    """Return 4 frustum-corner intersection points in canonical plane frame."""
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
            return None
        t = -d_rand / denom
        if t <= 1e-6:
            return None

        p_cam = t * ray
        # Inverse randomized transform to canonical plane frame.
        p_norm = r_rand.T @ (p_cam - plane_point)
        pts_norm.append(p_norm)

    return np.asarray(pts_norm, dtype=np.float64)


def _frustum_corners_on_raw_plane(camera_cfg: dict, plane_normal: np.ndarray, plane_offset: float) -> np.ndarray | None:
    """Return 4 frustum-corner intersection points on the raw fitted plane."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )

    pts = []
    for u, v in corners_uv:
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        ray /= max(np.linalg.norm(ray), 1e-12)

        denom = float(np.dot(plane_normal, ray))
        if abs(denom) < 1e-9:
            return None
        t = -float(plane_offset) / denom
        if t <= 1e-6:
            return None
        pts.append(t * ray)
    return np.asarray(pts, dtype=np.float64)


def _build_camera_ray_grid(camera_cfg: dict) -> np.ndarray:
    """Return normalized camera rays for each pixel [H, W, 3]."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Bilinear sampling of 2D image at floating coordinates."""
    h, w = image.shape
    out = np.full_like(x, fill_value, dtype=np.float64)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    if not np.any(inside):
        return out

    xi0 = x0[inside]
    yi0 = y0[inside]
    xi1 = x1[inside]
    yi1 = y1[inside]

    Ia = image[yi0, xi0]
    Ib = image[yi0, xi1]
    Ic = image[yi1, xi0]
    Id = image[yi1, xi1]

    xf = x[inside] - xi0
    yf = y[inside] - yi0

    wa = (1.0 - xf) * (1.0 - yf)
    wb = xf * (1.0 - yf)
    wc = (1.0 - xf) * yf
    wd = xf * yf

    out_inside = wa * Ia + wb * Ib + wc * Ic + wd * Id
    out[inside] = out_inside
    return out


def normalize_and_randomize_background_depth(
    depth_m: np.ndarray,
    camera_cfg: dict,
    fitted_plane: PlaneModel,
    rng: np.random.Generator,
    distance_range_m: tuple[float, float],
    pitch_deg_range: tuple[float, float],
    yaw_deg_range: tuple[float, float],
    fill_fov: bool = True,
    target_fill_ratio: float = 0.98,
    max_inplane_scale: float = 3.0,
    middle_percentile: float = 0.90,
    out_of_plane_range_m: tuple[float, float] = (0.0, 0.2),
    reference_plane_camera_normal: bool = False,
) -> tuple[np.ndarray, BackgroundTransformParams]:
    """Normalize captured background to canonical plane, then randomize pose.

    Steps:
    1) Build plane-relative residual texture from the input depth image.
    2) Normalize residuals using a robust middle-percentile range.
    3) Randomize plane pose (pitch, yaw, distance) and in-plane scale for frustum coverage.
    4) Warp residual texture onto the randomized plane to generate output depth.
    """
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)
    depth = np.asarray(depth_m, dtype=np.float64)
    valid_depth = np.isfinite(depth) & (depth > 1e-6)
    valid_tex = valid_depth.astype(np.float64)

    # Camera points from the original depth map.
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    z = depth.copy()
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=-1)

    # Canonical normalization: plane -> z=0, normal -> +Z.
    if reference_plane_camera_normal:
        # Force fronto-parallel reference plane and fit only depth offset from support region.
        n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        support = select_plane_support_mask(depth, middle_percentile=middle_percentile)
        if np.any(support):
            z_ref = float(np.median(depth[support]))
        else:
            z_ref = float(np.median(depth[valid_depth]))
        z_ref = max(z_ref, 1e-3)
        plane_offset = z_ref
    else:
        n = np.asarray(fitted_plane.normal, dtype=np.float64)
        n = n / max(np.linalg.norm(n), 1e-12)
        plane_offset = float(fitted_plane.offset)

    p0 = -plane_offset * n

    r_align = _rotation_align_vectors(n, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # Derive per-pixel residual along plane normal (signed distance to plane).
    # These residuals encode local outliers/roughness and become displacement texture.
    residual = np.full((height, width), np.nan, dtype=np.float64)
    residual[valid_depth] = (
        points_cam[..., 0][valid_depth] * n[0]
        + points_cam[..., 1][valid_depth] * n[1]
        + points_cam[..., 2][valid_depth] * n[2]
        + plane_offset
    )

    vals = residual[valid_depth]
    if vals.size == 0:
        raise RuntimeError("No valid depth values available for background normalization.")

    mid = np.clip(float(middle_percentile), 0.1, 0.999)
    lo_p = 50.0 * (1.0 - mid)
    hi_p = 100.0 - lo_p
    lo = float(np.percentile(vals, lo_p))
    hi = float(np.percentile(vals, hi_p))
    denom = max(hi - lo, 1e-9)

    # Keep outliers outside [0, 1] by not clipping here.
    residual_norm = np.zeros((height, width), dtype=np.float64)
    residual_norm[valid_depth] = (residual[valid_depth] - lo) / denom

    out_of_plane_scale_m = float(rng.uniform(*out_of_plane_range_m))
    displacement_tex = residual_norm * out_of_plane_scale_m
    displacement_tex = np.nan_to_num(displacement_tex, nan=0.0, posinf=0.0, neginf=0.0)

    # Build source rectangle in canonical plane frame.
    # For fill_fov, use valid depth footprint so scaling compensates sparse/partial captures.
    pts_valid = points_cam[valid_depth]
    if pts_valid.shape[0] < 3:
        raise RuntimeError("Could not derive source rectangle from depth frame.")
    pts_valid_norm = (r_align @ (pts_valid - p0).T).T

    corners_raw = _frustum_corners_on_raw_plane(camera_cfg, n, plane_offset)
    src_min = None
    src_max = None

    if fill_fov:
        # Robust footprint from valid points; trim outliers at image borders.
        px = pts_valid_norm[:, 0]
        py = pts_valid_norm[:, 1]
        src_min = np.array([
            float(np.percentile(px, 1.0)),
            float(np.percentile(py, 1.0)),
        ])
        src_max = np.array([
            float(np.percentile(px, 99.0)),
            float(np.percentile(py, 99.0)),
        ])

    if src_min is None or src_max is None:
        if corners_raw is not None:
            corners_src_norm = (r_align @ (corners_raw - p0).T).T
            src_min = np.min(corners_src_norm[:, :2], axis=0)
            src_max = np.max(corners_src_norm[:, :2], axis=0)
        else:
            src_min = np.min(pts_valid_norm[:, :2], axis=0)
            src_max = np.max(pts_valid_norm[:, :2], axis=0)

    src_center = 0.5 * (src_min + src_max)
    src_half = 0.5 * (src_max - src_min)

    # Randomized canonical background pose.
    pitch_deg = float(rng.uniform(*pitch_deg_range))
    yaw_deg = float(rng.uniform(*yaw_deg_range))
    distance_m = float(rng.uniform(*distance_range_m))

    r_rand = Rotation.from_euler("xy", [pitch_deg, yaw_deg], degrees=True).as_matrix()

    inplane_scale_xy = 1.0
    fill_u, fill_v = 0.0, 0.0

    if fill_fov:
        frustum_corners_norm = _frustum_corners_on_canonical_plane(
            camera_cfg=camera_cfg,
            r_rand=r_rand,
            distance_m=distance_m,
        )

        if frustum_corners_norm is not None:
            # Minimum isotropic scale around bg_center so all frustum-corner intersections
            # lie inside the scaled background rectangle.
            corner_xy = frustum_corners_norm[:, :2]
            rel = np.abs(corner_xy - src_center[None, :])
            rel_x = rel[:, 0] / max(src_half[0], 1e-9)
            rel_y = rel[:, 1] / max(src_half[1], 1e-9)
            required_scale = float(np.max(np.maximum(rel_x, rel_y)))

            # target_fill_ratio < 1 adds margin beyond strict containment.
            required_scale /= max(float(target_fill_ratio), 1e-6)
            inplane_scale_xy = np.clip(required_scale, 1.0, max(float(max_inplane_scale), 1.0))

    scaled_half = src_half * inplane_scale_xy
    rect_min = src_center - scaled_half
    rect_max = src_center + scaled_half

    # Warp displacement texture to the randomized plane over the full image.
    rays = _build_camera_ray_grid(camera_cfg)
    n_rand = r_rand @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    plane_point = np.array([0.0, 0.0, distance_m], dtype=np.float64)
    d_rand = -float(np.dot(n_rand, plane_point))

    denom = np.einsum("ijk,k->ij", rays, n_rand)
    valid_ray = np.abs(denom) > 1e-9
    t = np.zeros((height, width), dtype=np.float64)
    t[valid_ray] = -d_rand / denom[valid_ray]
    valid_ray &= t > 1e-6

    p_cam = rays * t[..., None]
    p_norm = np.einsum("ji,hwj->hwi", r_rand, (p_cam - plane_point[None, None, :]))

    x_norm = p_norm[..., 0]
    y_norm = p_norm[..., 1]

    sx = (x_norm - rect_min[0]) / max(rect_max[0] - rect_min[0], 1e-9) * (width - 1)
    sy = (y_norm - rect_min[1]) / max(rect_max[1] - rect_min[1], 1e-9) * (height - 1)

    disp = _bilinear_sample(displacement_tex, sx, sy, fill_value=0.0)
    src_valid_weight = _bilinear_sample(valid_tex, sx, sy, fill_value=0.0)
    src_valid = src_valid_weight > 0.5

    base_depth = p_cam[..., 2]
    depth_rand = np.zeros((height, width), dtype=np.float32)
    out_valid = valid_ray & src_valid
    depth_rand[out_valid] = (base_depth[out_valid] + disp[out_valid] * n_rand[2]).astype(np.float32)
    depth_rand = np.maximum(depth_rand, 0.0)

    # Fill metrics after warp.
    uu_proj = fx * p_cam[..., 0] / np.maximum(base_depth, 1e-9) + cx
    vv_proj = fy * p_cam[..., 1] / np.maximum(base_depth, 1e-9) + cy
    mask_proj = out_valid & np.isfinite(uu_proj) & np.isfinite(vv_proj)
    if np.any(mask_proj):
        span_u = float(np.max(uu_proj[mask_proj]) - np.min(uu_proj[mask_proj]))
        span_v = float(np.max(vv_proj[mask_proj]) - np.min(vv_proj[mask_proj]))
        fill_u = span_u / float(width)
        fill_v = span_v / float(height)
    params = BackgroundTransformParams(
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        distance_m=distance_m,
        inplane_scale_xy=float(inplane_scale_xy),
        projected_fill_u=float(fill_u),
        projected_fill_v=float(fill_v),
        out_of_plane_scale_m=float(out_of_plane_scale_m),
    )
    return depth_rand, params
