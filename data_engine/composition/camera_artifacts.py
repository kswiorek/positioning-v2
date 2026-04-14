"""Camera depth artifact simulation utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from data_engine.geometry.camera import intrinsics_from_camera_config


def _build_structure(connectivity: int) -> np.ndarray:
    if int(connectivity) >= 8:
        return np.ones((3, 3), dtype=bool)
    return np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )


def _depth_discontinuity_mask(
    depth_m: np.ndarray,
    min_depth_m: float,
    abs_threshold_m: float,
    rel_threshold: float,
    threshold_mode: str,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = depth > float(min_depth_m)

    edge = np.zeros_like(valid, dtype=bool)

    def _edge_pair(a: np.ndarray, b: np.ndarray, va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        pair_valid = va & vb
        if not np.any(pair_valid):
            return np.zeros_like(pair_valid, dtype=bool)

        diff = np.abs(a - b)
        denom = np.maximum(np.minimum(a, b), 1e-6)
        rel = diff / denom

        abs_ok = diff >= float(abs_threshold_m)
        rel_ok = rel >= float(rel_threshold)

        mode = str(threshold_mode).strip().lower()
        if mode == "and":
            return pair_valid & abs_ok & rel_ok
        return pair_valid & (abs_ok | rel_ok)

    # Horizontal neighbor pairs.
    h_edge = _edge_pair(depth[:, :-1], depth[:, 1:], valid[:, :-1], valid[:, 1:])
    edge[:, :-1] |= h_edge
    edge[:, 1:] |= h_edge

    # Vertical neighbor pairs.
    v_edge = _edge_pair(depth[:-1, :], depth[1:, :], valid[:-1, :], valid[1:, :])
    edge[:-1, :] |= v_edge
    edge[1:, :] |= v_edge

    return edge


def _surface_grazing_mask(
    depth_m: np.ndarray,
    camera_cfg: dict,
    min_depth_m: float,
    angle_threshold_deg: float,
    use_abs_dot: bool,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape
    valid = depth > float(min_depth_m)
    if h < 3 or w < 3:
        return np.zeros_like(valid, dtype=bool)

    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth
    x = (uu - float(cx)) * z / float(fx)
    y = (vv - float(cy)) * z / float(fy)
    p = np.stack([x, y, z], axis=-1).astype(np.float32)

    mask = np.zeros((h, w), dtype=bool)

    c = p[1:-1, 1:-1, :]
    l = p[1:-1, :-2, :]
    r = p[1:-1, 2:, :]
    u_p = p[:-2, 1:-1, :]
    d_p = p[2:, 1:-1, :]

    valid_center = valid[1:-1, 1:-1]
    valid_lr = valid[1:-1, :-2] & valid[1:-1, 2:]
    valid_ud = valid[:-2, 1:-1] & valid[2:, 1:-1]
    local_valid = valid_center & valid_lr & valid_ud
    if not np.any(local_valid):
        return mask

    dx = r - l
    dy = d_p - u_p
    n = np.cross(dx, dy)
    n_norm = np.linalg.norm(n, axis=-1)
    n_valid = n_norm > 1e-8

    view = c
    view_norm = np.linalg.norm(view, axis=-1)
    v_valid = view_norm > 1e-8

    ok = local_valid & n_valid & v_valid
    if not np.any(ok):
        return mask

    dot = np.zeros_like(n_norm, dtype=np.float32)
    dot[ok] = np.einsum("ijk,ijk->ij", n, view)[ok] / (n_norm[ok] * view_norm[ok])

    cos_thr = float(np.cos(np.deg2rad(float(angle_threshold_deg))))
    dot_eval = np.abs(dot) if use_abs_dot else dot
    grazing = ok & (dot_eval < cos_thr)

    mask[1:-1, 1:-1] = grazing
    return mask


def apply_camera_artifacts(
    depth_m: np.ndarray,
    camera_artifacts_cfg: dict | None,
    background_depth_m: np.ndarray | None = None,
    object_depth_m: np.ndarray | None = None,
    camera_cfg: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply configurable camera-depth artifacts to a depth image."""
    cfg = camera_artifacts_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return depth_m.astype(np.float32), {
            "enabled": False,
            "edge_border": {"enabled": False, "pixels_invalidated": 0, "ratio_invalidated": 0.0},
            "normal_angle_blank": {"enabled": False, "pixels_invalidated": 0, "ratio_invalidated": 0.0},
        }

    source_depth = depth_m.astype(np.float32)
    out = source_depth.copy()
    stats = {
        "enabled": True,
        "edge_border": {
            "enabled": False,
            "pixels_invalidated": 0,
            "ratio_invalidated": 0.0,
        },
        "normal_angle_blank": {
            "enabled": False,
            "pixels_invalidated": 0,
            "ratio_invalidated": 0.0,
        },
    }

    apply_to_background = bool(cfg.get("apply_to_background", False))

    def _build_scope_mask(min_depth_m: float, connectivity: int, interaction_radius_px: int):
        if apply_to_background:
            return np.ones_like(out, dtype=bool), None

        if object_depth_m is None:
            return None, "object_depth_missing"

        obj = np.asarray(object_depth_m, dtype=np.float32)
        if obj.shape != out.shape:
            raise ValueError("object_depth_m shape must match depth_m shape")

        obj_valid = obj > float(min_depth_m)
        if not np.any(obj_valid):
            return None, "no_object_pixels"

        structure = _build_structure(connectivity)
        influence = binary_dilation(obj_valid, structure=structure, iterations=max(int(interaction_radius_px), 1))
        return influence, None

    edge_cfg = cfg.get("edge_border", {})
    edge_enabled = bool(edge_cfg.get("enabled", True))
    if edge_enabled:
        min_depth_m = float(edge_cfg.get("min_depth_m", 1e-6))
        abs_threshold_m = float(edge_cfg.get("abs_threshold_m", 0.08))
        rel_threshold = float(edge_cfg.get("rel_threshold", 0.06))
        threshold_mode = str(edge_cfg.get("threshold_mode", "or"))
        border_radius_px = max(int(edge_cfg.get("border_radius_px", 1)), 0)
        connectivity = int(edge_cfg.get("connectivity", 8))
        interaction_radius_px = max(int(edge_cfg.get("interaction_radius_px", border_radius_px + 1)), 0)

        base_edge = _depth_discontinuity_mask(
            out,
            min_depth_m=min_depth_m,
            abs_threshold_m=abs_threshold_m,
            rel_threshold=rel_threshold,
            threshold_mode=threshold_mode,
        )

        scope_mask, scope_reason = _build_scope_mask(min_depth_m, connectivity, interaction_radius_px)
        if scope_mask is not None:
            base_edge &= scope_mask

            if border_radius_px > 0 and np.any(base_edge):
                structure = _build_structure(connectivity)
                edge_band = binary_dilation(base_edge, structure=structure, iterations=border_radius_px)
            else:
                edge_band = base_edge

            valid_before = out > min_depth_m
            invalidate = edge_band & valid_before
            out[invalidate] = 0.0

            invalid_pixels = int(np.count_nonzero(invalidate))
            stats["edge_border"] = {
                "enabled": True,
                "pixels_invalidated": invalid_pixels,
                "ratio_invalidated": float(invalid_pixels) / float(out.size),
                "border_radius_px": border_radius_px,
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "abs_threshold_m": abs_threshold_m,
                "rel_threshold": rel_threshold,
                "threshold_mode": threshold_mode,
                "connectivity": connectivity,
            }
        else:
            stats["edge_border"] = {
                "enabled": False,
                "pixels_invalidated": 0,
                "ratio_invalidated": 0.0,
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "reason": scope_reason,
            }

    normal_cfg = cfg.get("normal_angle_blank", {})
    normal_enabled = bool(normal_cfg.get("enabled", False))
    if normal_enabled:
        min_depth_m = float(normal_cfg.get("min_depth_m", 1e-6))
        angle_threshold_deg = float(normal_cfg.get("angle_threshold_deg", 80.0))
        use_abs_dot = bool(normal_cfg.get("use_abs_dot", True))
        connectivity = int(normal_cfg.get("connectivity", 8))
        interaction_radius_px = max(int(normal_cfg.get("interaction_radius_px", 2)), 0)
        dilation_radius_px = max(int(normal_cfg.get("dilation_radius_px", 1)), 0)

        if camera_cfg is None:
            stats["normal_angle_blank"] = {
                "enabled": False,
                "pixels_invalidated": 0,
                "ratio_invalidated": 0.0,
                "reason": "camera_cfg_missing",
            }
            return out, stats

        grazing = _surface_grazing_mask(
            source_depth,
            camera_cfg=camera_cfg,
            min_depth_m=min_depth_m,
            angle_threshold_deg=angle_threshold_deg,
            use_abs_dot=use_abs_dot,
        )

        scope_mask, scope_reason = _build_scope_mask(min_depth_m, connectivity, interaction_radius_px)
        if scope_mask is not None:
            grazing &= scope_mask
            if dilation_radius_px > 0 and np.any(grazing):
                structure = _build_structure(connectivity)
                grazing = binary_dilation(grazing, structure=structure, iterations=dilation_radius_px)

            valid_before = out > min_depth_m
            invalidate = grazing & valid_before
            out[invalidate] = 0.0

            invalid_pixels = int(np.count_nonzero(invalidate))
            stats["normal_angle_blank"] = {
                "enabled": True,
                "pixels_invalidated": invalid_pixels,
                "ratio_invalidated": float(invalid_pixels) / float(out.size),
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "angle_threshold_deg": angle_threshold_deg,
                "use_abs_dot": use_abs_dot,
                "dilation_radius_px": dilation_radius_px,
                "connectivity": connectivity,
            }
        else:
            stats["normal_angle_blank"] = {
                "enabled": False,
                "pixels_invalidated": 0,
                "ratio_invalidated": 0.0,
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "reason": scope_reason,
            }

    return out, stats
