"""Camera depth artifact simulation utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

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


def _surface_view_alignment(
    depth_m: np.ndarray,
    camera_cfg: dict,
    min_depth_m: float,
    use_abs_dot: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine(view, normal) map and validity mask for interior pixels."""
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape
    valid = depth > float(min_depth_m)
    dot_map = np.zeros((h, w), dtype=np.float32)
    ok_map = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return dot_map, ok_map

    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth
    x = (uu - float(cx)) * z / float(fx)
    y = (vv - float(cy)) * z / float(fy)
    p = np.stack([x, y, z], axis=-1).astype(np.float32)

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
        return dot_map, ok_map

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
        return dot_map, ok_map

    dot = np.zeros_like(n_norm, dtype=np.float32)
    dot[ok] = np.einsum("ijk,ijk->ij", n, view)[ok] / (n_norm[ok] * view_norm[ok])
    if use_abs_dot:
        dot = np.abs(dot)

    dot_map[1:-1, 1:-1] = dot
    ok_map[1:-1, 1:-1] = ok
    return dot_map, ok_map


def _linear_step_by_depth(
    depth_m: np.ndarray,
    z_near_m: float,
    z_far_m: float,
    step_near_m: float,
    step_far_m: float,
) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float32)
    denom = max(float(z_far_m) - float(z_near_m), 1e-6)
    t = (z - float(z_near_m)) / denom
    t = np.clip(t, 0.0, 1.0)
    return (1.0 - t) * float(step_near_m) + t * float(step_far_m)


def _normalized_smooth_random_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma_px: float,
) -> np.ndarray:
    """Return [0,1] random field with spatial correlation for blob-like masks."""
    noise = rng.standard_normal(size=shape).astype(np.float32)
    sigma_px = max(float(sigma_px), 0.0)
    if sigma_px > 1e-6:
        noise = gaussian_filter(noise, sigma=sigma_px)

    lo = float(np.percentile(noise, 1.0))
    hi = float(np.percentile(noise, 99.0))
    if hi <= lo + 1e-9:
        return np.full(shape, 0.5, dtype=np.float32)

    out = (noise - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_camera_artifacts(
    depth_m: np.ndarray,
    camera_artifacts_cfg: dict | None,
    background_depth_m: np.ndarray | None = None,
    object_depth_m: np.ndarray | None = None,
    camera_cfg: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply configurable camera-depth artifacts to a depth image."""
    cfg = camera_artifacts_cfg or {}
    rng_local = rng if rng is not None else np.random.default_rng()
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return depth_m.astype(np.float32), {
            "enabled": False,
            "range_noise": {"enabled": False, "pixels_modified": 0, "ratio_modified": 0.0},
            "depth_quantization": {"enabled": False, "pixels_modified": 0, "ratio_modified": 0.0},
            "edge_border": {"enabled": False, "pixels_invalidated": 0, "ratio_invalidated": 0.0},
            "normal_confidence_falloff": {"enabled": False, "pixels_invalidated": 0, "ratio_invalidated": 0.0},
            "normal_angle_blank": {"enabled": False, "pixels_invalidated": 0, "ratio_invalidated": 0.0},
        }

    source_depth = depth_m.astype(np.float32)
    out = source_depth.copy()
    stats = {
        "enabled": True,
        "range_noise": {
            "enabled": False,
            "pixels_modified": 0,
            "ratio_modified": 0.0,
        },
        "depth_quantization": {
            "enabled": False,
            "pixels_modified": 0,
            "ratio_modified": 0.0,
        },
        "edge_border": {
            "enabled": False,
            "pixels_invalidated": 0,
            "ratio_invalidated": 0.0,
        },
        "normal_confidence_falloff": {
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
    alignment_cache: dict[tuple[bool, float], tuple[np.ndarray, np.ndarray]] = {}

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

    def _get_alignment(use_abs_dot: bool, min_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
        key = (bool(use_abs_dot), float(min_depth_m))
        cached = alignment_cache.get(key)
        if cached is None:
            cached = _surface_view_alignment(
                source_depth,
                camera_cfg=camera_cfg,
                min_depth_m=min_depth_m,
                use_abs_dot=use_abs_dot,
            )
            alignment_cache[key] = cached
        return cached

    # L515-oriented range-dependent noise.
    noise_cfg = cfg.get("range_noise", {})
    noise_enabled = bool(noise_cfg.get("enabled", False))
    if noise_enabled:
        min_depth_m = float(noise_cfg.get("min_depth_m", 1e-6))
        connectivity = int(noise_cfg.get("connectivity", 8))
        interaction_radius_px = max(int(noise_cfg.get("interaction_radius_px", 2)), 0)

        sigma0_m = float(noise_cfg.get("sigma0_m", 0.0008))
        sigma_z2_m = float(noise_cfg.get("sigma_z2_m", 0.00045))
        sigma_min_m = float(noise_cfg.get("sigma_min_m", 0.0005))
        sigma_max_m = float(noise_cfg.get("sigma_max_m", 0.0040))

        scope_mask, scope_reason = _build_scope_mask(min_depth_m, connectivity, interaction_radius_px)
        if scope_mask is not None:
            valid = (out > min_depth_m) & scope_mask
            if np.any(valid):
                z = out.astype(np.float32)
                sigma = sigma0_m + sigma_z2_m * (z * z)
                sigma = np.clip(sigma, sigma_min_m, sigma_max_m)
                noise = rng_local.normal(loc=0.0, scale=sigma, size=out.shape).astype(np.float32)
                out[valid] = np.maximum(out[valid] + noise[valid], 0.0)
                px = int(np.count_nonzero(valid))
                stats["range_noise"] = {
                    "enabled": True,
                    "pixels_modified": px,
                    "ratio_modified": float(px) / float(out.size),
                    "apply_to_background": apply_to_background,
                    "interaction_radius_px": interaction_radius_px,
                    "sigma0_m": sigma0_m,
                    "sigma_z2_m": sigma_z2_m,
                    "sigma_min_m": sigma_min_m,
                    "sigma_max_m": sigma_max_m,
                    "mean_sigma_m": float(np.mean(sigma[valid])),
                }
            else:
                stats["range_noise"] = {
                    "enabled": False,
                    "pixels_modified": 0,
                    "ratio_modified": 0.0,
                    "reason": "no_valid_pixels",
                }
        else:
            stats["range_noise"] = {
                "enabled": False,
                "pixels_modified": 0,
                "ratio_modified": 0.0,
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "reason": scope_reason,
            }

    # L515-like depth quantization with depth-varying step.
    quant_cfg = cfg.get("depth_quantization", {})
    quant_enabled = bool(quant_cfg.get("enabled", False))
    if quant_enabled:
        min_depth_m = float(quant_cfg.get("min_depth_m", 1e-6))
        connectivity = int(quant_cfg.get("connectivity", 8))
        interaction_radius_px = max(int(quant_cfg.get("interaction_radius_px", 2)), 0)

        z_near_m = float(quant_cfg.get("z_near_m", 0.25))
        z_far_m = float(quant_cfg.get("z_far_m", 2.5))
        step_near_m = float(quant_cfg.get("step_m_near", 0.00025))
        step_far_m = float(quant_cfg.get("step_m_far", 0.0010))
        dither_m = float(quant_cfg.get("dither_m", 0.00005))

        scope_mask, scope_reason = _build_scope_mask(min_depth_m, connectivity, interaction_radius_px)
        if scope_mask is not None:
            valid = (out > min_depth_m) & scope_mask
            if np.any(valid):
                step = _linear_step_by_depth(
                    out,
                    z_near_m=z_near_m,
                    z_far_m=z_far_m,
                    step_near_m=step_near_m,
                    step_far_m=step_far_m,
                )
                jitter = rng_local.uniform(-dither_m, dither_m, size=out.shape).astype(np.float32)
                z_noisy = out + jitter
                z_q = np.round(z_noisy / np.maximum(step, 1e-7)) * step
                out[valid] = np.maximum(z_q[valid], 0.0)
                px = int(np.count_nonzero(valid))
                stats["depth_quantization"] = {
                    "enabled": True,
                    "pixels_modified": px,
                    "ratio_modified": float(px) / float(out.size),
                    "apply_to_background": apply_to_background,
                    "interaction_radius_px": interaction_radius_px,
                    "step_m_near": step_near_m,
                    "step_m_far": step_far_m,
                    "z_near_m": z_near_m,
                    "z_far_m": z_far_m,
                    "dither_m": dither_m,
                    "mean_step_m": float(np.mean(step[valid])),
                }
            else:
                stats["depth_quantization"] = {
                    "enabled": False,
                    "pixels_modified": 0,
                    "ratio_modified": 0.0,
                    "reason": "no_valid_pixels",
                }
        else:
            stats["depth_quantization"] = {
                "enabled": False,
                "pixels_modified": 0,
                "ratio_modified": 0.0,
                "apply_to_background": apply_to_background,
                "interaction_radius_px": interaction_radius_px,
                "reason": scope_reason,
            }

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

    # Soft probabilistic falloff for grazing normals.
    conf_cfg = cfg.get("normal_confidence_falloff", {})
    conf_enabled = bool(conf_cfg.get("enabled", False))
    if conf_enabled:
        min_depth_m = float(conf_cfg.get("min_depth_m", 1e-6))
        use_abs_dot = bool(conf_cfg.get("use_abs_dot", True))
        connectivity = int(conf_cfg.get("connectivity", 8))
        interaction_radius_px = max(int(conf_cfg.get("interaction_radius_px", 2)), 0)
        dilation_radius_px = max(int(conf_cfg.get("dilation_radius_px", 0)), 0)

        angle_center_deg = float(conf_cfg.get("angle_center_deg", 70.0))
        angle_span_deg = max(float(conf_cfg.get("angle_span_deg", 20.0)), 1e-6)
        drop_prob_max = float(np.clip(conf_cfg.get("drop_prob_max", 0.8), 0.0, 1.0))
        structured_dropout = bool(conf_cfg.get("structured_dropout", True))
        blob_sigma_px = float(conf_cfg.get("blob_sigma_px", 1.8))
        blob_gamma = max(float(conf_cfg.get("blob_gamma", 1.0)), 1e-6)

        if camera_cfg is None:
            stats["normal_confidence_falloff"] = {
                "enabled": False,
                "pixels_invalidated": 0,
                "ratio_invalidated": 0.0,
                "reason": "camera_cfg_missing",
            }
            return out, stats

        dot_map, ok_map = _get_alignment(use_abs_dot=use_abs_dot, min_depth_m=min_depth_m)
        angle_deg = np.degrees(np.arccos(np.clip(dot_map, -1.0, 1.0))).astype(np.float32)
        lo = angle_center_deg - 0.5 * angle_span_deg
        hi = angle_center_deg + 0.5 * angle_span_deg
        t = np.clip((angle_deg - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        p_drop = drop_prob_max * t

        scope_mask, scope_reason = _build_scope_mask(min_depth_m, connectivity, interaction_radius_px)
        if scope_mask is not None:
            candidates = ok_map & scope_mask & (out > min_depth_m)
            if np.any(candidates):
                if structured_dropout:
                    field = _normalized_smooth_random_field(out.shape, rng_local, sigma_px=blob_sigma_px)
                    if abs(blob_gamma - 1.0) > 1e-6:
                        field = np.power(field, blob_gamma, dtype=np.float32)
                    invalidate = candidates & (field < p_drop)
                else:
                    rand = rng_local.uniform(0.0, 1.0, size=out.shape).astype(np.float32)
                    invalidate = candidates & (rand < p_drop)
                if dilation_radius_px > 0 and np.any(invalidate):
                    structure = _build_structure(connectivity)
                    invalidate = binary_dilation(invalidate, structure=structure, iterations=dilation_radius_px)
                    invalidate &= scope_mask
                out[invalidate] = 0.0

                px = int(np.count_nonzero(invalidate))
                stats["normal_confidence_falloff"] = {
                    "enabled": True,
                    "pixels_invalidated": px,
                    "ratio_invalidated": float(px) / float(out.size),
                    "apply_to_background": apply_to_background,
                    "interaction_radius_px": interaction_radius_px,
                    "angle_center_deg": angle_center_deg,
                    "angle_span_deg": angle_span_deg,
                    "drop_prob_max": drop_prob_max,
                    "structured_dropout": structured_dropout,
                    "blob_sigma_px": blob_sigma_px,
                    "blob_gamma": blob_gamma,
                    "dilation_radius_px": dilation_radius_px,
                    "connectivity": connectivity,
                }
            else:
                stats["normal_confidence_falloff"] = {
                    "enabled": False,
                    "pixels_invalidated": 0,
                    "ratio_invalidated": 0.0,
                    "reason": "no_candidate_pixels",
                }
        else:
            stats["normal_confidence_falloff"] = {
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

        dot_map, ok_map = _get_alignment(use_abs_dot=use_abs_dot, min_depth_m=min_depth_m)
        cos_thr = float(np.cos(np.deg2rad(float(angle_threshold_deg))))
        grazing = ok_map & (dot_map < cos_thr)

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
