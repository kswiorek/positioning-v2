"""Bulk dataset generator with multiprocessing and in-worker background RAM caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from data_engine.composition.background_normalization import normalize_and_randomize_background_depth
from data_engine.composition.camera_artifacts import apply_camera_artifacts
from data_engine.composition.depth_compositor import compose_depth, render_mesh_depth, transform_mesh
from data_engine.composition.plane_fit import fit_plane_from_depth
from data_engine.composition.plane_placement import PlacementConstraints
from data_engine.composition.placement_sampling import sample_pose_from_camera
from data_engine.generators import generate_superquadric_canonical_model
from data_engine.generators.mixed import choose_object_source
from data_engine.generators.stl import build_stl_canonical_model_from_cache, preload_stl_asset_chunk

_WORKER_SCENE_CFG: dict[str, Any] | None = None
_WORKER_BG_DEPTHS: list[np.ndarray] = []
_WORKER_BG_IDS: list[str] = []
_WORKER_STL_ASSETS: dict[str, Any] = {}


def _perlin_fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _perlin_grad(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = h & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def _perlin_noise_batch(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    xi = np.floor(x).astype(np.int32) & 255
    yi = np.floor(y).astype(np.int32) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    u = _perlin_fade(xf)
    v = _perlin_fade(yf)

    a = p[xi] + yi
    aa = p[a]
    ab = p[a + 1]
    b = p[xi + 1] + yi
    ba = p[b]
    bb = p[b + 1]

    return _perlin_lerp(
        v,
        _perlin_lerp(u, _perlin_grad(p[aa], xf, yf), _perlin_grad(p[ba], xf - 1.0, yf)),
        _perlin_lerp(u, _perlin_grad(p[ab], xf, yf - 1.0), _perlin_grad(p[bb], xf - 1.0, yf - 1.0)),
    )


def _sample_background_weights(scene_cfg: dict[str, Any]) -> dict[str, float]:
    cfg = scene_cfg.get("background_sources", {})
    weights_cfg = cfg.get("weights", {})
    real_w = float(weights_cfg.get("real", 1.0))
    perlin_w = float(weights_cfg.get("perlin", 0.0))
    blank_w = float(weights_cfg.get("blank", 0.0))
    real_w = max(real_w, 0.0)
    perlin_w = max(perlin_w, 0.0)
    blank_w = max(blank_w, 0.0)
    total = real_w + perlin_w + blank_w
    if total <= 0.0:
        return {"real": 1.0, "perlin": 0.0, "blank": 0.0}
    return {
        "real": float(real_w / total),
        "perlin": float(perlin_w / total),
        "blank": float(blank_w / total),
    }


def _generate_perlin_background(camera_cfg: dict[str, Any], perlin_cfg: dict[str, Any], seed: int) -> np.ndarray:
    cam_res = camera_cfg.get("resolution", {})
    width = int(cam_res.get("width", 0))
    height = int(cam_res.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera.resolution.width and camera.resolution.height must be > 0")

    rng = np.random.default_rng(int(seed))

    base_range = perlin_cfg.get("base_depth_m_range", [2.0, 2.0])
    amp_range = perlin_cfg.get("amplitude_m_range", [0.15, 0.15])
    scale_range = perlin_cfg.get("noise_scale_range", [2.5, 2.5])
    octaves_range = perlin_cfg.get("octaves_range", [4, 4])
    persistence = float(perlin_cfg.get("persistence", 0.5))

    base_lo, base_hi = float(base_range[0]), float(base_range[1])
    amp_lo, amp_hi = float(amp_range[0]), float(amp_range[1])
    scale_lo, scale_hi = float(scale_range[0]), float(scale_range[1])
    oct_lo, oct_hi = int(octaves_range[0]), int(octaves_range[1])
    if base_hi < base_lo:
        base_lo, base_hi = base_hi, base_lo
    if amp_hi < amp_lo:
        amp_lo, amp_hi = amp_hi, amp_lo
    if scale_hi < scale_lo:
        scale_lo, scale_hi = scale_hi, scale_lo
    if oct_hi < oct_lo:
        oct_lo, oct_hi = oct_hi, oct_lo

    base_depth = float(rng.uniform(base_lo, base_hi)) if base_hi > base_lo else base_lo
    amplitude = float(rng.uniform(amp_lo, amp_hi)) if amp_hi > amp_lo else amp_lo
    noise_scale = float(rng.uniform(scale_lo, scale_hi)) if scale_hi > scale_lo else scale_lo
    octaves = int(rng.integers(oct_lo, oct_hi + 1)) if oct_hi > oct_lo else oct_lo
    octaves = max(octaves, 1)

    p0 = rng.permutation(256).astype(np.int32)
    p = np.concatenate([p0, p0])

    xv = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yv = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(xv, yv)
    x = xx * noise_scale
    y = yy * noise_scale

    total = np.zeros_like(x, dtype=np.float32)
    freq = 1.0
    amp = 1.0
    amp_sum = 0.0
    for _ in range(octaves):
        total += _perlin_noise_batch(x * freq, y * freq, p).astype(np.float32) * amp
        amp_sum += amp
        amp *= persistence
        freq *= 2.0
    noise = total / max(amp_sum, 1e-8)

    depth = base_depth + amplitude * noise
    return np.maximum(depth, 0.05).astype(np.float32)


def _sample_background_depth(
    scene_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    rng_master: np.random.Generator,
    forced_source: str | None = None,
) -> tuple[np.ndarray, str, str]:
    weights = _sample_background_weights(scene_cfg)
    if len(_WORKER_BG_DEPTHS) == 0 and (weights["real"] > 0.0 or forced_source == "real"):
        raise RuntimeError("Background source 'real' has non-zero weight but no real backgrounds were loaded.")
    if forced_source is not None:
        source = str(forced_source).strip().lower()
        if source not in {"real", "perlin", "blank"}:
            raise ValueError(f"Unsupported forced background source: {forced_source!r}")
    else:
        u = float(rng_master.random())
        if u < weights["real"]:
            source = "real"
        elif u < weights["real"] + weights["perlin"]:
            source = "perlin"
        else:
            source = "blank"

    if source == "real":
        bg_idx = int(rng_master.integers(0, len(_WORKER_BG_DEPTHS)))
        bg_depth = _WORKER_BG_DEPTHS[bg_idx]
        bg_id = _WORKER_BG_IDS[bg_idx]
        return bg_depth, bg_id, "real"

    if source == "perlin":
        perlin_cfg = scene_cfg.get("background_sources", {}).get("perlin", {})
        perlin_seed = int(rng_master.integers(0, 2**31 - 1))
        bg_depth = _generate_perlin_background(camera_cfg=camera_cfg, perlin_cfg=perlin_cfg, seed=perlin_seed)
        return bg_depth, f"synthetic_perlin:{perlin_seed}", "perlin"

    cam_res = camera_cfg.get("resolution", {})
    width = int(cam_res.get("width", 0))
    height = int(cam_res.get("height", 0))
    blank_depth_value_m = float(scene_cfg.get("background_sources", {}).get("blank", {}).get("depth_value_m", 0.0))
    blank = np.full((height, width), blank_depth_value_m, dtype=np.float32)
    return blank, f"synthetic_blank:{blank_depth_value_m:.4f}", "blank"


def _counts_from_weights(weights: dict[str, float], n_total: int, rng: np.random.Generator) -> dict[str, int]:
    keys = ["real", "perlin", "blank"]
    expected = {k: float(weights.get(k, 0.0)) * float(n_total) for k in keys}
    counts = {k: int(math.floor(expected[k])) for k in keys}
    remaining = int(n_total - sum(counts.values()))
    if remaining > 0:
        order = list(keys)
        rng.shuffle(order)
        order.sort(key=lambda k: expected[k] - counts[k], reverse=True)
        for i in range(remaining):
            counts[order[i % len(order)]] += 1
    return counts


def _assign_background_sources_stratified(
    sample_plans: list["SamplePlan"],
    background_weights: dict[str, float],
    seed: int,
) -> list["SamplePlan"]:
    if not sample_plans:
        return sample_plans

    rng = np.random.default_rng(int(seed))
    plans_out = list(sample_plans)
    by_object_source: dict[str, list[int]] = {}
    for idx, plan in enumerate(sample_plans):
        by_object_source.setdefault(plan.source, []).append(idx)

    for _, indices in by_object_source.items():
        local_indices = list(indices)
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        local_rng.shuffle(local_indices)
        counts = _counts_from_weights(background_weights, len(local_indices), local_rng)

        cursor = 0
        for bg_source in ("real", "perlin", "blank"):
            n = counts[bg_source]
            for idx in local_indices[cursor : cursor + n]:
                plans_out[idx] = replace(plans_out[idx], background_source=bg_source)
            cursor += n
    return plans_out


@dataclass(frozen=True)
class SamplePlan:
    sample_index: int
    sample_seed: int
    split: str
    source: str
    shape_seed: int
    allow_truncation: bool
    background_source: str | None = None
    stl_path: str | None = None
    stl_chunk_id: int | None = None


def _stable_seed_for_path(path: Path, extra: str = "") -> int:
    digest = hashlib.sha1(f"{path.resolve().as_posix()}::{extra}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_counts_from_config(dataset_cfg: dict[str, Any]) -> dict[str, int]:
    total = int(dataset_cfg.get("num_samples", 0))
    if total <= 0:
        raise ValueError("dataset_config.num_samples must be > 0")

    split_cfg = dataset_cfg.get("train_validation_split", {"train": 0.9, "val": 0.1})
    train_v = split_cfg.get("train", 0.9)
    val_v = split_cfg.get("val", 0.1)

    is_ratio = (
        isinstance(train_v, (int, float))
        and isinstance(val_v, (int, float))
        and float(train_v) >= 0.0
        and float(val_v) >= 0.0
        and float(train_v) <= 1.0
        and float(val_v) <= 1.0
    )

    if is_ratio:
        train_r = float(train_v)
        val_r = float(val_v)
        s = train_r + val_r
        if s <= 0.0:
            raise ValueError("train_validation_split train+val must be > 0")
        train_n = int(round(total * (train_r / s)))
        train_n = max(0, min(train_n, total))
        val_n = total - train_n
        return {"train": train_n, "val": val_n}

    train_n = int(train_v)
    val_n = int(val_v)
    if train_n < 0 or val_n < 0:
        raise ValueError("train_validation_split counts must be >= 0")
    if train_n + val_n != total:
        raise ValueError(
            "When using count-based split, train+val must equal num_samples in dataset config"
        )
    return {"train": train_n, "val": val_n}


def _seed_for_sample(base_seed: int, sample_index: int, attempt: int = 0) -> int:
    # Deterministic seed derivation with low collision risk for large runs.
    return int((base_seed * 1000003 + sample_index * 9176 + attempt * 1315423911) % (2**63 - 1))


def _init_worker_scene(scene_cfg: dict[str, Any]) -> None:
    global _WORKER_SCENE_CFG
    _WORKER_SCENE_CFG = scene_cfg


def _load_background_cache(scene_cfg: dict[str, Any], background_paths: list[str], max_backgrounds_in_ram: int) -> None:
    global _WORKER_BG_DEPTHS, _WORKER_BG_IDS

    _init_worker_scene(scene_cfg)
    paths = list(background_paths)
    if max_backgrounds_in_ram > 0:
        paths = paths[: max_backgrounds_in_ram]

    cam_res = scene_cfg.get("camera", {}).get("resolution", {})
    expected_w = int(cam_res.get("width", 0))
    expected_h = int(cam_res.get("height", 0))

    depths: list[np.ndarray] = []
    ids: list[str] = []
    for p in paths:
        arr = np.load(p)["depth_m"].astype(np.float32)
        if expected_w > 0 and expected_h > 0 and arr.shape != (expected_h, expected_w):
            continue
        depths.append(arr)
        ids.append(str(Path(p).as_posix()))

    if not depths:
        raise RuntimeError(
            "Could not load any resolution-compatible backgrounds into RAM. "
            f"Expected shape {(expected_h, expected_w)}."
        )

    _WORKER_BG_DEPTHS = depths
    _WORKER_BG_IDS = ids


def _set_stl_cache(stl_assets: dict[str, Any]) -> None:
    global _WORKER_STL_ASSETS
    _WORKER_STL_ASSETS = stl_assets


def _list_stl_files(scene_cfg: dict[str, Any]) -> list[Path]:
    stl_cfg = scene_cfg.get("scene", {}).get("stl", {})
    input_cfg = stl_cfg.get("input", {})
    directory = Path(input_cfg.get("directory", "data/stl"))
    pattern = str(input_cfg.get("glob", "*.stl"))
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def _build_sample_plan(
    sample_index: int,
    base_seed: int,
    split: str,
    scene_cfg: dict[str, Any],
    stl_files: list[Path],
    stl_available: bool,
    stl_chunk_size: int,
    source_override: str,
    allow_truncation: bool,
) -> SamplePlan:
    sample_seed = _seed_for_sample(base_seed, sample_index, 0)
    rng = np.random.default_rng(sample_seed)
    source = choose_object_source(
        scene_cfg,
        seed=sample_seed,
        source_override=source_override,
        stl_available=stl_available,
    )
    shape_seed = int(rng.integers(0, 2**31 - 1))

    stl_path = None
    if source == "stl":
        if not stl_files:
            raise RuntimeError("No STL files are available for STL sample planning")
        stl_rng = np.random.default_rng(shape_seed)
        stl_idx = int(stl_rng.integers(0, len(stl_files)))
        stl_path = stl_files[stl_idx].resolve().as_posix()
        stl_chunk_id = stl_idx // max(stl_chunk_size, 1)
    else:
        stl_chunk_id = None

    return SamplePlan(
        sample_index=sample_index,
        sample_seed=sample_seed,
        split=split,
        source=source,
        shape_seed=shape_seed,
        allow_truncation=allow_truncation,
        stl_path=stl_path,
        stl_chunk_id=stl_chunk_id,
    )


def _compute_object_view_metrics(object_depth: np.ndarray) -> dict[str, float | int | bool]:
    valid_mask = np.isfinite(object_depth) & (object_depth > 0.0)
    visible_pixels = int(np.count_nonzero(valid_mask))
    h, w = object_depth.shape
    total_pixels = max(int(h * w), 1)
    if visible_pixels <= 0:
        return {
            "visible_pixels": 0,
            "pixel_coverage": 0.0,
            "bbox_width_px": 0,
            "bbox_height_px": 0,
            "bbox_min_dimension_px": 0,
            "border_touch_pixels": 0,
            "border_touch_ratio": 0.0,
            "touches_border": False,
        }

    ys, xs = np.nonzero(valid_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    border_mask = np.zeros_like(valid_mask, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_touch_pixels = int(np.count_nonzero(valid_mask & border_mask))

    return {
        "visible_pixels": visible_pixels,
        "pixel_coverage": float(visible_pixels / total_pixels),
        "bbox_width_px": int(bbox_width),
        "bbox_height_px": int(bbox_height),
        "bbox_min_dimension_px": int(min(bbox_width, bbox_height)),
        "border_touch_pixels": border_touch_pixels,
        "border_touch_ratio": float(border_touch_pixels / max(visible_pixels, 1)),
        "touches_border": bool(border_touch_pixels > 0),
    }


def _generate_sample_from_plan(
    plan: SamplePlan,
    out_samples_dir: str,
    save_components: bool,
    compressed_npz: bool,
    max_attempts_per_sample: int,
) -> dict[str, Any]:
    if _WORKER_SCENE_CFG is None:
        raise RuntimeError("Worker not initialized")

    scene_cfg = _WORKER_SCENE_CFG
    plane_cfg = scene_cfg.get("plane_fit", {})
    bg_norm_cfg = scene_cfg.get("background_normalization", {})
    camera_cfg = scene_cfg["camera"]
    place_cfg = scene_cfg["placement"]
    camera_artifacts_cfg = scene_cfg.get("camera_artifacts", {})

    for attempt in range(max_attempts_per_sample):
        sample_seed = plan.sample_seed if attempt == 0 else _seed_for_sample(plan.sample_seed, plan.sample_index, attempt)
        rng_master = np.random.default_rng(sample_seed)

        background_depth_raw, background_id, background_source = _sample_background_depth(
            scene_cfg=scene_cfg,
            camera_cfg=camera_cfg,
            rng_master=rng_master,
            forced_source=plan.background_source,
        )

        norm_enabled_cfg = bool(bg_norm_cfg.get("enabled", True))
        bg_has_valid_depth = bool(np.any(np.isfinite(background_depth_raw) & (background_depth_raw > 1e-6)))
        norm_enabled = bool(norm_enabled_cfg and bg_has_valid_depth)
        if norm_enabled:
            bg_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            background_depth, bg_transform = normalize_and_randomize_background_depth(
                depth_m=background_depth_raw,
                camera_cfg=camera_cfg,
                rng=bg_rng,
                distance_range_m=tuple(bg_norm_cfg.get("distance_range_m", [1.8, 2.5])),
                pitch_deg_range=tuple(bg_norm_cfg.get("pitch_deg_range", [-20.0, 20.0])),
                yaw_deg_range=tuple(bg_norm_cfg.get("yaw_deg_range", [-20.0, 20.0])),
                fill_fov=bool(bg_norm_cfg.get("fill_fov", True)),
                target_fill_ratio=float(bg_norm_cfg.get("target_fill_ratio", 0.98)),
                max_inplane_scale=float(bg_norm_cfg.get("max_inplane_scale", 3.0)),
                middle_percentile=float(bg_norm_cfg.get("middle_percentile", 0.90)),
                out_of_plane_range_m=tuple(bg_norm_cfg.get("out_of_plane_range_m", [0.0, 0.2])),
            )
        else:
            background_depth = background_depth_raw
            bg_transform = None

        plane = fit_plane_from_depth(
            depth_m=background_depth,
            camera_cfg=camera_cfg,
            stride=int(plane_cfg.get("stride", 2)),
            middle_percentile=float(plane_cfg.get("middle_percentile", 0.90)),
            seed=int(rng_master.integers(0, 2**31 - 1)),
        )

        object_seed = plan.shape_seed
        try:
            if plan.source == "stl":
                if not plan.stl_path:
                    raise RuntimeError("STL sample plan is missing stl_path")
                if not _WORKER_STL_ASSETS:
                    raise RuntimeError("No preloaded STL assets available in worker cache")

                # Prefer the planned STL; if it was skipped during preload, fall back to another valid STL.
                candidate_paths: list[str] = []
                if plan.stl_path in _WORKER_STL_ASSETS:
                    candidate_paths.append(plan.stl_path)

                alternatives = [p for p in _WORKER_STL_ASSETS.keys() if p != plan.stl_path]
                if alternatives:
                    alt_offset = int(rng_master.integers(0, len(alternatives)))
                    alternatives = alternatives[alt_offset:] + alternatives[:alt_offset]
                    candidate_paths.extend(alternatives)

                stl_build_error: RuntimeError | None = None
                for selected_stl_path in candidate_paths:
                    cached_asset = _WORKER_STL_ASSETS.get(selected_stl_path)
                    if cached_asset is None:
                        continue
                    try:
                        canonical_mesh, model_cloud, bbox_corners, shape_params = build_stl_canonical_model_from_cache(
                            scene_cfg,
                            selected_path=Path(selected_stl_path),
                            cached_asset=cached_asset,
                            seed=object_seed,
                            include_point_cloud=True,
                        )
                        break
                    except RuntimeError as e:
                        stl_build_error = e
                else:
                    if stl_build_error is not None:
                        raise stl_build_error
                    raise RuntimeError(f"No usable STL assets found for sample {plan.sample_index}")
            else:
                canonical_mesh, model_cloud, bbox_corners, shape_params = generate_superquadric_canonical_model(
                    scene_cfg,
                    seed=object_seed,
                    include_point_cloud=True,
                )
                shape_params = dict(shape_params)
                shape_params.setdefault("object_source", "superquadric")
        except RuntimeError:
            continue

        if model_cloud is None:
            continue

        bbox_extent = (bbox_corners.max(axis=0) - bbox_corners.min(axis=0)).astype(np.float64)
        model_points = np.asarray(model_cloud.points, dtype=np.float32)

        constraints = PlacementConstraints(
            min_camera_distance_m=float(place_cfg["min_camera_distance_m"]),
            max_camera_distance_m=float(place_cfg["max_camera_distance_m"]),
        )

        place_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        try:
            placement = sample_pose_from_camera(
                camera_cfg=camera_cfg,
                bbox_extent_m=bbox_extent,
                constraints=constraints,
                rng=place_rng,
                max_tries=int(place_cfg.get("max_attempts", 400)),
                center_margin_ratio=float(place_cfg.get("center_margin_ratio_core", 0.12)),
                allow_edge_sampling=bool(plan.allow_truncation),
            )
        except RuntimeError:
            continue

        mesh_world = transform_mesh(
            canonical_mesh,
            position_xyz=placement.position_xyz,
            euler_deg_xyz=placement.orientation_euler_deg_xyz,
        )

        object_depth = render_mesh_depth(mesh_world, camera_cfg)
        view_metrics = _compute_object_view_metrics(object_depth)

        min_pixel_coverage = float(place_cfg.get("min_pixel_coverage", 0.10))
        max_pixel_coverage = float(place_cfg.get("max_pixel_coverage", 0.95))
        min_bbox_dimension_px = int(place_cfg.get("min_bbox_dimension_px", 20))
        max_border_touch_ratio_core = float(place_cfg.get("max_border_touch_ratio_core", 0.10))

        if not (min_pixel_coverage <= float(view_metrics["pixel_coverage"]) <= max_pixel_coverage):
            continue
        if int(view_metrics["bbox_min_dimension_px"]) < min_bbox_dimension_px:
            continue
        # Truncated samples are best-effort: require border touch in early tries, then
        # relax to avoid disproportionately high late-stage generation failures.
        require_truncated = bool(plan.allow_truncation) and attempt < max(max_attempts_per_sample // 2, 1)
        if require_truncated:
            if not bool(view_metrics["touches_border"]):
                continue
        elif not plan.allow_truncation:
            if float(view_metrics["border_touch_ratio"]) >= max_border_touch_ratio_core:
                continue

        composite_depth = compose_depth(background_depth, object_depth)

        artifacts_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        composite_depth, artifact_stats = apply_camera_artifacts(
            composite_depth,
            camera_artifacts_cfg,
            background_depth_m=background_depth,
            object_depth_m=object_depth,
            camera_cfg=camera_cfg,
            rng=artifacts_rng,
        )

        # Binary mask in image space: object has valid rendered depth (same grid as composite_depth).
        object_mask = (np.isfinite(object_depth) & (object_depth > 0.0)).astype(np.float32)

        sample_id = f"{plan.sample_index:06d}"
        out_path = Path(out_samples_dir) / f"{sample_id}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if save_components:
            save_dict = {
                "background_depth_raw_m": background_depth_raw.astype(np.float32),
                "background_depth_m": background_depth.astype(np.float32),
                "object_depth_m": object_depth.astype(np.float32),
                "composite_depth_m": composite_depth.astype(np.float32),
                "object_mask": object_mask,
                "model_points": model_points,
            }
        else:
            save_dict = {
                "composite_depth_m": composite_depth.astype(np.float32),
                "object_mask": object_mask,
                "model_points": model_points,
            }

        if compressed_npz:
            np.savez_compressed(out_path, **save_dict)
        else:
            np.savez(out_path, **save_dict)

        rot = Rotation.from_quat(placement.orientation_quat_xyzw).as_matrix().astype(np.float64)
        t = placement.position_xyz.astype(np.float64)
        t_cam_from_obj = np.eye(4, dtype=np.float64)
        t_cam_from_obj[:3, :3] = rot
        t_cam_from_obj[:3, 3] = t

        domain_tags = [
            "depth",
            "synthetic_object",
            "background_composite",
            f"background_source:{background_source}",
            f"object_source:{shape_params.get('object_source', 'unknown')}",
            "visibility:truncated" if plan.allow_truncation else "visibility:core",
        ]

        metadata = {
            "sample_id": sample_id,
            "split": plan.split,
            "sample_seed": int(plan.sample_seed),
            "domain_tags": domain_tags,
            "object_id": shape_params.get("object_id", ""),
            "object_source": shape_params.get("object_source", "unknown"),
            "object_asset_path": shape_params.get("object_asset_path", None),
            "background_id": background_id,
            "background_asset_path": background_id,
            "background_source": background_source,
            "bbox_extent_m": bbox_extent.tolist(),
            "bbox_corners_m": bbox_corners.astype(np.float32).tolist(),
            "gt_transform_camera_from_object": t_cam_from_obj.tolist(),
            "depth_npz": str(out_path).replace("\\", "/"),
        }

        debug_metadata = {
            "sample_id": sample_id,
            "success": True,
            "seed": plan.sample_seed,
            "background_id": background_id,
            "background_source": background_source,
            "object_source": shape_params.get("object_source", "unknown"),
            "object_id": shape_params.get("object_id", ""),
            "object_asset_path": shape_params.get("object_asset_path", None),
            "background_asset_path": background_id,
            "npz": str(out_path).replace("\\", "/"),
            "bbox_extent_m": bbox_extent.tolist(),
            "plane": {
                "normal": plane.normal.tolist(),
                "offset": float(plane.offset),
                "inlier_ratio": float(plane.inlier_ratio),
            },
            "placement": {
                "position_xyz": placement.position_xyz.tolist(),
                "orientation_euler_deg_xyz": placement.orientation_euler_deg_xyz.tolist(),
                "orientation_quat_xyzw": placement.orientation_quat_xyzw.tolist(),
                "plane_offset_m": float(placement.plane_offset_m),
                "center_pixel_uv": placement.center_pixel_uv.tolist(),
                "allow_truncation": bool(plan.allow_truncation),
            },
            "object_view_metrics": view_metrics,
            "background_normalization": {
                "enabled": norm_enabled,
                "transform": None
                if bg_transform is None
                else {
                    "pitch_deg": float(bg_transform.pitch_deg),
                    "yaw_deg": float(bg_transform.yaw_deg),
                    "distance_m": float(bg_transform.distance_m),
                    "inplane_scale_xy": float(bg_transform.inplane_scale_xy),
                    "projected_fill_u": float(bg_transform.projected_fill_u),
                    "projected_fill_v": float(bg_transform.projected_fill_v),
                    "out_of_plane_scale_m": float(bg_transform.out_of_plane_scale_m),
                },
            },
            "camera_artifacts": artifact_stats,
            "shape_params": shape_params,
            "gt_transform_camera_from_object": t_cam_from_obj.tolist(),
        }

        return {
            "success": True,
            "metadata": metadata,
            "debug_metadata": debug_metadata,
        }

    return {
        "success": False,
        "metadata": {
            "sample_id": f"{plan.sample_index:06d}",
            "split": plan.split,
            "sample_seed": int(plan.sample_seed),
            "domain_tags": ["failed_generation"],
            "object_id": "",
            "object_source": "unknown",
            "object_asset_path": None,
            "background_id": "",
            "background_asset_path": None,
            "gt_transform_camera_from_object": np.eye(4, dtype=np.float64).tolist(),
        },
        "debug_metadata": {
            "sample_id": f"{plan.sample_index:06d}",
            "success": False,
        },
        "error": f"failed_after_{max_attempts_per_sample}_attempts",
    }


def _generate_plan_batch(
    plans: list[SamplePlan],
    out_samples_dir: str,
    save_components: bool,
    compressed_npz: bool,
    max_attempts_per_sample: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for plan in plans:
        out.append(
            _generate_sample_from_plan(
                plan=plan,
                out_samples_dir=out_samples_dir,
                save_components=save_components,
                compressed_npz=compressed_npz,
                max_attempts_per_sample=max_attempts_per_sample,
            )
        )
    return out


def build_dataset(
    scene_cfg: dict[str, Any],
    background_paths: list[str],
    output_dir: Path,
    num_samples: int,
    base_seed: int,
    split: str,
    workers: int,
    samples_per_task: int,
    max_backgrounds_in_ram: int,
    object_source: str,
    debug_metadata: bool,
    save_components: bool,
    compressed_npz: bool,
    stl_chunk_size: int,
    max_attempts_per_sample: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    metadata_jsonl = output_dir / "metadata.jsonl"
    debug_metadata_jsonl = output_dir / "metadata_debug.jsonl"
    summary_json = output_dir / "summary.json"

    if stl_chunk_size <= 0:
        stl_chunk_size = 1

    background_weights = _sample_background_weights(scene_cfg)
    _init_worker_scene(scene_cfg)
    if background_weights["real"] > 0.0:
        _load_background_cache(scene_cfg, background_paths, max_backgrounds_in_ram)
    else:
        _WORKER_BG_DEPTHS.clear()
        _WORKER_BG_IDS.clear()
    place_cfg = scene_cfg.get("placement", {})

    stl_files = _list_stl_files(scene_cfg)
    stl_chunk_size = min(stl_chunk_size, max(len(stl_files), 1))
    stl_available = bool(stl_files)

    truncated_fraction_min = float(place_cfg.get("truncated_fraction_min", 0.10))
    truncated_fraction_max = float(place_cfg.get("truncated_fraction_max", 0.20))
    truncated_fraction_min = float(np.clip(truncated_fraction_min, 0.0, 1.0))
    truncated_fraction_max = float(np.clip(truncated_fraction_max, 0.0, 1.0))
    if truncated_fraction_max < truncated_fraction_min:
        truncated_fraction_min, truncated_fraction_max = truncated_fraction_max, truncated_fraction_min
    split_rng = np.random.default_rng(int(base_seed + 917623))
    target_truncated_fraction = float(
        split_rng.uniform(truncated_fraction_min, truncated_fraction_max)
        if truncated_fraction_max > truncated_fraction_min
        else truncated_fraction_min
    )
    target_truncated_count = int(round(num_samples * target_truncated_fraction))
    target_truncated_count = max(0, min(target_truncated_count, num_samples))
    truncation_indices = set(
        int(i)
        for i in split_rng.choice(num_samples, size=target_truncated_count, replace=False)
    )

    sample_plans: list[SamplePlan] = []
    for sample_index in range(num_samples):
        plan = _build_sample_plan(
            sample_index=sample_index,
            base_seed=base_seed,
            split=split,
            scene_cfg=scene_cfg,
            stl_files=stl_files,
            stl_available=stl_available,
            stl_chunk_size=stl_chunk_size,
            source_override=object_source,
            allow_truncation=sample_index in truncation_indices,
        )
        sample_plans.append(plan)
    sample_plans = _assign_background_sources_stratified(
        sample_plans=sample_plans,
        background_weights=background_weights,
        seed=int(base_seed + 294117),
    )
    superquadric_plans: list[SamplePlan] = []
    stl_plans_by_chunk: dict[int, list[SamplePlan]] = {}
    for plan in sample_plans:
        if plan.source == "stl" and plan.stl_chunk_id is not None:
            stl_plans_by_chunk.setdefault(plan.stl_chunk_id, []).append(plan)
        else:
            superquadric_plans.append(plan)

    def _write_records(batch_records: list[dict[str, Any]], meta_f, debug_f) -> tuple[int, int]:
        batch_success = 0
        batch_fail = 0
        for rec in batch_records:
            meta_f.write(json.dumps(rec["metadata"]) + "\n")
            meta_f.flush()
            os.fsync(meta_f.fileno())
            if debug_f is not None:
                debug_f.write(json.dumps(rec["debug_metadata"]) + "\n")
                debug_f.flush()
                os.fsync(debug_f.fileno())
            if rec.get("success", False):
                batch_success += 1
            else:
                batch_fail += 1
        return batch_success, batch_fail

    def _run_batch(plans: list[SamplePlan], stl_assets: dict[str, Any]) -> list[dict[str, Any]]:
        if not plans:
            return 0, 0
        _set_stl_cache(stl_assets)
        batch_success = 0
        batch_fail = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_sample_from_plan,
                    plan,
                    str(samples_dir),
                    save_components,
                    compressed_npz,
                    max_attempts_per_sample,
                )
                for plan in plans
            ]
            for fut in as_completed(futures):
                rec = fut.result()
                meta_f.write(json.dumps(rec["metadata"]) + "\n")
                meta_f.flush()
                os.fsync(meta_f.fileno())
                if debug_f is not None:
                    debug_f.write(json.dumps(rec["debug_metadata"]) + "\n")
                    debug_f.flush()
                    os.fsync(debug_f.fileno())
                if rec.get("success", False):
                    batch_success += 1
                else:
                    batch_fail += 1
                progress_bar.update(1)
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                progress_bar.set_postfix(
                    success=success_count + batch_success,
                    failed=fail_count + batch_fail,
                    rate=f"{progress_bar.n / elapsed:.2f}/s",
                )
        return batch_success, batch_fail

    start_time = time.perf_counter()
    success_count = 0
    fail_count = 0

    with metadata_jsonl.open("w", encoding="utf-8") as meta_f:
        debug_f = None
        if debug_metadata:
            debug_f = debug_metadata_jsonl.open("w", encoding="utf-8")
        with tqdm(total=num_samples, desc=f"Generating {split}", unit="sample") as progress_bar:
            batch_success, batch_fail = _run_batch(superquadric_plans, {})
            success_count += batch_success
            fail_count += batch_fail

            for chunk_id in tqdm(sorted(stl_plans_by_chunk.keys()), desc="STL chunks", unit="chunk", leave=False):
                chunk_start = chunk_id * stl_chunk_size
                chunk_end = min(len(stl_files), chunk_start + stl_chunk_size)
                chunk_paths = stl_files[chunk_start:chunk_end]
                stl_assets = preload_stl_asset_chunk(
                    scene_cfg,
                    chunk_paths,
                    int(scene_cfg.get("scene", {}).get("stl", {}).get("points", {}).get("sample_count", 5000)),
                )
                batch_success, batch_fail = _run_batch(stl_plans_by_chunk[chunk_id], stl_assets)
                success_count += batch_success
                fail_count += batch_fail

        if debug_f is not None:
            debug_f.close()

    elapsed = time.perf_counter() - start_time
    summary = {
        "num_requested": num_samples,
        "num_success": success_count,
        "num_failed": fail_count,
        "elapsed_sec": elapsed,
        "throughput_samples_per_sec": 0.0 if elapsed <= 0 else float(success_count / elapsed),
        "split": split,
        "workers": workers,
        "samples_per_task": samples_per_task,
        "backgrounds_available": len(background_paths),
        "background_source_weights": background_weights,
        "max_backgrounds_in_ram": max_backgrounds_in_ram,
        "debug_metadata": debug_metadata,
        "save_components": save_components,
        "compressed_npz": compressed_npz,
        "object_source": object_source,
        "metadata_jsonl": str(metadata_jsonl).replace("\\", "/"),
        "metadata_debug_jsonl": None
        if not debug_metadata
        else str(debug_metadata_jsonl).replace("\\", "/"),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk sample generation with multiprocessing")
    parser.add_argument(
        "--scene_config",
        default="data_engine/config/scene_config.superquadric.example.json",
        help="Path to scene config JSON",
    )
    parser.add_argument(
        "--dataset_config",
        default="data_engine/config/dataset_config.example.json",
        help="Path to dataset-generation config JSON",
    )
    args = parser.parse_args()

    scene_cfg = load_json(Path(args.scene_config))
    dataset_cfg = load_json(Path(args.dataset_config))

    output_root = Path(dataset_cfg.get("output_dir", "data/generated/bulk_dataset"))
    output_root.mkdir(parents=True, exist_ok=True)

    background_glob = str(dataset_cfg.get("background_glob", "data/backgrounds/raw/**/depth/*.npz"))
    background_paths = sorted(str(p) for p in Path(".").glob(background_glob) if p.is_file())
    if not background_paths and _sample_background_weights(scene_cfg).get("real", 0.0) > 0.0:
        raise RuntimeError(f"No background files matched: {background_glob}")

    workers = int(dataset_cfg.get("workers", 0))
    if workers <= 0:
        workers = max((os.cpu_count() or 4) - 1, 1)

    max_backgrounds_in_ram = int(dataset_cfg.get("max_backgrounds_in_ram", 0))
    if max_backgrounds_in_ram < 0:
        max_backgrounds_in_ram = 0

    samples_per_task = max(int(dataset_cfg.get("samples_per_task", 8)), 1)
    object_source = str(dataset_cfg.get("object_source", "random"))
    debug_metadata = bool(dataset_cfg.get("debug_metadata", False))
    save_components = bool(dataset_cfg.get("save_components", False))
    compressed_npz = bool(dataset_cfg.get("compressed_npz", False))
    stl_chunk_size = max(int(dataset_cfg.get("stl_chunk_size", 32)), 1)
    max_attempts_per_sample = max(int(dataset_cfg.get("max_attempts_per_sample", 20)), 1)
    base_seed = int(dataset_cfg.get("seed", 12345))

    split_counts = _split_counts_from_config(dataset_cfg)

    split_summaries: dict[str, Any] = {}
    total_requested = 0
    total_success = 0
    total_failed = 0
    total_elapsed = 0.0

    for split_index, (split_name, split_n) in enumerate(split_counts.items()):
        if split_n <= 0:
            continue

        split_out = output_root / split_name
        split_seed = base_seed + split_index * 10000019

        print(
            f"Starting split '{split_name}': samples={split_n}, workers={workers}, "
            f"backgrounds={len(background_paths)}, max_backgrounds_in_ram={max_backgrounds_in_ram or 'all'}"
        )

        summary = build_dataset(
            scene_cfg=scene_cfg,
            background_paths=background_paths,
            output_dir=split_out,
            num_samples=int(split_n),
            base_seed=int(split_seed),
            split=str(split_name),
            workers=workers,
            samples_per_task=samples_per_task,
            max_backgrounds_in_ram=max_backgrounds_in_ram,
            object_source=object_source,
            debug_metadata=debug_metadata,
            save_components=save_components,
            compressed_npz=compressed_npz,
            stl_chunk_size=stl_chunk_size,
            max_attempts_per_sample=max_attempts_per_sample,
        )
        split_summaries[split_name] = summary

        total_requested += int(summary["num_requested"])
        total_success += int(summary["num_success"])
        total_failed += int(summary["num_failed"])
        total_elapsed += float(summary["elapsed_sec"])

    overall = {
        "num_requested": total_requested,
        "num_success": total_success,
        "num_failed": total_failed,
        "elapsed_sec_sum_splits": total_elapsed,
        "throughput_samples_per_sec": 0.0 if total_elapsed <= 0 else float(total_success / total_elapsed),
        "workers": workers,
        "backgrounds_available": len(background_paths),
        "max_backgrounds_in_ram": max_backgrounds_in_ram,
        "object_source": object_source,
        "debug_metadata": debug_metadata,
        "save_components": save_components,
        "compressed_npz": compressed_npz,
        "seed": base_seed,
        "train_validation_split": split_counts,
        "splits": split_summaries,
    }

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(
        f"Done: success={overall['num_success']}, failed={overall['num_failed']}, "
        f"throughput={overall['throughput_samples_per_sec']:.2f} samples/s"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
