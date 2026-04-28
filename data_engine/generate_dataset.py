"""Bulk dataset generator with multiprocessing and in-worker background RAM caching."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from data_engine.composition.background_normalization import normalize_and_randomize_background_depth
from data_engine.composition.camera_artifacts import apply_camera_artifacts
from data_engine.composition.depth_compositor import compose_depth, render_mesh_depth, transform_mesh
from data_engine.composition.plane_fit import fit_plane_from_depth
from data_engine.composition.plane_placement import PlacementConstraints
from data_engine.composition.placement_sampling import sample_pose_on_plane
from data_engine.generators import generate_mixed_canonical_model

_WORKER_SCENE_CFG: dict[str, Any] | None = None
_WORKER_BG_DEPTHS: list[np.ndarray] = []
_WORKER_BG_IDS: list[str] = []


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


def _init_worker(scene_cfg: dict[str, Any], background_paths: list[str], max_backgrounds_in_ram: int) -> None:
    global _WORKER_SCENE_CFG, _WORKER_BG_DEPTHS, _WORKER_BG_IDS

    _WORKER_SCENE_CFG = scene_cfg
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
        if expected_w > 0 and expected_h > 0:
            if arr.shape != (expected_h, expected_w):
                continue
        depths.append(arr)
        ids.append(str(Path(p).as_posix()))

    if not depths:
        raise RuntimeError(
            "Worker could not load any resolution-compatible backgrounds into RAM. "
            f"Expected shape {(expected_h, expected_w)}."
        )

    _WORKER_BG_DEPTHS = depths
    _WORKER_BG_IDS = ids


def _generate_one_sample(
    sample_index: int,
    base_seed: int,
    split: str,
    object_source: str,
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
        sample_seed = _seed_for_sample(base_seed, sample_index, attempt)
        rng_master = np.random.default_rng(sample_seed)

        bg_idx = int(rng_master.integers(0, len(_WORKER_BG_DEPTHS)))
        background_depth_raw = _WORKER_BG_DEPTHS[bg_idx]
        background_id = _WORKER_BG_IDS[bg_idx]

        norm_enabled = bool(bg_norm_cfg.get("enabled", True))
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

        object_seed = int(rng_master.integers(0, 2**31 - 1))
        try:
            canonical_mesh, model_cloud, bbox_corners, shape_params = generate_mixed_canonical_model(
                scene_cfg,
                seed=object_seed,
                source_override=object_source,
                include_point_cloud=True,
            )
        except RuntimeError:
            continue

        if model_cloud is None:
            continue

        bbox_extent = (bbox_corners.max(axis=0) - bbox_corners.min(axis=0)).astype(np.float64)
        model_points = np.asarray(model_cloud.points, dtype=np.float32)

        constraints = PlacementConstraints(
            min_plane_distance_m=float(place_cfg["min_plane_distance_m"]),
            max_plane_distance_m=float(place_cfg["max_plane_distance_m"]),
        )

        place_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        try:
            placement = sample_pose_on_plane(
                plane=plane,
                camera_cfg=camera_cfg,
                bbox_extent_m=bbox_extent,
                constraints=constraints,
                rng=place_rng,
                max_tries=int(place_cfg.get("max_attempts", 400)),
            )
        except RuntimeError:
            continue

        mesh_world = transform_mesh(
            canonical_mesh,
            position_xyz=placement.position_xyz,
            euler_deg_xyz=placement.orientation_euler_deg_xyz,
        )

        object_depth = render_mesh_depth(mesh_world, camera_cfg)
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

        sample_id = f"{sample_index:06d}"
        out_path = Path(out_samples_dir) / f"{sample_id}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if save_components:
            save_dict = {
                "background_depth_raw_m": background_depth_raw.astype(np.float32),
                "background_depth_m": background_depth.astype(np.float32),
                "object_depth_m": object_depth.astype(np.float32),
                "composite_depth_m": composite_depth.astype(np.float32),
                "model_points": model_points,
            }
        else:
            save_dict = {
                "composite_depth_m": composite_depth.astype(np.float32),
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
            "real_background_composite",
            f"object_source:{shape_params.get('object_source', 'unknown')}",
        ]

        metadata = {
            "sample_id": sample_id,
            "split": split,
            "sample_seed": int(sample_seed),
            "domain_tags": domain_tags,
            "object_id": shape_params.get("object_id", ""),
            "object_source": shape_params.get("object_source", "unknown"),
            "object_asset_path": shape_params.get("object_asset_path", None),
            "background_id": background_id,
            "background_asset_path": background_id,
            "bbox_extent_m": bbox_extent.tolist(),
            "bbox_corners_m": bbox_corners.astype(np.float32).tolist(),
            "gt_transform_camera_from_object": t_cam_from_obj.tolist(),
            "depth_npz": str(out_path).replace("\\", "/"),
        }

        debug_metadata = {
            "sample_id": sample_id,
            "success": True,
            "seed": sample_seed,
            "background_id": background_id,
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
            },
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
            "sample_id": f"{sample_index:06d}",
            "split": split,
            "sample_seed": int(_seed_for_sample(base_seed, sample_index, 0)),
            "domain_tags": ["failed_generation"],
            "object_id": "",
            "object_source": "unknown",
            "object_asset_path": None,
            "background_id": "",
            "background_asset_path": None,
            "gt_transform_camera_from_object": np.eye(4, dtype=np.float64).tolist(),
        },
        "debug_metadata": {
            "sample_id": f"{sample_index:06d}",
            "success": False,
        },
        "error": f"failed_after_{max_attempts_per_sample}_attempts",
    }


def _generate_chunk(
    start_idx: int,
    end_idx: int,
    base_seed: int,
    split: str,
    object_source: str,
    out_samples_dir: str,
    save_components: bool,
    compressed_npz: bool,
    max_attempts_per_sample: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(start_idx, end_idx):
        out.append(
            _generate_one_sample(
                sample_index=i,
                base_seed=base_seed,
                split=split,
                object_source=object_source,
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
    max_attempts_per_sample: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    metadata_jsonl = output_dir / "metadata.jsonl"
    debug_metadata_jsonl = output_dir / "metadata_debug.jsonl"
    summary_json = output_dir / "summary.json"

    chunk_count = int(math.ceil(num_samples / samples_per_task))
    tasks: list[tuple[int, int]] = []
    for chunk_idx in range(chunk_count):
        s = chunk_idx * samples_per_task
        e = min(num_samples, s + samples_per_task)
        tasks.append((s, e))

    start_time = time.perf_counter()
    success_count = 0
    fail_count = 0

    with metadata_jsonl.open("w", encoding="utf-8") as meta_f:
        debug_f = None
        if debug_metadata:
            debug_f = debug_metadata_jsonl.open("w", encoding="utf-8")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(scene_cfg, background_paths, max_backgrounds_in_ram),
        ) as ex:
            futures = [
                ex.submit(
                    _generate_chunk,
                    s,
                    e,
                    base_seed,
                    split,
                    object_source,
                    str(samples_dir),
                    save_components,
                    compressed_npz,
                    max_attempts_per_sample,
                )
                for s, e in tasks
            ]

            processed = 0
            for fut in as_completed(futures):
                records = fut.result()
                for rec in records:
                    meta_f.write(json.dumps(rec["metadata"]) + "\n")
                    if debug_f is not None:
                        debug_f.write(json.dumps(rec["debug_metadata"]) + "\n")
                    processed += 1
                    if rec.get("success", False):
                        success_count += 1
                    else:
                        fail_count += 1

                if processed % max(50, samples_per_task) == 0 or processed == num_samples:
                    elapsed = max(time.perf_counter() - start_time, 1e-9)
                    rate = processed / elapsed
                    print(
                        f"Progress: {processed}/{num_samples} samples, "
                        f"success={success_count}, fail={fail_count}, "
                        f"rate={rate:.2f} samples/s"
                    )

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
    if not background_paths:
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
