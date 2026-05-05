"""Debug utility: inspect one generated dataset sample with the existing sample viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from data_engine.visualization import visualize_sample


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_depth_components(sample: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    composite_key = None
    for candidate in ("composite_depth_m", "depth_image", "depth_m"):
        if candidate in sample.files:
            composite_key = candidate
            break
    if composite_key is None:
        raise KeyError("Sample does not contain a depth array")

    composite_depth = np.asarray(sample[composite_key], dtype=np.float32)
    background_depth = np.asarray(sample["background_depth_m"], dtype=np.float32) if "background_depth_m" in sample.files else None
    object_depth = np.asarray(sample["object_depth_m"], dtype=np.float32) if "object_depth_m" in sample.files else None
    return composite_depth, background_depth, object_depth


def _points_to_cloud(points_xyz: np.ndarray, color_rgb: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
    cloud.paint_uniform_color(list(color_rgb))
    return cloud


def _transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = (np.asarray(transform_4x4, dtype=np.float64) @ points_h.T).T
    return transformed[:, :3]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_npz_path(split_dir: Path, row: dict) -> Path:
    raw_path = row.get("depth_npz")
    if raw_path:
        candidate = Path(str(raw_path))
        search_paths = [
            candidate,
            split_dir / candidate,
            split_dir.parent / candidate,
            split_dir / "samples" / candidate.name,
        ]
        for search_path in search_paths:
            if search_path.exists():
                return search_path

    fallback = split_dir / "samples" / f"{row['sample_id']}.npz"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not resolve sample file for {row.get('sample_id')!r}")


def _load_records(dataset_root: Path, split: str | None) -> list[dict]:
    if split:
        split_dir = dataset_root / split
        metadata_path = split_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        return [
            {
                "split_dir": split_dir,
                "sample_id": str(row["sample_id"]),
                "split": str(row.get("split", split_dir.name)),
                "sample_seed": int(row.get("sample_seed", 0)),
                "depth_npz": _resolve_npz_path(split_dir, row),
                "gt_transform_camera_from_object": np.asarray(row["gt_transform_camera_from_object"], dtype=np.float32),
                "object_source": str(row.get("object_source", "unknown")),
            }
            for row in _read_jsonl(metadata_path)
        ]

    records: list[dict] = []
    for metadata_path in dataset_root.rglob("metadata.jsonl"):
        split_dir = metadata_path.parent
        try:
            rows = _read_jsonl(metadata_path)
        except FileNotFoundError:
            continue
        for row in rows:
            records.append(
                {
                    "split_dir": split_dir,
                    "sample_id": str(row["sample_id"]),
                    "split": str(row.get("split", split_dir.name)),
                    "sample_seed": int(row.get("sample_seed", 0)),
                    "depth_npz": _resolve_npz_path(split_dir, row),
                    "gt_transform_camera_from_object": np.asarray(row["gt_transform_camera_from_object"], dtype=np.float32),
                    "object_source": str(row.get("object_source", "unknown")),
                }
            )
    return records


def _pick_record(
    records: list[dict],
    sample_id: str | None,
    random_sample: bool,
) -> dict:
    if not records:
        raise RuntimeError("No generated samples were found")

    if sample_id:
        for record in records:
            if record["sample_id"] == sample_id:
                return record
        raise FileNotFoundError(f"Could not find sample_id={sample_id!r}")

    rng = np.random.default_rng()
    index = int(rng.integers(0, len(records)))
    return records[index]


def _resolve_sample_path(args: argparse.Namespace) -> tuple[Path, dict | None]:
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    split = args.split.strip() if args.split else None

    if args.sample_npz:
        sample_path = Path(args.sample_npz).expanduser().resolve()
        record = None
        if dataset_root.exists():
            all_records = _load_records(dataset_root, split)
            stem = sample_path.stem
            for candidate in all_records:
                if candidate["sample_id"] == stem or candidate["depth_npz"].resolve() == sample_path:
                    record = candidate
                    break
        return sample_path, record

    records = _load_records(dataset_root, split)
    if not records:
        raise RuntimeError(f"No metadata.jsonl files found under {dataset_root}")

    record = _pick_record(records, args.sample_id, args.random_sample)
    return record["depth_npz"].resolve(), record


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a generated dataset sample with visualization")
    parser.add_argument(
        "--dataset_dir",
        default="data/generated/dataset_v2",
        help="Root directory containing generated split folders",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to search within (useful for random or sample_id selection)",
    )
    parser.add_argument("--sample_id", default="", help="Specific sample_id to load")
    parser.add_argument("--sample_npz", default="", help="Explicit path to a sample NPZ to inspect")
    parser.add_argument("--random_sample", action="store_true", help="Pick a random sample from the selected split")
    parser.add_argument(
        "--scene_config",
        default="data_engine/config/scene_config.json",
        help="Path to scene config JSON providing camera intrinsics",
    )
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization windows")
    args = parser.parse_args()

    if bool(args.sample_id.strip()) and args.random_sample:
        raise ValueError("--sample_id and --random_sample are mutually exclusive")
    if not args.sample_npz and not args.sample_id.strip() and not args.random_sample:
        args.random_sample = True

    scene_cfg = load_json(Path(args.scene_config))
    sample_path, record = _resolve_sample_path(args)

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample NPZ does not exist: {sample_path}")

    with np.load(sample_path, allow_pickle=False) as sample:
        composite_depth, background_depth, object_depth = _load_depth_components(sample)
        model_points = np.asarray(sample["model_points"], dtype=np.float32) if "model_points" in sample.files else None

    object_cloud = None
    if model_points is not None and record is not None:
        object_points_cam = _transform_points(model_points, record["gt_transform_camera_from_object"])
        object_cloud = _points_to_cloud(object_points_cam, color_rgb=(0.9, 0.3, 0.2))

    background_plot = background_depth if background_depth is not None else composite_depth
    object_plot = object_depth

    print(f"Loaded sample: {sample_path}")
    if record is not None:
        print(f"Sample id: {record['sample_id']} | split: {record['split']} | object_source: {record['object_source']}")
    else:
        print("No metadata record was found for this NPZ; showing depth data only.")

    if not args.no_vis:
        visualize_sample(
            background_depth_m=background_plot,
            object_depth_m=object_plot,
            composite_depth_m=composite_depth,
            object_mesh_world=None,
            camera_cfg=scene_cfg["camera"],
            extra_clouds=[object_cloud] if object_cloud is not None else None,
        )


if __name__ == "__main__":
    main()