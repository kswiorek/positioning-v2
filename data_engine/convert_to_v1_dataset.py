"""Convert a v2 generated dataset into the legacy v1 positioning training layout.

v2 layout (per split):
  <input>/<split>/samples/{sample_id}.npz
  <input>/<split>/metadata.jsonl

v1 layout:
  <output>/<split>/{index:06d}.npz   keys: depth_image, model_points, bbox_corners, gt_transform
  <output>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

V1_SAMPLE_KEYS = ("depth_image", "model_points", "bbox_corners", "gt_transform")
DEPTH_KEYS = ("composite_depth_m", "depth_image", "depth_m")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return records


def is_successful_record(meta: dict[str, Any]) -> bool:
    tags = meta.get("domain_tags") or []
    if "failed_generation" in tags:
        return False
    if not meta.get("depth_npz"):
        return False
    if meta.get("gt_transform_camera_from_object") is None:
        return False
    if meta.get("bbox_corners_m") is None:
        return False
    return True


def resolve_npz_path(meta: dict[str, Any], split_dir: Path) -> Path:
    depth_npz = meta.get("depth_npz")
    if not depth_npz:
        raise FileNotFoundError(f"Record {meta.get('sample_id')} has no depth_npz path")

    path = Path(str(depth_npz))
    if path.is_file():
        return path

    # Paths in metadata are often relative to the process cwd at generation time.
    candidates = [
        split_dir / path.name,
        split_dir / "samples" / path.name,
        split_dir / path,
    ]
    sample_id = meta.get("sample_id")
    if sample_id:
        candidates.append(split_dir / "samples" / f"{sample_id}.npz")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not locate NPZ for sample {meta.get('sample_id')}: {depth_npz}")


def extract_depth_m(sample: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in DEPTH_KEYS:
        if key in sample.files:
            return np.asarray(sample[key], dtype=np.float32)
    raise KeyError(f"NPZ has no depth array (tried {DEPTH_KEYS}); keys={sample.files}")


def convert_record(
    meta: dict[str, Any],
    split_dir: Path,
) -> dict[str, np.ndarray]:
    npz_path = resolve_npz_path(meta, split_dir)
    with np.load(npz_path) as sample:
        depth_m = extract_depth_m(sample)
        if "model_points" not in sample.files:
            raise KeyError(f"{npz_path} is missing model_points")
        model_points = np.asarray(sample["model_points"], dtype=np.float32)

    gt = np.asarray(meta["gt_transform_camera_from_object"], dtype=np.float32)
    if gt.shape != (4, 4):
        raise ValueError(f"Invalid gt_transform shape for {meta.get('sample_id')}: {gt.shape}")

    bbox = np.asarray(meta["bbox_corners_m"], dtype=np.float32)
    if bbox.shape != (8, 3):
        raise ValueError(f"Invalid bbox_corners_m shape for {meta.get('sample_id')}: {bbox.shape}")

    return {
        "depth_image": depth_m.astype(np.float16),
        "model_points": model_points,
        "bbox_corners": bbox,
        "gt_transform": gt,
    }


def convert_split(
    split_name: str,
    input_split_dir: Path,
    output_split_dir: Path,
    *,
    renumber: bool,
) -> dict[str, Any]:
    metadata_path = input_split_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.jsonl for split '{split_name}': {metadata_path}")

    records = load_metadata_jsonl(metadata_path)
    successful = [r for r in records if is_successful_record(r)]
    successful.sort(key=lambda r: int(str(r.get("sample_id", "0"))))

    output_split_dir.mkdir(parents=True, exist_ok=True)

    skipped_failed = sum(1 for r in records if "failed_generation" in (r.get("domain_tags") or []))
    skipped_other = len(records) - len(successful) - skipped_failed

    seeds_used: list[int] = []
    v2_sample_ids: list[str] = []
    converted = 0
    errors: list[str] = []

    for out_index, meta in enumerate(successful):
        sample_id = str(meta.get("sample_id", out_index))
        out_name = f"{out_index:06d}.npz" if renumber else f"{sample_id}.npz"
        out_path = output_split_dir / out_name

        try:
            arrays = convert_record(meta, input_split_dir)
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{sample_id}: {exc}")
            continue

        np.savez_compressed(out_path, **arrays)
        seeds_used.append(int(meta.get("sample_seed", 0)))
        v2_sample_ids.append(sample_id)
        converted += 1

    summary_path = input_split_dir / "summary.json"
    v2_summary = None
    if summary_path.is_file():
        v2_summary = load_json(summary_path)

    return {
        "split": split_name,
        "input_dir": str(input_split_dir).replace("\\", "/"),
        "output_dir": str(output_split_dir).replace("\\", "/"),
        "records_in_jsonl": len(records),
        "num_converted": converted,
        "num_skipped_failed": skipped_failed,
        "num_skipped_other": skipped_other,
        "num_errors": len(errors),
        "errors": errors[:20],
        "renumbered": renumber,
        "seeds_used": seeds_used,
        "v2_sample_ids": v2_sample_ids if renumber else None,
        "v2_summary": v2_summary,
    }


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    renumber: bool = True,
    scene_config: dict[str, Any] | None = None,
    network_config: dict[str, Any] | None = None,
    shape_type: str | None = None,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    split_summaries: dict[str, Any] = {}

    for split in splits:
        input_split = input_dir / split
        if not input_split.is_dir():
            raise FileNotFoundError(f"Split directory not found: {input_split}")
        split_summaries[split] = convert_split(
            split,
            input_split,
            output_dir / split,
            renumber=renumber,
        )

    if shape_type is None:
        shape_type = _infer_shape_type(split_summaries)

    metadata: dict[str, Any] = {
        "format": "positioning_v1",
        "converted_from": "positioning_v2",
        "shape_type": shape_type,
        "sample_keys": list(V1_SAMPLE_KEYS),
        "splits": {
            name: {
                "num_samples": summary["num_converted"],
                "seeds_used": summary["seeds_used"],
                **(
                    {"v2_sample_id_map": summary["v2_sample_ids"]}
                    if summary.get("v2_sample_ids")
                    else {}
                ),
            }
            for name, summary in split_summaries.items()
        },
        "v2_input_dir": str(input_dir).replace("\\", "/"),
        "conversion": {
            "renumbered": renumber,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
    }
    if scene_config is not None:
        metadata["scene_config"] = scene_config
    if network_config is not None:
        metadata["network_config"] = network_config

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "output_dir": str(output_dir).replace("\\", "/"),
        "metadata_json": str(metadata_path).replace("\\", "/"),
        "splits": split_summaries,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _infer_shape_type(split_summaries: dict[str, Any]) -> str:
    for summary in split_summaries.values():
        v2_summary = summary.get("v2_summary") or {}
        source = v2_summary.get("object_source")
        if source:
            if source == "superquadric":
                return "superquadric"
            if source in ("stl", "random"):
                return str(source)
    return "composite"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert v2 dataset output to legacy v1 training layout (flat split/*.npz + metadata.json)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="v2 dataset root containing train/ and val/ split folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for v1-style dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split names to convert (default: train val)",
    )
    parser.add_argument(
        "--scene_config",
        type=Path,
        default=None,
        help="Optional scene config JSON copied into output metadata.json",
    )
    parser.add_argument(
        "--network_config",
        type=Path,
        default=None,
        help="Optional network config JSON copied into output metadata.json",
    )
    parser.add_argument(
        "--shape_type",
        default=None,
        help="Override shape_type written to metadata.json (default: inferred)",
    )
    parser.add_argument(
        "--keep_sample_ids",
        action="store_true",
        help="Keep v2 sample_id filenames instead of renumbering 000000..N-1",
    )
    args = parser.parse_args()

    scene_config = load_json(args.scene_config) if args.scene_config else None
    network_config = load_json(args.network_config) if args.network_config else None

    result = convert_dataset(
        args.input_dir,
        args.output_dir,
        splits=tuple(args.splits),
        renumber=not args.keep_sample_ids,
        scene_config=scene_config,
        network_config=network_config,
        shape_type=args.shape_type,
    )

    print("Conversion complete")
    print(f"  Output:   {result['output_dir']}")
    print(f"  Metadata: {result['metadata_json']}")
    for split, summary in result["splits"].items():
        print(
            f"  {split}: {summary['num_converted']} samples "
            f"(skipped {summary['num_skipped_failed']} failed, "
            f"{summary['num_skipped_other']} other)"
        )
        if summary["num_errors"]:
            print(f"    errors: {summary['num_errors']} (see metadata or rerun with verbose logs)")
            for err in summary["errors"][:5]:
                print(f"      - {err}")
    print(f"  Elapsed: {result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
