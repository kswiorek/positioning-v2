"""Offline STL preprocessing utility (repair/remesh/copy to separate folder)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymeshlab


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mk_percentage(value: float):
    if hasattr(pymeshlab, "Percentage"):
        return pymeshlab.Percentage(value)
    if hasattr(pymeshlab, "PercentageValue"):
        return pymeshlab.PercentageValue(value)
    raise RuntimeError("Unsupported pymeshlab version: missing Percentage/PercentageValue")


def _try_apply_filter(ms: pymeshlab.MeshSet, filter_name: str, **kwargs) -> bool:
    if not hasattr(ms, filter_name):
        return False
    fn = getattr(ms, filter_name)
    fn(**kwargs)
    return True


def preprocess_one(src_path: Path, dst_path: Path, cfg: dict) -> dict:
    repair_cfg = cfg.get("repair", {})
    remesh_cfg = cfg.get("remesh", {})

    close_holes_max_size = int(repair_cfg.get("close_holes_max_size", 200))
    remesh_enabled = bool(remesh_cfg.get("enabled", False))
    remesh_iterations = int(remesh_cfg.get("iterations", 4))
    remesh_target_perc = float(remesh_cfg.get("target_perc", 1.0))

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src_path))

    _try_apply_filter(ms, "meshing_remove_duplicate_vertices")
    _try_apply_filter(ms, "meshing_remove_duplicate_faces")
    _try_apply_filter(ms, "meshing_remove_null_faces")
    _try_apply_filter(ms, "meshing_repair_non_manifold_edges")
    _try_apply_filter(ms, "meshing_repair_non_manifold_vertices")
    _try_apply_filter(ms, "meshing_close_holes", maxholesize=close_holes_max_size)

    if remesh_enabled:
        target_len = _mk_percentage(remesh_target_perc)
        _try_apply_filter(
            ms,
            "meshing_isotropic_explicit_remeshing",
            iterations=remesh_iterations,
            targetlen=target_len,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(dst_path))

    mesh = ms.current_mesh()
    return {
        "source": str(src_path).replace("\\", "/"),
        "output": str(dst_path).replace("\\", "/"),
        "vertex_count": int(mesh.vertex_number()),
        "face_count": int(mesh.face_number()),
        "close_holes_max_size": close_holes_max_size,
        "remesh_enabled": remesh_enabled,
        "remesh_iterations": remesh_iterations,
        "remesh_target_perc": remesh_target_perc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess STL assets into a separate output folder")
    parser.add_argument(
        "--config",
        default="data_engine/config/stl_preprocess_config.example.json",
        help="Path to STL preprocessing config JSON",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output STL files")
    parser.add_argument("--report_json", default="", help="Optional path to write processing report JSON")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    io_cfg = cfg.get("io", {})

    input_dir = Path(io_cfg.get("input_directory", "data/stl"))
    output_dir = Path(io_cfg.get("output_directory", "data/stl_processed"))
    pattern = str(io_cfg.get("glob", "*.stl"))

    src_files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not src_files:
        raise RuntimeError(f"No STL files found in '{input_dir}' with pattern '{pattern}'")

    records = []
    skipped = 0
    for src in src_files:
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            records.append(
                {
                    "source": str(src).replace("\\", "/"),
                    "output": str(dst).replace("\\", "/"),
                    "skipped": True,
                    "reason": "exists",
                }
            )
            continue

        rec = preprocess_one(src, dst, cfg)
        rec["skipped"] = False
        records.append(rec)
        print(f"Processed: {src.name} -> {dst}")

    summary = {
        "input_directory": str(input_dir).replace("\\", "/"),
        "output_directory": str(output_dir).replace("\\", "/"),
        "count_total": len(src_files),
        "count_processed": len(src_files) - skipped,
        "count_skipped": skipped,
        "records": records,
    }

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
