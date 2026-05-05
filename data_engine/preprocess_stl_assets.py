"""Offline STL preprocessing utility (repair/remesh/copy to separate folder)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymeshlab
from tqdm import tqdm


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

    # Some pymeshlab operations can affect process cwd after load errors;
    # keep paths absolute and restore cwd to avoid breaking subsequent files.
    original_cwd = Path.cwd()
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(str(src_path))
    except pymeshlab.pmeshlab.PyMeshLabException as e:
        print(f"Error loading mesh {src_path}: {e}")
        os.chdir(original_cwd)
        return None

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
    result = {
        "source": str(src_path).replace("\\", "/"),
        "output": str(dst_path).replace("\\", "/"),
        "vertex_count": int(mesh.vertex_number()),
        "face_count": int(mesh.face_number()),
        "close_holes_max_size": close_holes_max_size,
        "remesh_enabled": remesh_enabled,
        "remesh_iterations": remesh_iterations,
        "remesh_target_perc": remesh_target_perc,
    }
    os.chdir(original_cwd)
    return result


def _preprocess_worker(src_path: str, dst_path: str, cfg: dict, result_queue: mp.Queue) -> None:
    try:
        rec = preprocess_one(Path(src_path), Path(dst_path), cfg)
        result_queue.put({"ok": True, "record": rec})
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def preprocess_one_with_timeout(src_path: Path, dst_path: Path, cfg: dict, timeout_sec: float) -> tuple[dict | None, str | None]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_preprocess_worker, args=(str(src_path), str(dst_path), cfg, result_queue))

    try:
        proc.start()
        proc.join(timeout=timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return None, f"timeout after {timeout_sec:.1f}s"

        if result_queue.empty():
            if proc.exitcode == 0:
                return None, "processing failed with no details"
            return None, f"worker exited with code {proc.exitcode}"

        payload = result_queue.get_nowait()
        if payload.get("ok"):
            return payload.get("record"), None
        return None, payload.get("error", "processing failed")
    finally:
        result_queue.close()
        result_queue.join_thread()


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
    runtime_cfg = cfg.get("runtime", {})
    timeout_sec = float(runtime_cfg.get("per_file_timeout_sec", 3.0))
    if timeout_sec <= 0:
        raise ValueError("runtime.per_file_timeout_sec must be > 0")
    max_workers = int(runtime_cfg.get("max_workers", max(1, min((os.cpu_count() or 1), 4))))
    if max_workers <= 0:
        raise ValueError("runtime.max_workers must be > 0")

    workspace_root = Path.cwd().resolve()
    input_dir = Path(io_cfg.get("input_directory", "data/stl"))
    output_dir = Path(io_cfg.get("output_directory", "data/stl_processed"))
    if not input_dir.is_absolute():
        input_dir = (workspace_root / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()
    pattern = str(io_cfg.get("glob", "*.stl"))

    src_files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not src_files:
        raise RuntimeError(f"No STL files found in '{input_dir}' with pattern '{pattern}'")

    records: list[dict] = []
    skipped = 0
    failed = 0
    work_items: list[tuple[int, Path, Path]] = []
    for index, src in enumerate(src_files):
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            records.append(
                {
                    "index": index,
                    "source": str(src).replace("\\", "/"),
                    "output": str(dst).replace("\\", "/"),
                    "skipped": True,
                    "reason": "exists",
                }
            )
        else:
            work_items.append((index, src, dst))

    with tqdm(total=len(src_files), desc="Preprocessing STL", unit="file") as progress_bar:
        progress_bar.update(skipped)
        if work_items:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(preprocess_one_with_timeout, src, dst, cfg, timeout_sec): (index, src, dst)
                    for index, src, dst in work_items
                }
                for future in as_completed(future_map):
                    index, src, dst = future_map[future]
                    try:
                        rec, err = future.result()
                    except Exception as e:
                        rec, err = None, str(e)

                    if rec is None:
                        failed += 1
                        records.append(
                            {
                                "index": index,
                                "source": str(src).replace("\\", "/"),
                                "output": str(dst).replace("\\", "/"),
                                "skipped": True,
                                "reason": err or "corrupted or unreadable",
                            }
                        )
                        tqdm.write(f"Failed: {src.name} ({err or 'corrupted or unreadable'})")
                    else:
                        rec["index"] = index
                        rec["skipped"] = False
                        records.append(rec)
                    progress_bar.update(1)

    records.sort(key=lambda item: item.get("index", 0))

    summary = {
        "input_directory": str(input_dir).replace("\\", "/"),
        "output_directory": str(output_dir).replace("\\", "/"),
        "count_total": len(src_files),
        "count_processed": len(src_files) - skipped - failed,
        "count_skipped": skipped,
        "count_failed": failed,
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
