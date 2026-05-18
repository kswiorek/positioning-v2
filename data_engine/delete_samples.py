"""Delete dataset samples listed in a high-loss log.

Usage (from workspace root):

python -m data_engine.delete_samples \
    --high_loss_file runs/hybrid_pose_v2/high_loss_samples.txt \
    --dataset_root data/generated/<run_name> \
    [--dry-run] [--trash-dir trash] [--confirm]

The script parses lines written by the training engine (containing a `samples=[...]`
fragment), collects all sample IDs, and for each split under `--dataset_root` will:
- remove (or move to `--trash-dir`) files under `<split>/samples/<sample_id>.npz`
- remove corresponding lines from `<split>/metadata.jsonl` and optionally
  `<split>/metadata_debug.jsonl` (backups are made)

This is intended to be safe-by-default: use `--dry-run` to preview changes.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set


def gather_sample_ids_from_high_loss(path: Path) -> Set[str]:
    ids: Set[str] = set()
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        # find a bracketed samples=[...] fragment
        if "samples=" not in line:
            continue
        try:
            # crude extraction: find the first '[' after 'samples=' and the matching ']'
            start = line.index("samples=") + len("samples=")
            bracket_start = line.index("[", start)
            bracket_end = line.rindex("]")
            list_text = line[bracket_start:bracket_end + 1]
            parsed = ast.literal_eval(list_text)
            if isinstance(parsed, (list, tuple)):
                for v in parsed:
                    s = str(v)
                    # Accept either plain numeric ids ("000123") or strings like "sample_000123"
                    if s.startswith("sample_"):
                        s = s.split("sample_", 1)[1]
                    # extract digits
                    digits = "".join(ch for ch in s if ch.isdigit())
                    if digits:
                        ids.add(digits.zfill(6))
        except Exception:
            # ignore parse errors on a line
            continue
    return ids


def _backup_file(p: Path) -> Path:
    if not p.exists():
        return p
    bak = p.with_suffix(p.suffix + ".bak")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    bak = bak.with_name(bak.stem + f".{timestamp}" + bak.suffix)
    shutil.copy2(p, bak)
    return bak


def rewrite_jsonl_excluding(path: Path, out_path: Path, exclude_ids: Set[str]) -> int:
    kept = 0
    removed = 0
    with path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                import json

                rec = json.loads(line_strip)
                sid = str(rec.get("sample_id", ""))
                if sid in exclude_ids:
                    removed += 1
                    continue
            except Exception:
                # If the line isn't parseable, keep it to be safe
                pass
            dst.write(line)
            kept += 1
    return removed


def find_splits(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        return []
    for p in dataset_root.iterdir():
        if p.is_dir() and (p / "samples").exists():
            yield p


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--high_loss_file", required=True, help="Path to high_loss_samples.txt")
    p.add_argument("--dataset_root", required=True, help="Generated dataset root (contains splits)")
    p.add_argument("--dry-run", action="store_true", help="Do not perform deletions; only report")
    p.add_argument("--trash-dir", help="If set, move deleted NPZs to this directory instead of deleting")
    p.add_argument("--confirm", action="store_true", help="Actually perform deletions (unsafe without --confirm)")
    args = p.parse_args(argv)

    high_loss = Path(args.high_loss_file)
    if not high_loss.exists():
        print(f"High-loss file not found: {high_loss}")
        return 2

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 2

    ids = gather_sample_ids_from_high_loss(high_loss)
    if not ids:
        print("No sample ids parsed from high-loss file.")
        return 0

    print(f"Found {len(ids)} unique sample ids to remove (dry_run={args.dry_run}).")

    trash_dir = None
    if args.trash_dir:
        trash_dir = Path(args.trash_dir)
        trash_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # report where samples would be removed from
        for split in find_splits(dataset_root):
            samples_dir = split / "samples"
            found = [s for s in ids if (samples_dir / f"{s}.npz").exists()]
            if found:
                print(f"Would remove {len(found)} files from split: {split.name}")
        return 0

    if not args.confirm:
        print("Refusing to delete files without --confirm. Rerun with --confirm to proceed.")
        return 3

    total_deleted = 0
    total_removed_meta = 0
    for split in find_splits(dataset_root):
        samples_dir = split / "samples"
        # Delete/move NPZs
        for sid in sorted(ids):
            src = samples_dir / f"{sid}.npz"
            if not src.exists():
                continue
            if trash_dir is not None:
                dest = trash_dir / f"{split.name}_{sid}.npz"
                shutil.move(str(src), str(dest))
                total_deleted += 1
            else:
                src.unlink()
                total_deleted += 1

        # Update metadata.jsonl
        meta = split / "metadata.jsonl"
        if meta.exists():
            _backup_file(meta)
            tmp = split / "metadata.jsonl.tmp"
            removed = rewrite_jsonl_excluding(meta, tmp, ids)
            tmp.replace(meta)
            total_removed_meta += removed

        # Update debug metadata if present
        debug_meta = split / "metadata_debug.jsonl"
        if debug_meta.exists():
            _backup_file(debug_meta)
            tmp = split / "metadata_debug.jsonl.tmp"
            removed = rewrite_jsonl_excluding(debug_meta, tmp, ids)
            tmp.replace(debug_meta)
            # don't double-count removed lines across files

    print(f"Done. deleted_files={total_deleted}, removed_metadata_entries={total_removed_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
