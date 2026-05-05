"""Mixed object-source generator (superquadric and STL)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .stl import generate_stl_canonical_model
from .superquadric import generate_superquadric_canonical_model


def _stl_available(config: dict) -> bool:
    scene_cfg = config.get("scene", {})
    stl_cfg = scene_cfg.get("stl", {})
    input_cfg = stl_cfg.get("input", {})
    directory = Path(input_cfg.get("directory", "data/stl"))
    pattern = str(input_cfg.get("glob", "*.stl"))
    return any(p.is_file() for p in directory.glob(pattern))


def _choose_source(
    config: dict,
    rng: np.random.Generator,
    source_override: str | None,
    stl_available: bool | None = None,
) -> str:
    if source_override is not None:
        source = source_override.strip().lower()
        if source not in {"superquadric", "stl", "random"}:
            raise ValueError("source_override must be one of: superquadric, stl, random")
        if source != "random":
            return source

    scene_cfg = config.get("scene", {})
    src_cfg = scene_cfg.get("source_selection", {})

    mode = str(src_cfg.get("mode", "random")).strip().lower()
    if mode in {"superquadric", "stl"}:
        return mode

    stl_ok = _stl_available(config) if stl_available is None else bool(stl_available)
    if not stl_ok:
        return "superquadric"

    weights = src_cfg.get("weights", {})
    w_sq = float(weights.get("superquadric", 1.0))
    w_stl = float(weights.get("stl", 1.0))
    w_sq = max(w_sq, 0.0)
    w_stl = max(w_stl, 0.0)

    if w_sq <= 0.0 and w_stl <= 0.0:
        w_sq = 1.0
        w_stl = 1.0

    p_stl = w_stl / max(w_sq + w_stl, 1e-12)
    return "stl" if float(rng.uniform(0.0, 1.0)) < p_stl else "superquadric"


def choose_object_source(
    config: dict,
    seed: int | None = None,
    source_override: str | None = None,
    stl_available: bool | None = None,
) -> str:
    """Deterministically resolve the object source used for a sample plan."""
    rng = np.random.default_rng(seed)
    return _choose_source(config, rng=rng, source_override=source_override, stl_available=stl_available)


def generate_mixed_canonical_model(
    config: dict,
    seed: int | None = None,
    source_override: str | None = None,
    include_point_cloud: bool = False,
):
    """Generate canonical object from randomized source pool."""
    rng = np.random.default_rng(seed)
    source = _choose_source(config, rng=rng, source_override=source_override)

    shape_seed = int(rng.integers(0, 2**31 - 1))
    if source == "stl":
        mesh, cloud, bbox_corners, shape_params = generate_stl_canonical_model(
            config,
            seed=shape_seed,
            include_point_cloud=include_point_cloud,
        )
    else:
        mesh, cloud, bbox_corners, shape_params = generate_superquadric_canonical_model(
            config,
            seed=shape_seed,
            include_point_cloud=include_point_cloud,
        )

    shape_params = dict(shape_params)
    shape_params.setdefault("object_source", source)
    if source == "superquadric":
        shape_params.setdefault("object_id", "superquadric")
        shape_params.setdefault("object_asset_path", None)

    return mesh, cloud, bbox_corners, shape_params
