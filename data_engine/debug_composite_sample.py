"""Debug utility: full sample generation with depth compositing and Open3D visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_engine.composition.background_normalization import normalize_and_randomize_background_depth
from data_engine.composition.camera_artifacts import apply_camera_artifacts
from data_engine.composition.depth_compositor import compose_depth, render_mesh_depth, transform_mesh
from data_engine.composition.plane_fit import fit_plane_from_depth
from data_engine.composition.plane_placement import PlacementConstraints
from data_engine.composition.placement_sampling import sample_pose_on_plane
from data_engine.generators import generate_mixed_canonical_model
from data_engine.visualization import visualize_sample


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug full sample compositing pipeline")
    parser.add_argument("--depth_npz", required=True, help="Path to background depth npz (expects key depth_m)")
    parser.add_argument(
        "--scene_config",
        default="data_engine/config/scene_config.json",
        help="Path to scene config JSON",
    )
    parser.add_argument(
        "--object_source",
        choices=["superquadric", "stl", "random"],
        default="superquadric",
        help="Shape source override for debug runs.",
    )
    parser.add_argument("--out_npz", default="data/debug/composite_sample_000000.npz")
    parser.add_argument("--out_json", default="data/debug/composite_sample_000000.json")
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization windows")
    args = parser.parse_args()

    scene_cfg = load_json(Path(args.scene_config))
    seed_cfg = scene_cfg.get("seed", None)
    if seed_cfg is None:
        seed = int(np.random.SeedSequence().entropy)
        seed_source = "random"
    else:
        seed = int(seed_cfg)
        seed_source = "config"

    rng_master = np.random.default_rng(seed)

    depth_data = np.load(args.depth_npz)
    background_depth_raw = depth_data["depth_m"].astype(np.float32)

    plane_cfg = scene_cfg.get("plane_fit", {})
    bg_norm_cfg = scene_cfg.get("background_normalization", {})
    norm_enabled = bool(bg_norm_cfg.get("enabled", True))
    bg_transform = None
    if norm_enabled:
        bg_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        background_depth, bg_transform = normalize_and_randomize_background_depth(
            depth_m=background_depth_raw,
            camera_cfg=scene_cfg["camera"],
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

    # Fit placement plane on the normalized/randomized background depth.
    plane = fit_plane_from_depth(
        depth_m=background_depth,
        camera_cfg=scene_cfg["camera"],
        stride=int(plane_cfg.get("stride", 2)),
        middle_percentile=float(plane_cfg.get("middle_percentile", 0.90)),
        seed=int(rng_master.integers(0, 2**31 - 1)),
    )

    object_seed = int(rng_master.integers(0, 2**31 - 1))
    canonical_mesh, _, bbox_corners, shape_params = generate_mixed_canonical_model(
        scene_cfg,
        seed=object_seed,
        source_override=args.object_source,
    )
    bbox_extent = (bbox_corners.max(axis=0) - bbox_corners.min(axis=0)).astype(np.float64)

    place_cfg = scene_cfg["placement"]
    constraints = PlacementConstraints(
        min_plane_distance_m=float(place_cfg["min_plane_distance_m"]),
        max_plane_distance_m=float(place_cfg["max_plane_distance_m"]),
    )

    place_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
    placement = sample_pose_on_plane(
        plane=plane,
        camera_cfg=scene_cfg["camera"],
        bbox_extent_m=bbox_extent,
        constraints=constraints,
        rng=place_rng,
        max_tries=int(place_cfg.get("max_attempts", 400)),
    )

    mesh_world = transform_mesh(
        canonical_mesh,
        position_xyz=placement.position_xyz,
        euler_deg_xyz=placement.orientation_euler_deg_xyz,
    )

    object_depth = render_mesh_depth(mesh_world, scene_cfg["camera"])
    composite_depth = compose_depth(background_depth, object_depth)

    camera_artifacts_cfg = scene_cfg.get("camera_artifacts", {})
    artifacts_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
    composite_depth, artifact_stats = apply_camera_artifacts(
        composite_depth,
        camera_artifacts_cfg,
        background_depth_m=background_depth,
        object_depth_m=object_depth,
        camera_cfg=scene_cfg["camera"],
        rng=artifacts_rng,
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        background_depth_raw_m=background_depth_raw.astype(np.float32),
        background_depth_m=background_depth.astype(np.float32),
        object_depth_m=object_depth.astype(np.float32),
        composite_depth_m=composite_depth.astype(np.float32),
    )

    result = {
        "seed": seed,
        "seed_source": seed_source,
        "plane": {
            "normal": plane.normal.tolist(),
            "offset": float(plane.offset),
            "inlier_ratio": float(plane.inlier_ratio),
        },
        "background_normalization": {
            "enabled": norm_enabled,
            "transform": {
                "pitch_deg": None if bg_transform is None else float(bg_transform.pitch_deg),
                "yaw_deg": None if bg_transform is None else float(bg_transform.yaw_deg),
                "distance_m": None if bg_transform is None else float(bg_transform.distance_m),
                "inplane_scale_xy": None if bg_transform is None else float(bg_transform.inplane_scale_xy),
                "projected_fill_u": None if bg_transform is None else float(bg_transform.projected_fill_u),
                "projected_fill_v": None if bg_transform is None else float(bg_transform.projected_fill_v),
                "out_of_plane_scale_m": None if bg_transform is None else float(bg_transform.out_of_plane_scale_m),
            },
        },
        "camera_artifacts": artifact_stats,
        "shape_params": shape_params,
        "object_source": shape_params.get("object_source", "unknown"),
        "bbox_extent_m": bbox_extent.tolist(),
        "placement": {
            "position_xyz": placement.position_xyz.tolist(),
            "orientation_euler_deg_xyz": placement.orientation_euler_deg_xyz.tolist(),
            "orientation_quat_xyzw": placement.orientation_quat_xyzw.tolist(),
            "plane_offset_m": float(placement.plane_offset_m),
            "center_pixel_uv": placement.center_pixel_uv.tolist(),
        },
        "outputs": {
            "npz": str(out_npz),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved sample depth arrays to: {out_npz}")
    print(f"Saved sample metadata to: {out_json}")

    if not args.no_vis:
        visualize_sample(
            background_depth_m=background_depth,
            object_depth_m=object_depth,
            composite_depth_m=composite_depth,
            object_mesh_world=mesh_world,
            camera_cfg=scene_cfg["camera"],
            plane_normal=plane.normal,
            plane_offset=plane.offset,
        )


if __name__ == "__main__":
    main()
