"""Debug utility: fit plane from depth and sample valid object placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_engine.composition.plane_fit import fit_plane_from_depth
from data_engine.composition.plane_placement import PlacementConstraints
from data_engine.composition.placement_sampling import sample_pose_on_plane
from data_engine.generators import generate_superquadric_canonical_model


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug plane fitting and placement sampling")
    parser.add_argument("--depth_npz", required=True, help="Path to captured depth npz (expects key depth_m)")
    parser.add_argument(
        "--scene_config",
        default="data_engine/config/scene_config.superquadric.example.json",
        help="Path to scene config JSON",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_json", default="", help="Optional output path for sampled placement JSON")
    args = parser.parse_args()

    scene_cfg = load_json(Path(args.scene_config))
    depth_data = np.load(args.depth_npz)
    depth_m = depth_data["depth_m"].astype(np.float32)

    plane_cfg = scene_cfg.get("plane_fit", {})
    plane = fit_plane_from_depth(
        depth_m=depth_m,
        camera_cfg=scene_cfg["camera"],
        stride=int(plane_cfg.get("stride", 2)),
        threshold_m=float(plane_cfg.get("ransac_threshold_m", 0.01)),
        max_iterations=int(plane_cfg.get("ransac_iterations", 300)),
        seed=args.seed,
    )

    _, _, bbox_corners, shape_params = generate_superquadric_canonical_model(scene_cfg, seed=args.seed)
    bbox_extent = (bbox_corners.max(axis=0) - bbox_corners.min(axis=0)).astype(np.float64)

    place_cfg = scene_cfg["placement"]
    constraints = PlacementConstraints(
        min_camera_distance_m=float(place_cfg["min_camera_distance_m"]),
        max_camera_distance_m=float(place_cfg["max_camera_distance_m"]),
    )

    rng = np.random.default_rng(args.seed)
    placement = sample_pose_on_plane(
        plane=plane,
        camera_cfg=scene_cfg["camera"],
        bbox_extent_m=bbox_extent,
        constraints=constraints,
        rng=rng,
        max_tries=int(place_cfg.get("max_attempts", 400)),
    )

    result = {
        "seed": args.seed,
        "plane": {
            "normal": plane.normal.tolist(),
            "offset": plane.offset,
            "inlier_ratio": plane.inlier_ratio,
        },
        "shape_params": shape_params,
        "bbox_extent_m": bbox_extent.tolist(),
        "placement": {
            "position_xyz": placement.position_xyz.tolist(),
            "orientation_euler_deg_xyz": placement.orientation_euler_deg_xyz.tolist(),
            "orientation_quat_xyzw": placement.orientation_quat_xyzw.tolist(),
            "plane_offset_m": placement.plane_offset_m,
            "center_pixel_uv": placement.center_pixel_uv.tolist(),
        },
    }

    out_json = args.out_json.strip()
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved placement debug JSON to: {out_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
