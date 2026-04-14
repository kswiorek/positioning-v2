"""Debug utility: reconstruct selected background frame with identity compositing.

Goal: verify that compositing/rendering preserves the original background depth
when no object is inserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from data_engine.composition.depth_compositor import compose_depth
from data_engine.composition.plane_fit import fit_plane_from_depth, select_plane_support_mask
from data_engine.geometry.camera import intrinsics_from_camera_config
from data_engine.visualization import visualize_sample


def _rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return rotation matrix mapping src direction to dst direction."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / max(s * s, 1e-12))


def _compute_centering_translation(points_cam: np.ndarray, camera_cfg: dict, rotation: np.ndarray) -> np.ndarray:
    """Compute camera-frame XY translation that centers projected cloud bounds."""
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)

    rot = np.asarray(rotation, dtype=np.float64)
    pts_rot = (rot @ pts.T).T
    z = pts_rot[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros(3, dtype=np.float64)

    pts_rot = pts_rot[valid]
    z = pts_rot[:, 2]
    u = fx * pts_rot[:, 0] / z + cx
    v = fy * pts_rot[:, 1] / z + cy

    u_center = 0.5 * (float(np.min(u)) + float(np.max(u)))
    v_center = 0.5 * (float(np.min(v)) + float(np.max(v)))
    du = cx - u_center
    dv = cy - v_center

    z_ref = float(np.median(z))
    tx = du * z_ref / fx
    ty = dv * z_ref / fy
    return np.array([tx, ty, 0.0], dtype=np.float64)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct raw background depth for sanity testing")
    parser.add_argument("--depth_npz", required=True, help="Path to background depth npz (expects key depth_m)")
    parser.add_argument(
        "--scene_config",
        default="data_engine/config/scene_config.reconstruct_background.example.json",
        help="Path to scene config JSON",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_npz", default="data/debug/reconstruct_background_000000.npz")
    parser.add_argument("--out_json", default="data/debug/reconstruct_background_000000.json")
    parser.add_argument(
        "--rectify_plane_view",
        action="store_true",
        help="Rotate scene cloud for visualization so fitted plane normal aligns with camera axis",
    )
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D/matplotlib visualization")
    args = parser.parse_args()

    scene_cfg = load_json(Path(args.scene_config))

    depth_data = np.load(args.depth_npz)
    background_depth = depth_data["depth_m"].astype(np.float32)

    plane_cfg = scene_cfg.get("plane_fit", {})
    plane = fit_plane_from_depth(
        depth_m=background_depth,
        camera_cfg=scene_cfg["camera"],
        stride=int(plane_cfg.get("stride", 2)),
        middle_percentile=float(plane_cfg.get("middle_percentile", 0.9)),
        allow_tilt=bool(plane_cfg.get("allow_tilt", True)),
        seed=args.seed,
    )

    support_mask = select_plane_support_mask(
        background_depth,
        middle_percentile=float(plane_cfg.get("middle_percentile", 0.9)),
    )

    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(scene_cfg["camera"])
    sy, sx = np.where(support_mask)
    sz = background_depth[sy, sx].astype(np.float64)
    sx3 = (sx.astype(np.float64) - cx) * sz / fx
    sy3 = (sy.astype(np.float64) - cy) * sz / fy
    support_points = np.stack([sx3, sy3, sz], axis=1) if sz.size > 0 else np.zeros((0, 3), dtype=np.float64)

    support_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(support_points))
    support_cloud.paint_uniform_color([1.0, 0.1, 0.8])

    valid_mask = background_depth > 1e-6
    by, bx = np.where(valid_mask)
    bz = background_depth[by, bx].astype(np.float64)
    bx3 = (bx.astype(np.float64) - cx) * bz / fx
    by3 = (by.astype(np.float64) - cy) * bz / fy
    bg_points = np.stack([bx3, by3, bz], axis=1) if bz.size > 0 else np.zeros((0, 3), dtype=np.float64)

    bg_colors = np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float64), (bg_points.shape[0], 1))
    support_vals = support_mask[by, bx]
    bg_colors[support_vals] = np.array([1.0, 0.1, 0.8], dtype=np.float64)

    background_cloud_colored = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_points))
    background_cloud_colored.colors = o3d.utility.Vector3dVector(bg_colors)

    object_depth = np.zeros_like(background_depth, dtype=np.float32)
    composite_depth = compose_depth(background_depth, object_depth)

    abs_err = np.abs(composite_depth.astype(np.float64) - background_depth.astype(np.float64))
    valid = background_depth > 1e-6
    max_abs_err = float(abs_err.max())
    mean_abs_err_valid = float(abs_err[valid].mean()) if np.any(valid) else 0.0

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        background_depth_m=background_depth,
        object_depth_m=object_depth,
        composite_depth_m=composite_depth,
        abs_error_m=abs_err.astype(np.float32),
    )

    result = {
        "seed": args.seed,
        "plane": {
            "normal": plane.normal.tolist(),
            "offset": float(plane.offset),
            "inlier_ratio": float(plane.inlier_ratio),
        },
        "plane_support": {
            "pixels": int(np.count_nonzero(support_mask)),
            "ratio": float(np.count_nonzero(support_mask)) / float(background_depth.size),
        },
        "reconstruction_error": {
            "max_abs_error_m": max_abs_err,
            "mean_abs_error_valid_m": mean_abs_err_valid,
        },
        "outputs": {
            "npz": str(out_npz),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved reconstruction arrays to: {out_npz}")
    print(f"Saved reconstruction metadata to: {out_json}")
    print(f"Reconstruction error: max={max_abs_err:.6e} m, mean(valid)={mean_abs_err_valid:.6e} m")

    if not args.no_vis:
        scene_rotation = None
        scene_translation = None
        if args.rectify_plane_view:
            # Keep camera fixed and rotate scene so plane appears fronto-parallel.
            scene_rotation = _rotation_align_vectors(np.asarray(plane.normal, dtype=np.float64), np.array([0.0, 0.0, -1.0]))
            # Recenter projected scene after rectification so principal ray points near cloud center.
            scene_translation = _compute_centering_translation(bg_points, scene_cfg["camera"], scene_rotation)

        visualize_sample(
            background_depth_m=background_depth,
            object_depth_m=object_depth,
            composite_depth_m=composite_depth,
            object_mesh_world=o3d.geometry.TriangleMesh(),
            camera_cfg=scene_cfg["camera"],
            plane_normal=plane.normal,
            plane_offset=plane.offset,
            extra_clouds=[support_cloud],
            background_cloud_override=background_cloud_colored,
            scene_rotation=scene_rotation,
            scene_translation=scene_translation,
        )


if __name__ == "__main__":
    main()
