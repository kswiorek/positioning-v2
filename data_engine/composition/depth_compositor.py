"""Depth rendering and compositing utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from data_engine.geometry.camera import intrinsics_from_camera_config


@lru_cache(maxsize=16)
def _cached_camera_rays(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[o3d.core.Tensor, np.ndarray]:
    """Return cached ray tensor and ray-dir z component map for camera intrinsics."""
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - float(cx)) / float(fx)
    y = (vv - float(cy)) / float(fy)
    z = np.ones_like(x, dtype=np.float32)

    ray_dirs = np.stack([x, y, z], axis=-1)
    ray_dirs /= np.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    ray_origins = np.zeros_like(ray_dirs, dtype=np.float32)
    rays = np.concatenate([ray_origins.reshape(-1, 3), ray_dirs.reshape(-1, 3)], axis=1)
    rays_t = o3d.core.Tensor(rays.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ray_dir_z = ray_dirs[:, :, 2].astype(np.float32)
    return rays_t, ray_dir_z


def transform_mesh(mesh: o3d.geometry.TriangleMesh, position_xyz: np.ndarray, euler_deg_xyz: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Return a transformed mesh copy in camera/world frame."""
    mesh_out = o3d.geometry.TriangleMesh(mesh)
    rot = Rotation.from_euler("xyz", euler_deg_xyz, degrees=True).as_matrix()
    mesh_out.rotate(rot, center=(0.0, 0.0, 0.0))
    mesh_out.translate(position_xyz.astype(np.float64))
    mesh_out.compute_vertex_normals()
    return mesh_out


def render_mesh_depth(mesh: o3d.geometry.TriangleMesh, camera_cfg: dict) -> np.ndarray:
    """Render mesh depth image from camera origin looking along +Z."""
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    rays_t, ray_dir_z = _cached_camera_rays(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )

    result = scene.cast_rays(rays_t)
    t_hit = result["t_hit"].numpy().reshape(height, width)

    depth = t_hit * ray_dir_z
    depth[np.isinf(depth)] = 0.0
    depth = np.maximum(depth, 0.0)
    return depth.astype(np.float32)


def compose_depth(background_depth_m: np.ndarray, object_depth_m: np.ndarray) -> np.ndarray:
    """Compose depth maps with z-buffer style nearest-surface rule."""
    if background_depth_m.shape != object_depth_m.shape:
        raise ValueError("Background and object depth shapes must match.")

    bg = background_depth_m.astype(np.float32)
    obj = object_depth_m.astype(np.float32)

    bg_valid = bg > 1e-6
    obj_valid = obj > 1e-6

    out = np.zeros_like(bg, dtype=np.float32)

    only_bg = bg_valid & (~obj_valid)
    only_obj = obj_valid & (~bg_valid)
    both = bg_valid & obj_valid

    out[only_bg] = bg[only_bg]
    out[only_obj] = obj[only_obj]
    out[both] = np.minimum(bg[both], obj[both])
    return out
