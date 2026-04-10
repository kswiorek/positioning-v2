"""Open3D visualization for generated debug samples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from data_engine.geometry.camera import intrinsics_from_camera_config


def _depth_to_pointcloud(depth_m: np.ndarray, camera_cfg: dict, color_rgb: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)

    depth = depth_m.astype(np.float32)
    valid = depth > 1e-6
    vv, uu = np.where(valid)
    z = depth[vv, uu].astype(np.float64)

    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color(list(color_rgb))
    return pcd


def _camera_frame(scale: float = 0.15) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0.0, 0.0, 0.0])


def _camera_frustum(camera_cfg: dict, scale: float = 0.5) -> o3d.geometry.LineSet:
    fx, fy, cx, cy, width, height = intrinsics_from_camera_config(camera_cfg)

    corners_2d = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )

    corners_3d = []
    for u, v in corners_2d:
        x = (u - cx) / fx * scale
        y = (v - cy) / fy * scale
        z = scale
        corners_3d.append([x, y, z])

    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    points = np.vstack([origin, np.asarray(corners_3d, dtype=np.float64)])

    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.1, 0.4, 1.0]]), (len(lines), 1)))
    return frustum


def _plane_patch(normal: np.ndarray, offset: float, size: float = 0.8) -> o3d.geometry.TriangleMesh:
    normal = normal.astype(np.float64)
    normal /= np.linalg.norm(normal)

    center = -offset * normal

    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    corners = np.array([
        center + size * (+u + v),
        center + size * (+u - v),
        center + size * (-u - v),
        center + size * (-u + v),
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.8, 0.3])
    return mesh


def visualize_sample(
    background_depth_m: np.ndarray,
    object_depth_m: np.ndarray,
    composite_depth_m: np.ndarray,
    object_mesh_world: o3d.geometry.TriangleMesh,
    camera_cfg: dict,
    plane_normal: np.ndarray | None = None,
    plane_offset: float | None = None,
) -> None:
    """Show 3D scene in Open3D and depth maps as 2D matplotlib images."""
    bg_cloud = _depth_to_pointcloud(background_depth_m, camera_cfg, color_rgb=(0.6, 0.6, 0.6))
    obj_depth_cloud = _depth_to_pointcloud(object_depth_m, camera_cfg, color_rgb=(0.8, 0.3, 0.2))

    object_mesh_world = o3d.geometry.TriangleMesh(object_mesh_world)
    object_mesh_world.paint_uniform_color([0.9, 0.3, 0.3])

    geometries_scene = [_camera_frame(), _camera_frustum(camera_cfg), bg_cloud, object_mesh_world, obj_depth_cloud]
    if plane_normal is not None and plane_offset is not None:
        geometries_scene.append(_plane_patch(np.asarray(plane_normal), float(plane_offset)))

    o3d.visualization.draw_geometries(
        geometries_scene,
        window_name="Sample Scene (camera frustum + background + object)",
        width=1280,
        height=720,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _plot_depth(ax, depth_m: np.ndarray, title: str) -> None:
        depth_viz = depth_m.astype(np.float32).copy()
        depth_viz[depth_viz <= 1e-6] = np.nan
        im = ax.imshow(depth_viz, cmap="turbo", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _plot_depth(axes[0], background_depth_m, "Background Depth")
    _plot_depth(axes[1], object_depth_m, "Object Depth")
    _plot_depth(axes[2], composite_depth_m, "Composite Depth")
    plt.tight_layout()
    plt.show()
