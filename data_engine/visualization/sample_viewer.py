"""Open3D visualization for generated debug samples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from data_engine.geometry.camera import camera_points_to_depth, intrinsics_from_camera_config


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


def _transform_depth_to_camera_view(
    depth_m: np.ndarray,
    camera_cfg: dict,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply scene transform in camera frame and reproject to a depth image."""
    fx, fy, cx, cy, _, _ = intrinsics_from_camera_config(camera_cfg)

    depth = depth_m.astype(np.float32)
    valid = depth > 1e-6
    vv, uu = np.where(valid)
    if vv.size == 0:
        return np.zeros_like(depth, dtype=np.float32)

    z = depth[vv, uu].astype(np.float64)
    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    pts_tf = pts
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64)
        pts_tf = (rot @ pts_tf.T).T
    if translation is not None:
        tr = np.asarray(translation, dtype=np.float64).reshape(1, 3)
        pts_tf = pts_tf + tr

    # Small splat radius reduces tiny holes from discrete reprojection.
    return camera_points_to_depth(pts_tf, camera_cfg, min_depth_m=1e-6, splat_radius_px=1)


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
    object_depth_m: np.ndarray | None,
    composite_depth_m: np.ndarray,
    object_mesh_world: o3d.geometry.TriangleMesh | None,
    camera_cfg: dict,
    plane_normal: np.ndarray | None = None,
    plane_offset: float | None = None,
    extra_clouds: list[o3d.geometry.PointCloud] | None = None,
    background_cloud_override: o3d.geometry.PointCloud | None = None,
    scene_rotation: np.ndarray | None = None,
    scene_translation: np.ndarray | None = None,
) -> None:
    """Show 3D scene in Open3D and depth maps as 2D matplotlib images."""
    if background_cloud_override is not None:
        bg_cloud = o3d.geometry.PointCloud(background_cloud_override)
    else:
        bg_cloud = _depth_to_pointcloud(background_depth_m, camera_cfg, color_rgb=(0.6, 0.6, 0.6))
    if object_depth_m is None:
        object_depth_m = np.zeros_like(background_depth_m, dtype=np.float32)
    obj_depth_cloud = _depth_to_pointcloud(object_depth_m, camera_cfg, color_rgb=(0.8, 0.3, 0.2))

    rot = None
    tr = None
    if scene_rotation is not None:
        rot = np.asarray(scene_rotation, dtype=np.float64)
        bg_cloud.rotate(rot, center=(0.0, 0.0, 0.0))
        obj_depth_cloud.rotate(rot, center=(0.0, 0.0, 0.0))
    if scene_translation is not None:
        tr = np.asarray(scene_translation, dtype=np.float64)
        bg_cloud.translate(tr)
        obj_depth_cloud.translate(tr)

    geometries_scene = [_camera_frame(), _camera_frustum(camera_cfg), bg_cloud, obj_depth_cloud]
    if object_mesh_world is not None and len(np.asarray(object_mesh_world.vertices)) > 0:
        mesh_vis = o3d.geometry.TriangleMesh(object_mesh_world)
        mesh_vis.paint_uniform_color([0.9, 0.3, 0.3])
        if rot is not None:
            mesh_vis.rotate(rot, center=(0.0, 0.0, 0.0))
        if tr is not None:
            mesh_vis.translate(tr)
        geometries_scene.append(mesh_vis)
    if extra_clouds:
        for cloud in extra_clouds:
            c = o3d.geometry.PointCloud(cloud)
            if rot is not None:
                c.rotate(rot, center=(0.0, 0.0, 0.0))
            if tr is not None:
                c.translate(tr)
            geometries_scene.append(c)
    if plane_normal is not None and plane_offset is not None:
        n = np.asarray(plane_normal, dtype=np.float64)
        if rot is not None:
            n = rot @ n
        plane_vis = _plane_patch(n, float(plane_offset))
        if tr is not None:
            plane_vis.translate(tr)
        geometries_scene.append(plane_vis)

    o3d.visualization.draw_geometries(
        geometries_scene,
        window_name="Sample Scene (camera frustum + background)",
        width=1280,
        height=720,
    )

    background_plot = background_depth_m
    object_plot = object_depth_m
    composite_plot = composite_depth_m
    if rot is not None or tr is not None:
        background_plot = _transform_depth_to_camera_view(background_depth_m, camera_cfg, rot, tr)
        object_plot = _transform_depth_to_camera_view(object_depth_m, camera_cfg, rot, tr)
        composite_plot = _transform_depth_to_camera_view(composite_depth_m, camera_cfg, rot, tr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _plot_depth(ax, depth_m: np.ndarray, title: str) -> None:
        depth_viz = depth_m.astype(np.float32).copy()
        depth_viz[depth_viz <= 1e-6] = np.nan
        im = ax.imshow(depth_viz, cmap="turbo", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title_suffix = " (Rectified View)" if (rot is not None or tr is not None) else ""
    _plot_depth(axes[0], background_plot, f"Background Depth{title_suffix}")
    _plot_depth(axes[1], object_plot, f"Object Depth{title_suffix}")
    _plot_depth(axes[2], composite_plot, f"Composite Depth{title_suffix}")
    plt.tight_layout()
    plt.show()
