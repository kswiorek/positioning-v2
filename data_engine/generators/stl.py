"""STL-backed canonical shape generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


_STL_BASE_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _ensure_outward_normals(mesh: o3d.geometry.TriangleMesh) -> None:
    """Flip winding if normals point inward relative to mesh center."""
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if len(vertices) == 0 or len(normals) == 0:
        return

    score = float(np.mean(np.einsum("ij,ij->i", vertices, normals)))
    if score < 0.0:
        triangles = np.asarray(mesh.triangles).copy()
        triangles = triangles[:, [0, 2, 1]]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()


def _bbox_corners_from_mesh(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    return np.array(
        [
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
        ],
        dtype=np.float32,
    )


def _list_stl_files(stl_cfg: dict) -> list[Path]:
    input_cfg = stl_cfg.get("input", {})
    directory = Path(input_cfg.get("directory", "data/stl"))
    pattern = str(input_cfg.get("glob", "*.stl"))
    files = sorted(p for p in directory.glob(pattern) if p.is_file())
    return files


def _get_cached_stl_mesh(stl_path: Path) -> o3d.geometry.TriangleMesh:
    key = str(stl_path.resolve())
    cached = _STL_BASE_MESH_CACHE.get(key)
    if cached is None:
        base_mesh = o3d.io.read_triangle_mesh(str(stl_path))
        if not base_mesh.has_vertices() or not base_mesh.has_triangles():
            raise RuntimeError(f"Failed to load valid triangle mesh from STL: {stl_path}")

        base_mesh.remove_duplicated_vertices()
        base_mesh.remove_degenerate_triangles()
        base_mesh.remove_duplicated_triangles()
        base_mesh.remove_non_manifold_edges()
        base_mesh.translate(-base_mesh.get_center())
        _ensure_outward_normals(base_mesh)

        vertices = np.asarray(base_mesh.vertices).astype(np.float64)
        triangles = np.asarray(base_mesh.triangles).astype(np.int32)
        _STL_BASE_MESH_CACHE[key] = (vertices, triangles)
        cached = _STL_BASE_MESH_CACHE[key]

    vertices, triangles = cached
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.copy())
    mesh.triangles = o3d.utility.Vector3iVector(triangles.copy())
    return mesh


def generate_stl_canonical_model(
    config: dict,
    seed: int | None = None,
    include_point_cloud: bool = False,
):
    """Load random STL, normalize scale by max AABB dimension, and return canonical outputs."""
    scene_cfg = config.get("scene", {})
    stl_cfg = scene_cfg.get("stl", {})
    rng = np.random.default_rng(seed)

    stl_files = _list_stl_files(stl_cfg)
    if not stl_files:
        input_cfg = stl_cfg.get("input", {})
        directory = Path(input_cfg.get("directory", "data/stl"))
        pattern = str(input_cfg.get("glob", "*.stl"))
        raise RuntimeError(f"No STL files found in '{directory}' with pattern '{pattern}'.")

    selected_idx = int(rng.integers(0, len(stl_files)))
    selected_path = stl_files[selected_idx]

    mesh = _get_cached_stl_mesh(selected_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"Failed to load valid triangle mesh from STL: {selected_path}")

    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    max_dim = float(np.max(np.asarray(extent, dtype=np.float64)))
    if max_dim <= 1e-9:
        raise RuntimeError(f"STL mesh appears degenerate (max AABB dim near zero): {selected_path}")

    norm_cfg = stl_cfg.get("normalization", {})
    target_range = norm_cfg.get("max_aabb_dimension_range_m", [0.8, 1.2])
    target_min = float(target_range[0])
    target_max = float(target_range[1])
    if target_max < target_min:
        target_min, target_max = target_max, target_min

    target_max_dim = float(rng.uniform(target_min, target_max))
    scale_factor = target_max_dim / max_dim
    mesh.scale(scale_factor, center=(0.0, 0.0, 0.0))

    mesh.translate(-mesh.get_center())

    bbox_corners = _bbox_corners_from_mesh(mesh)

    points_cfg = stl_cfg.get("points", {})
    sq_cfg = scene_cfg.get("superquadric", {})
    points_n = int(points_cfg.get("sample_count", sq_cfg.get("point_sample_count", 5000)))
    model_cloud = None
    if include_point_cloud:
        if seed is not None:
            o3d.utility.random.seed(int(seed) & 0x7FFFFFFF)
        model_cloud = mesh.sample_points_uniformly(number_of_points=points_n)
        model_cloud.translate(-model_cloud.get_center())

    shape_params = {
        "object_source": "stl",
        "object_id": selected_path.stem,
        "object_asset_path": str(selected_path).replace("\\", "/"),
        "max_aabb_dim_before_scale_m": max_dim,
        "target_max_aabb_dim_m": target_max_dim,
        "normalization_scale": scale_factor,
        "sample_points": points_n,
    }

    return mesh, model_cloud, bbox_corners, shape_params
