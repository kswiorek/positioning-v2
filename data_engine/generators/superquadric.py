"""Superquadric shape generator (v2 copy of proven v1 logic)."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _signed_power(x: np.ndarray, epsilon: float) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** epsilon


def _ensure_outward_normals(mesh: o3d.geometry.TriangleMesh) -> None:
    """Flip winding if normals point inward relative to mesh center."""
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if len(vertices) == 0 or len(normals) == 0:
        return

    # For centered meshes, outward normals should have positive radial dot product.
    score = float(np.mean(np.einsum("ij,ij->i", vertices, normals)))
    if score < 0.0:
        triangles = np.asarray(mesh.triangles).copy()
        triangles = triangles[:, [0, 2, 1]]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()


def _create_tapered_superquadric_mesh(
    length: float,
    major_axis_root: float,
    minor_axis_root: float,
    major_axis_tip: float,
    minor_axis_tip: float,
    twist_angle_deg: float,
    epsilon_1: float,
    epsilon_2: float,
    resolution: int = 50,
) -> o3d.geometry.TriangleMesh:
    n_theta = int(resolution)
    n_z = max(20, int(resolution * 0.4))

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_param = np.linspace(-1.0, 1.0, n_z)

    t_vals = (z_param + 1.0) / 2.0
    t_grid, theta_grid = np.meshgrid(t_vals, theta, indexing="ij")

    major = major_axis_root * (1.0 - t_grid) + major_axis_tip * t_grid
    minor = minor_axis_root * (1.0 - t_grid) + minor_axis_tip * t_grid
    twist = np.deg2rad(twist_angle_deg) * t_grid
    z = (t_grid * 2.0 - 1.0) * (length / 2.0)

    x = major * _signed_power(np.cos(theta_grid), epsilon_1)
    y = minor * _signed_power(np.sin(theta_grid), epsilon_2)

    x_rot = x * np.cos(twist) - y * np.sin(twist)
    y_rot = x * np.sin(twist) + y * np.cos(twist)

    vertices = np.stack([x_rot.ravel(), y_rot.ravel(), z.ravel()], axis=1)

    i_arr = np.arange(n_z - 1)
    j_arr = np.arange(n_theta)
    ii, jj = np.meshgrid(i_arr, j_arr, indexing="ij")

    idx1 = ii * n_theta + jj
    idx2 = ii * n_theta + (jj + 1) % n_theta
    idx3 = (ii + 1) * n_theta + jj
    idx4 = (ii + 1) * n_theta + (jj + 1) % n_theta

    tri1 = np.stack([idx1, idx3, idx2], axis=-1).reshape(-1, 3)
    tri2 = np.stack([idx2, idx3, idx4], axis=-1).reshape(-1, 3)

    root_center_idx = len(vertices)
    tip_center_idx = root_center_idx + 1
    vertices = np.vstack([vertices, [[0.0, 0.0, -length / 2.0]], [[0.0, 0.0, length / 2.0]]])

    j_arr = np.arange(n_theta)
    root_cap = np.stack(
        [np.full(n_theta, root_center_idx), j_arr, (j_arr + 1) % n_theta], axis=1
    )
    base_idx = (n_z - 1) * n_theta
    tip_cap = np.stack(
        [np.full(n_theta, tip_center_idx), base_idx + (j_arr + 1) % n_theta, base_idx + j_arr], axis=1
    )

    triangles = np.vstack([tri1, tri2, root_cap, tip_cap])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def generate_superquadric_canonical_model(
    config: dict,
    seed: int | None = None,
    include_point_cloud: bool = False,
):
    """Generate canonical mesh, model cloud, and bbox corners for superquadrics."""
    sq_cfg = config["scene"]["superquadric"]
    rng = np.random.default_rng(seed)

    length = float(sq_cfg["base_length_m"]) * float(rng.uniform(*sq_cfg["length_scale_range"]))

    cross = sq_cfg["cross_section"]
    major_axis = float(rng.uniform(*cross["major_axis_range"]))
    minor_axis = float(rng.uniform(*cross["minor_axis_range"]))
    min_aspect = float(cross.get("min_aspect_ratio", 1.0))
    if major_axis / max(minor_axis, 1e-8) < min_aspect:
        major_axis = minor_axis * float(rng.uniform(min_aspect, min_aspect + 0.3))

    taper_ratio = float(rng.uniform(*sq_cfg["taper_ratio_range"]))
    twist_deg = float(rng.uniform(*sq_cfg["twist_angle_range_deg"]))

    exponents = sq_cfg["exponents"]
    eps1 = float(rng.uniform(*exponents["epsilon_1_range"]))
    eps2 = float(rng.uniform(*exponents["epsilon_2_range"]))

    resolution = int(sq_cfg.get("mesh_resolution", 50))

    mesh = _create_tapered_superquadric_mesh(
        length=length,
        major_axis_root=major_axis * taper_ratio,
        minor_axis_root=minor_axis * taper_ratio,
        major_axis_tip=major_axis,
        minor_axis_tip=minor_axis,
        twist_angle_deg=twist_deg,
        epsilon_1=eps1,
        epsilon_2=eps2,
        resolution=resolution,
    )

    mesh.translate(-mesh.get_center())
    _ensure_outward_normals(mesh)

    vertices = np.asarray(mesh.vertices)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    bbox_corners = np.array(
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

    points_n = int(sq_cfg.get("point_sample_count", 5000))
    model_cloud = None
    if include_point_cloud:
        if seed is not None:
            o3d.utility.random.seed(int(seed) & 0x7FFFFFFF)
        model_cloud = mesh.sample_points_uniformly(number_of_points=points_n)
        model_cloud.translate(-model_cloud.get_center())

    shape_params = {
        "length": length,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "taper_ratio": taper_ratio,
        "twist_deg": twist_deg,
        "epsilon_1": eps1,
        "epsilon_2": eps2,
        "resolution": resolution,
        "point_sample_count": points_n,
    }

    return mesh, model_cloud, bbox_corners, shape_params
