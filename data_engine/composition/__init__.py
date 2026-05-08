"""Compositing and placement utilities."""

from .background_normalization import (
    BackgroundTransformParams,
    normalize_and_randomize_background_depth,
)
from .camera_artifacts import apply_camera_artifacts
from .depth_compositor import compose_depth, render_mesh_depth, transform_mesh
from .plane_fit import PlaneModel, fit_plane_from_depth, select_plane_support_mask
from .plane_placement import (
    PlacementConstraints,
    center_projects_inside_fov,
    is_camera_inside_aabb,
    sample_plane_offset_distance,
)
from .placement_sampling import PlacementSample, sample_pose_from_camera, sample_pose_on_plane

__all__ = [
    "BackgroundTransformParams",
    "apply_camera_artifacts",
    "compose_depth",
    "PlaneModel",
    "normalize_and_randomize_background_depth",
    "render_mesh_depth",
    "transform_mesh",
    "fit_plane_from_depth",
    "select_plane_support_mask",
    "PlacementConstraints",
    "PlacementSample",
    "center_projects_inside_fov",
    "is_camera_inside_aabb",
    "sample_plane_offset_distance",
    "sample_pose_from_camera",
    "sample_pose_on_plane",
]
