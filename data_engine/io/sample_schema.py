"""Dataset sample metadata schema used by the v2 data engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlaneFitInfo:
    normal_camera: List[float]
    offset_camera: float
    inlier_ratio: float


@dataclass
class PlacementInfo:
    object_center_camera: List[float]
    orientation_quat_xyzw: List[float]
    object_plane_distance_m: float
    center_in_fov: bool
    camera_outside_object_bbox: bool


@dataclass
class ObjectInfo:
    object_id: str
    object_source: str
    object_asset_path: Optional[str]
    normalization_scale: float
    bbox_extent_m: List[float]
    shape_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactInfo:
    enabled: bool
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleMetadata:
    sample_id: str
    split: str
    global_seed: int
    pipeline_version: str
    domain_tags: List[str]
    object_info: ObjectInfo
    background_id: str
    capture_session_id: Optional[str]
    plane_fit: PlaneFitInfo
    placement: PlacementInfo
    artifacts: ArtifactInfo
    gt_transform_camera_from_object: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
