"""Lean dataset sample metadata schema used by the v2 data engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SampleMetadata:
    sample_id: str
    split: str
    sample_seed: int
    domain_tags: List[str]
    object_id: str
    object_source: str
    object_asset_path: Optional[str]
    background_id: str
    background_asset_path: Optional[str]
    bbox_extent_m: Optional[List[float]]
    bbox_corners_m: Optional[List[List[float]]]
    gt_transform_camera_from_object: List[List[float]]
    depth_npz: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
