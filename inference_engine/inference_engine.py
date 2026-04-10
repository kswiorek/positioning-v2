"""Dummy inference engine module.

This module will own runtime model loading and pose estimation APIs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class InferenceConfig:
    checkpoint_path: Path
    device: str = "cpu"


class InferenceEngine:
    """Entry point for runtime pose estimation workflows."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def estimate_pose(self, depth_image: Any, model_point_cloud: Any) -> Dict[str, Any]:
        """Estimate object pose from depth image and model point cloud."""
        raise NotImplementedError("Inference is not implemented yet.")
