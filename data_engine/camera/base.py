"""Camera backend interfaces for depth capture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DepthFrame:
    """Single depth frame with optional color image."""

    depth_m: np.ndarray
    color_bgr: Optional[np.ndarray] = None
    timestamp_s: Optional[float] = None


class DepthCameraBackend(ABC):
    """Abstract depth camera backend interface."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the camera stream."""

    @abstractmethod
    def read(self) -> Optional[DepthFrame]:
        """Read one frame from the active stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and release all resources."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return backend metadata (camera type, intrinsics if known, etc.)."""
