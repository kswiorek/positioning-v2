"""Generic OpenCV backend.

This backend is intentionally simple and treats incoming single-channel or BGR
frames as depth-like input for quick prototyping with non-RealSense cameras.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from .base import DepthCameraBackend, DepthFrame


class OpenCVDepthBackend(DepthCameraBackend):
    """Generic backend using cv2.VideoCapture."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        depth_scale: float = 1.0,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = float(depth_scale)
        self._cap = None
        self._cv2 = None

    def start(self) -> None:
        try:
            self._cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for OpenCV backend.") from exc

        self._cap = self._cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        if self.width is not None:
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps is not None:
            self._cap.set(self._cv2.CAP_PROP_FPS, int(self.fps))

    def read(self) -> Optional[DepthFrame]:
        if self._cap is None:
            raise RuntimeError("OpenCV backend is not started.")

        ok, frame = self._cap.read()
        if not ok:
            return None

        if frame.ndim == 3:
            gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            depth_m = gray.astype(np.float32) * self.depth_scale
            color_bgr = frame
        else:
            depth_m = frame.astype(np.float32) * self.depth_scale
            color_bgr = None

        return DepthFrame(depth_m=depth_m, color_bgr=color_bgr, timestamp_s=time.time())

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "opencv",
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "depth_scale_m": self.depth_scale,
        }
