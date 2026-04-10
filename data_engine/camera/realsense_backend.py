"""Intel RealSense depth backend (L515 compatible)."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import numpy as np

from .base import DepthCameraBackend, DepthFrame


class RealSenseDepthBackend(DepthCameraBackend):
    """Depth backend powered by pyrealsense2."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color: bool = False,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_color = bool(enable_color)

        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._intrinsics: Dict[str, float] = {}

    def start(self) -> None:
        try:
            rs = importlib.import_module("pyrealsense2")
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for RealSense backend. "
                "Install Intel RealSense Python bindings first."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if self.enable_color:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self._pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        depth_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream_profile.get_intrinsics()
        self._intrinsics = {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
        }

        if self.enable_color:
            self._align = rs.align(rs.stream.color)

    def read(self) -> Optional[DepthFrame]:
        if self._pipeline is None or self._rs is None:
            raise RuntimeError("RealSense backend is not started.")

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None

        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * float(self._depth_scale)

        color_bgr = None
        if self.enable_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                color_bgr = np.asanyarray(color_frame.get_data())

        return DepthFrame(
            depth_m=depth_m,
            color_bgr=color_bgr,
            timestamp_s=float(depth_frame.get_timestamp()) / 1000.0,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": "realsense",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "enable_color": self.enable_color,
            "depth_scale_m": self._depth_scale,
            "intrinsics": self._intrinsics,
        }
