"""Interactive background depth capture utility."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data_engine.camera.base import DepthCameraBackend


@dataclass
class CaptureConfig:
    output_root: Path
    session_name: Optional[str] = None
    max_frames: int = 0
    preview_max_depth_m: float = 5.0
    save_preview_png: bool = True


class BackgroundCaptureSession:
    """Interactive capture loop for depth backgrounds."""

    def __init__(self, camera: DepthCameraBackend, config: CaptureConfig) -> None:
        self.camera = camera
        self.config = config

    def run(self) -> Path:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for background capture.") from exc

        session_name = self.config.session_name or time.strftime("session_%Y%m%d_%H%M%S")
        session_dir = self.config.output_root / session_name
        depth_dir = session_dir / "depth"
        preview_dir = session_dir / "preview"

        depth_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_preview_png:
            preview_dir.mkdir(parents=True, exist_ok=True)

        self.camera.start()
        print("Capture controls: [s]=save frame, [q]=quit")

        saved = 0
        started = time.time()
        metadata: Dict[str, Any] = {
            "session_name": session_name,
            "started_unix_s": started,
            "capture_config": {
                "output_root": str(self.config.output_root),
                "session_name": self.config.session_name,
                "max_frames": self.config.max_frames,
                "preview_max_depth_m": self.config.preview_max_depth_m,
                "save_preview_png": self.config.save_preview_png,
            },
            "camera": self.camera.metadata(),
            "frames": [],
        }

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                preview = self._depth_preview(frame.depth_m, self.config.preview_max_depth_m)
                cv2.imshow("Background Capture - Depth", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key == ord("s"):
                    frame_name = f"frame_{saved:06d}"
                    depth_path = depth_dir / f"{frame_name}.npz"
                    np.savez_compressed(depth_path, depth_m=frame.depth_m.astype(np.float32))

                    if self.config.save_preview_png:
                        preview_path = preview_dir / f"{frame_name}.png"
                        cv2.imwrite(str(preview_path), preview)

                    metadata["frames"].append(
                        {
                            "frame_index": saved,
                            "frame_name": frame_name,
                            "timestamp_s": frame.timestamp_s,
                            "depth_file": str(Path("depth") / f"{frame_name}.npz"),
                            "preview_file": str(Path("preview") / f"{frame_name}.png")
                            if self.config.save_preview_png
                            else None,
                        }
                    )
                    saved += 1
                    print(f"Saved {frame_name}")

                    if self.config.max_frames > 0 and saved >= self.config.max_frames:
                        print("Reached max_frames; stopping capture.")
                        break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

        metadata["ended_unix_s"] = time.time()
        metadata["num_saved_frames"] = saved

        metadata_path = session_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Session saved to: {session_dir}")
        return session_dir

    @staticmethod
    def _depth_preview(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
        cv2 = importlib.import_module("cv2")
        depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth / max(max_depth_m, 1e-6), 0.0, 1.0)
        depth_u8 = (depth * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
