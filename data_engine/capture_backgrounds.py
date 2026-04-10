"""CLI entry point for interactive background capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data_engine.camera import create_camera_backend
from data_engine.capture import BackgroundCaptureSession, CaptureConfig


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture background depth frames")
    parser.add_argument(
        "--config",
        type=str,
        default="data_engine/config/capture_config.example.json",
        help="Path to capture config JSON",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_json(cfg_path)

    camera = create_camera_backend(cfg.get("camera", {}))
    cap_cfg_raw = cfg.get("capture", {})
    cap_cfg = CaptureConfig(
        output_root=Path(cap_cfg_raw.get("output_root", "data/backgrounds/raw")),
        session_name=cap_cfg_raw.get("session_name"),
        max_frames=int(cap_cfg_raw.get("max_frames", 0)),
        preview_max_depth_m=float(cap_cfg_raw.get("preview_max_depth_m", 5.0)),
        save_preview_png=bool(cap_cfg_raw.get("save_preview_png", True)),
    )

    session = BackgroundCaptureSession(camera=camera, config=cap_cfg)
    session.run()


if __name__ == "__main__":
    main()
