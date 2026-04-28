"""Command-line entrypoint for the v2 training engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import TrainingConfig
from .model import build_model
from .training_engine import TrainingEngine, load_and_resume


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return TrainingConfig.from_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a v2 pose model")
    parser.add_argument(
        "--config",
        default="training_engine/training_config.example.json",
        help="Path to the training config JSON",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint configured in the JSON file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_training_config(config_path)
    model = build_model(config)

    engine = TrainingEngine(config=config, model=model)

    resume_path = config.resume_from
    if args.resume and resume_path is not None and resume_path.exists():
        load_and_resume(engine, resume_path)

    summary = engine.train()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()