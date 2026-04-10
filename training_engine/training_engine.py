"""Dummy training engine module.

This module will own training loops, checkpoints, and validation.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    run_dir: Path
    max_epochs: int = 0
    seed: int = 0


class TrainingEngine:
    """Entry point for model training workflows."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Run end-to-end training and validation."""
        raise NotImplementedError("Training is not implemented yet.")
