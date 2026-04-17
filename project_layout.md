# Positioning v2 Layout

This workspace is split into three independent parts:

1. data_engine: dataset creation (synthetic generation, real background ingestion, compositing)
2. training_engine: model training, validation, checkpointing
3. inference_engine: runtime pose estimation and optional refinement

## Status

- data_engine: implemented and operational (capture, STL preprocessing, mixed object generation,
  compositing, artifacts, split-aware bulk generation, metadata writing).
- training_engine: scaffold only, API placeholder for future training/validation loop.
- inference_engine: scaffold only, API placeholder for future runtime estimator/refinement.

## Boundary Rules

- data_engine owns generation logic, scene/dataset configs, and dataset metadata contracts.
- training_engine and inference_engine consume generated artifacts; they do not import data_engine internals.
- Dataset run authority (seed, counts/splits, workers, output path) is defined in dataset config.
