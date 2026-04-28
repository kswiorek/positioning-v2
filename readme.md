Updated version of the pose estimator project

## Current Status

- Project split into three modules: data engine, training engine, inference engine
- Data engine is functional end-to-end for dataset generation
- Training engine now has a modular dataset loader, loss functions, checkpoint helpers, and a reusable loop
- Inference engine is intentionally scaffolded for the next implementation phase

## Architecture Notes

- `data_engine` owns synthetic + real-background compositing, preprocessing, and metadata writing.
- `training_engine` consumes exported dataset artifacts through a dedicated dataset adapter and owns optimization/checkpointing.
- `inference_engine` stays decoupled and is still scaffolded.
- Dataset run authority is centralized in `data_engine/config/dataset_config*.json`.

## Dataset Generation (Implemented)

Run split-aware generation from workspace root:

```bash
python -m data_engine.generate_dataset --scene_config data_engine/config/scene_config.superquadric.example.json --dataset_config data_engine/config/dataset_config.example.json
```

The generator writes:

- `data/generated/<run_name>/<split>/samples/*.npz`
- `data/generated/<run_name>/<split>/metadata.jsonl`
- `data/generated/<run_name>/<split>/summary.json`
- `data/generated/<run_name>/summary.json`

Lean metadata format (single contract) is documented in
`data_engine/config/dataset_schema.example.json`.

## Training (Implemented)

Train from a config file in the same style as dataset generation:

```bash
python -m training_engine.train --config training_engine/training_config.example.json
```

The training config defines:

- dataset location and split names
- dataset storage mode (`ram` preloads each split into CPU memory, `disk` streams NPZs)
- model architecture settings
- optimizer, scheduler, and loss weights
- checkpoint/run directory and resume behavior

## Background Capture (Implemented)

Run from workspace root:

```bash
python -m data_engine.capture_backgrounds --config data_engine/config/capture_config.example.json
```

Synthetic debug mode (no camera required):

```bash
python -m data_engine.capture_backgrounds --config data_engine/config/capture_config.synthetic_perlin.json
```

Controls in preview window:

- `s`: save current depth frame
- `q`: quit session

Output is written to `data/backgrounds/raw/<session_name>/` with:

- `depth/frame_XXXXXX.npz`
- `preview/frame_XXXXXX.png` (optional)
- `metadata.json`

## Plane Fit + Placement Debug (Implemented)

Given a saved depth frame, fit the dominant plane and sample a valid
superquadric pose with constraints:

- center projects inside FOV
- camera origin is outside oriented object bounding box
- object is sampled in front of the fitted plane

Run:

```bash
python -m data_engine.debug_plane_sampling --depth_npz data/backgrounds/raw/synthetic_debug/depth/frame_000000.npz --out_json data/debug/placement_000000.json
```

## Full Composite Sample Debug (Implemented)

This command performs plane fitting, superquadric placement, object depth
rendering, depth compositing, and Open3D visualization.

Background handling now follows a normalized pipeline:

- fit dominant plane from captured depth
- normalize background so plane is canonical (normal aligned with +Z and plane through origin)
- randomize plane pose via pitch/yaw and camera distance from config
- optionally scale background in-plane (XY on canonical plane) to fill image FOV
- sample object placement relative to this randomized plane

```bash
python -m data_engine.debug_composite_sample --depth_npz data/backgrounds/raw/synthetic_debug/depth/frame_000000.npz
```

Outputs:

- `data/debug/composite_sample_000000.npz`
- `data/debug/composite_sample_000000.json`

Open3D windows:

- Scene view: camera frame + frustum + background cloud + sampled object mesh

Matplotlib figure:

- Background depth, object depth, and composite depth heatmaps (2D)