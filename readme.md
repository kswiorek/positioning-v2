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
python -m data_engine.generate_dataset --scene_config data_engine/config/scene_config.json --dataset_config data_engine/config/dataset_config.json
```

The generator writes:

- `data/generated/<run_name>/<split>/samples/*.npz`
- `data/generated/<run_name>/<split>/metadata.jsonl`
- `data/generated/<run_name>/<split>/summary.json`
- `data/generated/<run_name>/summary.json`

Lean metadata format (single contract) is documented in
`data_engine/config/dataset_schema.example.json`.

## Export for legacy v1 training (`positioning`)

Convert a v2 run into the flat NPZ layout expected by the original project
(`depth_image`, `model_points`, `bbox_corners`, `gt_transform` under
`train/` and `val/`):

```bash
python -m data_engine.convert_to_v1_dataset \
  --input_dir data/generated/dataset_test \
  --output_dir data/exported/v1_dataset \
  --scene_config data_engine/config/scene_config.json
```

Optional: pass `--network_config` from the v1 repo so `metadata.json` matches
what `generate_dataset.py` would have written. Use `--keep_sample_ids` to
preserve v2 filenames instead of renumbering `000000`..`N-1`.

Point the v1 `network_config.json` dataset path (or `train.py` `--dataset_dir`)
at `--output_dir`.

## Training (Implemented)

Train from a config file in the same style as dataset generation:

```bash
python -m training_engine.train --config training_engine/training_config.json
```

The training config defines:

- dataset location and split names
- dataset storage mode (`ram` preloads each split into CPU memory, `disk` streams NPZs)
- model architecture settings
- optimizer, scheduler, and loss weights
- checkpoint/run directory and resume behavior
- batch-level console progress under `monitoring`
- per-epoch TensorBoard scalar groups when enabled: `train_epoch`, `val_pred_mask`, and `val_gt_mask` (only if segmentation is enabled)

If TensorBoard is installed, you can watch a run with:

```bash
tensorboard --logdir runs/hybrid_pose_v2/tensorboard
```

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