
Your direction is good, and your instincts are exactly right: this is mainly a simulation-to-reality and project-architecture problem, not a “make the network deeper” problem.

Given your current strengths in positioning/models/hybrid_pose_net.py, positioning/losses.py, and positioning/estimators/network_estimator.py, these are the highest-impact structural improvements for a clean v2 rewrite:

1. Split the project into 3 independent products

Data engine (scene generation, domain randomization, real background ingestion)
Training engine (model, losses, optimization, checkpoints)
Inference engine (runtime estimator, optional refinement, diagnostics)
This prevents dataset experiments from tangling with runtime code and makes your proof-of-concept easier to defend.

2. Make “sim-to-real” a first-class module, not a side setting

Build a dedicated realism pipeline for depth images:
Sensor artifacts: holes, edge fattening, multipath-like noise, quantization, flying pixels
Background artifacts: rough geometry, clutter, partial occluders
Material artifacts: depth dropout on dark/shiny surfaces
Pose artifacts: slight calibration drift and lens distortion residuals
Your idea of capturing real backgrounds and compositing synthetic objects is excellent. Treat it as a formal data source with versioning and metadata, not ad hoc samples.

3. Add a geometry-aware preprocessor stage before the network

Optional foreground proposal from depth geometry
Surface normal and local roughness channels
Confidence mask for valid depth regions
Right now the network in positioning/estimators/network_estimator.py sees raw normalized depth; adding geometric priors makes it less sensitive to rough backgrounds and reduces pressure on the model to “discover” basic 3D cues each time.

4. Train as two tasks: objectness + pose

First head: “is target object present / where likely”
Second head: pose only on confident regions/features
This is a major robustness upgrade in real scenes. It avoids forcing a pure pose regressor to solve detection and pose implicitly in one step.

5. Use teacher-student adaptation with your classical pipeline

Run the classical estimator in positioning/estimators/algorithm_estimator.py on real captures
Keep only high-confidence outputs as pseudo-labels
Fine-tune the network on these real samples with confidence weighting
This turns your benchmark into a data engine for adaptation, not just a comparator.

6. Build explicit uncertainty and fallback logic into inference

Predict pose + confidence/uncertainty
If uncertainty is high, trigger ICP or classical fallback
Return both estimate and trust score
For real deployment this is fundamental. A fast model that knows when it is unsure beats a faster model that is confidently wrong.

7. Redefine the dataset around “domains,” not just samples

Domain labels: synthetic clean, synthetic randomized, real background composite, real captured
Balanced batch sampler across domains
Per-domain metrics in validation
This makes generalization measurable and prevents overfitting to whichever generator currently dominates.

8. Move to protocol-based evaluation, not single loss curves

Pose metrics stratified by clutter level, occlusion level, depth quality
Robustness curves: accuracy vs background roughness
Failure taxonomy: symmetric flips, translation bias, false lock on clutter
This will reveal whether changes actually improve real-world behavior, not just validation loss.

9. Introduce shape-family curriculum

Start with broad desktop-like object families for robust geometric priors
Then progressively bias training toward blade-like elongated objects
Keep a fraction of broad data to prevent specialization collapse
This directly fits your “varied items for generalization” idea while preserving relevance to turbine-blade-like geometry.

10. Enforce plugin interfaces from day one in v2

Estimator interface: estimate + confidence + debug artifacts
Data backend interface: sample scene, sample object, sample background
Refinement interface: ICP/post-filter interchangeable
Model registry: swap encoders/heads without touching training loop
That keeps v2 clean as it grows and avoids the “messy monolith” problem again.

If you want, I can draft a concrete v2 repository skeleton next (folders, interfaces, experiment config schema, and the minimum viable training/inference flow) so you can start rewriting immediately with clear boundaries.


You want to start the rewrite with only the core architectural split, so I’ll scaffold a minimal v2 structure with dummy modules for data, training, and inference in the new workspace, without adding any heavy implementation yet. I’ll quickly inspect positioning v2 and then create the placeholder files with clear interfaces so we can begin data generation next.

I’m creating a minimal but structured scaffold now so each subsystem has a clear entry point and interface, and the outcome will be a runnable skeleton in positioning v2 ready for implementing data generation next.

The scaffold files are in place; I’m doing a quick directory check to confirm the v2 split is complete before we move on to implementing data generation.

Done. I created a clean v2 scaffold with dummy modules for the three independent parts:

Data engine: positioning v2/data_engine/data_engine.py
Training engine: positioning v2/training_engine/training_engine.py
Inference engine: positioning v2/inference_engine/inference_engine.py
Package entry files were also added:

positioning v2/data_engine/init.py
positioning v2/training_engine/init.py
positioning v2/inference_engine/init.py
And I added a short layout note:

positioning v2/project_layout.md
These are intentionally placeholders with minimal config dataclasses and NotImplemented methods, so we can now focus directly on data generation in a structured way.

Next step I can do now:

Create the full data generation folder structure and configs (background ingestion, object library, compositing, metadata schema).
Add a first dummy pipeline that writes sample metadata and empty sample files so the training side can already read a consistent format.

You are asking for a practical v2 dataset-generation blueprint that stays modular and manageable, so I’ll quickly inspect the current generation pipeline files and then map your idea into a step-by-step build plan with folder structure, milestones, and what to implement first.

Your direction is solid, and yes, skipping heavy handcrafted depth preprocessing is reasonable if the CNN already performs well. In your case, the biggest gains will come from data realism, clean modular tooling, and consistent geometry conventions.

What you already covered well:

Modular camera capture with driver abstraction.
Mixing superquadrics with STL objects.
Plane fitting for realistic placement.
Background-only capture and compositing idea.
What you are still missing (important):

Coordinate and unit conventions.
You need one canonical convention for all assets: meters, right-handed frame, forward axis, and camera frame. Most sim-to-real bugs come from hidden convention mismatches.
Symmetry metadata per object.
For elongated or near-symmetric shapes, multiple rotations are equivalent. Store symmetry info with each object so training and evaluation do not punish valid equivalent poses.
Object quality filtering for STL pool.
Many public STLs are non-watertight, huge/small scale, or weird topology. Add an offline sanitize step before they enter the generation pool.
Occlusion and truncation control.
Explicitly generate partial visibility cases, not just clean full-object views.
Domain tags in metadata.
Mark each sample as synthetic_only, real_bg_composite, superquadric, stl, occluded, etc. This makes debugging and later ablations much easier.
Real camera artifact simulation.
For L515-like depth, add missing-depth holes, edge artifacts, quantization, and depth jitter. This is often more important than shape diversity alone.
Train/val split by object identity.
For generalization, keep some object IDs only in validation so you can measure unseen-object performance honestly.
Detailed implementation plan (student-friendly, structured but not heavy)

Phase 1: Lock interfaces and folder layout

Keep the three-part split already created in positioning v2/project_layout.md and positioning v2/data_engine/data_engine.py.
In data engine, define these submodules:
capture for camera background collection
assets for STL ingestion and normalization
generators for superquadric and STL scene objects
compose for object-on-background placement
render for depth rendering and sensor artifacts
io for dataset writing and metadata
Define one sample schema now and never improvise later.
Phase 2: Background capture tool

Build a live preview app with a camera backend interface:
RealSense backend using pyrealsense2
Generic backend using OpenCV VideoCapture or another depth camera SDK
Features:
Show live depth preview
Click or keypress to save frame
Save intrinsic/extrinsic info per frame if available
Save session metadata (camera type, resolution, timestamp, operator notes)
Output format:
backgrounds/raw/session_x/frame_000123.(png or npz)
backgrounds/raw/session_x/metadata.json
Phase 3: Object asset pipeline (STL + superquadric)

Reuse concepts from positioning/shape_generation.py for superquadrics.
Add STL importer pipeline:
Load mesh
Repair if possible
Normalize scale to meters
Center canonical pose
Compute canonical point cloud
Store object metadata:
object_id
source (superquadric or stl)
scale factor
bbox extents
symmetry_type
quality flags
Save processed objects in one cache so generation is fast.
Phase 4: Plane-aware compositing logic

For each captured background depth:
Fit dominant plane with RANSAC
Sample placement point on plane inside valid FOV region
Sample orientation and standoff distance
Apply optional distance amplification parameter as you suggested
Compose by rendering object depth then z-buffer merge with background depth.
Keep per-sample placement diagnostics:
plane normal
plane fit inlier ratio
object-to-plane distance
visible fraction
Phase 5: Sensor realism augmentation

After depth composition, apply camera-like artifacts:
Random holes/dropout
Edge noise near depth discontinuities
Quantization
Mild spatial noise and occasional outliers
Keep knobs simple and few. Save applied augmentation params into sample metadata for traceability.
Phase 6: Dataset writer and splits

Write train/val splits with deterministic seeds.
Store each sample with:
depth_image
model_points
gt_transform
object_id
domain tags
generation parameters snapshot
Reuse ideas from positioning/generate_dataset.py, but make schema richer from day one.
Add identity-based split option for STL objects.
Phase 7: Fast sanity checks (lightweight, not formal QA)

Small visualizer script: overlay rendered object depth vs final composed depth.
Histogram checks:
depth ranges
object pixel coverage
invalid pixel ratio
Plane-fit failure rate report.
One quick notebook or script generating 20 random samples for manual inspection.
Suggested first build order (so you do not get lost)

Background capture tool with backend abstraction.
STL normalization cache pipeline.
Plane-fit + placement + compose for one sample.
Batch generator + metadata writer.
Artifact augmentation.
Optional parallelization and speed-ups.
Practical mapping from your old code

Reuse base generation concepts from positioning/generate_dataset.py.
Reuse superquadric generation from positioning/shape_generation.py.
Reuse scene/noise ideas from positioning/scene_generation.py, but convert them into explicit modules and metadata-driven steps.
If you want, next I can draft the exact v2 data schema and config files first (single source of truth), then we can implement the camera capture module against that schema.

Agreed. Your constraints are good, and they simplify generation in a useful way.

Decisions We Lock In

Canonical convention: one fixed metric and frame convention across all assets and samples.
No explicit symmetry handling in labels by default.
Placement constraints:
Object center must be inside camera FOV.
Camera cannot be inside the object bounding box.
Object must be in front of fitted background plane by a sampled offset.
Let truncation and partial visibility emerge naturally from geometry.
Domain tags and rich metadata are required for every sample.
Sensor artifact simulation is included, but controlled and toggleable.
On your symmetry question: yes, you can compute a quick automatic asymmetry score and use it only as a filter, not as a label correction.

Fast automatic symmetry check (offline, optional):

Convert object to normalized point cloud.
Mirror across candidate planes (xy, yz, xz, and PCA-aligned variants).
Compute Chamfer distance from original to mirrored cloud.
Lower distance means more mirror symmetry.
Also test 180-degree rotational symmetry around principal axis similarly.
Keep objects/shapes with asymmetry score above a threshold.
This gives you a practical way to reject nearly symmetric STLs while keeping your twisted-superquadric strategy.

On watertight meshes: you are mostly right for your use case, but a caveat.

Not strictly required for single-view depth compositing.
Still useful to detect and filter extreme mesh issues:
Degenerate triangles
Broken scale or huge coordinate outliers
Very sparse or spiky geometry
So keep a light sanitize step, not a strict watertight requirement.
Metadata fields to include per sample (recommended minimum):

sample_id
split
global_seed
pipeline_version
domain tags array
object_id
object_source: superquadric or stl
object_asset_path (for stl)
object_scale_normalization_factor
object_bbox_extent
shape_params (for procedural shapes)
background_id and capture_session_id
plane_fit:
normal
offset
inlier_ratio
placement:
object_center_camera_frame
orientation_euler or quaternion
object_plane_distance
visibility:
projected_center_in_fov boolean
bbox_in_frame_ratio
depth_valid_ratio
sensor_artifacts parameters
camera intrinsics snapshot
gt_transform_camera_from_object
Extra useful field:

rejection_reason for failed generation attempts, so you can tune constraints with evidence.
For artifact realism:

Start simple and realistic, not heavy:
Missing-depth masks
Edge noise near discontinuities
Depth quantization
Small random jitter
Keep all artifact parameters logged in metadata.
Add one config switch to disable all artifacts for ablation.
This is a strong plan for your student project and stays practical.

Next step to start implementation cleanly:

I draft the concrete config schema and sample metadata schema first.
Then we implement background capture with modular camera backends against that schema.
Then object asset pipeline and compositing.
If you want, I can provide the exact schema structure in the next message so you can implement directly without guessing names later.