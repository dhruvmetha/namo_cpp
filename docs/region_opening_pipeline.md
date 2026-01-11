# Region Opening with ML-Guided Primitive Search

This document is written to serve as both a **Method** section and an
**Experimental Setup** section. It describes the region opening algorithm,
how ML is attached to inference, the exact model inputs/outputs, and the
specific model/config used for evaluation. MCTS is not used anywhere in this
pipeline.

## Method

### 1) Task and representation

We solve NAMO (navigation among movable obstacles). The robot must reach a
specified goal pose by pushing movable objects. We represent the free space
as **regions** computed by a wavefront-style reachability analysis. Regions
are connected by boundaries (edges) that can be blocked by movable objects.
A successful action is one that **creates a new opening** from the robot’s
current region to a neighboring region.

### 2) Region connectivity snapshot

Before planning, we compute a fast connectivity snapshot of the current scene:

- A grid-based reachability map is created at fixed resolution.
- Regions are identified as connected components of free space.
- The robot’s current region is labeled.
- For each adjacent region, we identify **blocking objects** that lie along
  the boundary between regions.
- For each non-robot region, we sample a small set of **region goals** used
  for validation.

This snapshot is recomputed per environment state so it reflects current object
poses rather than the original XML.

### 3) Opening validation criterion

To decide whether a push created a valid opening, we use a reachability test:

```
reachable_before = number of sampled region goals reachable before push
reachable_after  = number of sampled region goals reachable after push
success = (reachable_after > reachable_before) and (reachable_after >= 1)
```

This ensures success is attributed to the push itself (a new opening), not just
that a goal is reachable at some point.

### 4) Region opening search (core algorithm)

For each neighbor region, we try to open a path by pushing blocking objects.
We explicitly reset the simulator state before each object trial so all objects
are evaluated from the same baseline.

**Algorithm outline (per neighbor):**

1. **Pre-check**: if the neighbor region is already reachable, record a zero-push
   success and move on.
2. **Blocking objects**: collect objects on the boundary between the robot region
   and the neighbor region.
3. **Reachable objects**: filter to objects the robot can currently reach.
4. **Multi-push search**:
   - Outer BFS over chain depth (1-push, 2-push, ... up to `region_max_chain_depth`).
   - Inner BFS over primitive candidates (edge points x push depth).
   - Each candidate push is executed and validated using the reachability test.

**Cost and pruning:**

- Each primitive has a depth (push length). Chain cost is the sum of primitive
  depths plus a flat chain penalty if multiple pushes are used.
- If a chain with lower cost is found, higher-cost chains are pruned.
- Collisions or stuck events blacklist deeper depths on the same edge.
- Optional beam width limits the number of frontier states per depth.

### 5) Goal generation strategies (how we produce candidate pushes)

The planner never directly executes a continuous ML pose. Instead, it operates
on a fixed library of **motion primitives** and uses ML only to prioritize which
primitives to try first.

#### 5.1 Primitive-only

- A motion primitive database encodes relative pushes for three object shapes
  (square, tall, wide).
- Each object face has a fixed set of edge points (e.g., 15 per face).
- For each edge point, we have a short sequence of pushes of increasing length
  (depth 1..10).
- These are transformed into world coordinates based on the object’s current
  pose and returned as a fixed grid of candidates.

#### 5.2 ML-aligned primitives (goal_strategy: ml)

- The diffusion model predicts continuous SE(2) goal samples.
- Each ML sample is aligned to nearby primitive slots within position/angle
  tolerances.
- Only aligned slots are returned. If no aligned slots exist, the object yields
  no candidates.

#### 5.3 ML with fallback (goal_strategy: ml_fallback)

- The entire primitive grid is returned as fallback (score 0).
- ML samples vote for nearby primitive slots; votes become scores.
- Candidates are sorted by score first, then by primitive depth.

### 6) Search ordering and ML attachment

The search process ties ML to inference by **using ML scores to order which
primitives are tried first**.

- `ml_first`: ML-scored candidates are tried before fallback primitives.
- `cost_first`: lower chain cost is prioritized, ML only breaks ties.
- Two-phase behavior with `ml_first` + `ml_fallback`:
  1) Try ML-scored candidates only across all depths.
  2) If no ML success, try fallback primitives across all depths.

This ensures ML influences inference **without changing the execution
representation**. The planner still executes primitives; ML only selects the
ordering.

### 7) ML input/output (exact)

The inference model is called with a JSON scene description plus a selected
object. That JSON is converted into local, object-centered masks that match
training.

#### 7.1 Input JSON schema

```
{
  "xml_path": "...",
  "robot_goal": [x, y],
  "reachable_objects": ["obj_a", "obj_b", ...],
  "robot": {"position": [x, y, theta]},
  "objects": {
    "obj_name": {
      "position": [x, y, theta],
      "quaternion": [w, x, y, z]
    },
    ...
  }
}
```

The object list includes movable objects from observations and static objects
from environment metadata. The selected object must be present.

#### 7.2 Local masks (used by the current 2-push model)

The model is trained with **local, object-centered** masks. The mask generator
is the same code used for training, ensuring the exact same semantics. The
context channels are stacked in this order:

1. `local_static`: static obstacles
2. `local_movable`: other movable objects
3. `local_target_object`: selected object
4. `local_robot_region`: BFS-reachable region from robot on inflated obstacles
5. `local_goal_sample_region`: BFS-reachable region from a sampled goal

These are resized to `context_size` (64) and scaled to [-1, 1]. If `use_coord_grid`
was enabled during training, a 2-channel (x,y) grid is appended.

**Target during training:** a center-cropped goal mask (crop size 32 in this model)
created from the same local mask pipeline (`local_goal_mask_a1`, fallback to
`local_target_goal`). The model therefore learns to predict a **goal mask** in
local pixel coordinates.

#### 7.3 Local output decoding (exact algorithm)

After diffusion sampling:

1. Threshold the predicted goal mask and run connected components.
2. Reject samples with multiple disconnected regions.
3. Compute the predicted goal center and orientation using rectangle fitting.
4. Convert the predicted local pixel center to world coordinates using the
   crop’s world bounds.
5. Compute orientation by comparing goal-mask angle to object-mask angle, then
   normalize to [-pi, pi].

**Output per sample:**

```
{
  "index": i,
  "x": world_x,
  "y": world_y,
  "theta": goal_theta,
  "goal_sample": goal_mask,
  "input_channels": input_tensor
}
```

These outputs are **continuous SE(2) goals**; the planner aligns them to
primitive slots before execution.

## ML Architecture (the model used in experiments)

The 2-push experiments use the **cropped diffusion with cross-attention**
model. This is a Diffusion Transformer (DiT) variant with a separate context
encoder and cross-attention conditioning.

### Architecture summary

- **Backbone:** DiT with cross-attention (cropped output)
- **Context input:** 5 channels at 64x64 (local masks)
- **Target input:** 1 channel at 32x32 (center-cropped goal mask)
- **Patch size:** 4
- **Model dim:** 256
- **Depth:** 8 transformer blocks
- **Heads:** 8 attention heads

### Data path inside the network

1. **Context encoder (CNN):**
   - Downsamples 64x64 context masks to 8x8 spatial tokens.
2. **Target embedding:**
   - The noisy 32x32 target is patch-embedded into tokens.
3. **Transformer blocks:**
   - Self-attention on target tokens.
   - Cross-attention where target tokens attend to context tokens.
   - MLP with time-dependent modulation.
4. **Output head:**
   - Tokens are unpatched to reconstruct a 32x32 goal mask.

### Diffusion / sampling details

- **Forward path:** HuggingFace diffusion path, 1000 timesteps,
  squaredcos_cap_v2 schedule.
- **Prediction type:** epsilon (noise prediction).
- **Sampler:** DDIM by default, eta=0.0.
- **Inference overrides:** `ml_sampler_method` and `ml_num_steps` can change
  the sampler and number of steps at runtime.

## Experimental Setup

### 1) Model used in 2-push evaluation

The 2-push evaluation uses a single diffusion model trained under the run:

- `/common/users/dm1487/namo_data/outputs/cropped_diffusion_crossattn_2push/2025-12-16/05-36-44`

The Hydra config for this run specifies the architecture above (DiTCroppedCrossAttn,
context_size 64, crop_size 32, depth 8, etc.).

### 2) Evaluation conditions

Two learned modes are evaluated on the same model weights:

- **Diffusion (ML-only):** ML-aligned primitives only (`goal_strategy: ml`).
- **Hybrid (ML + fallback):** ML scores + full primitive fallback
  (`goal_strategy: ml_fallback`).

These produce different result directories but share identical weights. The
“Hybrid” vs “Diffusion” difference is purely in inference strategy.

### 3) Planner settings used in ML runs

Key parameters (from the ML collection config):

- `region_max_chain_depth = 2`
- `region_chain_link_cost = 11`
- `region_selection_strategy = ml_first`
- `ml_samples = 32`, `ml_seed = 42`
- `ml_sampler_method = ddim`, `ml_num_steps = 5`
- `ml_match_position_tolerance = 0.2`
- `ml_match_angle_tolerance = 0.2`
- `ml_k_nearest = 1`

These settings control search depth, chain cost, ML sampling, and alignment
behavior during inference.

### 4) Metrics and evaluation protocol

The evaluation script aggregates results across environments and reports:

- **Success rate** vs time cutoff (time-based success curve)
- **Push count distributions**
- **Failure reasons** (e.g., no reachable objects, ML goals not aligned)

The reference curve is generated from a non-learned search baseline; learned
models are compared to this reference.

## Summary

- The planner searches over discrete push primitives; ML only prioritizes them.
- The diffusion model predicts a local goal mask conditioned on local context.
- Goals are decoded to SE(2), aligned to primitives, and executed by the planner.
- The 2-push experiments use a cropped, cross-attention DiT diffusion model.

MCTS is not part of this pipeline.
