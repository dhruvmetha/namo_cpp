# Full NAMO + Region-Opening + ML Pipeline (Filename-Free Description)

This document provides a paper-ready, implementation-faithful description of the full pipeline:

- A **hierarchical Full NAMO** planner that reaches a final robot goal by repeatedly solving **region-opening** subproblems.
- A **region-opening** planner that searches over discrete push primitives, optionally guided by a learned model.
- A learned **diffusion image-to-image goal sampler** that proposes object goal poses, used only to prioritize which primitives to try.
- The **data collection** protocol and **evaluation** protocol used for the 2-push benchmark.

No internal codebase filenames are referenced; the description is purely methodological.

---

## 1) Problem and key abstractions

We study navigation among movable obstacles (NAMO) where:

- The robot must reach a specified **robot goal pose**.
- The workspace contains **static obstacles** (e.g., walls) and **movable objects** that can be pushed.
- Planning proceeds by simulating candidate push actions and validating whether they make progress toward reachability.

### Regions and region connectivity

A central abstraction is a decomposition of free space into **regions**:

- A region is a connected component of robot-navigable free space in an **inflated occupancy grid**, where obstacles are expanded by the robot footprint.
- Two regions are **adjacent** if removing (or moving) at least one blocking movable object would connect them.
- Each region (except the robot’s region) is associated with a small set of **region-goal samples** (positions inside that region) used to validate that the robot can reach the region.

The region graph provides a compact representation of large-scale connectivity while the planner uses low-level pushes to modify that connectivity.

---

## 2) Region connectivity snapshot (computed from the current state)

Before solving a region-opening subproblem, we compute a fast snapshot of the scene that returns:

- A region adjacency graph.
- For each adjacency edge, the set of **blocking movable objects** on that boundary.
- Region labels (robot region, goal region, other regions).
- For each non-robot region, a bundle of **region-goal samples**.

### 2.1 Grid and inflation

- The snapshot is computed on a 2D grid at **1 cm resolution**.
- Obstacles are inflated by the robot’s half-extents plus a small epsilon (**0.005 m**) to match wavefront-style collision margins.
- Free space connectivity uses **8-connected** adjacency.

### 2.2 Region labeling

- The connected component containing the robot is labeled as the robot region.
- If the global goal pose lies in a different connected component (under inflation), that component is labeled as a distinct goal region.
- Remaining enclosed connected components are labeled as additional regions.

### 2.3 Blocking-object attribution and adjacency

To identify which movable objects block which region boundaries:

1. Rasterize all static obstacles and movable objects into the inflated grid.
2. For each movable object:
   - Temporarily remove the object’s inflated footprint from the grid.
   - Flood-fill through the removed footprint to discover which regions become connected.
   - If two or more regions become connected, record adjacency between those regions and attribute that movable object as a blocker for that region edge.

This produces both an adjacency graph and a mapping from region edges to blocking objects.

### 2.4 Region-goal sampling (for validation)

For each region (excluding the robot region), we sample a small number of interior cells uniformly at random and convert them to world coordinates.

These region-goal samples serve as lightweight “proxies” for region reachability in validation tests.

---

## 3) Opening validation criterion (what counts as success)

A region-opening attempt targets a specific neighboring region.

We validate that an opening has been created by testing reachability of region-goal samples.

For a target region, let:

- `reachable_before` = number of sampled region-goal poses reachable before the push.
- `reachable_after`  = number of sampled region-goal poses reachable after the push.

A push is credited as creating a new opening if:

- `reachable_after >= 1`, and
- `reachable_after > reachable_before`.

This ensures we only declare success when the push **increases** region reachability and results in at least one reachable sample.

---

## 4) Region-opening subproblem (local planner)

The region-opening planner solves, from a fixed baseline state:

> For each immediate neighbor of the robot region in the region graph, attempt to create connectivity to that neighbor by pushing boundary-blocking objects.

### 4.1 Candidate neighbor enumeration

- Compute the region connectivity snapshot at the current state.
- Identify the robot region label.
- Enumerate immediate neighbor regions from the adjacency graph.

### 4.2 Candidate blocking objects

For a given neighbor region:

- Retrieve the set of blocking objects that appear on the robot–neighbor region edge.
- Filter to objects that are **currently reachable by the robot**.

If no reachable blocking objects exist, the neighbor attempt is recorded as a failure with a specific reason.

### 4.3 Fair per-object trials

For evaluation fairness and clean triplet accounting, each blocking object is tried from the **same baseline state**:

- Reset simulator state to baseline before trying object A.
- Run the search to find opening(s) using pushes of object A.
- Reset simulator state to baseline before trying object B.
- Repeat.

This isolates the difficulty of each object’s interaction with the neighbor region boundary.

### 4.4 Recorded outputs per attempt

For each (neighbor region, candidate object) trial, the system records:

- Whether a valid opening was created.
- The action sequence (single push or multi-push chain).
- Timing and counts (pushes executed, solutions found).
- Collision indicators (wall collision; number of distinct movable objects contacted).
- Optional ML grounding signals (aligned primitive votes and which edges were reachable).

These logs support both training data generation and evaluation.

---

## 5) Primitive action space (discrete push proposals)

Push actions are enumerated from a precomputed primitive library.

### 5.1 Primitive indexing

- **60 discrete “edge points”** around the object footprint (conceptually 4 sides × 15 points per side).
- **10 discrete push depths** per edge point (push lengths / durations).

Thus, each object has up to **600 discrete primitive slots**.

### 5.2 Shape-conditioned primitive library

Objects are mapped to one of three primitive libraries based on aspect ratio:

- square-like
- wide
- tall

Each primitive stores a relative transform `(Δx, Δy, Δθ)` in the object frame, plus its `(edge_idx, depth_idx)`.

### 5.3 Converting primitives to world-frame goals

At planning time, for an object at pose `(x, y, θ)`, each primitive’s relative displacement is rotated into the world frame and added to the object pose to obtain an absolute SE(2) goal `(x_goal, y_goal, θ_goal)`.

### 5.4 Reachability gating at the primitive level

Not all edge points are executable from the current robot configuration.

A simulator-provided predicate returns the set of **reachable edge indices** (subset of `{0..59}`) for a given object.

The planner only considers primitive slots whose `edge_idx` lies in the reachable set.

---

## 6) Region-opening search algorithm

Region opening uses a **two-level search**:

- An inner search that tries primitive slots for a single push (a “skill execution”).
- An outer search that chains multiple pushes to allow 2-push (or more) solutions.

### 6.1 Inner search (single-push primitive search)

For a fixed baseline state and chosen object:

1. Generate a grid of candidate goals: `goals[edge_idx][depth_idx]`.
2. Filter to reachable `edge_idx`.
3. Sort candidates by a priority key:
   - Primary: descending goal score (when ML assigns scores),
   - Secondary: shallower depth first,
   - Tertiary: edge index for determinism.
4. For each candidate:
   - Reset to baseline state.
   - Measure reachability of the target region (before).
   - Execute the push in simulation.
   - Measure reachability (after).
   - If it creates an opening, record success.

**Collision/stuck pruning.** If collision termination is enabled, a collision or stuck event blacklists deeper depths on that edge for the remainder of that inner search call.

**Depth pruning.** Once any candidate succeeds at some primitive depth, candidates strictly deeper than that depth may be skipped to reduce unnecessary long pushes.

### 6.2 Outer search (multi-push chaining)

To find 2-push solutions:

- The planner performs a BFS over chain depth.
- Each node stores a simulator state resulting from an earlier push and the goal that produced it.
- Expanding a node runs the inner search from that node’s state. Any “valid but not successful” post-push states are collected as frontier nodes for deeper chains.

### 6.3 Cost model

Each primitive has a depth index `d ∈ {1..10}`.

- Single-push cost is proportional to the depth used.
- Multi-push chains have cost:

`total_cost = sum(depth_costs) + chain_link_cost`  (only if number of pushes > 1)

This cost model penalizes long pushes and optionally penalizes multi-push chains with a flat link cost.

The search maintains the best cost found so far and prunes expansions that cannot beat it.

### 6.4 Beam width

Optionally, the number of frontier nodes carried forward at each chain depth can be capped (beam search) to reduce compute.

---

## 7) Goal strategies (primitive-only vs ML-guided)

All strategies ultimately execute the same discrete primitives; the difference is how candidates are ordered and/or filtered. The active set of strategies (`region_opening.py` dispatch):

- `primitive` — exhaustive primitive enumeration (with optional edge shuffling).
- `geometric` — primitive enumeration with geometric transport priority scoring.
- `ml` (alias `ml_primitive`) — ML-aligned primitives only.
- `ml_fallback` — ML-first scored slots + full primitive fallback.
- `ml_async` — same semantics as ml/ml_fallback but ML inference is dispatched on a background thread.
- `ml_driven_async` — event-driven async search that prioritizes ML results while keeping CPU busy with fallback (see `ML_DRIVEN_ASYNC_ALGORITHM.md`).

### 7.1 Primitive strategy

- Enumerate all primitive slots for reachable edges.
- Order is deterministic unless optional edge shuffling is enabled.

This is an exhaustive discrete baseline.

### 7.2 ML strategy (aligned primitives only)

- A learned model produces a set of continuous SE(2) goal proposals for the chosen object.
- Each continuous proposal is aligned to the nearest discrete primitive slots within tolerances.
- Only aligned primitive slots are returned to the planner; unaligned slots are set to “empty” and are never executed.

This yields a sparse candidate set focused on the model’s predictions.

### 7.3 ML-fallback strategy (ML-first with full primitive fallback)

- Start with the full primitive grid.
- Assign a positive score to slots supported by the ML model, and score 0 to all others.
- Search is run in two phases when ML-first behavior is enabled:

1. **ML-only phase:** execute only primitives with score > 0.
2. **Fallback phase:** if no solution is found, execute only primitives with score = 0.

The system logs which phase produced the solution and how many pushes were spent in each phase.

---

## 8) Learned diffusion goal sampler (img2img)

The learned component is a diffusion image-to-image model that predicts a **goal mask** for the selected object.

The planner does not directly execute model outputs. Instead, predicted goals are:

1. decoded into SE(2) proposals, and
2. aligned to the discrete primitive set.

### 8.1 Inputs: local object-centered mask context

Inference begins from a scene description (robot pose, robot goal, object poses), then constructs **local, object-centered masks**.

Local masks are generated by:

- Rendering static and movable obstacles into a high-resolution grid.
- Inflating obstacles by robot half-extents + epsilon.
- Computing connected components of free space (8-connected).
- Extracting binary masks for:
  1. static obstacles
  2. other movable objects
  3. selected target object
  4. robot-reachable free-space component
  5. goal-sample-reachable free-space component
- Cropping a fixed-size window around the selected object (e.g., 5 m × 5 m) and resizing to a standard image size.

Two important implementation details ensure robustness:

- If the robot (or goal sample) lies inside an inflated obstacle cell due to discretization, a 3×3 neighborhood around that cell is cleared before connected-component labeling.
- The crop is padded as needed to keep the selected object centered even near world boundaries.

### 8.2 Preprocessing

- Input masks are normalized to `[0,1]`, then scaled to `[-1,1]`.
- Masks are resized to a **context resolution** of 64×64.

### 8.3 Output: goal mask decoding to SE(2)

The model outputs a single-channel goal mask. Each diffusion sample is decoded as:

1. Threshold the mask.
2. Reject samples with multiple disconnected components.
3. Fit a minimum-area rectangle to obtain a goal center and rectangle orientation.
4. Convert the local pixel center to world coordinates using the local crop bounds.
5. Recover object orientation by comparing the predicted rectangle orientation to the current object mask orientation.
   - Resolve 180° symmetry by folding angle differences into a canonical range.
   - Add the angle difference to the current object yaw and wrap to `[-π, π]`.

Each sample yields a continuous SE(2) goal `(x, y, θ)`.

### 8.4 Aligning continuous goals to discrete primitives

Each predicted goal is aligned to discrete primitive slots by nearest-neighbor matching under tolerances:

- Position tolerance (meters)
- Angle tolerance (radians)

For each predicted goal, the alignment procedure:

- finds all primitive slots within the tolerances,
- ranks them by a combined error (position + weighted angle),
- votes for the top-`k` nearest slots.

The number of votes becomes the discrete slot score used for planning priority.

### 8.5 Inference request schema (concrete)

The inference model is invoked with a JSON scene description plus a selected object. The selected object must be present in `objects`.

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

The JSON is converted into the local masks of §8.1 using the same code used for training. Context channels are stacked in this order (must match training): `local_static`, `local_movable`, `local_target_object`, `local_robot_region`, `local_goal_sample_region`. If `use_coord_grid` was enabled during training, a 2-channel (x, y) grid is appended.

### 8.6 Per-sample model output

After diffusion sampling, each accepted sample is decoded to:

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

These continuous SE(2) goals are then aligned to primitive slots (§8.4) before any of them is executed.

---

## 9) Diffusion model architecture and sampling details

The model is a cropped-output diffusion transformer with cross-attention context conditioning.

### 9.1 Architecture

- **Context input:** 5 channels at 64×64.
- **Target prediction:** 1 channel goal mask.
- **Cropped output:** model predicts a 32×32 crop, later embedded into the 64×64 coordinate frame for compatibility.

Key components:

1. **Spatial context encoder (CNN):**
   - Downsamples 64×64 context to an 8×8 grid of context tokens (64 tokens).

2. **Patch embedding for the noisy target crop:**
   - Patch size 4.
   - 32×32 crop becomes 8×8 = 64 target tokens.

3. **Transformer stack (8 blocks):**
   - Self-attention on target tokens.
   - Cross-attention from target tokens to context tokens.
   - MLP with time-dependent modulation.

4. **Unpatchify head:**
   - Converts tokens back into a 32×32 goal mask prediction.

Representative model size settings:

- model dimension: 256
- transformer depth: 8
- attention heads: 8

### 9.2 Diffusion configuration

- Training diffusion steps: 1000 timesteps.
- Noise schedule: cosine-style (`squaredcos` variant).
- Prediction target: noise (epsilon prediction).

### 9.3 Sampling

- Default sampler: DDIM with `η=0` (deterministic).
- At inference time, the number of sampling steps can be reduced for speed.

Representative inference settings in the 2-push experiments:

- samples per query: 32
- sampler: DDIM
- inference steps: 5
- fixed random seed for reproducibility

---

## 10) Full NAMO planner (hierarchical)

The Full NAMO planner uses region opening as a sub-solver to reach the final robot goal.

### 10.1 Iterative loop

Repeatedly:

1. If the robot goal is currently reachable, terminate successfully.
2. Compute a **full** region connectivity snapshot (entire region graph, not restricted to local neighbors).
3. Determine the goal region label by looking up which region contains the goal position.
4. Determine the current robot region label.
5. Find the shortest path in the region adjacency graph from robot region to goal region.
6. Select the next region along that path.
7. Invoke the region-opening sub-solver targeted specifically at that next region.
8. Apply the resulting post-push simulator state.

Stop if:

- the goal becomes reachable, or
- a maximum number of region-opening iterations is exceeded.

This realizes a hierarchical planner where region opening provides a local “connectivity-improving” operator and the region graph provides the global plan skeleton.

---

## 11) Data collection protocol

### 11.1 Units of data

Data is collected over a large set of environments.

For each environment, the pipeline records results as a collection of “episodes” corresponding to region-opening attempts. Each attempt includes:

- the neighbor region being targeted,
- the specific blocking object being tested,
- the push sequence (possibly multi-push),
- success/failure status and failure reason,
- intermediate observations.

### 11.2 Goal consistency for mask generation

For region-opening episodes, the recorded robot goal used for mask generation is the **first reachable region-goal sample** that validated the opening (when available). This aligns the learning signal with the planner’s validation criterion.

### 11.3 Filtering

Episodes that correspond to trivial successes with zero actions (e.g., target region already accessible) are excluded from training datasets because they contain no push interaction.

---

## 12) Evaluation protocol (2-push benchmark)

Evaluation is performed over triplets:

- environment instance
- neighbor region label
- object identifier

### 12.1 Success definition

A run is counted as successful if:

- a solution is reported, and
- at least one push was executed.

### 12.2 Oracle categorization

A strong primitive-search reference is used to categorize each triplet into:

- blocked (no reachable object interactions)
- solvable with one push
- solvable with two pushes
- unsolved / exceeded budget

These categories are determined by the reference solution’s chain depth.

### 12.3 Matched-triplet filtering

For fair comparisons across methods, evaluation is restricted to the intersection of triplets that:

- appear in all compared methods’ logs, and
- are successful in the reference set (when evaluating solvable-only statistics).

### 12.4 Reported metrics

The evaluation suite reports:

1. **Success vs time cutoff:** success rate as a function of wall-clock budget.
2. **Success vs push budget:** success rate as a function of number of simulated push evaluations.
3. **Success@B:** success rate at fixed push budgets (e.g., 50 / 100 / 200).
4. **Success@T:** success rate at fixed time budgets (e.g., 5s / 10s / 30s).
5. **ReachableAttachment@K (RA@K):**
   - Rank ML-aligned primitive slots by vote score.
   - Measure the fraction of top-K ranked slots that lie on reachable edges.
   - Report both macro-average (mean per instance) and micro-average (global ratio).
6. **Hybrid decomposition:** for ML-fallback strategies, report the fraction of problems solved in the ML-only phase vs fallback phase, and the compute spent before fallback.
7. **Collision stratification:** success/efficiency as a function of collision category.
8. **Chain-depth confusion:** compare oracle-required depth vs achieved depth.

### 12.5 Difficulty stratification with multiple oracles

To estimate problem difficulty robustly, multiple independent oracle runs are performed with different randomized primitive orderings.

For each triplet, compute the median number of pushes across oracle runs.

Define difficulty bins by the 33rd and 67th percentiles of median pushes:

- easy: ≤ p33
- medium: p33–p66
- hard: > p66

These bins are used to stratify success and efficiency metrics.

---

## 13) Representative experimental settings

The following settings are representative of the runs described in this pipeline.

### 13.1 Region-opening baseline (primitive)

- collisions allowed during pushing
- max chain depth: 2
- region selection priority: cost-first
- chain link cost: 11
- region-goal samples per region: 10

### 13.2 Region-opening learned (ML-fallback)

- collisions allowed during pushing
- max chain depth: 2
- region selection priority: ML-first (two-phase ML-only then fallback)
- chain link cost: 11

ML inference:

- diffusion samples per query: 32
- DDIM sampling steps: 5
- fixed random seed

Alignment:

- position tolerance: 0.2 m
- angle tolerance: 0.2 rad
- k-nearest voting: 5
- angle weight: 1.0

### 13.3 Full NAMO

- maximum region-opening iterations: 20
- each iteration opens the next region along the current shortest region-graph path

---

## 14) Notes for paper writing (what to emphasize)

- The learned model does **not** output actions; it outputs a spatial prior over promising object goal placements.
- The planner remains a **discrete, verification-based search** over physically grounded push primitives.
- The region abstraction provides a bridge between local pushing and global navigation.
- The hybrid strategy is naturally decomposable into “learned-only compute” and “fallback compute,” enabling clear experimental ablations.
