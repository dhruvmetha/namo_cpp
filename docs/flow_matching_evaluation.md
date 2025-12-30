# Flow-Matching Evaluation in NAMO (Detailed Reference)

This document describes the end-to-end evaluation pipeline for flow‑matching (vector) goal models in the NAMO stack, with emphasis on how `namo_cpp` components are used during evaluation. It covers data inputs, mask generation, goal inference, primitive mapping, simulation execution, collision accounting, and metrics.

---

## 1) High-level evaluation flow

Evaluation is driven by `sage_learning/scripts/evaluate_flow_matching.py`, which wraps the NAMO environment and goal strategies to run a standardized loop. The flow is:

1. **Select test samples** from the dataset (HDF5) using the same random split logic as training.
2. **Initialize a MuJoCo NAMO environment** from each XML (via `namo_cpp` bindings).
3. **Generate model inputs** (masks) either by:
   - Using **precomputed H5 masks** (fast path), or
   - **Regenerating masks** via the same visualizer pipeline used for training.
4. **Run flow‑matching inference** to produce a set of candidate object‑relative SE(2) goal deltas.
5. **Map each goal to a motion primitive slot** using the same alignment logic used by the planner.
6. **Vote and rank** primitives by frequency (vote‑only ranking; no fallback to “all primitives”).
7. **Execute ranked primitives** in simulation, re‑starting from the same baseline state for each trial.
8. **Record metrics**, collisions, and difficulty breakdown, then aggregate summary statistics.

All modeling‑specific logic (inference, mask creation) stays consistent with training through `namo_cpp`’s visualization adapters and `GoalVectorInferenceModel`.

---

## 2) Key files and roles

### Primary evaluation scripts
- `sage_learning/scripts/evaluate_flow_matching.py`
  - Main evaluator and metrics aggregator.
  - Uses NAMO environment + NAMO goal strategies for mapping.
  - Supports both **H5 masks** and **regenerated masks**.

- `sage_learning/scripts/visualize_flow_matching_eval.py`
  - Visualizes a small number of examples in MuJoCo.
  - Reuses the same evaluator logic for inference + mapping.

- `sage_learning/job_scripts/evaluate_flow_matching.sh`
  - Example Slurm job for evaluation (partial test set).

- `sage_learning/job_scripts/evaluate_flow_matching_full.sh`
  - Full test sweep using H5 masks by default.

### NAMO components used during evaluation
- `namo_cpp/python/namo/strategies/ml_strategies.py`
  - Builds the JSON “planning scene” that feeds into `GoalVectorInferenceModel`.
  - Injects `region_goals_sampled` for vector models to recreate the goal‑sampling mask.

- `namo_cpp/python/namo/strategies/primitive_goal_strategy.py`
  - Holds the primitive library and alignment utilities.
  - Defines how ML goals are snapped to discrete primitive slots.

- `namo_cpp/python/namo/visualization/ml_image_converter_adapter.py`
  - Converts JSON planning scene to the exact mask format used in training.
  - Calls `NAMODataVisualizer.generate_all_masks_highres`.

- `namo_cpp/python/namo/visualization/mask_generation/visualizer.py`
  - Training‑grade mask generator. Used for both H5 masks and regenerated masks.
  - Consumes `state_observations`, `static_object_info`, and `region_goals_sampled`.

- `namo_cpp/src/environment/namo_environment.cpp` and `namo_cpp/python/namo/cpp_bindings/rl_env.cpp`
  - Provide static object info: `pos_x`, `pos_y`, `pos_z`, `quat_w/x/y/z`, `size_x/y/z`.
  - These fields are used to build static masks consistent with training.

---

## 3) Dataset split and test selection

**Source dataset:**
- Example path: `/common/users/shared/robot_learning/dm1487/namo/datasets/images/dec2/aug9_envs/1_push_train/h5/training_data.h5`

**Split logic:**
- Uses a **deterministic shuffle** with seed `42`.
- `train_split=0.9` implies:
  - First 90% are “train”,
  - Last 10% are treated as “test/val” for evaluation.

**In `evaluate_flow_matching.py`:**
- `get_test_indices(...)` loads `len(h5f['local_static'])`, shuffles indices, and selects the tail slice.

---

## 4) Environment initialization (simulation)

**Environment creation:**
- `env = namo_rl.RLEnvironment(xml_path, config_file, visualize=False)`
- `env.reset()`
- `env.set_robot_goal(...)` sets the navigation goal.

**Config file:**
- Default: `namo_cpp/config/namo_config_complete_skill15.yaml`.

**Collision checking behavior:**
- Evaluation may run with `allow_collisions=True`. This disables hard collision failure but tracks collision statistics.

**Baseline state:**
- The evaluator captures a baseline `state = env.get_full_state()`.
- Every primitive trial resets to this baseline before execution so each trial is independent.

---

## 5) Mask generation and model inputs

Flow‑matching models operate on **local masks** (object‑centric) derived from the planning scene. Evaluation supports two modes:

### 5.1 Fast path: H5 masks

If `--use-h5-masks` is enabled:
- Masks are read directly from the H5 dataset (`local_static`, `local_movable`, `local_target_object`, etc.).
- This guarantees **exact match** to training inputs.
- Inference is faster since mask rendering is skipped.

### 5.2 Regenerated masks (training‑equivalent)

If `--use-h5-masks` is disabled:
- The evaluator regenerates masks using **the same pipeline as training**:
  1. Build JSON planning message with object poses and static objects.
  2. `GoalVectorInferenceModel` uses `MLImageConverterAdapter`.
  3. Adapter converts the JSON to an `episode_data` structure.
  4. `NAMODataVisualizer.generate_all_masks_highres()` renders the mask stack.

**Key matching details:**
- **Static objects** are represented using `pos_x`, `pos_y`, `pos_z`, and quaternion fields from `env.get_object_info()`.
- **Movable objects** are included from `env.get_observation()`.
- **Region goal samples** are included (`region_goals_sampled`) so the `goal_sample_region` mask matches training.

This regenerated path is now aligned to the training data format, so masks are consistent with data collection.

---

## 6) Flow‑matching inference (vector model)

**Model wrapper:**
- `ktamp_learning.goal_vector_inference_model.GoalVectorInferenceModel`
- Loaded with:
  - `model_path`, `device`, `sampler_method`, `num_steps`

**Inference behavior:**
- Produces `N` samples (`--num-samples`, default 32), each containing `{x, y, theta}` in object‑relative space.
- For vector models, the evaluator includes `region_goals_sampled` to match training behavior.

**Sampling control:**
- `num_steps` controls the ODE solver steps used by the flow‑matching sampler.
- `sampler_method` can override (e.g., `euler`, `midpoint`, `rk4`, `dopri5`).

---

## 7) Primitive mapping and voting

Evaluation uses the exact primitive mapping logic used by the region‑opening planner.

### 7.1 Primitive library

- Motion primitives are loaded from `namo_cpp/data/motion_primitives_15_*.dat`.
- Shapes: `square`, `wide`, `tall`.

### 7.2 Mapping ML goals to primitives

- `PrimitiveGoalStrategy.generate_goals()` provides all candidate primitive target slots.
- For each ML goal:
  - Compute positional error and angular error vs each primitive slot.
  - Score = `pos_err + angle_weight * ang_err`.
  - If within tolerance (`match_position_tolerance`, `match_angle_tolerance`), map to best slot.

### 7.3 Voting

- Every ML goal casts a vote for the nearest matching primitive.
- Primitives are ranked by vote count.
- **Important:** evaluation uses **vote‑only ranking** (no fallback to try all primitives). If no matches exist, the episode fails.

This aligns with the “ML aligned primitive” sampling in the planner.

---

## 8) Primitive execution in simulation

For each ranked primitive:

1. Restore baseline state (`env.set_full_state(baseline_state)`).
2. Execute the primitive:
   - If `env.execute_push_primitive` exists, call directly for speed.
   - Otherwise, issue an `namo_rl.Action` with the primitive goal.
3. Check success:
   - `region_opened = env.is_robot_goal_reachable()`

**Push execution semantics:**
- Each primitive is tried **independently** from the baseline state.
- The first primitive that makes the robot goal reachable counts as success.

---

## 9) Collision accounting

Even when collisions are allowed, evaluation tracks collision statistics. The evaluator uses:

- `env.get_object_info()` to build OBBs for static objects.
- `env.get_observation()` for movable object pose changes.

Collision counters include:
- `collision_total`
- `collision_static_only`
- `collision_movable_only`
- `collision_both`

These are aggregated per environment and summarized across the test set (avg/median).

---

## 10) Metrics recorded

### Per‑environment metrics
- `success` (bool)
- `pushes_to_success` (count until success)
- `time_to_success_ms` (cumulative sim time only up to success)
- `total_pushes_attempted`
- `total_time_ms`
- `primitive_votes` (vote counts)
- `chosen_primitive_idx`
- `difficulty_label`, `difficulty_score` (from H5, if present)
- `collision_*` counts
- timing breakdown: init / inference / mapping / simulation

### Aggregate metrics
- `success_rate`
- `avg` + `median` pushes‑to‑success
- `avg` + `median` time‑to‑success
- collision stats (avg/median for total/static/movable/both)
- timing breakdown averages
- difficulty breakdown (easy/medium/hard/unknown)

### Difficulty breakdown
The evaluation groups results by `difficulty_label` from the dataset. Each difficulty tier reports:
- success rate
- average/median pushes to success
- average/median time to success
- average/median collision counts

---

## 11) Evaluation modes (H5 vs regenerated)

| Mode | How masks are built | Pros | Cons |
|---|---|---|---|
| `--use-h5-masks` | Loaded directly from H5 datasets | Fast, exact match to training | Doesn’t test mask regeneration |
| Regenerated masks | Built from planning scene + static/movable info | Verifies end‑to‑end pipeline | Slower, requires correct static info and region goal sampling |

The regeneration path now matches the training mask pipeline, including static object quaternion usage and region goal sampling.

---

## 12) Visualization workflow

`visualize_flow_matching_eval.py` renders a small number of H5 samples in MuJoCo:
- Loads the same masks (H5 or regenerated).
- Executes primitives in ranked order.
- Prints success/failure and collision flags per primitive.

A convenience wrapper is provided:
- `sage_learning/scripts/visualize_flow_matching_sim.sh`

---

## 13) Practical command examples

### Small eval sweep (regenerated masks)
```
python sage_learning/scripts/evaluate_flow_matching.py \
  --model-path /common/users/tdn39/Robotics/Mujoco/sage_learning/outputs/2025-12-28/max_abs \
  --max-test-envs 5 \
  --max-pushes 5 \
  --allow-collisions \
  --device cuda
```

### Fast eval sweep (H5 masks)
```
python sage_learning/scripts/evaluate_flow_matching.py \
  --model-path /common/users/tdn39/Robotics/Mujoco/sage_learning/outputs/2025-12-28/max_abs \
  --use-h5-masks \
  --allow-collisions \
  --device cuda
```

### Visualization
```
./sage_learning/scripts/visualize_flow_matching_sim.sh \
  /common/users/tdn39/Robotics/Mujoco/sage_learning/outputs/2025-12-28/max_abs \
  --num-examples 3
```

### Full test set (Slurm)
```
sbatch sage_learning/job_scripts/evaluate_flow_matching_full.sh \
  /common/users/tdn39/Robotics/Mujoco/sage_learning/outputs/2025-12-28/max_abs
```

---

## 14) Important invariants for correctness

1. **Mask generation must match training**
   - Static objects require `pos_x`, `pos_y`, `pos_z` and quaternion fields.
   - Region goal sampling must be included for vector models.

2. **Primitive mapping must occur before voting**
   - Votes are computed only after mapping ML goals to primitives.

3. **Vote‑only ranking**
   - No fallback to all primitives; the candidate set is limited to the mapped subset.

4. **Time‑to‑success metrics**
   - Cumulative simulation time only for successful environments.

5. **Independent primitive trials**
   - Each primitive is tested from a baseline state to isolate success per primitive.

---

## 15) Troubleshooting checklist

If evaluation results look wrong (e.g., low success rate or too many pushes):

- Confirm `--use-h5-masks` to eliminate mask‑regen mismatch.
- Ensure `goals_per_region` matches data collection (default 5).
- Verify `allow_collisions` matches the training data policy.
- Check `num_steps` and `sampler_method` align with training.
- Inspect `ml_match_*` tolerances (too strict → no matches).

---

## 16) Glossary

- **Flow‑matching model:** A vector‑based generative model predicting object‑relative SE(2) deltas.
- **Primitive slot:** A discrete `(edge_idx, depth_idx)` slot in the motion primitive library.
- **Region opening:** Task where a single push makes the robot goal region reachable.
- **region_goals_sampled:** Goal samples per region used to generate `goal_sample_region` masks.

---

## 17) Summary

The NAMO flow‑matching evaluation pipeline uses the same visual and planning components as training: masks are generated by the same visualizer, goal samples are aligned to primitives using the same planner logic, and success is judged by the same reachability checks. This ensures evaluation is faithful to the training semantics while providing clear metrics, collision accounting, and difficulty‑stratified analysis.
