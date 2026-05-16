# ML‑Goal Model vs Ground‑Truth F: Evaluation Plan (Point Robot, 1‑Push & 2‑Push)

Status: **planning + reference only**. Nothing in this doc has been executed.
Author: claude session 2026‑05‑16. All commands below should be re‑verified once
before you launch long runs.

> **Goal.** Measure how well the cropped diffusion model's per‑object goal
> predictions overlap with the ground‑truth feasible set **F** of region‑opening
> primitives. For the 1‑push horizon F = F₁; for the 2‑push horizon we score
> against F₁′ = {push₁ : F₂(state_after(push₁)) ≠ ∅}. Robot is the original
> 30 cm holonomic **point** robot (not the car). Metric is **Top‑K hit‑rate**
> against GT F — no end‑to‑end planner runs in this evaluation.

---

## 0. TL;DR — what you'll actually run

Two things have to happen in order:

1. **Collect ground‑truth F** (point robot) on the 2‑push held‑out split:
   - F₁ (1‑push exhaustive): variant of `region_opening_exhaustive.yaml` pointed
     at `manifest_2push_test_minus_1push_test_filtered.txt` (1038 envs) or the
     stratified `..._difficulty_100each.txt` (303 envs) subset.
   - F₂ + F₁′ (2‑push exhaustive): same config with
     `region_max_chain_depth: 2`. **This requires a small code change** to the
     trial log (see §5.4) so push‑1 vs push‑2 are distinguishable.

2. **Evaluate ML predictions** against the GT trial logs:
   - For each (env, neighbour region, object) instance, run the diffusion model
     once with `ml_samples=32` (matches inference defaults).
   - Decode each sample to SE(2) and align it to nearby primitive slots using
     the existing alignment in `ml_strategies.py` (position 0.2 m,
     angle 0.2 rad, k_nearest=1 by default).
   - Compute Top‑K hit‑rate: of the unique aligned primitive slots, how many
     fall inside GT F₁ (or F₁′ for 2‑push)? Report Top‑1, Top‑5, Top‑K=|aligned|.

No new planner; offline scoring only.

---

## 1. What the ML model is and how it is loaded

**Model run dir:**

```
/common/users/dm1487/namo_data/outputs/cropped_diffusion_crossattn_2push/2025-12-16/05-36-44/
```

**Hydra config** (`.hydra/config.yaml`) — key fields:

| Field | Value |
|---|---|
| `data.data_dir` | `/common/users/dm1487/namo_data/h5_files/dec2/aug9_envs/2_push_train_corrected_overlaps_2` |
| `data.train_split` | 0.9 |
| `data._target_` | `src.data.mask_diffusion_data_cropped.MaskDiffusionCroppedDataModule` |
| `model._target_` | `src.model.generative_module_cropped.GenerativeModuleCropped` |
| `network._target_` | `src.model.dit.dit_cropped_crossattn.DiTCroppedCrossAttn` |
| context_size / crop_size | 64 / 32 |
| context_channels | 5 (local_static, local_movable, local_target_object, local_robot_region, local_goal_sample_region) |
| sampler | HF DDIM, ε‑prediction, 1000 train steps |

**Checkpoint auto‑selection** — in
[`sage_learning/goal_inference_model.py:120-140`](../../sage_learning/sage_learning/goal_inference_model.py):

```python
for checkpoint_file in checkpoint_files:
    if "epoch" in checkpoint_file.name:
        checkpoint_path = checkpoint_file
        break
# fallback: last.ckpt
```

So passing the run dir (no `--ckpt-path`) makes it auto‑pick the first
`epoch*` file in `checkpoints/` (in this case `epoch223‑val_loss0.0032.ckpt`).
`last.ckpt` is only used as fallback.

**Local mask schema** (5 channels @ 64×64, scaled to [−1, 1]):

```
ch0  local_static               (walls / fixed obstacles)
ch1  local_movable              (other movable objects)
ch2  local_target_object        (selected object)
ch3  local_robot_region         (BFS reachable from robot)
ch4  local_goal_sample_region   (BFS reachable from sampled goal site)
```

Target is a 32×32 center‑cropped goal mask. See
[`docs/region_opening_pipeline.md`](region_opening_pipeline.md) §7.2.

**Important:** the model was trained on **point‑robot** data (the 2‑push h5
under `dec2/aug9_envs/`). The car branch's primitive‑prefix wiring (just
committed in `90a8242`) does *not* affect this evaluation — we'll use
`config/namo_config_complete_skill15.yaml` and the legacy
`motion_primitives_15_*.dat` (no `primitive_prefix`).

---

## 2. Region‑opening + ML wiring (where the model plugs in)

End‑to‑end, the planner only uses ML to **order** primitive candidates — it
never executes a continuous ML pose. Concretely:

1. `modular_parallel_collection.py` parses `--ml-goal-model`, `--ml-samples`,
   `--ml-match-position-tolerance` (default 0.2 m), `--ml-match-angle-tolerance`
   (default 0.35 rad), etc., and packs them into `algorithm_params`
   ([`modular_parallel_collection.py:1246-1283`](../python/namo/data_collection/modular_parallel_collection.py#L1246)).
2. `RegionOpeningPlanner` reads those params and constructs goal strategies
   ([`region_opening.py:455-580`](../python/namo/planners/opening/region_opening.py#L455)):
   - `goal_strategy="ml"` → ML‑aligned primitives only (model misses → 0 candidates for that object).
   - `goal_strategy="ml_fallback"` → ML scores + full primitive fallback (score 0).
   - `goal_strategy="ml_driven_async"` → async ML w/ primitive fallback.
3. Two‑phase semantics when `region_selection_strategy: ml_first`
   ([`region_opening.py:1656-1665`](../python/namo/planners/opening/region_opening.py#L1656)):
   - Phase 1 "ML‑only" tries only score>0 candidates across all depths.
   - Phase 2 "primitives" runs the fallback grid if phase 1 didn't solve.
4. ML inference call site is `MLPrimitiveGoalStrategy._load_model` →
   `GoalInferenceModel.infer(...)` in
   [`ml_strategies.py:441-499`](../python/namo/strategies/ml_strategies.py#L441).
5. Knobs only exposed via YAML (no CLI flag in
   `modular_parallel_collection.py`): `ml_seed`, `ml_sampler_method`,
   `ml_num_steps`, `ml_k_nearest`. These are read directly from
   `algorithm_params` inside `region_opening.py` (see §1 of grep output below).

```
region_opening.py:484  k_nearest=algo_params.get("ml_k_nearest", 1)
region_opening.py:485  seed=algo_params.get("ml_seed")
# also at 508, 529, 558 for the other strategy variants
```

So **for our offline eval we will skip the planner entirely** and call
`GoalInferenceModel.infer(...)` directly, then run the same alignment logic
that `MLPrimitiveGoalStrategy` uses (see §5.3).

---

## 3. The held‑out split for the 2‑push model

Training h5: `2_push_train_corrected_overlaps_2.h5`. Held‑out test pool sits at:

```
/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/
  manifest_2push.txt                                              (master)
  manifest_2push_train.txt                                   8 134 envs (model train)
  manifest_2push_test_filtered.txt                              54 envs (strict 2-push test)
  manifest_2push_test_minus_1push_test_filtered.txt          1 038 envs (broad test, recommended)
  manifest_2push_test_minus_1push_test_filtered_easy_100.txt   100 envs
  manifest_2push_test_minus_1push_test_filtered_medium_100.txt 100 envs
  manifest_2push_test_minus_1push_test_filtered_hard_100.txt   100 envs
  manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt   303 envs
                                                              (concatenation w/ headers)
  manifest_2push_test_minus_1push_test_filtered_triplets_*    (alternate triplet form)
```

Each line is `<absolute_xml_path>\t<region_label_or_label:object>`. The base XMLs
live under
`/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/medium/`.

**Recommendation for first run:** use
`manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt` (303 envs,
already stratified easy / medium / hard). Cheap to compute, gives difficulty
buckets out of the box, lines up with the existing eval directories under the
model run (`results_32samples_*` were generated against the same split).

---

## 4. Existing F‑characterization data (what is and isn't on disk)

Directory: `/common/users/dm1487/namo_data/f_characterization/`.

| Path | Manifest | Chain depth | Envs | Status |
|---|---|---|---|---|
| `1_push_exhaustive_full/modular_data_rlab7/` | `manifest_test.txt` (1 767 envs) | 1 | 1 767 envs × ≤ ~50 instances | ✅ complete |
| `1_push_exhaustive_train/modular_data_rlab{5,6,7}/` | `manifest_train.txt` (15 900 envs) | 1 | 5 925 pkls total | ✅ partial training‑set sweep |
| `2_push_exhaustive_*` | — | 2 | — | **❌ does not exist** |

**Per‑instance pkl layout** (`*_results.pkl`):

```
top-level dict
  task_id              str
  success              bool
  episodes_collected   int
  processing_time      float
  episode_results      list[dict]                # one entry per (neighbour, object) instance
```

Each `episode_results[i]` is a dict with the keys you'll see if you `vars()`‑print
it; the ones that matter for F characterization:

```
xml_file              str (abs path)
robot_goal            (x, y, theta)
algorithm_stats       dict
  chosen_object_id        str
  neighbour_region_label  str
  primitive_trial_log     list[dict]   # ONLY present when exhaustive_mode=True
state_observations          [pre]      list of dict (per-object SE(2) poses)
post_action_state_observations  [post] same shape
action_sequence       list[{object_id, target=(x,y,theta)}]
```

A `primitive_trial_log` entry (see `region_opening.py:2419`):

```python
{
  'edge_idx': int,              # 0..(4*points_per_face - 1) → 0..59 at points_per_face=15
  'depth':    int,              # 0..9 (push length)
  'success':  bool,             # reachable_after > reachable_before AND >= 1
  'wall_collision':       bool,
  'movable_collisions':   str,  # space-separated object names
  'stuck':    bool,
  'collision':bool,
  'reachable_after':      int,
}
```

[`analyze_F.py:60-65`](f_characterization/analyze_F.py#L60) shows how to fold a
trial log into the `(60, 10)` F grid (NaN = not reachable, 0 = reachable‑failed,
1 = feasible).

### 4.1 What we lack for the 2‑push case

Two hard gaps:

1. **No 2‑push exhaustive collection has been done.** Running
   `region_opening_exhaustive.yaml` with `region_max_chain_depth: 2` gives us
   the chained outcomes, but…
2. **The current `primitive_trial_log` entries do not record `chain_depth`.**
   Every BFS call appends to a flat `object_trial_log` (see
   `region_opening.py:1834` `all_trial_logs.extend(bfs_trial_log)`), so once
   chain_depth ≥ 2 you cannot tell which `(edge_idx, depth)` entries were
   push‑1 vs push‑2. To compute F₁′ we need the planner to tag each trial with
   the chain depth and (ideally) a back‑pointer to its parent push.

That is the only known structural change required before we can run 2‑push F.
Suggested minimal patch — at `_search_bfs(..., current_chain_depth, parent_node, ...)`:

```python
# region_opening.py: ~2419
trial_log.append({
    'edge_idx': edge_idx,
    'depth': depth,
    'success': is_accessible_after and not is_accessible_before,
    'wall_collision': step_result.info.get("wall_collision", "false") == "true",
    'movable_collisions': step_result.info.get("movable_collisions", ""),
    'stuck': stuck_detected,
    'collision': collision_detected,
    'reachable_after': reachable_count_after,
    'chain_depth': current_chain_depth,                           # NEW
    'parent_edge_idx': parent_node.edge_idx if parent_node else None,  # NEW
    'parent_depth':    parent_node.step_cost - 1 if parent_node else None,  # NEW
})
```

With those two parent‑pointer fields, F₁′ is computable as:

```
push1_succeeded_via_push2 = { (e1, d1) :
    exists trial with chain_depth=2,
                     parent_edge_idx=e1, parent_depth=d1,
                     success=True }
```

(Equivalently we can rebuild it from successful chains' `action_sequence` —
that does not need a code change — but only successful chains exist there;
the trial log is the only place that has all the *attempted* push‑1s with
their downstream push‑2 outcomes.)

---

## 5. The evaluation, step by step

### 5.1 Pick the env split

Use:

```
MANIFEST=/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt
```

303 envs total (easy/medium/hard 100 each). Skip the `# EASY/MEDIUM/HARD`
header lines when reading.

### 5.2 Collect GT F (point robot)

Copy `region_opening_exhaustive.yaml` to two new files. **Do not edit the
existing yaml in place** (it's referenced by `run_f_char_full.sh`).

`python/namo/data_collection/region_opening_exhaustive_2push_test.yaml`
(1‑push F₁ on the held‑out 2‑push split):

```yaml
output_dir: /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_2push_test_difficulty_100each
algorithm: region_opening
workers: 48
episodes_per_env: 1

max_depth: 5
max_goals_per_object: 5
max_terminal_checks: 5000
search_timeout: 600.0
goals_per_region: 10

points_per_face: 15
region_allow_collisions: true
region_max_chain_depth: 1
region_max_solutions_per_neighbor: 9999
region_max_recorded_solutions_per_neighbor: 9999
region_frontier_beam_width: 10000
region_chain_link_cost: 11
region_ml_ignore_blacklist: false
region_selection_strategy: cost_first
region_exhaustive_mode: true

goal_strategy: primitive
# primitive_prefix: ""   (default — point-robot, 30 cm)

xml_dir: /common/users/shared/robot_learning/dm1487/namo/mj_env_configs/aug9/medium
config_file: config/namo_config_complete_skill15.yaml
manifest: /common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt

verbose: false
filter_minimum_length: false
smooth_solutions: false
refine_actions: false
validate_refinement: false
```

And `region_opening_exhaustive_2push_test_chain2.yaml` — same as above but
`region_max_chain_depth: 2` and a different `output_dir` suffix
(`..._chain2`). **Block on the trial‑log patch in §4.1 before launching this
one**, otherwise F₁′ is unrecoverable.

Run:

```bash
export MJ_PATH=/common/users/dm1487/ktamp/mujoco
export PYTHONPATH=/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_rlab7:$PYTHONPATH
PYTHON=/common/users/dm1487/envs/mjxrl/bin/python

# 1-push F (cheap — single push per primitive)
$PYTHON python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_exhaustive_2push_test.yaml \
  --start-idx 0 --end-idx 303 --workers 48

# 2-push F (much heavier — quadratic primitive grid expansion)
$PYTHON python/namo/data_collection/modular_parallel_collection.py \
  --config-yaml python/namo/data_collection/region_opening_exhaustive_2push_test_chain2.yaml \
  --start-idx 0 --end-idx 303 --workers 48
```

Expected runtime for 1‑push F at 1 767 envs / 100 workers ≈ tens of minutes
(based on existing rlab7 runtimes). For 2‑push on 303 envs / 48 workers,
order‑of‑magnitude estimate is hours, possibly overnight — please re‑benchmark
on 5 envs first.

### 5.3 Run ML inference on the same instances

There is no off‑the‑shelf "score the model offline" script. We need a small new
one (~150 LOC) that:

1. Iterates over the same `(xml, neighbour_label, object_id)` instances used by
   the GT collection. The instance enumeration is deterministic given the
   manifest + the planner's region snapshot. Easiest is to **read the GT pkls**
   produced in §5.2 and use their `episode_id`, `xml_file`,
   `chosen_object_id`, `neighbour_region_label`, plus the pre‑push state from
   `state_observations[0]`.
2. For each instance, instantiates `RLEnvironment` on the XML, restores the
   pre‑push state via `set_full_state` (the state lives in the env at start
   for 1‑push; for 2‑push push‑1 candidates also use the initial state).
3. Builds the JSON message exactly as `MLPrimitiveGoalStrategy._create_json_message_for_goals`
   does ([`ml_strategies.py:896`](../python/namo/strategies/ml_strategies.py#L896)).
4. Calls `GoalInferenceModel.infer(json_message, xml_path, robot_goal, selected_object,
   samples=32, seed=42)`.
5. Aligns each returned SE(2) sample to primitive slots using
   `MLPrimitiveGoalStrategy._align_samples_to_primitives` (or whatever it is
   called — see the same file). Tolerances: `position 0.2 m`, `angle 0.2 rad`,
   `k_nearest=1`. These exactly match the 2‑push evaluation defaults from
   `region_opening_ml_collection.yaml`.
6. Returns the set of unique aligned primitive slots
   `{(edge_idx, depth)}` with their ML scores (vote counts).

Output: one pkl/jsonl per instance with `{instance_key, ml_aligned: [(edge, depth, score), ...]}`.

I will scaffold this script next session under
`python/namo/data_collection/ml_prediction_offline.py` — flagged below as an
open code task.

### 5.4 Compute Top‑K hit‑rate

Given GT and ML outputs per instance:

```python
# F characterization → set of feasible primitives
F = {(t['edge_idx'], t['depth']) for t in trial_log if t['success']}
R = {(t['edge_idx'], t['depth']) for t in trial_log}            # reachable set

# ML aligned slots sorted by score descending
ml_sorted = sorted(ml_aligned, key=lambda s: -s.score)
topK_set = {(s.edge_idx, s.depth) for s in ml_sorted[:K]}

# Metrics
hit@K        = int(len(topK_set & F) > 0)              # any-hit
precision@K  = len(topK_set & F) / max(1, len(topK_set))
recall@K     = len(topK_set & F) / max(1, len(F))
coverage@K   = len(topK_set & R) / max(1, len(topK_set)) # how many ML preds even reachable
```

Aggregate across instances, then stratify by:
- difficulty bucket (easy / medium / hard from the manifest);
- `|F|/|R|` (continuous version of difficulty, matches `analyze_F.py`);
- chain depth (1 vs 2).

For the 2‑push setting, replace `F` with `F₁'` derived per §4.1 (set of push‑1
`(edge_idx, depth)` whose downstream chain succeeded).

### 5.5 Reports to produce

- `topk_hit_rate.png` — hit@K vs K (1..32) per difficulty bucket. One line for
  1‑push F₁, one for 2‑push F₁′.
- `precision_recall.png` — PR curve scored over all aligned ML slots, ranked
  by score; one curve per difficulty.
- `coverage.png` — what fraction of ML predictions are even in R (sanity).
- A single CSV with per‑instance numbers so we can re‑slice later.

These mirror the existing oracle/realistic plots already in
[`docs/f_characterization/eval_results/`](f_characterization/eval_results/) so
the new ML curves can be overlaid on the same figure.

---

## 6. Where this fits in the broader research thread

From `docs/research_notes_F_characterization.md`:

> Research Question #6 — *"How does F chain for multi-step? For 2‑push
> problems, define F₁' = {push1 : F₂(state_after(push1)) ≠ ∅}. … How does
> F₁' compare to the full reachable set? Is it a small fraction (tight
> bottleneck) or a large fraction (generous)?"*

And from the Methodology section:

> Step 5: Add 2‑push intermediate state data — retrain on 1‑push + 2‑push.
> Step 6: Evaluate on multi‑push. **Works at intermediate states → classifier
> generalizes to dynamics‑generated scenes. No world model needed.**
> Fails → that is the evidence that a dynamics model is needed.

This evaluation is the **Step 6 measurement** for the cropped diffusion model:
does the 2‑push‑trained diffusion model predict push‑1 primitives that lie in
F₁′? If the Top‑K hit‑rate is comparable between 1‑push and 2‑push horizons,
the model has internalized enough chain dynamics to act as a multi‑push goal
sampler. If it collapses at the 2‑push horizon, that is the *first concrete
signal* that classifier‑only is insufficient and a world model is warranted.

Keep [`docs/research_notes_F_characterization.md`](research_notes_F_characterization.md)
Hypothesis 1 (`|F|/|R|` predicts difficulty) in mind — the easiest way to
detect "ML hits F where F is dense, misses where F is sparse" is to bucket the
results by `|F|/|R|`. The existing `analyze_F.py` already computes this; we
should reuse the histogram bins so figures are comparable.

---

## 7. Open decisions / blockers (read before running anything)

1. **Trial‑log patch for chain_depth ≥ 2.** Without it we cannot recover F₁′.
   See §4.1 for the minimal diff. Estimated effort: ~20 LOC + a regression
   re‑run of `analyze_F.py` on the existing 1‑push data to confirm no schema
   regression.
2. **Offline ML scoring script.** §5.3 needs ~150 LOC of new code. It does not
   modify the planner. It is the right place to live next to
   `modular_parallel_collection.py`.
3. **Should we re‑use the model run dir's `jan20/1push/results_*` outputs?**
   The dir already contains several `results_32samples_*` end‑to‑end planner
   sweeps. They give end‑to‑end success rate but **not** Top‑K vs GT F. So
   they're complementary, not substitutes. Decision: keep them for end‑to‑end
   reporting; do not retrofit them into the F overlap metric.
4. **Manifest size.** 303 envs × ≤ ~5 (neighbour × object) instances ≈ ~1.5k
   instances; each ML inference is 32 samples × 5 DDIM steps. At ~0.2 s per
   sample on a single GPU that is ~25 min wall clock. Cheap. The expensive
   side is the GT F collection (especially chain_depth 2).
5. **Random seed.** Pin `ml_seed=42` and `ml_sampler_method=ddim`,
   `ml_num_steps=5` to match the existing 2‑push eval runs.
6. **Re‑verify** that
   `manifest_2push_test_minus_1push_test_filtered_difficulty_100each.txt` is
   truly disjoint from the model's training split. The naming says it is
   ("test minus 1push test filtered"), but a quick set‑difference check
   against `manifest_2push_train.txt` is cheap insurance.

---

## 8. Useful files (everything referenced above)

| Topic | Path |
|---|---|
| Region‑opening planner | `python/namo/planners/opening/region_opening.py` |
| ML strategy (where `GoalInferenceModel.infer` is called) | `python/namo/strategies/ml_strategies.py` |
| Modular collection CLI/YAML loader | `python/namo/data_collection/modular_parallel_collection.py` |
| 1‑push exhaustive YAML (template) | `python/namo/data_collection/region_opening_exhaustive.yaml` |
| ML region‑opening YAML (reference) | `python/namo/data_collection/region_opening_ml_collection.yaml` |
| `GoalInferenceModel` class | `../sage_learning/sage_learning/goal_inference_model.py` |
| F‑characterization analysis script | `docs/f_characterization/analyze_F.py` |
| Existing 1‑push GT (test 1767) | `/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/modular_data_rlab7/` |
| Existing 1‑push GT (train) | `/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab{5,6,7}/` |
| ML model run dir | `/common/users/dm1487/namo_data/outputs/cropped_diffusion_crossattn_2push/2025-12-16/05-36-44/` |
| 2‑push manifests | `/common/users/shared/robot_learning/dm1487/namo/manifests/aug9_medium/manifest_2push_*` |
| Pipeline overview (Method + Setup) | `docs/region_opening_pipeline.md` |
| Research framing & hypotheses | `docs/research_notes_F_characterization.md` |
