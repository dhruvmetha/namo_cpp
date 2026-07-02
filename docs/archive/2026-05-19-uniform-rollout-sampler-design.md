# Uniform 1-Push Sampler — Fresh F-Characterization with Chain-Extendable Schema (v0)

**Date:** 2026-05-19
**Status:** Draft — under brainstorming review
**Author session:** F-characterization brainstorm

---

## 1. Context

The May-2026 round-1 evaluation of the cropped diffusion model against ground-truth F₁ showed the model is *worse than uniform-random-from-R* at every difficulty stratum (very_hard hit@32: 11% vs random 96%). Root cause: the training-data distribution was the region-opening planner's *first* solutions per env (BFS-shallowest by construction). The model learned that shallow bias and missed all deep F primitives.

Conclusion: training data must match F by construction, not the planner's preferences. This spec describes how to collect such data — **v0 scope: 1-push (depth 0) only**.

**Why scope to depth 0:** the round-1 failure is a 1-push problem (model can't even predict F₁ correctly). Fixing it doesn't require chain data. The classifier-first methodology in the user's research notes prescribes exactly this — train a solid 1-push model first, let its failures drive whether chain data is needed.

**Why "chain-extendable schema":** even though we're collecting depth 0 only now, the pkl schema is designed so that a follow-up spec can add depth-1 (and depth-2) records to the same files without re-collecting depth 0. The schema is defensive against decisions we may want to make later.

---

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Produce fresh exhaustive 1-push F-characterization data at every initial scene in the env pool. Distribution matches F by construction (every reachable primitive tried).
- **G2.** Log enough per-transition information that every scene mask `batch_collection_classifier.py` needs — for both successful and failed pushes — is renderable offline from the pkls. No re-sim needed if the mask convention changes.
- **G3.** Backward-compatible with the existing `batch_collection_classifier.py` post-processing (emit `primitive_trial_log` field unchanged). Existing 1-push training paths keep working with zero downstream code edits.
- **G4.** Integrate into the existing `modular_parallel_collection.py` pipeline with no changes to the parallel infrastructure.
- **G5.** Schema is chain-extendable — a follow-up spec adding depth-1+ collection appends new records without touching depth-0 records or breaking existing readers.

### 2.2 Non-goals

- **Not chain data.** Depth ≥ 1 collection is explicitly deferred to a follow-up spec.
- **Not a goal-directed planner.** Even at depth 0, the sampler doesn't try to "solve" anything — it tries every reachable primitive uniformly.
- **Not a replacement for `RegionOpeningPlanner`.** The two coexist.
- **Not a training pipeline.** Model training (X or Y) is the subject of a separate spec.

---

## 3. The sampler — design

### 3.1 Name and role

**Class:** `UniformRolloutSampler`
**Registered name:** `"uniform_rollout_sampler"`
**Module:** `python/namo/planners/sampling/uniform_rollout_sampler.py`

The class implements `BasePlanner` to plug into the existing collection pipeline. The `BasePlanner` interface is overloaded — the worker code expects a "planner" — but the sampler does no planning. It implements `search()` as an exploration-and-logging loop. The name "sampler" in the class is intentional to clarify intent for readers; the registration name uses the same convention.

### 3.2 What it does, in one paragraph

For each call to `search(robot_goal)` (one env, one episode), the sampler exhaustively executes every primitive in R(s₀) — the initial scene's reachable-primitive set. Each push is sim'd from s₀ (state restored before each via `set_full_state`), the outcome labeled, and a transition record logged. The sampler does no search, makes no decisions, and never extends chains. Every reachable primitive is tried; every outcome is recorded.

### 3.3 Per-env algorithm

Let:
- `s₀` = initial scene state.
- `R(s₀)` = the set of reachable primitives at the initial scene, computed by the existing wavefront BFS.
- `r(s, a)` = 1 if `is_robot_goal_reachable(T(s, a))` else 0. This is the per-transition reward used by Design Y; for X it doubles as the witness label.

```
search(robot_goal):
    env.set_robot_goal(*robot_goal)
    s₀ = env.get_full_state()
    R₀ = env.get_reachable_primitives_at(s₀)   # full enumeration

    transitions = []
    next_id = 0

    # Depth 0: exhaustive sweep
    for a₀ in R₀:
        env.set_full_state(s₀)
        s_after, sim_info = env.step(a₀)
        r = env.is_robot_goal_reachable()
        rec = transition_record(
            transition_id=next_id, parent_id=None, depth=0,
            state_before=s₀, action=a₀, state_after=s_after, r=r,
            per_neighbor_opening=collect_per_neighbor_opening(env, s_after),
            per_neighbor_region_goals=sample_per_neighbor_region_goals(env, s₀),
            R_at_state_before=R₀,
            R_at_state_after=env.get_reachable_primitives_at(s_after),
            sim_info=sim_info,
        )
        transitions.append(rec)
        next_id += 1

    return PlannerResult(
        success=False, solution_found=False, action_sequence=None,
        algorithm_stats={
            "rollout_trace": transitions,
            "primitive_trial_log": derive_legacy_trial_log(transitions),  # see §4.6
            "env_metadata": {...},
        },
        search_time_ms=elapsed_ms,
    )
```

Order of iteration over R₀ is not significant for the data (each primitive is tried exactly once), but for reproducibility we iterate in sorted (object, edge_idx, depth_idx) order seeded deterministically per env.

### 3.4 The action-sampling rule (depth 0 only)

Exhaustive over R(s₀). Every reachable primitive is tried exactly once. No sampling decisions at v0 — "uniform" applies trivially because the entire population is enumerated.

The class is still named `UniformRolloutSampler` (not `ExhaustiveDepth0Sampler`) because the schema and structure are designed for the chain-extended version; calling it a one-shot thing would lock us out of the natural follow-up. The contract is "tries primitives without goal-directed bias" — at v0 that means exhaustive, at v1 it could mean uniform-random at deeper levels. Same class, same registration, same code paths.

### 3.5 What's deferred to a follow-up spec

These pieces of the original design are explicitly **not in v0** but the schema and class structure keep the door open:

- **Depth-1 chain expansion** (random first-push expansion into second pushes). Adds 2-push training data.
- **Depth-2 chain expansion.** Adds 3-push training data.
- **The `set_full_state` branching trick.** Only relevant once we expand chains.
- **K₁/K₂/K₃ sampling budgets.** No K knobs at v0.
- **Chain-termination logic** (`stop_on_reward`). Trivially "stop after depth 0" at v0.
- **Optional planner fallback** (Pass 1) for envs where random rollouts can't find a 2-push chain.

When the follow-up spec is written, these are added as new code paths inside the same class. Existing v0 pkls remain valid; new pkls have additional records appended to `rollout_trace` at depths ≥ 1.

### 3.6 Reachability set R(s) — what counts as "reachable"

`R(s)` must be the same notion used by `RegionOpeningPlanner` so the data is comparable. Specifically: the set of (object, face, contact-point, depth) primitives whose contact point the robot can reach via wavefront BFS, restricted to movable objects in the env.

The relevant building blocks already exist in the codebase:
- `env.get_reachable_objects()` — Python binding returning reachable object names.
- `env.is_object_reachable(name)` — boolean per-object check.
- The primitive motion database loader used by `region_opening`'s goal strategies — already shared utility.

What is *not* exposed as a standalone function: enumerating the (object, edge_idx, depth_idx) tuples whose contact points are individually reachable. This logic lives inside `region_opening`'s goal-strategy classes (`PrimitiveGoalStrategy`, `MLPrimitiveGoalStrategy`, etc.), entangled with goal-direction code the sampler doesn't need.

**Plan: a small local helper inside `uniform_rollout_sampler.py`** — ~20–40 LOC that loads the primitive motion database once at sampler init and, given a state, returns the list of (object, edge_idx, depth_idx) tuples whose contact points are reachable. No new shared module. If a third caller ever wants the same enumeration, extract to `python/namo/core/reachable_primitives.py` at that point, not earlier.

The sampler does **not** reimplement reachability — it calls `env.get_reachable_objects()` / `is_object_reachable` for the per-object check and uses the existing motion-database loader. The new code is only the enumeration loop, not the underlying reachability primitive.

### 3.7 Per-neighbor opening evaluation + region-goal sampling

Per transition, the sampler does two things using `snapshot_region_connectivity` and `find_robot_label` from `python/namo/planners/connectivity_snapshot.py` (already imported by `region_opening`):

1. **Per-neighbor opening labels.** Compute the snapshot at `state_before` and at `state_after`; diff which neighbor labels became reachable. Result → `transition.per_neighbor_opening`.

2. **Per-neighbor region-goal sampling.** At `state_before`, for each current neighbor region, sample K points (e.g. K=5) uniformly inside that neighbor's polygon. Result → `transition.per_neighbor_region_goals`. This is required because region polygons reshape as objects move, so points sampled at s₀ become stale at s₁/s₂.

Both reuse the same connectivity-snapshot calls; the second only adds a per-polygon point sampler (the same utility `region_opening`'s goal strategies already use). **No new evaluator module. No refactor of `region_opening`.**

---

## 4. Integration with the existing data-collection infrastructure

This is the most important section of the spec per the brainstorming-session ask. The integration is designed to require **zero changes to `ModularParallelCollectionManager`, the worker function's overall structure, the parallel-pool setup, or the pkl-output convention.** The sampler plugs into the existing extension points.

### 4.1 PlannerFactory registration

`UniformRolloutSampler` self-registers on import, following the pattern in lines 42–43 of `modular_parallel_collection.py`:

```python
# At the bottom of uniform_rollout_sampler.py
from namo.core import PlannerFactory
PlannerFactory.register_planner("uniform_rollout_sampler", UniformRolloutSampler)
```

The import is added to `modular_parallel_collection.py`:

```python
# Add to imports near line 43
from namo.planners.sampling.uniform_rollout_sampler import UniformRolloutSampler
```

After this, `PlannerFactory.list_available_planners()` includes `"uniform_rollout_sampler"`, and the existing `--algorithm` argparse choice (line 1011) auto-picks it up.

### 4.2 Worker flow

The worker code path in `modular_worker_process()` (lines 353–717) has three branches:

1. **`region_opening` branch** (lines 398–520): special-cased because the planner returns `attempt_results` and the worker fans out one `ModularEpisodeResult` per neighbor attempt.
2. **Optimal-planner branch** (lines 522–580): special-cased because the planner returns multiple minimum-length solutions.
3. **Standard branch** (lines 581–630): for any other planner. Takes the `PlannerResult` and emits one `ModularEpisodeResult` per env.

**`UniformRolloutSampler` uses the standard branch (3).** It returns one `PlannerResult` per env; the worker emits one `ModularEpisodeResult` per env. The rollout data lives in `PlannerResult.algorithm_stats["rollout_trace"]`, which the worker already serializes verbatim into the pkl (line 599: `algorithm_stats=planner_result.algorithm_stats`).

**No new branch is needed in the worker.** This is the central infrastructure-integration claim of this spec.

There is one detail: the standard branch sets `success=planner_result.success` and `solution_found=planner_result.solution_found` (lines 591–592). The sampler returns `success=False, solution_found=False` since it never tries to solve. That's correct — these fields reflect *whether the env was solved*, not whether the sampler ran cleanly. We adopt the convention that `success=False, solution_found=False, algorithm_stats != None` means "data collection completed normally." If the sampler crashes mid-env, the exception path at line 632 logs the failure with the existing failure-classification system; the partial transitions collected so far are discarded along with the rest of the env's data (acceptable — re-run that env).

### 4.3 `ModularWorkerTask` and `PlannerConfig`

The sampler has minimal v0 knobs:

```python
algorithm_params = {
    "max_chain_depth": 1,                # v0: must be 1; chain expansion is a follow-up spec
    "seed": <int>,                       # per-env determinism
    "log_sim_info": True,                # collision flags, push-terminated-early, etc.
    "region_goal_samples_per_neighbor": 5,  # K points per neighbor for goal_sample_region mask
}
```

These pass through the existing `algorithm_params` plumbing (line 779). The sampler reads `config.algorithm_params` in its `_initialize_algorithm()` method.

When the follow-up spec adds chain expansion, new keys (`K_first_push_expansions`, `K_second_push_branches`, `stop_on_reward`) join this dict without disturbing existing v0 invocations.

### 4.4 Argparse / CLI surface

Add to `modular_parallel_collection.py`'s argparse (under the existing planner-specific args block, around line 1080):

```python
# Uniform rollout sampler arguments (v0 = depth 0 only)
parser.add_argument("--sampler-max-chain-depth", type=int, default=1, choices=[1],
                    help="v0 supports depth 0 only (max_chain_depth=1). "
                         "Deeper depths are a follow-up spec.")
parser.add_argument("--sampler-region-goal-samples", type=int, default=5,
                    help="K points to sample per neighbor region for goal_sample_region mask.")
```

And in the algorithm_params construction (around line 1148):

```python
if args.algorithm == "uniform_rollout_sampler":
    algorithm_params["max_chain_depth"] = args.sampler_max_chain_depth
    algorithm_params["region_goal_samples_per_neighbor"] = args.sampler_region_goal_samples
    algorithm_params["seed"] = args.seed if args.seed is not None else DEFAULT_GLOBAL_SEED
```

### 4.5 Output: pkl convention

Existing pkl path: `<output_dir>/modular_data_<hostname>/<task_id>_results.pkl` (lines 697–698).

Contents are `worker_result_data` (lines 687–696) — a dict with `episode_results` containing a list of `ModularEpisodeResult` records. For the sampler, each env produces one record where:

- `success`, `solution_found` = False
- `action_sequence` = None
- `algorithm_stats` = `{"rollout_trace": [...], "env_metadata": {...}}`
- `xml_file`, `robot_goal`, `static_object_info` = populated as today

**No new pkl files. No new directory layout. No new naming convention.** The downstream analysis code (existing `analyze_F.py`) needs an extension to read `rollout_trace` instead of (or in addition to) `primitive_trial_log`, but the file structure is identical.

### 4.6 Backward-compatibility shim: `primitive_trial_log`

The existing `batch_collection_classifier.py` post-processing script reads `episode.algorithm_stats['primitive_trial_log']` to build the (60, 10) `f_grid` consumed by sage_learning's classifier and diffusion data loaders. The sampler emits a `primitive_trial_log` field alongside `rollout_trace`, derived deterministically from depth-0 records:

```python
primitive_trial_log = [
    {
        "edge_idx": rec.edge_idx,
        "depth": rec.push_depth_idx,
        "success": bool(rec.r),
        "wall_collision": rec.wall_collision,
        "movable_collisions": rec.movable_collisions,
        "stuck": rec.push_terminated_early,
        "collision": bool(rec.wall_collision or rec.movable_collisions),
        "reachable_after": int(rec.r),
    }
    for rec in rollout_trace if rec.depth == 0
]
```

Result: `batch_collection_classifier.py` runs unchanged on the new pkls. Existing 1-push training paths keep working with zero downstream code edits.

### 4.7 Reused vs. new vs. modified

**Reused unchanged:**
- `ModularParallelCollectionManager` (parallel pool, signal handling, run-dir layout)
- `modular_worker_process` standard branch
- `ModularEpisodeResult`, `ModularWorkerResult` dataclasses
- pkl serialization (dataclass `asdict` + pickle)
- Manifest loading
- Failure-classification system

**New:**
- `python/namo/planners/sampling/uniform_rollout_sampler.py` — the sampler class plus a small local helper for enumerating reachable (object, edge, depth) primitives.
- CLI args in `modular_parallel_collection.py`.
- A loader / analysis utility for reading `rollout_trace` records (small, can be added incrementally).

**Modified:**
- `python/namo/data_collection/modular_parallel_collection.py` — one import line plus the new argparse args. No control-flow changes.

**Reused as-is (not modified):**
- `python/namo/planners/connectivity_snapshot.py` — `snapshot_region_connectivity` and `find_robot_label` already shared, no refactor needed.
- The primitive motion database loader — already a shared utility used by `region_opening`'s goal strategies; the sampler imports and uses it without modification.
- `python/namo/planners/opening/region_opening.py` — **untouched** by Spec B. No refactor, no behavior change. This was an earlier overproposal in this spec that has been corrected.

---

## 5. Data schema

This is the contract that downstream Spec A (1-push training) and any chain-training spec will read against. It is defensively designed: every field has a stated reason for being logged. No field is logged "just in case."

### 5.1 Per-env top-level structure

Inside `ModularEpisodeResult.algorithm_stats`:

```python
{
    "rollout_trace": [TransitionRecord, ...],     # at v0: only depth-0 records
    "primitive_trial_log": [...],                  # legacy-shape shim (§4.6) for batch_collection_classifier.py
    "env_metadata": EnvMetadata,
    "sampler_config": {
        "max_chain_depth": 1,                      # v0 only supports depth 0
        "seed": ...,
        "sampler_version": "0.1",
        "region_goal_samples_per_neighbor": 5,
    },
    "summary_stats": {
        "n_transitions": int,
        "n_r1": int,                               # how many depth-0 pushes opened the passage
        "n_sim_failures": int,
        "total_sim_time_ms": float,
    },
}
```

### 5.2 `TransitionRecord`

```python
@dataclass
class TransitionRecord:
    # Identity
    transition_id: int                    # unique within env, dense from 0
    parent_id: Optional[int]              # always None at v0 (depth 0); kept for chain-extendability
    depth: int                            # always 0 at v0; field kept for chain-extendability

    # State
    state_before_qpos: np.ndarray         # full MuJoCo qpos at state_before
    state_before_qvel: np.ndarray         # always zeros by env convention but stored for safety
    state_before_se2: Dict[str, Tuple[float, float, float]]  # cheap per-object SE(2)
    state_after_qpos: np.ndarray
    state_after_qvel: np.ndarray
    state_after_se2: Dict[str, Tuple[float, float, float]]

    # Action
    object_id: str
    edge_idx: int                         # 0..59
    push_depth_idx: int                   # 0..9
    target_pose: Tuple[float, float, float]    # (x, y, θ) the push aimed for

    # Reward and per-neighbor opening
    r: int                                # 0 or 1; 1 iff is_robot_goal_reachable(state_after)
    per_neighbor_opening: Dict[str, bool] # {neighbor_label: opened?}

    # Per-neighbor region goal samples — sampled at state_before from each
    # current neighbor region's polygon. Required by NAMODataVisualizer to
    # render the goal_sample_region / local_goal_sample_region mask channels.
    # Sampled at state_before only; state_after of transition N equals
    # state_before of transition N+1 (already covered), and the terminal
    # state_after is never a decision point we render masks for.
    per_neighbor_region_goals: Dict[str, List[Tuple[float, float, float]]]

    # Reachable-set bookkeeping (for Q-learning's max_a' term)
    R_at_state_before: List[Tuple[str, int, int]]   # (object, edge_idx, depth_idx) tuples
    R_at_state_after: List[Tuple[str, int, int]]    # same shape; needed for Q max over R(s')

    # Sim diagnostics
    sim_time_ms: float
    wall_collision: bool
    movable_collisions: List[str]         # object_ids hit during push
    push_terminated_early: bool           # e.g. robot stuck
    sim_failure: bool                     # MuJoCo error / NaN
```

**Per-field justification** (what each field defends against, anti-recollection):

| Field | What it defends |
|---|---|
| `state_before_qpos/qvel`, `state_after_qpos/qvel` | World-model training, restart-sim analyses, custom feature extraction. SE(2) is a lossy projection; qpos is the full picture. Logging both is a small storage cost vs. re-sim cost. |
| `state_before_se2`, `state_after_se2` | Fast inspection / plotting without unpacking qpos. Derived from qpos but cached for analysis convenience. |
| `R_at_state_after` | Required by Q-learning's `max_a' Q(s', a')` term. Without it, the Q-loss either has to re-run wavefront BFS at every training step (slow) or learn to ignore unreachable actions (wrong). |
| `per_neighbor_opening` | Allows per-neighbor Q (v1) without re-collection. The scalar `r` is a useful summary but loses per-neighbor structure. |
| `per_neighbor_region_goals` | Required to render the `goal_sample_region` mask channel (one of the 5 NN inputs). Region polygons reshape between transitions as objects move, so points must be sampled at each transition's `state_before`, not pre-sampled once per env. |
| `wall_collision`, `movable_collisions` | Structural feature analysis (your finding: 76% of very-hard-F involves wall contact). Already used by existing F-char analysis. |
| `parent_id` chain | Reconstruct any chain (s₀ → s₁ → s₂ → s₃) by walking parents. Required for X (witness reconstruction). |
| `sim_time_ms`, `sim_failure`, `push_terminated_early` | Filter pathological transitions during analysis without re-sim. Sim failures are real data, not bugs. |

### 5.3 `EnvMetadata`

```python
@dataclass
class EnvMetadata:
    xml_file: str
    robot_goal: Tuple[float, float, float]
    initial_state_qpos: np.ndarray
    initial_state_qvel: np.ndarray
    initial_state_se2: Dict[str, Tuple[float, float, float]]
    initial_R: List[Tuple[str, int, int]]         # R(s₀) for sanity-checking sampling
    neighbor_regions: List[Dict]                  # [{label, geometry, ...}, ...] from region_opening
    static_object_info: Dict[str, Dict]            # sizes / types — same as today (line 365)
    collection_timestamp_utc: str
    git_commit: Optional[str]                     # for reproducibility, populated if available
```

### 5.4 Size estimates

Per transition: ~4 KB uncompressed (qpos×2, qvel×2, SE(2)×2, R lists, per_neighbor dicts, sim diagnostics).

Per env at v0: up to 600 transitions = ~2.4 MB uncompressed, ~0.5–1 MB compressed pickle.

For ~5,925 train envs: ~3–6 GB total. Trivial to store.

When chain expansion is added in the follow-up spec, per-env size grows ~5× (to ~5,500 transitions worst case). Still manageable; switch to sharded zarr/HDF5 only if pkl size becomes operationally annoying.

---

## 6. What this enables downstream

### 6.1 Scene-mask rendering — the key invariant

The pkl logs enough per-transition state to render scene masks for **any** action — successful or failed — without re-simulating. For every transition the schema records:

| Need to render | Where it lives |
|---|---|
| Scene at `state_before` (static / movable / target_object / robot_region) | `transition.state_before_se2` + `env_metadata.xml_file` + `env_metadata.static_object_info` |
| Target-pose mask (where the action was *aimed*) | `transition.target_pose` + `transition.object_id` |
| Scene at `state_after` (where the object actually landed) | `transition.state_after_se2` |
| `goal_sample_region` mask (the NN input channel for the neighbor region) | `transition.per_neighbor_region_goals` — points sampled at this transition's state_before |
| Goal-pose mask (robot's target XML pose) | `env_metadata.robot_goal` |
| Robot's current region (for `robot_region` mask) | derivable from `state_before_se2` via wavefront BFS at that state |

Successful and failed transitions log the same fields with the same shape. Any mask convention — current (`local_static`, `local_target_goal`, etc.) or future — can be derived offline from the pkls.

### 6.2 1-push training (X or Y at v0 — both 1-push)

With v0's depth-0-only data, both Design X and Design Y reduce to **1-push models**:

**Design X (generative on witnesses):**
- Positives = every depth-0 transition with `r=1`.
- Train diffusion / flow / autoregressive to predict an action that opens the passage at the initial scene.
- Existing `batch_collection_classifier.py` pipeline → NPZ files → existing sage_learning trainers. Drop-in replacement for round-1's biased data.

**Design Y (Q-function):**
- Every depth-0 transition is a Bellman tuple `(s₀, a, s₁, r)`.
- With no data from s₁ onwards, the Bellman discount γ doesn't bootstrap anywhere — Q collapses to a classifier on F₁ at s₀.
- Effectively the same model as X with a different loss.

Both train from the same pkls. Pick whichever matches the existing infrastructure (X is the natural fit for the existing cropped-DiT pipeline).

### 6.3 Why depth 0 is enough for v0

The round-1 failure was a 1-push problem: the model can't predict F₁ correctly at the initial scene. Depth-0 data fixes that — uniform over R(s₀) by construction, F₁ recovered without planner bias.

If the resulting 1-push model still doesn't beat random-from-R on hard envs, the failure is architectural (not data). If it does, that becomes the calibrated baseline against which any chain model has to lift — and the follow-up spec adds chain data with a clear performance target.

---

## 7. Compute budget and cluster strategy

### 7.1 Per-env sim count

v0 is depth-0 only:
- Depth 0: |R(s₀)| ≤ 600 sims (exhaustive)
- **Per-env total: ≤ 600 sims.**

Same compute as the existing 1-push F-char collection. No new compute cost.

### 7.2 Cluster strategy: sharded across hosts

The collection runs as **N independent shards** — one per cluster host (rlab5, rlab6, rlab7, etc.). Each shard processes a disjoint env range via `--start-idx` / `--end-idx`. Output goes to per-host subdirectories (`modular_data_<hostname>`) under a shared output root. This is exactly how the existing 1-push F-char data was collected (`/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab{5,6,7}/*.pkl`).

Concretely, three sbatch jobs (one per node) with non-overlapping ranges:

```
# rlab5: envs 0..1975
python modular_parallel_collection.py --algorithm uniform_rollout_sampler \
  --manifest <train_manifest.txt> --start-idx 0 --end-idx 1975 --workers 100 \
  --output-dir /common/users/.../uniform_rollout_train --sampler-max-chain-depth 1

# rlab6: envs 1975..3950   (analogous)
# rlab7: envs 3950..5925   (analogous)
```

Wall-clock estimate at ~0.1 s/sim, ~600 sims/env, 100 workers per host: ~10–20 minutes per shard. With three hosts running in parallel, the full train pool collects in well under an hour. The test pool (1,767 envs) collects in proportionally less time on one host.

**Sharding is strictly faster** than a single job — more workers, more cores, no coordination overhead between shards (each shard writes independent pkls to its own per-host directory). The existing manifest mechanism (`--start-idx` / `--end-idx`) already enforces non-overlapping slices, so there is no risk of duplicate work.

### 7.3 Per-shard determinism

Each shard derives its RNG seeds from `seed_base + env_global_idx`, so the same env produces the same sampling regardless of which shard processes it. Reproducibility is unaffected by shard boundaries.

### 7.4 Run sequencing within a shard

Within a single shard, each worker processes its assigned envs sequentially. Per env, the worker runs the exhaustive depth-0 sweep and writes the per-env pkl on completion. Standard pattern — no chain bookkeeping at v0.

---

## 8. Testing strategy

### 8.1 Unit: exhaustive coverage

Fix an env. Run the sampler. Verify that the set of `(object, edge_idx, push_depth_idx)` tuples in `rollout_trace` exactly equals `R(s₀)`. No duplicates, no omissions.

### 8.2 Unit: reproducibility

Run the sampler twice on the same env with the same seed. Verify the transition records are bit-identical (modulo timing fields).

### 8.3 Integration: F-char regression

Run the sampler on a 10-env manifest sample. Compare the resulting depth-0 records to the existing F-char pkls (`/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/`) for the same envs. The set of `(object, edge, depth, r)` tuples must match exactly. Confirms the new sampler reproduces existing F-char outputs.

### 8.4 Integration: full pipeline on small manifest

Run `modular_parallel_collection.py --algorithm uniform_rollout_sampler --manifest <small_manifest.txt> --workers 4 --start-idx 0 --end-idx 10`. Verify: all pkls written, schema loads without error, `primitive_trial_log` shim is present and well-formed, `batch_collection_classifier.py` runs on the new pkls and produces valid NPZ files.

### 8.5 Sanity: 1-push classifier convergence on a single env

Train a tiny per-primitive classifier on one env's depth-0 records (~600 labels). Verify the loss converges and the classifier recovers F₁(s₀) exactly on the training env (memorization sanity check). Confirms data is wired correctly end-to-end.

---

## 9. Failure modes and error handling

### 9.1 Sim failures mid-transition

Caught at the `env.step()` call site. The transition is logged with `sim_failure=True`. Other transitions in the env continue normally.

### 9.2 Empty `R(s₀)`

Skip the env entirely (nothing to sample). Logged once in worker output.

### 9.3 Worker crashes during env

The existing `modular_worker_process` exception path (line 632) catches and logs. The env's pkl is not written; the env is silently dropped from the dataset. Acceptable — re-run that env, or accept the loss.

### 9.4 RNG management

Each env's RNG is seeded by `seed_base + env_idx`, where `seed_base` is `args.seed` (or default 42). Reproducibility tested in §8.2. At v0 the only randomness is in `per_neighbor_region_goals` sampling (sampling K points inside each polygon).

---

## 10. What this spec does not cover

- **Chain expansion (depth ≥ 1).** Explicitly deferred to a follow-up spec. Adds depth-1 random rollouts (and optionally depth-2). Schema is designed so the follow-up adds new transition records without breaking existing readers.
- **Optional planner-fallback pass.** Some envs (≤ 5% per round-1 data) need ≥ 2 pushes to open, so random rollouts may find zero positives. A follow-up may add a directed-pass (region_opening with shuffled edges) for envs where depth-1 random comes up empty. Out of scope for v0.
- **1-push training pipeline (Spec A).** Separate spec; blocks only on v0 pkls being on disk.
- **World-model training (Thread 2).** The schema supports it (full qpos at every transition), but training itself is out of scope.
- **Active-learning / model-guided augmentation collection.** v1+ question.
- **Schema evolution / format migration.** If the schema needs to change after data is collected, write a migration script.

---

## Open questions / risks

1. **Per-env runtime.** v0 is ~600 sims/env — well under the 300s default `--search-timeout`. Not a concern at v0.
2. **`set_full_state` correctness.** Each depth-0 push restores from saved s₀ via `set_full_state` before being executed. The qvel-zeroing convention (line 114 of `rl_env.cpp` per memory) ensures each push starts from rest, which matches what region_opening does today. Confirm with a smoke test against the existing F-char data (§8.3).
3. **F-char regression baseline differences.** Some primitives may produce slightly different outcomes in the new sampler vs. existing F-char data if there have been any code changes to the env, push primitive, or wavefront since the existing data was collected. §8.3 will flag this — and any difference is itself useful information.

---

## Appendix: file inventory

**New files:**
- `python/namo/planners/sampling/uniform_rollout_sampler.py`
- `python/namo/data_collection/rollout_trace_loader.py` (analysis utility)
- `tests/test_uniform_rollout_sampler.py`

**Modified files:**
- `python/namo/data_collection/modular_parallel_collection.py` (one import + CLI args; no control flow changes)

**Reused as-is, not modified:**
- `python/namo/planners/connectivity_snapshot.py` (per-neighbor opening evaluator)
- The primitive motion database loader (shared by `region_opening`'s goal strategies)
- `python/namo/core/base_planner.py`
- All other planners (including `region_opening.py` — no refactor for Spec B)
- The parallel-pool / worker / pkl-writer code paths

---

*End of Spec B.*
