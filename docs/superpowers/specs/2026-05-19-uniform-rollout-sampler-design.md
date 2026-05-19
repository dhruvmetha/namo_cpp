# Uniform Rollout Sampler for Chain-Aware Region-Opening Data Collection

**Date:** 2026-05-19
**Status:** Draft — under brainstorming review
**Author session:** F-characterization brainstorm

---

## 1. Context

The May-2026 round-1 evaluation of the cropped diffusion model against ground-truth F₁ showed the model is *worse than uniform-random-from-R* at every difficulty stratum (very_hard hit@32: 11% vs random 96%). The root cause was traced to the training-data distribution: targets were the region-opening planner's *first* solutions per env, which are BFS-shallowest pushes by construction. The model faithfully learned that shallow-displacement distribution and missed all deep F primitives.

Conclusion from round 1: any future model training must use data whose distribution matches F by construction, not a planner's preferences. This spec describes the data-collection mechanism that produces such data.

The scope is the **data collection only**. Two downstream design questions — which model family to train (X: generative on witnesses, Y: Q-function on transitions) and whether to scope first to 1-push or directly to chains — are deliberately deferred. The collection plan in this spec produces data that supports *both* X and Y, *both* 1-push and chain training, from one cluster run.

---

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Produce fresh exhaustive 1-push F-characterization data (depth-0 records) at every initial scene in the env pool.
- **G2.** Produce sampled multi-push chain data (depth-1 and depth-2 records) that captures chain feasibility for both X and Y.
- **G3.** The distribution of recorded actions at every state must be **uniform over R(s)** (depth 0 trivially; depths ≥ 1 by sampling). No goal-directed bias, no model-guided bias, no shallow-depth-first bias.
- **G4.** The pkl schema must support both X (witness positives) and Y (Bellman transitions) and any reasonable variant *without re-collection*.
- **G5.** Integrate into the existing `modular_parallel_collection.py` pipeline with no changes to the parallel infrastructure (manager, worker, signal handling, manifest loading).

### 2.2 Non-goals

- Not a goal-directed planner. The sampler does no search; it does not try to open any neighbor; it does not stop the env when a solution is found.
- Not a replacement for `RegionOpeningPlanner`. The two coexist; the existing planner is used at inference time (with model-guided action selection) and for goal-directed analyses.
- Not a world model. State transitions are sim'd with MuJoCo, not learned.
- Not a training pipeline. Model training (X or Y) is the subject of a separate spec.

---

## 3. The sampler — design

### 3.1 Name and role

**Class:** `UniformRolloutSampler`
**Registered name:** `"uniform_rollout_sampler"`
**Module:** `python/namo/planners/sampling/uniform_rollout_sampler.py`

The class implements `BasePlanner` to plug into the existing collection pipeline. The `BasePlanner` interface is overloaded — the worker code expects a "planner" — but the sampler does no planning. It implements `search()` as an exploration-and-logging loop. The name "sampler" in the class is intentional to clarify intent for readers; the registration name uses the same convention.

### 3.2 What it does, in one paragraph

For each call to `search(robot_goal)` (one env, one episode), the sampler walks a bounded-depth tree of states rooted at the env's initial scene. At depth 0, it exhaustively executes every primitive in R(s₀). At each depth ≥ 1, it draws K_d primitives uniformly from R(s) at the current state, sims each, and recurses on those whose terminal-reachability flag is still 0. The chain stops as soon as `is_robot_goal_reachable()` becomes True for that branch, or at `max_chain_depth` (= 3 for the planned cluster run). Every primitive tried — successful, failed, dead-end — is logged.

### 3.3 Per-env algorithm

Let:
- `s₀` = initial scene state.
- `R(s)` = the set of reachable primitives at state s, computed by the existing wavefront BFS.
- `r(s, a)` = 1 if `is_robot_goal_reachable(T(s, a))` else 0. This is the per-transition reward used by Design Y; for X it doubles as the witness label.
- `K₁`, `K₂`, `K₃` = sampling budgets at depths 1, 2, 3 (only K₁ and K₂ used when `max_chain_depth = 3`, since depth-0 is exhaustive and depth-3 final-pushes use K₃).

```
search(robot_goal):
    env.set_robot_goal(*robot_goal)
    s₀ = env.get_full_state()
    R₀ = env.get_reachable_primitives_at(s₀)   # full enumeration

    transitions = []                            # flat list, ordered by depth then sample
    next_id = 0

    # ----------------- Depth 0: exhaustive -----------------
    for a₀ in R₀:
        env.set_full_state(s₀)
        s_after, sim_info = env.step(a₀)
        r = env.is_robot_goal_reachable()
        rec = transition_record(
            transition_id=next_id, parent_id=None, depth=0,
            state_before=s₀, action=a₀, state_after=s_after, r=r,
            per_neighbor=collect_per_neighbor_opening(env, s_after),
            sim_info=sim_info, R_at_state_before=R₀,
        )
        transitions.append(rec)
        next_id += 1

    # ----------------- Depth 1: K₁ first-pushes that failed at depth 0 -----------------
    if max_chain_depth >= 2:
        failed_at_0 = [t for t in depth0 if t.r == 0]
        depth1_seeds = uniform_sample_without_replacement(failed_at_0, K=K₁)

        for seed in depth1_seeds:
            s₁ = seed.state_after
            env.set_full_state(s₁)
            R₁ = env.get_reachable_primitives_at(s₁)
            if len(R₁) == 0:                    # no reachable actions, dead-end
                continue
            depth1_actions = uniform_sample_without_replacement(R₁, K=K₂)

            for a₁ in depth1_actions:
                env.set_full_state(s₁)
                s_after, sim_info = env.step(a₁)
                r = env.is_robot_goal_reachable()
                rec = transition_record(
                    transition_id=next_id, parent_id=seed.transition_id, depth=1,
                    state_before=s₁, action=a₁, state_after=s_after, r=r,
                    per_neighbor=..., sim_info=..., R_at_state_before=R₁,
                )
                transitions.append(rec)
                next_id += 1

    # ----------------- Depth 2: K₃ third-pushes for chains that still haven't opened -----------------
    if max_chain_depth >= 3:
        for d1_rec in depth1_records:
            if d1_rec.r == 1:           # chain already opened at depth 1, no expansion
                continue
            s₂ = d1_rec.state_after
            env.set_full_state(s₂)
            R₂ = env.get_reachable_primitives_at(s₂)
            if len(R₂) == 0:
                continue
            depth2_actions = uniform_sample_without_replacement(R₂, K=K₃)

            for a₂ in depth2_actions:
                env.set_full_state(s₂)
                s_after, sim_info = env.step(a₂)
                r = env.is_robot_goal_reachable()
                rec = transition_record(
                    transition_id=next_id, parent_id=d1_rec.transition_id, depth=2,
                    state_before=s₂, action=a₂, state_after=s_after, r=r,
                    per_neighbor=..., sim_info=..., R_at_state_before=R₂,
                )
                transitions.append(rec)
                next_id += 1

    return PlannerResult(
        success=False,                    # sampler never declares success
        solution_found=False,
        action_sequence=None,
        algorithm_stats={
            "rollout_trace": transitions,
            "env_metadata": {...},
        },
        search_time_ms=elapsed_ms,
    )
```

### 3.4 The action-sampling rule

**At depth 0:** enumerate every primitive in R(s₀). Same as `region_exhaustive_mode=True` today.

**At depth ≥ 1:** draw K_d primitives **uniformly at random, without replacement** from R(s). Implemented as `numpy.random.default_rng(seed).choice(R, size=K, replace=False)`.

No heuristic, no model, no cost ordering, no shuffle-then-take-first (which has subtle bias if K < |R|). Sampling without replacement is non-negotiable — sampling with replacement would oversample some primitives within a single env, distorting the per-env distribution. The Mersenne-twister / PCG-based numpy choice on a deduplicated R is the standard correct primitive.

If `K > |R|`, we degrade to exhaustive at that node (sample without replacement = enumerate). Logged in the env metadata: `max_K_clamped_to_|R|` counter.

### 3.5 Chain termination

For a given chain `(a₀, a₁, …)`:
- **Stop on r=1.** If `r(s, a) = 1` (action opens the passage), record the transition and **do not expand** that chain further. Extending an already-successful chain produces only redundant positives.
- **Stop on `max_chain_depth`.** Chains never exceed configured depth (= 3 for the planned run).
- **Stop on dead-end.** If `R(s) = ∅` at some state, record nothing more for that chain and move on. (This should not occur in practice for region-opening setups with the existing reachable-primitive enumeration, but we handle it defensively.)
- **Stop on sim failure.** If `env.step(a)` raises (NaN qpos, robot stuck, contact resolver failure), record the transition with `sim_info.failure=True` and do not expand. This information is itself a useful signal — *pushes that crash the simulator are real outcomes*, not bugs to filter.

### 3.6 The "state restore via `set_full_state`" trick

This is the central efficiency move. Each depth-1 expansion needs to fork the sim into K₂ different second-pushes from the same s₁. Naively, you'd re-execute the depth-0 first push K₂ times to land at s₁ before each second push. With `set_full_state(s₁)`, we skip the re-execution entirely — `s₁` is restored directly from the qpos/qvel we already saved at depth 0.

Same trick at depth 2: `s₂` is the state we wrote out at depth 1; we restore it K₃ times for the third-push branches.

Compute savings: each non-root state is sim'd exactly once. Without the trick, the total sim count would be K₁·(1 + K₂·(1 + K₃)) per chain; with the trick, it's K₁·K₂·K₃ + (lower-order terms).

### 3.7 Reachability set R(s) — what counts as "reachable"

`R(s)` must be the same notion used by `RegionOpeningPlanner` so the data is comparable. Specifically: the set of (object, face, contact-point, depth) primitives whose contact point the robot can reach via wavefront BFS, restricted to movable objects in the env.

The relevant building blocks already exist in the codebase:
- `env.get_reachable_objects()` — Python binding returning reachable object names.
- `env.is_object_reachable(name)` — boolean per-object check.
- The primitive motion database loader used by `region_opening`'s goal strategies — already shared utility.

What is *not* exposed as a standalone function: enumerating the (object, edge_idx, depth_idx) tuples whose contact points are individually reachable. This logic lives inside `region_opening`'s goal-strategy classes (`PrimitiveGoalStrategy`, `MLPrimitiveGoalStrategy`, etc.), entangled with goal-direction code the sampler doesn't need.

**Plan: a small local helper inside `uniform_rollout_sampler.py`** — ~20–40 LOC that loads the primitive motion database once at sampler init and, given a state, returns the list of (object, edge_idx, depth_idx) tuples whose contact points are reachable. No new shared module. If a third caller ever wants the same enumeration, extract to `python/namo/core/reachable_primitives.py` at that point, not earlier.

The sampler does **not** reimplement reachability — it calls `env.get_reachable_objects()` / `is_object_reachable` for the per-object check and uses the existing motion-database loader. The new code is only the enumeration loop, not the underlying reachability primitive.

### 3.8 Per-neighbor opening evaluation

For each transition's `state_after`, the sampler records which neighbor regions are now reachable. This uses the existing `snapshot_region_connectivity` and `find_robot_label` functions in `python/namo/planners/connectivity_snapshot.py` — the same module `RegionOpeningPlanner` already imports (line 24 of `region_opening.py`).

The sampler computes `snapshot_region_connectivity(env, state_before)` and `snapshot_region_connectivity(env, state_after)` once per transition, then diffs which neighbor labels became reachable. Output goes into `transition.per_neighbor_opening`.

**No new evaluator module. No refactor of `region_opening`.** The existing shared utility is used as-is.

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

The sampler's knobs live in `algorithm_params`, with three named K-budgets that map directly to the K₁/K₂/K₃ in §3.3:

```python
algorithm_params = {
    "max_chain_depth": 3,                # 1, 2, or 3
    "K_first_push_expansions": 30,       # K₁: how many depth-0 transitions (with r=0)
                                          #     to expand into depth-1 chains
    "K_second_push_branches": 15,        # K₂: how many depth-1 children per expansion
    "K_third_push_branches": 10,         # K₃: how many depth-2 children per depth-1 r=0 node
    "seed": <int>,                       # per-env determinism; sampler derives from this
    "stop_on_reward": True,              # default: don't expand chains past r=1
    "log_sim_info": True,                # collision flags, push-terminated-early, etc.
}
```

Depth 0 is always exhaustive over R(s₀); no K knob for it.

These are passed through the existing `algorithm_params` plumbing (line 779) — already supported, no changes.

The worker passes `task_planner_config.algorithm_params` to `PlannerFactory.create_planner`, which calls `UniformRolloutSampler.__init__(env, config)`. The sampler reads `config.algorithm_params` in its `_initialize_algorithm()` method.

### 4.4 Argparse / CLI surface

Add to `modular_parallel_collection.py`'s argparse (under the existing planner-specific args block, around line 1080):

```python
# Uniform rollout sampler arguments
parser.add_argument("--sampler-max-chain-depth", type=int, default=3, choices=[1, 2, 3])
parser.add_argument("--sampler-k1-first-push-expansions", type=int, default=30,
                    help="K1: how many depth-0 r=0 transitions to expand into chains")
parser.add_argument("--sampler-k2-second-push-branches", type=int, default=15,
                    help="K2: how many depth-1 children per expansion")
parser.add_argument("--sampler-k3-third-push-branches", type=int, default=10,
                    help="K3: how many depth-2 children per depth-1 r=0 node")
parser.add_argument("--sampler-stop-on-reward", action=argparse.BooleanOptionalAction, default=True)
```

And in the algorithm_params construction (around line 1148):

```python
if args.algorithm == "uniform_rollout_sampler":
    algorithm_params["max_chain_depth"] = args.sampler_max_chain_depth
    algorithm_params["K_first_push_expansions"] = args.sampler_k1_first_push_expansions
    algorithm_params["K_second_push_branches"] = args.sampler_k2_second_push_branches
    algorithm_params["K_third_push_branches"] = args.sampler_k3_third_push_branches
    algorithm_params["stop_on_reward"] = args.sampler_stop_on_reward
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

### 4.6 Reused vs. new vs. modified

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
    "rollout_trace": [TransitionRecord, ...],   # flat, ordered by depth then transition_id
    "env_metadata": EnvMetadata,
    "sampler_config": {
        "K_per_depth": [...],
        "max_chain_depth": 3,
        "seed": ...,
        "sampler_version": "0.1",
    },
    "summary_stats": {
        "n_transitions": int,
        "n_depth_0": int,
        "n_depth_1": int,
        "n_depth_2": int,
        "n_r1_at_depth_0": int,
        "n_r1_at_depth_1": int,
        "n_r1_at_depth_2": int,
        "n_dead_ends": int,
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
    parent_id: Optional[int]              # None at depth 0; transition_id of parent otherwise
    depth: int                            # 0, 1, or 2

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

Per transition: roughly
- `state_*_qpos`: O(50 floats) × 2 = 800 B (~)
- `state_*_qvel`: O(50 floats) × 2 = 800 B (~)
- `state_*_se2`: ~200 B
- R lists: ~2 KB worst case
- per_neighbor_opening: ~100 B
- collision info / sim info: ~200 B
- **Total ~4 KB per transition, uncompressed.**

Per env with worst-case ~5500 transitions: ~22 MB uncompressed, ~5–8 MB compressed pickle.

For ~5,925 train envs: ~30–50 GB total. Manageable. If pkl size becomes operationally annoying, switch to a sharded zarr / HDF5 dataset in a follow-up — but the schema design is dataset-format-agnostic.

---

## 6. What this enables downstream

### 6.1 Spec A (1-push model training)

A 1-push model trains on the `depth=0` slice of `rollout_trace`. Per-env:
- Positives: every record with `r=1`.
- Negatives: every record with `r=0` and `sim_failure=False`.
- Per-neighbor positives: per-neighbor opening labels available for per-neighbor models.

The fresh data fixes the round-1 biased-teacher problem at its root — the depth-0 distribution is uniform over R(s₀) by construction (it's exhaustive). Any model trained on these labels is learning F as it actually is, not a planner's preferred slice.

### 6.2 Chain training (Design X or Y, later spec)

**Design X (generative on witnesses):** walk `rollout_trace`, find every `r=1` transition, trace back via `parent_id` to enumerate the witness chain. Every `(state_before, action)` on a successful chain is a positive witness. Train diffusion / flow / autoregressive on these.

**Design Y (Q-function):** every `TransitionRecord` is a Bellman tuple. Reward = `r`. Terminal flag = `(r == 1)`. `max_a' Q(s', a')` restricted to `R_at_state_after`. Train via TD or fitted-Q.

Both designs can be trained from the *same* `rollout_trace` records without going back to the cluster.

---

## 7. Compute budget and cluster strategy

### 7.1 Per-env sim count

With `K₁=30, K₂=15, K₃=10` and `max_chain_depth = 3`, worst case (every chain fails):
- Depth 0: |R(s₀)| ≤ 600 sims (exhaustive)
- Depth 1: K₁ × K₂ = 30 × 15 = 450 sims (each restored from saved s₁ via `set_full_state`, no first-push re-execution)
- Depth 2: K₁ × K₂ × K₃ = 30 × 15 × 10 = 4,500 sims (each restored from saved s₂)
- **Worst-case total: ~5,550 sims/env.**

Realistic average is lower because:
- Many envs are 1-push solvable; depth-0 chains stop on `r=1` and don't expand into depth-1/2. (`stop_on_reward=True`.)
- Many depth-1 chains succeed too, saving the depth-2 expansion.

Empirical estimate from the existing F-char distribution (most envs are easy / very_easy): ~1,500–2,500 sims/env average. ~3–4× the current 1-push collection compute.

### 7.2 Cluster strategy: sharded across hosts

The collection runs as **N independent shards** — one per cluster host (rlab5, rlab6, rlab7, etc.). Each shard processes a disjoint env range via `--start-idx` / `--end-idx`. Output goes to per-host subdirectories (`modular_data_<hostname>`) under a shared output root. This is exactly how the existing 1-push F-char data was collected (`/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab{5,6,7}/*.pkl`).

Concretely, three sbatch jobs (one per node) with non-overlapping ranges:

```
# rlab5: envs 0..1975
python modular_parallel_collection.py --algorithm uniform_rollout_sampler \
  --manifest <train_manifest.txt> --start-idx 0 --end-idx 1975 --workers 100 \
  --output-dir /common/users/.../uniform_rollout_train --sampler-max-chain-depth 3

# rlab6: envs 1975..3950   (analogous)
# rlab7: envs 3950..5925   (analogous)
```

Wall-clock estimate at ~0.1 s/sim, ~1500–2500 sims/env average, 100 workers per host: ~30–80 minutes per shard. With three hosts running in parallel, the full train pool collects in ~1 wall-clock hour. The test pool (1,767 envs) collects in proportionally less time on one host.

**Sharding is strictly faster** than a single job — more workers, more cores, no coordination overhead between shards (each shard writes independent pkls to its own per-host directory). The existing manifest mechanism (`--start-idx` / `--end-idx`) already enforces non-overlapping slices, so there is no risk of duplicate work.

### 7.3 Per-shard determinism

Each shard derives its RNG seeds from `seed_base + env_global_idx`, so the same env produces the same sampling regardless of which shard processes it. Reproducibility is unaffected by shard boundaries.

### 7.4 Run sequencing within a shard

Within a single shard, each worker processes its assigned envs sequentially. Per env, the worker runs depth 0 first (exhaustive), then expands depth-1 / depth-2 branches, then writes the per-env pkl when the entire env's collection is done. v0 writes one pkl per env on completion; if early consumption of depth-0 records becomes operationally important (e.g. to start 1-push training before chain collection finishes), a follow-up can split into `depth0` and `chain` sub-pkls without schema changes.

**Optional staged-run alternative:** run `max_chain_depth=1` shards first (~10× faster — only depth-0 sims), train 1-push model on depth-0 data, then run `max_chain_depth=3` shards. Redundantly re-collects depth-0 in the second run. Useful if de-risking the chain code is more valuable than total compute efficiency. Decide at plan time.

---

## 8. Testing strategy

### 8.1 Unit: uniformity of sampling

Fix an env with |R(s₀)| = 60. Run the sampler with `K_per_depth = [None, 10000, 0, 0]`, `max_chain_depth = 2`. Verify the empirical distribution of sampled depth-1 first-pushes is uniform over the depth-0 r=0 subset (chi-square test, p > 0.05).

### 8.2 Unit: reproducibility

Run the sampler twice on the same env with the same seed. Verify the transition records are bit-identical (modulo timing fields).

### 8.3 Unit: chain termination

Construct a synthetic env (or use a known 1-push-solvable env) where a known a₀ has `r=1`. Verify the sampler records that transition at depth 0 and does **not** generate any depth-1/2 transitions stemming from it.

### 8.4 Integration: F-char regression

Run the sampler with `max_chain_depth=1` on a 10-env manifest sample. Compare the resulting depth-0 records to the existing F-char pkls (`/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/`) for the same envs. The set of (object, edge, depth, r) tuples must match exactly (modulo any intentional schema differences, which are documented).

### 8.5 Integration: full pipeline on small manifest

Run `modular_parallel_collection.py --algorithm uniform_rollout_sampler --manifest <small_manifest.txt> --workers 4 --start-idx 0 --end-idx 10 --sampler-max-chain-depth 3`. Verify: all pkls written, schema loads without error, summary stats are consistent with worst-case bounds.

### 8.6 Sanity: Bellman convergence on a single env

For one env's depth-0 records, train a tiny tabular Q (one parameter per (s, a) tuple, only need ~600 params) via Bellman. With γ=0.9, after enough iterations, Q*(s₀, a) should be 1.0 for a ∈ F₁(s₀) and ≤ γ elsewhere. Confirms the reward signal is wired correctly.

---

## 9. Failure modes and error handling

### 9.1 Sim failures mid-transition

Caught at the `env.step()` call site. The transition is logged with `sim_failure=True`; the chain rooted at that transition is not extended. Other chains in the env continue normally.

### 9.2 Empty `R(s)` at some intermediate state

Defensively handled by skipping the expansion at that node and continuing. Logged in `env_metadata.summary_stats.n_dead_ends`. Rare but possible — e.g. after a push that traps the robot.

### 9.3 Worker crashes during env

The existing `modular_worker_process` exception path (line 632) catches and logs. The env's pkl is not written; the env is silently dropped from the dataset. Acceptable — re-run that env, or accept the loss (single env out of ~6000 is noise).

### 9.4 Memory pressure from large `rollout_trace` lists

Per env: ~5–8 MB compressed worst case. Per worker process: one env at a time, so memory footprint is bounded. No concern at expected cluster sizing.

### 9.5 RNG management

Each env's RNG is seeded by `seed_base + env_idx`, where `seed_base` is `args.seed` (or default 42). Within an env, the sampler uses one numpy `Generator` for all depth-≥1 draws. Reproducibility tested in §8.2.

---

## 10. What this spec does not cover

- **Spec A (1-push training pipeline).** Separate spec, blocks on the depth-0 portion of the data being on disk.
- **Chain-training pipeline (Design X or Y).** Separate spec, written after 1-push results inform the choice.
- **World-model training (Thread 2).** The schema supports it (full qpos at every transition), but training itself is out of scope.
- **Active-learning / model-guided augmentation collection.** Out of scope for v0. The current spec produces an unbiased baseline; biased augmentation is a v1 question after v0 results are in hand.
- **Schema evolution / format migration.** If the schema needs to change after data is collected, write a migration script. Don't try to pre-anticipate every possible future field.

---

## Open questions / risks

1. **Per-env runtime variance.** Hard envs with `K_per_depth = [None, 30, 15, 10]` and many failed chains could take longer than the 300s default `--search-timeout`. The sampler should not respect this timeout (it's a planner timeout, not a data-collection timeout) — confirm this is the case in plan-review, otherwise add a sampler-specific timeout knob.
2. **`set_full_state` correctness for branched sims.** The qvel-zeroing convention (line 114 of `rl_env.cpp` per memory) means `set_full_state(state_after)` produces a state with `qvel=0`. For chain expansion, this is correct (we want each new push to start from rest). Confirm with a smoke test.
3. **Disk-write granularity.** Currently the pkl is written at the end of all `episodes_per_env` for an env (line 697). With one env taking minutes, that's fine. If a worker crashes mid-env we lose partial data — acceptable per §9.3.

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
