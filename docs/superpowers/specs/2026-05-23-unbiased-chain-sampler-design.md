# Unbiased N-Push Random Sampler — Chain Extension of UniformRolloutSampler (v1)

**Date:** 2026-05-23
**Status:** Draft — design lock complete, ready for implementation
**Branch:** `unbiased-chain-sampler` (from `car-baseline`)
**Author session:** F-characterization v1 extension

---

## 1. Context

The May-2026 evaluation of the cropped diffusion model showed it loses to uniform-random-from-R at every difficulty stratum. v0 (`UniformRolloutSampler`, 2026-05-19 spec) collected exhaustive depth-0 data per scene to fix this for the 1-push problem. v0 explicitly deferred chain extension to a follow-up spec — this is that spec.

The driving insight: **the new sampler is the random analog of `PrimitiveGoalStrategy`.**
- `PrimitiveGoalStrategy` (region_opening's default): enumerate the cookbook → try primitives in `(depth, edge)` order.
- **This sampler:** enumerate the cookbook → try primitives in **uniform random order**.

Same role. Different selection mechanism. Everything else — per-(object, neighbor) attempt structure, stop on opening, AttemptResult output — is borrowed wholesale from region_opening's conventions so we plug into existing infrastructure without inventing a new schema.

---

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Collect unbiased n-push chain data per scene by running N independent random rollouts per (blocking object, target neighbor).
- **G2.** Match region_opening's per-(object, neighbor) AttemptResult output. No new worker branch.
- **G3.** Stop per env on first solution found per (object, neighbor) — collect up to S solutions, capped by N rollouts.
- **G4.** Backward-compatible with v0: existing 1-push readers (`rollout_trace_loader`, `batch_collection_classifier`) keep working unchanged.
- **G5.** Replace **all** other "random" code paths in the codebase with this single sampler (audit report from 2026-05-23 explorer agent identifies them).

### 2.2 Non-goals

- **Not a solver.** Doesn't replace region_opening for solution-finding workflows.
- **Not BFS- or tree-shaped.** Each rollout is an independent random walk from s₀.
- **Not multi-object.** Chains stay on the same object from first push to last (locked decision).
- **Not biased toward depth or ordering.** No PW-MCTS, no exploitation, no scoring of any kind.
- **Not a continuous-action sampler.** Cookbook-aligned primitives only.

---

## 3. The sampler — design

### 3.1 Name and role

Extends `UniformRolloutSampler` in-place. v0 (exhaustive depth-0) and v1 (chain) live in the same class. Mode is implicit: `max_chain_depth == 1` falls back to v0; `max_chain_depth > 1` invokes chain mode with N random rollouts.

Registered name unchanged: `"uniform_rollout_sampler"`. No new factory entry.

### 3.2 What it does, per env

```
For each immediate neighbor of robot's region R₀ that is the "goal" region:
  For each blocking object on the R₀ ↔ goal boundary:
    committed_object = this blocking object
    solutions_found = 0
    for rollout in range(N):
      restore env to s₀
      for step in range(n):
        candidates = [(edge, push_depth_idx) for primitives reachable from current state on committed_object]
        if not candidates: end rollout (object wedged)
        chosen = uniform_random(candidates)
        execute push, log transition
        if per_neighbor_opening["goal"] == True: solutions_found += 1, end rollout
        if sim_failure: end rollout
        # stuck does NOT end rollout
      if solutions_found >= S: break
    emit one AttemptResult for (committed_object, "goal")
```

Per env: one AttemptResult per (blocking object). All have `neighbour_region_label = "goal"`.

If the goal region is not an immediate neighbor of R₀ at s₀: skip this env (degenerate for our purposes — can't be solved by opening a single neighbor).

### 3.3 The sampling rule

At every step k of every rollout:
```python
candidates_k = [
    (edge, push_depth_idx)
    for (obj, edge, push_depth_idx) in enumerate_reachable_primitives(env)
    if obj == committed_object
]
chosen = candidates_k[rng.integers(len(candidates_k))]
```

Uniform over `(edge, push_depth_idx)` for the committed object. No edge bias. No depth bias.

### 3.4 Termination

**Per rollout** ends on any of:
1. `per_neighbor_opening["goal"] == True` (success).
2. `sim_failure == True`.
3. Step count reached n.
4. No reachable primitives for the committed object (object wedged).

**`stuck == True` does NOT end the rollout.** Log the stuck flag and continue with a new random primitive at the next step. (Rationale: stuck is a property of the (state, primitive) pair, not the state alone — a different primitive on the same object may still work.)

**Per (object, "goal")** ends on:
1. `solutions_found >= S`.
2. `rollouts_executed >= N`.

### 3.5 Budget knobs

```python
algorithm_params = {
    "max_chain_depth": int,                  # n; default 2
    "num_rollouts_per_env": int,             # N; default 10000
    "max_solutions_per_env": int,            # S; default 1
    "num_depths": int,                       # cookbook depth count; default 10
    "region_goal_samples_per_neighbor": int, # default 5
    "seed": int,                             # base seed; default 42
    "primitive_data_dir": str,               # default "data"
    "primitive_prefix": str,                 # default "" (point robot)
    "xml_file": str,                         # per-env, injected by collection script
    "env_idx": int,                          # NEW: per-env index for derived seeding
    "config_file_path": str,
}
```

Default operating point: **N=10000, n=2, S=1.** n=1 falls back to v0 exhaustive behavior.

### 3.6 Per-env determinism

v0 has a constant `seed` per worker — every env uses the same RNG seed (line 364 of uniform_rollout_sampler.py). This is a bug we fix in v1: the per-env seed is derived as `seed_base + env_idx`.

The collection script injects `env_idx` into `algorithm_params` per-task (mirror of how `xml_file` is injected today at line 786 of modular_parallel_collection.py).

---

## 4. Integration with existing data-collection infrastructure

### 4.1 Worker flow — zero control-flow changes

The `emits_attempt_results` branch at line 400 of `modular_parallel_collection.py` already handles `UniformRolloutSampler`. We populate the same `SamplerAttemptResult` fields it reads. No new worker branch, no new dispatch.

### 4.2 CLI additions to `modular_parallel_collection.py`

Around line 1097:

```python
# Relax the existing arg:
parser.add_argument("--sampler-max-chain-depth", type=int, default=1)
#                                                  ^^^^^^^^ remove choices=[1]

# Add:
parser.add_argument("--sampler-num-rollouts-per-env", type=int, default=10000,
                    help="N: number of random chains per (object, neighbor)")
parser.add_argument("--sampler-max-solutions-per-env", type=int, default=1,
                    help="S: stop per (object, neighbor) after collecting this many solutions")
```

Pack into `algorithm_params` (around line 1220):

```python
algorithm_params["num_rollouts_per_env"] = args.sampler_num_rollouts_per_env
algorithm_params["max_solutions_per_env"] = args.sampler_max_solutions_per_env
```

And inject `env_idx` per-task in `create_tasks()` (around line 786):

```python
task_algorithm_params['xml_file'] = xml_file
task_algorithm_params['env_idx'] = self.config.start_idx + i  # NEW
```

### 4.3 `sequential_ml_collection.py`

Has a duplicate worker branch (~lines 258-393, per audit). If used for collection, apply identical CLI + algorithm_params changes there. If unused, defer to Phase 5 cleanup.

### 4.4 Pkl output

Same path template, same `worker_result_data` shape. No new files. No new directories.

---

## 5. Data schema

### 5.1 AttemptResult shape

Per (committed_object, "goal"), the sampler emits one `SamplerAttemptResult` with the same field set as v0 / region_opening. Key fields:

| Field | v1 chain semantics |
|---|---|
| `success` | True iff any rollout solved within budget |
| `neighbour_region_label` | always `"goal"` |
| `chosen_object_id` | the committed object |
| `chosen_goal` | (x, y, θ) of the final push of the solving chain, or None |
| `goal_chain` | list of Goal-shaped entries for the solving chain (one per push), or empty |
| `chain_depth` | length of the solving chain (1..n), or 0 if no solution |
| `primitive_trial_log` | per-rollout depth-0-only entries (one per rollout's first push) — see §5.2 |
| `state_observations` / `post_action_state_observations` | per-push, solving chain only |
| `reachable_objects_before_action` / `reachable_objects_after_action` | per-push, solving chain only |
| `solutions_found_for_neighbour` | int |
| `solutions_total_for_neighbour` | same |
| `pushes_total_for_neighbour` | total push executions across all rollouts |
| `any_wall_collision` | aggregated across all pushes in the solving chain |
| `unique_movable_collision_count` | aggregated across all pushes |
| `failure_reason` | "success" / "no_reachable_primitives" / "all_pushes_failed" / "goal_not_immediate_neighbor" |

`SamplerAttemptResult` is reused as-is from `uniform_rollout_sampler.py:304-335`. No new fields required.

### 5.2 Schema disambiguation — the `depth` field (Phase 1)

**Critical bug we prevent up front.** Existing `primitive_trial_log` entries have a `depth` key that means **cookbook push_depth_idx (0-9)**. Naively dumping chain records into this list would conflict: a chain record's natural `depth` field means **chain step (0..n-1)**. Same key, incompatible semantics. Five consumers (rollout_trace_loader, batch_collection_classifier, ml_prediction_offline, batch_collection, sequential_ml_collection) silently corrupt training labels if both meanings get mixed.

**Resolution:**
- `primitive_trial_log` is **filtered to records where chain_depth == 0**. Its `depth` key always means cookbook push_depth_idx (0-9). Backward-compatible.
- Chain records (chain_depth ≥ 1) live in a new `algorithm_stats["rollout_trace"]` field with explicit `chain_depth` AND `push_depth_idx` keys. No collision possible.
- Defensive assertion in the shim: if any `primitive_trial_log` entry has `chain_depth > 0`, raise.

This is the entirety of Phase 1. ~30 lines in two files.

---

## 6. Phasing and shipping order

Strict ordering. Each phase is independent.

1. **Phase 1 — Schema disambiguation (standalone, no behavior change).**
   - Filter `primitive_trial_log` to chain_depth=0; add assertion.
   - Files: `modular_parallel_collection.py`, `sequential_ml_collection.py` (if used).
   - Land before any chain code touches files.

2. **Phase 2 — Chain sampler code.**
   - Extend `UniformRolloutSampler.search()` with chain mode.
   - Remove the `max_chain_depth != 1` guard at lines 353-357.
   - Add `_search_random_chains_for_object` helper.
   - Files: `python/namo/planners/sampling/uniform_rollout_sampler.py`.

3. **Phase 3 — CLI wiring.**
   - Add 2 new args, relax 1 existing arg, inject `env_idx`.
   - Files: `modular_parallel_collection.py` (and `sequential_ml_collection.py` if used).

4. **Phase 4 — Chain-aware downstream reader (optional).**
   - New file `python/namo/data_collection/chain_rollout_loader.py`.
   - Reads `algorithm_stats["rollout_trace"]`, reconstructs chains by `parent_id`, exposes per-rollout / per-depth analysis records.
   - Independent of Phase 2/3 — can land after pkls exist.

5. **Phase 5 — Cleanup (kill the deadwood).**
   - Delete: `RandomSamplingPlanner`, `RandomGoalStrategy`, `AdaptiveGoalStrategy`, `DiscretizedGridGoalStrategy`, `RandomRolloutGoalStrategy`, `EpsilonGreedyGoalStrategy`, `NoHeuristicStrategy`, `plan_with_random_sampling`.
   - Delete `region_opening.py:596-609` (the `random_rollout` / `random` dispatch branch — confirmed unused by any active YAML).
   - Remove imports from `modular_parallel_collection.py:42` and `visual_test_single.py:46`.
   - README updates.
   - Order: must come last, after Phase 2-3 ships and a real collection run validates the new code.

---

## 7. Compute budget

At MuJoCo's ~0.1 s/sim with defaults (N=10000, n=2, S=1):

| Env type | Solution rate per chain | Expected sims | Wall time |
|---|---|---|---|
| Easy | 10% | ~20 | 2 sec |
| Medium | 1% | ~200 | 20 sec |
| Hard | 0.1% | ~2,000 | 3.3 min |
| Very hard | 0.01% | ~20,000 | 33 min |
| Unsolvable in 2 pushes | 0% | 20,000 (full budget) | 33 min |

Per-env worst case is bounded: `N × n = 20,000` sims = ~33 min. Across 5,925 train envs with mixed difficulty, expect 10-30 GB of pkls and a few hours wall-clock on 100-300 parallel workers.

Sharding strategy unchanged from v0: per-host directories, non-overlapping `--start-idx` / `--end-idx`.

**No wall-clock timeout knob.** Budget is deterministic (N, n). Locked decision.

---

## 8. Testing strategy

1. **Reproducibility.** Same `(seed_base, env_idx)` → identical `rollout_trace` byte-for-byte (modulo timing fields).
2. **Uniformity sanity.** N=10000 rollouts at n=1 on a fixed env. Histogram over `(edge, push_depth_idx)` for the committed object. χ² test against uniform.
3. **Single-object invariant.** Every entry in any `goal_chain` shares the same object_id (assertion in test).
4. **v0 regression.** With n=1, the sampler's output matches v0 exhaustive output byte-for-byte (modulo timing).
5. **Pipeline integration.** `modular_parallel_collection.py --algorithm uniform_rollout_sampler --sampler-max-chain-depth 2 --sampler-num-rollouts-per-env 100 --start-idx 0 --end-idx 5 --workers 2` produces valid pkls.
6. **Downstream mask gen.** Chain pkls flow through `batch_collection.py` and produce NPZ files (the existing trajectory-suffix splitter handles multi-push chains).
7. **Phase 1 standalone.** With v0 records only, the shim filter is a no-op. With synthetic chain_depth=1 records present, the assertion fires.

Existing tests that must keep passing:
- `python/tests/test_uniform_rollout_sampler.py`
- `python/tests/test_uniform_rollout_sampler_integration.py`
- `python/tests/test_uniform_rollout_sampler_fchar_regression.py`
- `python/tests/test_rollout_trace_loader.py`

---

## 9. Failure modes

| Mode | Handling |
|---|---|
| No reachable primitives at s₀ | Skip env (same as v0). |
| Goal region not immediate neighbor of R₀ | Skip env, log `failure_reason="goal_not_immediate_neighbor"`. |
| No reachable primitives for committed object mid-chain | End this rollout. Log records up to this point. Next rollout. |
| Sim failure mid-chain | Log failed push with `sim_failure=True`. End rollout. |
| All N rollouts exhausted without solution | Emit AttemptResult with `success=False`, `failure_reason="all_pushes_failed"`. |
| Worker crash mid-env | v0's exception path at line 645 handles. Env's pkl is not written. |

---

## 10. What this spec does NOT cover

- **Multi-object chains.** Locked as non-goal. If ever needed, the schema field `goal_chain` would gain per-entry `object_id`, and the worker's `action_sequence` builder would need a small fix.
- **Per-env wall-clock timeout.** Budget is N×n. Hard cap by primitive count.
- **Active learning / model-guided sampling.** Pure uniform random only.
- **Importance sampling weights.** Not needed — sampling is uniform.
- **Continuous-action samplers.** Cookbook only.
- **Replacing region_opening as the solver.** This sampler is for data collection, not solving.

---

## Open risks

1. **Compute blowup for unsolvable envs.** Worst case 20k sims/env. Mitigated by parallel collection — typical envs are much cheaper.
2. **Singleton solutions missed ~64% of the time at N=10000.** Accepted as the unbiasedness/coverage trade-off.
3. **Stuck push at step 0 → chain continues with another random primitive (possibly same one).** Acceptable; uniform random tolerates the repeat probabilistically.
4. **`sequential_ml_collection.py` drift.** Duplicate worker branch. Phase 1 + 3 must touch both if used. Audit report flagged this.
5. **`env_metadata` and `sampler_summary_stats` keys are dropped by the worker** (v0 bug not strictly part of this spec). If chain runs need per-env aggregate stats in pkls, attach them per-attempt instead.

---

## Appendix: File inventory

**Modified:**
- `python/namo/planners/sampling/uniform_rollout_sampler.py` (Phase 2)
- `python/namo/data_collection/modular_parallel_collection.py` (Phase 1 + 3)
- `python/namo/data_collection/sequential_ml_collection.py` (Phase 1 + 3, if used)

**New:**
- `python/namo/data_collection/chain_rollout_loader.py` (Phase 4, optional)
- `tests/test_chain_sampler.py` (Phase 2)

**Deleted (Phase 5):**
- `python/namo/planners/sampling/random_sampling.py` (entire file)
- `python/namo/strategies/goal_selection_strategy.py` (entire file: RandomGoalStrategy, AdaptiveGoalStrategy, DiscretizedGridGoalStrategy)
- `RandomRolloutGoalStrategy` from `primitive_goal_strategy.py:362-431`
- `EpsilonGreedyGoalStrategy` from `ml_strategies.py:1021+`
- `NoHeuristicStrategy` from `object_selection_strategy.py:41+`
- `plan_with_random_sampling` convenience function
- `region_opening.py:596-609` (random_rollout dispatch branch)

**Untouched:**
- `python/namo/planners/connectivity_snapshot.py`
- The motion primitive database loader (shared utility)
- `python/namo/planners/opening/region_opening.py` (except for Phase 5 deletion of the random_rollout branch)
- `python/namo/visualization/mask_generation/batch_collection.py` (trajectory-suffix splitter already handles multi-push chains)
- `sage_learning/` (NPZ-only consumer, unaffected)

---

*End of Spec v1 draft.*
