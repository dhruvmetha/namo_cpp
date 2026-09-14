# NAMO Evaluation Logging and Independence Statistics Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task, after implementation is authorized. Steps use checkbox syntax. Do not launch agents, evaluations, or mining jobs merely because they appear in this document.

**Goal:** Collect reproducible Full-NAMO behavioral and keyhole-independence evidence on ordinary Amarel CPUs, then measure the same searches on exclusive Icelake nodes without collecting that behavioral evidence a second time.

**Architecture:** Add two independent measurement switches to the existing runners and canonical search: `record_statistics` and `record_timing`. Keep a small mandatory outcome/reproducibility record in every mode; behavioral records and committed-state checkpoints are optional, timing records are independently optional, and the twelve research statistics and downstream-access labels are computed offline. Reuse the existing Full-NAMO planner, best-first search, timed launcher, and fixed-route access-audit infrastructure.

**Tech Stack:** Existing Python search/runners, C++ region snapshots and pybind11 bindings, YAML configuration, JSONL and compressed state sidecars, standard-library analysis, and the existing pytest/simulator fixtures. No new service, database, tracing framework, model, or search algorithm.

**Engineering Standards:** Follow `plan-coding-standards`: efficient reuse, focused interfaces, documented public functions, repository naming conventions, named protocol values, specific errors, environment-provided paths, and small coherent commits. Extend existing tests with a few meaningful scenarios instead of generating a large suite. Markdown prose stays one paragraph per source line.

---

## 1. Status, scope, and decisions

This is a documentation-only implementation plan. Writing it does not implement the instrumentation, merge another worktree, alter physics, freeze environments, or authorize submitting jobs.

The requested collection workflow is:

| Run purpose | Hardware | `record_statistics` | `record_timing` | Outputs used for |
|---|---|---:|---:|---|
| Candidate Random evaluation, five seeds | Ordinary/non-Icelake Amarel `main` CPUs | `true` | `false` | Difficulty labels, outcomes, calls, planner behavior, saved states for independence |
| Missing method/seed behavioral coverage on the selected environments | Ordinary/non-Icelake Amarel `main` CPUs | `true` | `false` | Behavioral statistics for the actual model/baseline being reported |
| Final frozen-set reruns | Exclusive Icelake Amarel `main` nodes | `false` | `true` | Wall time and component times for the same scene/method/seed executions |
| Minimal execution or reproduction | Either, not a timing result | `false` | `false` | Outcome, calls, provenance, reproducibility fingerprint only |
| Instrumentation development check | Small controlled sample | `true` | `true` | Validate schemas and timing accounting; not the final timing campaign |

There is no requirement to run both measurement modes together. Independence measurements belong to the non-timing statistics workflow. Final Icelake jobs must not construct independence references, export region maps, save full state histories, or compute research metrics.

Important dependency: five Random candidate runs do not supply model behavioral statistics. Before a timing-only model rerun can be joined to behavioral evidence, the same model checkpoint, execution mode, scene, and evaluation seed must have an untimed statistics run. Do those missing runs only on the eventual selected subset; do not evaluate every model on the entire candidate pool unnecessarily.

### 1.1 What must not change

- Measurement switches must not change the ranker, model inputs, scores, candidate enumeration/order, random-number consumption, heap policy, pruning, state restoration, push acceptance, or goal predicate. The explicit one-keyhole boundary-pool requirement in Section 1.3 is a separate evaluation-scope correction if an old wrapper still restricts the search to one object; establish that scope before recording the new baseline.
- Full NAMO still computes a graph, attempts an opening, commits the returned action/state when appropriate, recomputes, and continues. An unchanged or increased hop count is an observation, not a failure or rejection condition.
- Keep goal-clearance planning and robot-wall rejection enabled for the agreed campaign; keep the existing 150-physics-tick collision-check cadence. Record the resolved configuration rather than overriding it in logging code.
- Preserve the original end-goal success criterion. Goal clearance is distinct from real-robot goal-cell expansion/retargeting; this plan does not enable the latter in simulation.
- Do not enable a next-keyhole-preservation acceptance filter. Access deterioration is something to measure, not a reason to reject the scene, push, or solution.
- The approved removed-K1 shortcut for K2 horizon certification remains valid as that certification protocol. It is not replaced by a post-K1 horizon test.
- No mining is included. Reuse candidate pools, refresh labels under the unified simulator, and freeze after coverage is known.
- No per-physics-tick logging, full heap dumps, per-candidate rendered grids, or new exhaustive searches for these statistics.

### 1.2 Benchmark protocol to record, not silently redefine

| Population | Shared evaluation call cap | Search horizon | Difficulty |
|---|---:|---|---|
| One-keyhole, including both certified 1-push and 2-push episodes | 3,000 | `hmax=2` | Expected trials `E=(N+1)/(S+1)`: easy `E <= 3`; medium `3 < E <= 15`; hard `E > 15` |
| Two-keyhole Full NAMO | 9,000 for the complete problem | Local `hmax=2` | Five-seed Random median: easy `<30`; medium `30..299`; hard `>=300` or a censored median with at least one success; unresolved only `0/5` successes |

For one-keyhole 1-push episodes, `S` counts direct opening primitives; for genuine 2-push episodes it counts verified setup primitives, not complete two-action search costs. `N` is the number of initially reachable primitives over the complete allowed boundary-object pool in the exact label protocol. Both `N` and `S` must refer to the same pooled scope as evaluation; Section 1.3 specifies when old single-object labels are insufficient. Carry those label/certification references; do not derive exact `N/S` or genuine horizon from five stochastic evaluations.

For Full NAMO, a failure receives infinite success cost for medians even if it terminates early. Store finite consumed calls separately. Success at the configured cap is included. Technical errors and missing jobs are not valid failed seeds and must be repaired before assigning a five-seed label.

The current target sizes are 100 easy, 100 medium, and 100 hard one-keyhole episodes, and 100 of each Full-NAMO difficulty with up to 100 unresolved, emphasizing available wall-template diversity. The logging implementation does not select or freeze these rows.

### 1.3 One-keyhole evaluation must include all eligible boundary objects

**Additional user requirement:** One keyhole means opening the selected robot-to-goal-region boundary, not being restricted to one donor/manifest object. If several objects are boundary objects for that same target region, the existing local search must be allowed to evaluate their pushes in one pool. It must not expand to arbitrary reachable objects elsewhere in the room, or start solving a different region-opening problem.

The shared engine already supports a collection in `restrict_obj`: `rank_first_pushes_h2` filters reachable objects against that collection, scores each applicable object, and pools its primitives. `BestFirstRegionOpeningPlanner._run_boundary` already obtains the boundary's objects, checks the requested pool, and supplies the common target samples. Full NAMO already passes a boundary pool. However, in the inspected dhruv checkout, `scripts/sandbox/eval_bestfirst.py:279` reads a single manifest `object_id` and `:294` calls the engine with `restrict_obj=obj`. Existing pooled engine support alone therefore does not establish that historical one-keyhole evaluations used the pooled problem. Recheck the unified checkout rather than assuming this inspected wrapper is its final version.

The new evaluation contract is:

- Resolve the actual fixed target region and its adjacent robot-region boundary under the pinned simulator/configuration. Reuse the existing boundary-object resolution/validation and local search; do not infer the pool from every object bordering any region or from XML object order.
- Pass the entire eligible boundary pool to every evaluated method. Keep the pool fixed for that local invocation as the existing pooled engine does; recompute reachable actions from each search state. A boundary object that is unreachable initially can become usable after a setup push, so do not permanently remove it from the allowed pool based on initial reachability.
- Score/rank all currently reachable primitives across that pool with the existing method. Uniform means the existing uniform primitive ranking over this pool, not choosing an object uniformly first. Keep the same target samples and success predicate for all objects and all methods.
- A two-push opening may push A then B, provided both belong to the permitted boundary pool and the chain opens the same fixed target. It is still one keyhole, not two hops. Preserve the native distinction between alternative blockers and a joint plug; multiple eligible objects does not mean all must be moved.
- Keep `hmax=2` and one shared 3,000-call evaluation cap for the whole local problem, not a cap per object. The old “at most 300 one-push options” statement applies to one object's primitive table; pooled root candidates sum over all eligible objects and can exceed 300 before reachability filtering.
- Record `boundary_objects`, `object_scope="boundary_pool"`, the fixed target definition, reachable candidate counts per object, actual simulated objects, and committed objects. Include object scope/pool in problem/protocol identity, independently of optional detailed statistics.

**Re-evaluation implications:** Inventory candidate episodes for which the actual boundary pool differs from the historical single-object restriction. A nominal 2-push episode may become 1-push if another allowed blocker has a direct opener. A genuine pooled 2-push certificate requires exhausting all reachable one-push primitives across the allowed pool, plus a valid two-push witness, potentially using different objects. To label difficulty, use pooled `N`; direct-opener `S` counts all valid pooled one-push actions, and setup `S` counts first actions with at least one permitted finishing action on any allowed object. Do not just sum historical same-object setup counts because that misses cross-object finishing chains.

Keep the agreed cutoffs `E <= 3`, `3 < E <= 15`, and `E > 15`, but refresh the underlying exact label evidence for affected pooled episodes before freezing. If exact pooled evidence is missing, mark certification/difficulty pending rather than borrowing a donor label. This is evaluation certification, not an instruction to create exhaustive training data. Five Random runs measure search performance; they do not certify the absence of a one-push solution.

Reuse old results only when the complete allowed pool, target definition, simulator/configuration, seed semantics, and search protocol match. Pool expansion invalidates comparisons with the old single-object outcome, labels, and timings. Fold the corrected scope into the already planned five-seed candidate refresh, then run the selected model/baseline methods under that same scope; do not add a redundant separate campaign. If a source pool has only one boundary object, no object-scope change is needed, although the planned physics refresh still applies.

Legacy object-specific rows remain identifiable as source records. When several rows become the same pooled robot-to-target-region task, evaluate that pooled task once and keep all source links. Do not deduplicate different target regions, fixed target sample definitions, or starts merely because they share an XML. This extends the old per-object episode identity rather than silently rewriting historical results.

## 2. Code inspected and reuse points

The dhruv-linux source inspected for this document is `/home/dhruv/projects_dhruv/namo/namo_cpp`, commit `0920eee91ed4a9ec9f4a1183c0eca67019193d9d`. The aligned Amarel worktree inspected is `/scratch/tdn39/full_namo_dhruv_aligned_20260911_v1/code`, commit `1fa32fd68d7dac75e5401293bb770f5c45eeeb34`; it contains timing/runner integration that is absent from the inspected dhruv main checkout. The implementation must use the user's unified build and inspect its resolved versions, not overwrite it with either historical checkout.

| Existing file | Relevant behavior | Planned change |
|---|---|---|
| `python/namo/planners/full_namo/full_namo_planner.py` | Route choices, pooled boundary search, committed actions, iteration trace, goal clearance | Add passive attempt/decision/state observations at existing control-flow points; gate optional trace collection |
| `python/namo/planners/opening/best_first_region_opening.py` | Adapts the canonical local engine and charges the shared push budget | Pass measurement handles and return local counters/raw termination cause without duplicating charges |
| `python/namo/planners/opening/best_first_search.py` | Canonical best-first and simulated greedy execution; used by one-keyhole too | Optional primitive/board counters and streaming execution digest; explicitly optional timers |
| `python/namo/solvability_runner.py` | Task configuration, planner invocation, solved/unsolved rows | Wire both switches; always emit the minimal outcome; write requested sidecars for successes and failures |
| `python/namo/planners/__init__.py` | Rebuilds the native snapshot dictionary field by field | Explicitly pass through optional native timing and region-cell evidence |
| `python/namo/cpp_bindings/rl_env.hpp`, `rl_env.cpp`, `bindings.cpp` | Computes grid/components, connectivity graph, goal-clearance access, sampled goals | Optional phase timers and export of already-computed component cells |
| `scripts/sandbox/eval_bestfirst.py` | Existing one-keyhole evaluation entrypoint importing canonical search; inspected version still restricts to one manifest object | Verify/use the existing full boundary pool and fixed target definition, then wire measurement configuration and pooled episode/run identity |
| `scripts/pipeline/eval_full_namo_walltime.py` | Existing aligned Full-NAMO campaign driver | Keep strict Icelake validation; add timing-only mode and paired-run checks; replace old campaign-specific 20k assumptions with explicit new protocol |
| `scripts/slurm/eval_walltime.slurm` | Exclusive node and thread-pool controls | Reuse for the final campaign; do not repurpose it as the ordinary candidate launcher |
| `scripts/pipeline/aggregate_multihop_solvability.py` | Existing solved/unsolved aggregation | Extend its new-schema path to preserve all run identities and compute behavioral summaries; do not collapse five seeds by XML |
| `scripts/pipeline/compare_multihop_rankers.py` | Existing paired reporting | Consume explicit problem/run keys and censored all-run costs for new-schema results |

The old aggregate implementation keys rows by `xml_path` and preferentially keeps a solved row. That is unsuitable for combining methods or five seeds and would erase failures. Fix this in the existing aggregator for the new schema; do not create another almost-identical evaluation pipeline.

### 2.1 Existing independence implementation

Reuse the separate dhruv worktree `.worktrees/keyhole-access-audit-20260911` and its files:

- `python/namo/planners/full_namo/access_audit.py`: fixed reference, contact subset comparator, pose diagnostics.
- `scripts/pipeline/audit_keyhole_access.py`: offline evidence inventory, replay/reference construction, coverage and recovery requests.
- `scripts/pipeline/keyhole_access_stats.py`: five-seed access labels and censored subgroup summaries.
- `config/analysis/keyhole_access.yaml` and their existing tests.

Its measurement contract is useful, but its old population/reporting assumptions are not the new campaign: 624 scenes, S2-only model summaries, historical cost records, and a 20,000-call cap must not leak into the new S1/current-runtime/9,000-call results. Reuse or integrate only the needed files after checking which have already been merged. Keep the old retrospective analysis reproducible as a separate configured input, not overwritten.

### 2.2 Three important implementation traps

First, Full NAMO now sends the eligible blocker pool for a boundary to the local engine. `chosen_initial_blocker` is the scheduler's representative, not proof that this object was the one simulated or committed. Save the offered pool, actual simulation object sequence, and committed action objects separately.

Second, the local engine currently calls `perf_counter()` even when no timing dictionary is supplied. Merely dropping timing fields from output does not turn timing collection off. Guard the clock reads and accumulation, including legacy `search_time_ms` measurements that are only statistics.

Third, current `t_score` wraps the complete candidate-ranking pass, including reachability/enumeration. It is not pure neural inference time. Keep its legacy meaning and add a separately named model-scoring-pipeline duration; do not silently rename an existing column's meaning.

## 3. Two independent switches and minimal always-on data

Use one YAML measurement section, parsed by the runner and forwarded without inventing a second set of planner options:

```yaml
measurement:
  schema_version: 1
  record_statistics: true
  record_timing: false
```

For the final Icelake campaign, invert the two booleans. Both false and both true are valid. New campaign configs must state the pair explicitly. Keep machine paths, output roots, Python/environment locations, and checkpoint locations in the existing campaign environment/configuration files, not in source.

Add `record_statistics: bool = False` to `SolveTask` and use the aligned runner's existing `record_timing` field. Forward a shared measurement object through `PlannerConfig.algorithm_params["search_measurements"]`; it contains optional statistics and timing sinks plus the minimal execution fingerprint. Adapt the existing `full_namo_timing` dictionary plumbing in place so there is one accumulator, not two competing timer paths.

When statistics are off, do not create the expanded `iteration_trace`, route-choice lists, region maps, state history, object-access audits, or behavioral-summary accumulators. Required planner state such as attempt penalties, blacklists, and the push budget still exists because planning needs it. Do not remove those to disable reporting. Legacy detailed traces remain available only when explicitly requested by the new statistics configuration; document this output-schema change to callers.

When timing is off, do not read clocks for measurement or emit elapsed-time/component-time fields. An operational job timestamp is provenance, not a benchmark duration. Do not disable a functional timeout used elsewhere in the system; no new wall-time stopping condition is introduced here.

Always retain the following minimal record, even when both switches are off:

| Group | Required contents |
|---|---|
| Identity | `schema_version`, `problem_id`, `run_id`, source manifest/row identity, scene/XML hash, method, model training seed/checkpoint hash when applicable, evaluation/random seed, protocol hash |
| Outcome | `solved`, original-goal result for Full NAMO, exact termination kind/subkind, `total_calls`, configured call cap, success-call count or null, technical/complete status |
| Reproducibility | Source/binding/configuration/primitive fingerprints, initialization fingerprint, execution digest, terminal committed-state digest, attempted/committed action counts |
| Collection | Statistics/timing flags and sidecar references/hashes when present; hardware/job metadata and timing-campaign identity where relevant |

This exception to “other statistics off” is intentional: these are the outcome and evidence needed to identify and validate the run, not the twelve behavioral analyses. A timing result without its scene, method, seed, success, and call count is not usable.

### 3.1 Stable identities

`problem_id` identifies the actual task, not a filename or wall template. For new one-keyhole evaluations it includes the scene/initialized robot state, fixed target-region/sample definition, and complete boundary-object pool with `object_scope="boundary_pool"`; a legacy donor `object_id` is provenance, not the sole task identity. For Full NAMO it includes the scene plus initialized robot start and original goal. Multiple start/goal pairs in the same room are different tasks. Use existing geometry identity for leakage checks, but do not confuse geometry identity with problem identity.

`run_id = SHA256(canonical_json({problem_id, semantic_protocol_hash, method, checkpoint_hash, seeds}))`. The semantic protocol covers all behavior-affecting values: simulator/configuration, primitive table, goal definition, clearance, collision policy, search/pruning/budgets, initialization, and model/scorer versions. Do not include output paths, sharding, difficulty labels, measurement booleans, or hostname; those differ between the paired stages without defining a new search.

Record the complete source revision and runtime artifact hashes separately as well. The candidate and timed runs must use the same pinned algorithm/simulator artifacts and resolved semantic configuration. A different manifest containing a selected subset is allowed only through an explicit row-identity mapping; it must not change `problem_id` or the run's seed.

Use explicit seed values per problem/run, not array index or “number of scenes processed so far.” Preserve existing seed semantics rather than changing how the canonical engine reseeds each local invocation. Record sampler seed and search/shuffle seed separately; the five Random search seeds are `7000, 8000, 9000, 10000, 11000` unless the campaign explicitly changes them.

## 4. Behavioral record schema

Use three record types, not a new collection of per-metric loggers: a decision/local-attempt record, a committed-state/graph checkpoint, and the terminal run record. Records are indexed by monotonically increasing IDs. The existing trace is the integration point.

### 4.1 Decision and local-attempt record

Emit one decision record for each high-level route-selection decision, including no-route termination. A decision that invokes the local opener receives an `attempt_id`; a no-route or already-at-goal decision has `attempt_id: null`. A one-keyhole evaluation has one local attempt and no high-level routing record.

| Fields | Definition |
|---|---|
| `decision_id`, `attempt_id`, `previous_attempt_id` | Run-local IDs; all invocations, including zero-simulation outcomes, receive consecutive attempt IDs |
| `scene_version_before`, `scene_version_after`, `snapshot_id` | Links to committed physical state and the graph used for selection |
| `task_kind` | `boundary_opening` or `goal_clearance`; do not infer from path length |
| `selected_route` | Complete ordered region-label path, scoped to `snapshot_id` |
| `selected_boundary`, `target_region`, `target_reference` | Current boundary and the actual target definition used by the opener; goal-clearance target recorded separately |
| `representative_object`, `offered_objects` | Scheduler representative versus the full, sorted eligible object pool actually passed to the engine |
| `route_choices` | Already-computed pre-blacklist and eligible choices, each with path, boundary, blocker, graph hops, scheduling hops, attempt penalty, total cost, goal-clearance/reaches-goal status, and its existing tie-break fields |
| `blocked_choices`, `attempt_penalties_before/after`, `penalty_reset_reason` | State-local exclusions and existing scheduling history; reasons distinguish hop reduction from pruning vanished objects |
| `calls_before`, `calls_after`, `local_calls`, `local_call_limit` | Exact shared-budget boundaries and allowance actually passed to the local engine |
| `root_candidates_by_object`, `simulations_by_object`, `simulations_by_chain_depth` | Counts from candidate lists and actual simulator invocations already present in the local engine |
| `simulated_object_runs` | Run-length encoding of actual simulated object IDs in order, plus first/last object; no full candidate-grid dump |
| `local_end`, `local_success`, `failure_kind/subkind` | Raw canonical end (`solved`, `exhausted`, `budget`) plus adapter/high-level reason and technical error where applicable |
| `committed_action_range`, `opening_reported`, `final_goal_reachable` | Committed action indices, local predicate result, and global original-goal result when the planner actually checked it |
| `hop_observation_before/after` | Minimum unpenalized feasible graph hops, selected-path hops, destination kind, path existence, and any explicit goal-clearance step; unknown post-state values stay null |

Do not record all simple graph paths. The planner currently uses shortest paths constrained by each first hop and produces route/blocker choices from these. Capture the choices it actually computed. Save the graph as well so an analyst can derive additional alternatives offline without implying that the planner enumerated them.

Count simulator calls at actual `env.step(...)` invocations only. Pruned or unreachable primitives do not count; rejected simulator pushes do. `Action.depth` is a primitive duration index, not local chain depth. Use the engine's board/chain depth for the latter.

Root candidate counts must be taken before search pruning/order changes from the existing candidate pool, not from a second enumeration. They are descriptive runtime counts, not automatically an exhaustive certification of a gate.

### 4.2 Committed-state and graph checkpoint

Save the initial state, each committed transition's before/after linkage, and the terminal committed state, including unsuccessful runs. A rejected speculative rollout is not a committed state. A simulated policy may commit a push that has not opened a region; that still produces a checkpoint.

`scene_version` advances only when committed physical state changes. Maintain a separate `commit_id` for every commit event, including a possible no-op. Store before/after state digests so a no-op can be recognized. Search backtracking does not advance either committed-state history or physical-push count.

Each state checkpoint contains `qpos`, `qvel`, robot pose, all movable-object poses with stable object IDs, its triggering attempt/commit IDs, and the action-prefix range. Store each unique committed state once in a compressed sidecar; attempts reference it. Keep exact values and derive motion tolerances offline, rather than rounding the saved state.

`RLState` is currently `qpos/qvel`; canonical `set_full_state` zeroes velocity and follows the unified reset contract. Record that restore contract and the model/XML/configuration fingerprints. Do not describe this as a complete arbitrary MuJoCo integration-state checkpoint. Access reconstruction uses the canonical restored decision state; it must not be silently substituted for an unrecorded mid-trajectory state.

Each graph checkpoint contains the robot and goal labels, adjacency, full edge-to-object sets, `multi_object_edges`, fixed goal samples/definition used by decisions, goal-free/blocked information already returned by the snapshot, and the region-cell map. `multi_object_edges` distinguishes a joint plug from alternative single objects that individually connect the same regions; a candidate pool of two objects alone does not prove both must be manipulated.

The region-cell map must come from the native components already computed for that snapshot. Record grid dimensions, resolution, coordinate convention/origin, integer-region-to-label mapping, and compressed cells. Do not reconstruct it with the Python visualization exporter inside search. That exporter can restore/recompute state and is not a passive observer.

Regions are snapshot-local labels. Across snapshots, compute an overlap table of old and new region cells offline. Keep one-to-many and many-to-one relations for splits/merges; do not turn an ambiguous overlap into a fictional persistent `region_3`. Store exact overlap counts, and report route correspondence as confirmed, changed, or ambiguous under a versioned analysis rule.

For initial-route compatibility, a later route is not a departure merely because merged regions shorten its label sequence. Collapse initial-route regions that now share a connected component and compare the surviving route in order. New off-route regions/blockers are evidence of an alternative only when correspondence is unambiguous. This establishes selected region-route departure, not the exact continuous navigation trajectory.

### 4.3 Commit actions and failed-run prefixes

Store every committed action with `commit_id`, attempt ID, stable object ID, primitive edge/duration index, target pose, and existing simulator rejection/contact information when it is already returned. If a search returns a two-action chain, save both actions even when Full NAMO adopts only the final saved state.

When the engine already retains intermediate chain states, link them for offline replay verification. Do not replay winning pushes in the live environment to create logging records. If only the chain's end state is available, the audit can replay that chain in an isolated environment and must record the weaker intermediate-state verification coverage.

Do not condition state/action export on `goal_clearance=True`, as the old timed runner does. The statistics switch governs export for every method and both successful and failed runs. A failed task can have a valid first opening and is essential to the independence analysis.

## 5. Timing schema and timing-only execution

Time the canonical planner call, not a replay of its saved successful action sequence. A replay does not measure search, failed candidates, or model ranking cost. Icelake reruns repeat the full seeded search with the behavior sink disabled.

### 5.1 Timer boundaries

Use `time.perf_counter()` in Python and `std::chrono::steady_clock` for native durations; never compare their absolute clock origins. Convert durations to seconds. Timer instrumentation may not call the simulator, query a new wavefront, score another object, or sample a point.

| Field | Boundary / interpretation |
|---|---|
| `t_wall` | Existing canonical outer `planner.search(...)` envelope; setup/model load/warmup and output serialization outside it |
| `time_until_success` | The run's canonical completion duration if solved, otherwise null; never substitute early-failure elapsed time |
| `t_local_search` | Each whole local opener invocation, including its adapter work, ranking, verification, and simulator calls |
| `t_sim` | Sum of existing `env.step(...)` durations, including the work inside that simulator API; not claimed to be pure `mj_step` time |
| `t_rank` / legacy `t_score` | Entire candidate-construction/ranking call; retain legacy meaning |
| `t_model_score` | Actual `scorer.score_state(...)` pipeline calls inside model ranking; includes that function's preprocessing/rendering/inference, not claimed to be only the NN forward |
| `n_rank`, `n_model_score` | Ranking passes and scorer calls respectively; one pooled pass may score several objects |
| `t_local_verify` | Existing local opening/goal checks after simulator calls; separate from `t_sim` and candidate ranking |
| `t_global_snapshot` | Existing high-level native snapshot call plus wrapper; do not include local snapshots in this global column |
| `snapshot_phases` | Native grid/components, connectivity-graph construction, goal-access analysis, and goal sampling, nested inside the owning snapshot |
| `t_global_goal_check` | Existing standalone global end-goal reachability checks, excluding checks already owned by a snapshot/local-search interval |
| `t_route_selection` | Existing route enumeration, penalty handling, blacklist filtering, and selection; no extra path enumeration |
| `t_global_other` | Outer planner time not assigned to non-overlapping global components; keep raw residual and flag materially negative accounting |

`t_local_search`, `t_global_snapshot`, standalone global goal checks, route selection, and global residual partition the outer work. `t_sim`, `t_rank`, `t_model_score`, and local verification explain local-search work and are nested, not additional top-level costs. In particular, do not add `t_model_score` to `t_rank`, or add native snapshot phase totals on top of `t_global_snapshot`.

Native graph construction internally performs reachability work. Report it under graph construction. The separate native grid/components and goal-access fields explain other reachability costs; they do not magically split every BFS from its owning graph algorithm. This is an explicit operational decomposition rather than a claim to have isolated every operation by theoretical category.

Store one timing row per local attempt: `run_id`, `attempt_id`, `calls_before/after`, elapsed offset at invocation start/end, duration, and component durations. IDs and call boundaries are alignment keys, not duplicated behavioral statistics. Include zero-call attempts and termination spans. Save no object-access profiles or full route records in timing-only mode.

All measured work stays inside the same outer envelope across arms. Do not subtract logging overhead, final checks, or “uninteresting” planner work after the fact. With both switches true, statistics collection adds overhead; mark that mode as development-only for wall-time reporting.

### 5.2 Turning timers completely off

Add small shared timing helpers in `python/namo/planners/search_measurements.py`:

```python
from time import perf_counter


def clock_start(timing):
    """Read the measurement clock only when a timing sink is enabled."""
    return perf_counter() if timing is not None else None


def clock_finish(timing, key, started):
    """Accumulate an existing operation's duration; disabled means no clock read."""
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + perf_counter() - started
```

Replace the unconditional clock operations around existing code, keeping the operation itself exactly once. For example, the existing best-first simulator block becomes:

```python
started = clock_start(timing)
step_res = env.step(make_action(it["obj"], it["g"]))
sims += 1
clock_finish(timing, "t_sim", started)
```

Use one fresh local timing dictionary per opener invocation, sum it into the run accumulator, and retain its keyed timing row. Do not let the local engine's existing `t_wall` assignment overwrite the full-problem timer. Apply the same guarded mechanism to the greedy simulated path, root/child candidate ranking, scorer calls, and measurement-only adapter timers.

### 5.3 Native snapshot additions

Append default-false options to `RLEnvironment::get_region_snapshot`, its pybind binding, and the Python wrapper: `include_region_cells` and `record_timing`. Existing positional parameters retain their meaning. Request cells only for statistics checkpoints, and timings only for measured snapshots.

Add optional snapshot payloads, with these defined shapes:

```text
region_cells:
  width: integer
  height: integer
  resolution_m: number
  origin_xy: [number, number]
  coordinate_convention: "native_grid_index"
  cells_by_region_id: {integer: [[x_index, y_index], ...]}
snapshot_timing:
  t_grid_components: seconds
  t_connectivity_graph: seconds
  t_goal_access: seconds
  t_goal_sampling: seconds
```

Use the return value of the existing `grid.find_connected_components(robot_xy, goal_cells)` for the cell export; do not call it twice and do not add a second graph build. Its map is already the requested region-to-cell relation. Save only when requested, compress in the output layer, and use `grid.get_grid_width()`, `get_grid_height()`, `get_resolution()`, and `grid_to_world_x/y(0)` for coordinate metadata.

Wrap each existing native phase with guarded `steady_clock::now()` calls and accumulate disjoint durations into the four fields. Include construction of `WavefrontGrid` in `t_grid_components`. Export an empty/missing timing payload when disabled, not fabricated zeros. Preserve the original ordering, sampled-goal seed, and goal-clearance behavior.

Forward both payloads explicitly through `bindings.cpp` and `python/namo/planners/__init__.py`; that wrapper will otherwise discard them. Rebuild the bindings once after this stage. No changes to wavefront occupancy rules, inflation, connected-component construction, or goal tolerances are part of the stage.

## 6. Deterministic join between untimed and timed runs

Seeds are necessary, not sufficient evidence that two executions match. Pin code, simulator/binding, linked runtime, configuration, primitive tables, model/scorer, initialized state, seeds, model evaluation mode, thread settings, and tie-breaking behavior. For model behavioral runs intended to pair with CPU-only Icelake timing, use the same CPU execution path on ordinary CPU nodes. Do not assume a GPU/CPU score ordering will be identical.

The mandatory execution fingerprint is a streaming digest, not a retained rollout trace. Update it from data already available at these points:

1. Route selection: selected task, representative, offered pool, selected path, and effective local budget.
2. Actual local simulator invocation: attempt ID, board/parent-chain identity, chain depth, object, primitive indices/target pose, and observed acceptance/opening/rejection result.
3. Local completion: raw termination, exact calls, returned action chain.
4. Commit: action chain and exact committed-state digest.
5. Run end: outcome, goal result, calls, terminal committed-state digest.

Use canonical serialization with sorted mapping keys and ordered event sequences; encode floating-point fingerprint values without lossy display rounding. Do not hash timing values, hardware names, dictionary iteration accidents, analysis labels, or optional telemetry. Keep per-attempt digests as small alignment data in addition to the whole-run digest. The common fingerprint cost is present in both modes and every timed arm.

This verifies discrete search decisions, attempted actions, outcomes, and recorded committed states. It does not claim to save or prove bitwise equality of every unrecorded physics tick.

The join requires exact equality for `run_id`, semantic/runtime fingerprints, initialized state, total calls, outcome, attempt count and per-attempt digest, full execution digest, and terminal-state digest. Only then attach the Icelake timing rows to the untimed attempt records by `attempt_id` and verify equal call intervals.

If any check fails, preserve both originals and emit a `pairing_mismatch` row with the first differing field/attempt. Do not combine the previous behavioral statistics with the new timing as if they describe the same execution. Investigate that run; if necessary collect a separate untimed statistics reproduction on the matching execution hardware. This is targeted recovery, not automatic relabeling or a new whole-population rerun.

Changing only the measurement flags must not alter any digest. A source/simulator/configuration change after candidate labeling creates a new protocol and requires refreshed compatible behavioral results, not a permissive hash bypass.

## 7. Independence statistics: complete measurement contract

“Independent” needs a named operational definition. The primary statistic here is **fixed-route downstream contact-access preservation**, using the already implemented audit. It is not proof that the keyholes never interact mechanically or that every possible K2 solution is unchanged.

### 7.1 Keep certification and run-time effects separate

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Exhaustive reachable one-push checks plus a valid two-push witness | A gate is genuinely two-push under the recorded primitive/configuration/certification state | The gate remains two-push after an arbitrary K1 opening |
| K2 checked with K1 removed | K2's independent horizon under the approved shortcut | Preservation of K2 in the executed complete scene |
| Contact/action-access comparison after K1 | How one observed opening changes access to the fixed downstream objects | Unchanged finishing feasibility or exact solution cost |
| Full-NAMO outcome | Whether the complete planner eventually reaches the original goal | That it followed the initial route or maintained K2 access at the first opening |

Carry `gate_pattern`, certification status, certification artifact hash, K1/K2 object sets, certification configuration/runtime, reachable one-push coverage, and a witness reference where they already exist. Unknown/incomplete certification remains unknown; do not assign `11`, `12`, `21`, or `22` from a returned chain length alone. This plan does not launch new exhaustive certification.

Gate-horizon evidence must also state its allowed object scope. If an old certificate exhausted only one selected object while the evaluated gate admits a larger boundary pool, it is not a pooled genuine-2-push certificate. Flag the scope mismatch for targeted certification under Section 1.3. This does not change the approved removed-K1 state used when certifying K2 independently.

### 7.2 Construct one fixed reference per problem

Use the initial graph and the initial no-retry route selected by the same routing rule, ignoring later attempt penalties. Require exactly two ordinary graph boundaries for this particular audit. Save the entire route, all K1 boundary blockers, all K2 boundary blockers, and whether either is marked as a joint plug. The fixed route is shared across methods/seeds; later replanning must not redefine K2.

In an isolated environment with the recorded runtime/configuration, restore the saved initial state, leave all objects in their initial poses, and place only the robot at the first deterministic sampled pose in the intermediate region. Reuse `profile_next_keyhole_before_open`. Save that pose and the sample seed. Verify that relocation has not changed the objects; an invalid reference is coverage failure, not evidence of access loss.

This relocation lets us ask what K2 access would be available from its intended source region before K1 changes the environment. Measuring K2 from the actual initial robot region would normally report no access because K1 separates it, which is not the independence question. This reference is also not the K1-removal certification shortcut: K1 remains physically present.

For each fixed K2 blocker, store:

- Stable object ID and initial pose.
- The complete reachable contact-edge index set from the shadow middle-region reference.
- Canonical accessible primitive IDs `(object_id, edge_idx, primitive_depth)` obtained by filtering the existing primitive generator with that reachability result and the fixed primitive definition; store the primitive-table hash.
- Explicit counts, including an empty set for an object that is present but has no reachable contact; absence of an object's evidence is not an empty set.

Primitive enumeration here is a zero-search, offline access query. It does not simulate all primitives or certify that a reachable contact can execute collision-free or open K2. Save that distinction in the output metadata.

### 7.3 Identify and capture the first-opening observation

Find the first completed opening of the fixed initial K1 boundary using the task/path reference and region correspondence, not merely the first action's object ID. For greedy execution, a first committed setup push is not automatically an opening; use the first recorded opening event for that fixed task. If multiple commits precede it, retain their prefix.

Measure immediately after that first opening from the saved committed state, with the robot where the canonical run left it. Do not wait for a later repair, relocation, or final success. A run with no first opening, an opening on a different initial route, or ambiguous fixed-boundary correspondence has an explicit unavailable observation; it is not observed access loss.

Do not impose a one-hop decrease to accept an observation. Save minimum-hop changes as a separate diagnostic. The reference and post-state can be measured even when the global path is still two hops or Full NAMO later fails.

Load the saved state into a separate audit environment. Query the same fixed K2 object IDs, record their poses and the full edge/primitive sets, and compare against the reference. Prefer saved post-commit states over replay for new runs. For legacy rows without states, use the existing isolated prefix-replay path, restore before each primitive as required by the canonical contract, and label the evidence source and verification strength explicitly.

### 7.4 Per-run outputs and definitions

For reference set `B` and after set `A`, over `(object, edge)` pairs:

```python
retained = B & A
lost = B - A
gained = A - B
preserved_access = bool(B) and not lost
exact_access_unchanged = bool(B) and B == A
all_reference_access_lost = bool(B) and not retained
no_access_after = bool(B) and not A
```

Only apply this block after successful evidence validation. Empty `B`, missing objects, missing state, invalid restoration, and mismatched runtime/configuration produce a null preservation label and a coverage reason. An actually observed empty after-set is valid loss. Complete loss of the old contacts with new contacts gained is not “no access.”

Record the same retained/lost/gained sets for canonical primitive IDs, per object and pooled. Classify the transition as unchanged, additions-only, losses-only, mixed additions/losses, or unavailable. `preserved_access` uses the contact subset rule from the existing audit; the primitive-access result is an additional named result. Do not silently replace the approved contact definition with a stronger primitive-feasibility definition.

Also save all of these diagnostics:

- K2 translation in millimeters and wrapped yaw change in degrees, per blocker; raw poses are retained. Reuse the existing audit's `0.1 mm / 0.1 degree` unchanged-pose diagnostic as an explicitly configured analysis tolerance, not a physics or acceptance tolerance.
- Contact/primitive loss conditioned on K2 poses being unchanged. This is the specific evidence for access changing even when the downstream blocker does not move.
- Post-opening minimum route hops, whether a route exists, whether the original goal is already reachable, and whether the next selected boundary still involves the fixed K2 objects. None is an extra preservation criterion.
- Initial K1/K2 pool sizes and native joint-plug flags; actual distinct objects committed during K1; movement of all context objects across the opening. If collision evidence exists in recorded simulator information, preserve it separately. Object motion alone does not prove direct K1-to-K2 collision.
- First-opening calls and committed physical pushes, final run success/failure and calls, and subsequent return/recovery events linked by attempt IDs. Keep timing null until a verified Icelake pairing exists.
- Evidence source, reference ID, before/after state IDs, runtime/configuration hashes, restore contract, validity/coverage reason, and intermediate replay-verification status when relevant.

These fields support access preservation, stricter exact-access/pose diagnostics, mechanical-coupling observations, and outcome-conditioned analyses without introducing a single misleading universal “independent” boolean.

### 7.5 Five-seed scene labels, missingness, and subgroup reports

For the five Random seeds, reuse this witness-based rule:

- At least one valid preserving first opening gives `preserved_access`, even if that seed later fails the full task.
- With at least one valid measurement, no preserving witness, and all five seeds accounted for without unresolved missing/invalid evidence, assign `not_preserved_access_in_five` (compatible with the old audit's `not_preserved_access` category).
- Otherwise the scene label is null with an explicit coverage reason. All five having no first opening means no independence measurement, not an observed loss.

Known no-opening or alternative-route runs are accounted for but do not enter the measured-run denominator. A positive witness can establish the scene label despite other missing observations; still report incomplete per-seed coverage. A negative five-seed label means “no preserving witness observed in these five runs,” not impossibility of any preserving opening.

Keep three concepts separate: difficulty unresolved (`0/5` goal successes), unavailable independence evidence, and measured access not preserved. A scene can be goal-unresolved and still have a preserving first-opening witness.

Required independence outputs are `run_observations.jsonl`, `scene_labels.jsonl`, `coverage.json`, `recovery_requests.jsonl`, `subgroup_summary.csv`, and `provenance.json`, using the existing audit layout. Include counts of measured, preserved, lost, gained-only, mixed, unchanged-pose-with-loss, no-first-opening, alternative-route, and missing/invalid records. Preserve these counts by template, difficulty, gate pattern, method, and seed where applicable.

For subgroup method comparisons, use the same eligible scene IDs for each method, preserve every failed outcome, and show sample sizes and missing coverage. Random difficulty/cost summaries may use the explicitly labeled per-scene five-seed median. Random success-rate curves should average single-run success probabilities across seeds, not count “at least one of five solved” as comparable to one model run. The latter is a separate best-of-five statistic if ever shown.

The existing `summarize_groups` also calls a scene Random-solved when its five-seed median is finite, meaning at least three successes. Preserve that only as an explicitly named majority-of-five statistic in the historical report; the new single-run success-rate comparison must not reuse that column as if it were the model's one-run success rate.

The witness-defined subgroup is selected using Random itself; report that selection rule rather than implying an intrinsic method-independent property. Model runs can be audited against the same fixed reference and given their own per-run access outcomes without changing the Random-defined scene grouping.

## 8. Derive the twelve requested research statistics offline

Do not add twelve new conditional branches to the planner. The planner exports facts; the analysis layer classifies events and denominators. Each derived file records `analysis_version`, input hashes, and coverage.

### 8.1 Switching blockers: exhaustion versus deferral

Keep scheduler representative changes, changes of offered high-level task/pool, actual simulated object changes within a pooled search, and committed object changes in separate columns. A representative changing while the same pooled task is passed to the opener is not itself evidence that the high-level planner switched problems.

An exhaustion transition starts at a local `exhausted` result and ends at the next eligible ordinary-boundary attempt. Record whether its task/path/pool differs, and whether its first actually simulated object differs from the previous attempt's first actually simulated object. A zero-simulation attempt can establish task reselection but has no observed simulated-object switch. Report these denominators explicitly.

A deferral transition requires the old choice to remain eligible in the current graph and its attempt penalty to affect selection. Re-sort the already logged eligible choice list offline with the old choice's relevant penalty removed, using the exact existing tie-break rule. If the selected task would remain unchanged, classify the event only as “selection after deferral,” not a demonstrated penalty-caused switch. An exhausted/blacklisted choice is not a still-eligible deferral.

Exclude ordinary progression after an opened boundary disappears/merges and exclude goal-clearance selection. Use correspondence and logged reset reasons rather than changes in region-label text. Pool/route identity ambiguity stays ambiguous, not forced into the switch count. Save per-run any/count for exhaustion switches, deferral switches, and their actual-object corroboration.

### 8.2 Success after a local impasse

Let `E` mean at least one genuinely exhausted local invocation, `S` eventual goal success, and `W_E` a subsequent exhaustion-linked blocker/task switch under the stated definition. Report `P(S|E)`, `P(E|S)`, `P(S|E and W_E)`, and `P(E and W_E|S)`, always with numerator and denominator. Separately report budget interruptions; do not call a truncated local search exhaustive.

### 8.3 Alternatives explored after an impasse

For every exhaustion event, examine attempts until the next recorded opening and until run termination. Count distinct offered blockers, actually simulated blockers, selected boundary tasks, and region routes; keep the first-opening endpoint and eventual-completion endpoint separate. Multiple exhaustion windows may overlap; report both per-impasse distributions and run-level unions, not a misleading sum of overlapping work.

Use snapshot-qualified exact path keys for raw counts and correspondence-aware route families for cross-state counts; report ambiguity. A route's numeric labels changing is not by itself a new route.

### 8.4 Exhausted blockers becoming useful again

Index exhaustion by stable object ID/pool, scene version, task reference, and attempt ID. A later attempt is a post-change revisit only if its committed state differs from the exhaustion state. Record whether the object was merely offered, actually simulated, included in a successful committed opener, and whether the full task eventually succeeded. Separate a robot-only state change from changes to movable-object poses using the saved state data.

With pooled exhaustion, report the exhausted pool and evidence of which objects were actually tried; do not claim exhaustive impossibility of every object's unrestricted behavior. Keep unchanged-state retries distinct from post-change revisits.

### 8.5 Departures from the initial route

Save the initial complete path, candidate blockers on every edge, and all subsequent selected routes. Use the region-cell correspondence rule in Section 4.2 to distinguish continued/collapsed initial-route progression from confirmed alternative route selection. Report selected-route departures for all runs and departures in attempts contributing committed actions for successful solutions. Do not claim a continuous executed navigation path was logged.

### 8.6 K1 changing access to K2

Use all outputs and coverage rules in Section 7. Include preservation, lost/gained primitive/contact sets, all-old-access-lost versus no-access-after, pose changes, and unchanged-pose access changes. This is the independence analysis, not a second simplified counter based on whether K2 moved.

### 8.7 Useful progress with unchanged or increased route length

For each committed opening, compare minimum unpenalized remaining graph hops before/after, not the scheduler's selected-path cost. Record successful runs containing a comparable finite hop count that stays equal or increases. Track lost-route, goal-already-reached, and goal-blocked/clearance-access cases separately; zero hops to a clearance access region does not mean the task is solved. If the post-state had no canonical snapshot before termination, its hop measurement is unavailable or explicitly reconstructed offline, not invented.

### 8.8 Where search effort goes

Use attempt call intervals to compute calls spent on invocations that return no opening, calls spent on invocations that return no committed progress, largest local-call share, and calls from the first exhaustion to eventual success. The first two are different for simulated policy, which can commit setup actions without opening. Attempts ending in failure/budget with a committed prefix must retain that distinction.

After a verified timing join, compute the analogous local-wall-time quantities and elapsed wall time from the first impasse to success. Candidate-stage time fields stay absent. A failed run has consumed effort but no finite success cost.

### 8.9 Final manipulation-sequence complexity

Count committed physical pushes, distinct object IDs in the committed sequence, recorded successful local openings, contiguous manipulation segments per object, and returns to an object after intervening manipulation of others. A two-push chain on the same object contributes two physical pushes and one local opening. Simulator attempts spent finding it remain a separate number. Preserve metrics for failed prefixes too, and mark them as prefixes rather than complete solutions.

### 8.10 Goal-clearance contribution

Count invocations, distinct targeted/simulated/committed objects, committed pushes, and calls for `task_kind=goal_clearance`. Separate a clearance invocation directly followed by verified goal success from a run that merely invokes clearance and eventually succeeds later. Exclude these events from ordinary boundary-switching and initial-route statistics, while retaining their cost in total Full-NAMO effort.

### 8.11 High-level computational overhead

On matched Icelake results only, report global snapshot/graph, standalone goal checks, route selection, and remaining orchestration time, with local-search/model/simulator decomposition. State the inclusive/nested timer definitions from Section 5 and do not sum overlapping phases. Compare methods only within the identical-hardware final campaign.

### 8.12 Identifiers, outcomes, and reproducibility

Every metric links to run/problem IDs, method, checkpoint/training seed, evaluation seed, template, difficulty-label version, certified gate pattern or unknown, call cap, goal success, termination cause, source/runtime fingerprints, and input records. Per-attempt metrics link to scene version, target/pool, task kind, selected route, raw outcome, and cumulative calls; cumulative elapsed time is attached only from a verified timing pair.

## 9. Implementation stages and focused validation

Implement in an isolated worktree from the agreed unified build after approval. Preserve the current untracked `HIGH_LEVEL_PLANNER_README.md` and unrelated worktrees. Do not cherry-pick an entire historical branch merely to obtain a logger or timer.

Run commands below from the implementation checkout after sourcing its existing machine environment (`env.robotlearning.sh` on dhruv-linux or the configured Amarel environment). Use that environment's Python, MuJoCo paths, and matching `SAGE_REPO`. Ensure the import path includes the checkout's `build_python`, `python`, `scripts`, and `scripts/pipeline` directories; the existing audit's statistics module imports sibling pipeline modules. No tests or commands in this document have been executed as implementation work merely by writing this plan.

### Stage 1: Measurement controls and the common record contract

**Files:** Create `python/namo/planners/search_measurements.py`; modify `python/namo/solvability_runner.py`, `scripts/sandbox/eval_bestfirst.py`, and the aligned `scripts/pipeline/eval_full_namo_walltime.py`; create `config/analysis/namo_measurements.yaml`; extend `python/tests/test_solvability_runner.py`; create one focused `python/tests/test_search_measurements.py`.

- [ ] Add the two booleans and schema version to YAML parsing and `SolveTask`; reject invalid types at the runner configuration boundary with the option name. Keep all environment-specific paths caller supplied.
- [ ] Define one small `SearchMeasurements` holder with optional `statistics` and `timing` dictionaries, the mandatory fingerprint, monotonic attempt/commit IDs, and terminal completeness. It must not import simulator construction, restore state, enumerate primitives, or call a scorer. Use the timing helper code in Section 5.2; statistics append operations occur only when their sink exists.
- [ ] Build problem/run IDs from the existing task/configuration and recorded content fingerprints as specified in Section 3.1. Use the same helper in both runners; do not duplicate the identity algorithm.
- [ ] Before candidate evaluation, audit the one-keyhole wrapper's actual `restrict_obj` argument against the resolved robot-to-target boundary pool. Record affected source episodes and label/certification gaps. The user's boundary-pool requirement is a protocol precondition; do not treat it as an optional logging feature or assert historical results satisfy it merely because the engine supports tuples.
- [ ] Extend output writing so `total_calls`, success, termination, and provenance exist with timing off. Optional sections are absent when disabled, not populated with zero values that appear measured.
- [ ] Update existing test/caller configuration that explicitly consumes detailed traces to request `record_statistics=true`; keep ordinary planner outcome tests independent of optional telemetry. Do not make the new default silently break a trace-dependent caller.
- [ ] Test the disabled-clock contract before changing the engine:

```python
def test_disabled_timing_never_reads_clock(monkeypatch):
    import namo.planners.search_measurements as measurement

    def forbidden_clock():
        raise AssertionError("timing disabled must not read the measurement clock")

    monkeypatch.setattr(measurement, "perf_counter", forbidden_clock)
    started = measurement.clock_start(None)
    measurement.clock_finish(None, "t_sim", started)
    assert started is None


def test_enabled_timing_accumulates_existing_operation(monkeypatch):
    import namo.planners.search_measurements as measurement

    values = iter([10.0, 10.25])
    monkeypatch.setattr(measurement, "perf_counter", lambda: next(values))
    timing = {}
    started = measurement.clock_start(timing)
    measurement.clock_finish(timing, "t_sim", started)
    assert timing == {"t_sim": 0.25}
```

- [ ] Add one parameterized runner test covering the four boolean combinations and mandatory fields. Before implementation, the new assertions must fail because fields/gating do not exist; afterward all combinations have the same mocked outcome/call count and the requested optional sections only.
- [ ] Validate and commit this coherent runner/schema stage:

```bash
python -m pytest -q \
  python/tests/test_search_measurements.py \
  python/tests/test_solvability_runner.py
git diff --check
git add python/namo/planners/search_measurements.py \
  python/namo/solvability_runner.py \
  scripts/sandbox/eval_bestfirst.py \
  scripts/pipeline/eval_full_namo_walltime.py \
  config/analysis/namo_measurements.yaml \
  python/tests/test_search_measurements.py \
  python/tests/test_solvability_runner.py
git commit -m "feat(eval): separate behavioral and timing collection controls" \
  -m "Add stable run identity and minimal outcomes in every measurement mode; keep optional payloads independent and clocks disabled for untimed collection."
```

### Stage 2: Local-search and Full-NAMO evidence hooks

**Files:** Modify `python/namo/planners/opening/best_first_search.py`, `best_first_region_opening.py`, `python/namo/planners/full_namo/full_namo_planner.py`, and `python/namo/solvability_runner.py`; extend the existing strict-BFS, goal-clearance, greedy, and budget tests with the required assertions.

- [ ] At each existing high-level decision, copy the already-computed route choices and selected task when statistics are enabled. Assign an attempt ID immediately before invoking the opener, not by counting successful openings afterward.
- [ ] At root candidate creation and each actual simulator call, update the local object/depth counts and streaming execution digest. Pass through the exact local end reason and budget allowance. Do not charge another simulation or request `trace_out`/visualization capture.
- [ ] If the unified one-keyhole wrapper still passes a single donor object, wire it to the existing validated boundary pool and actual fixed target samples for every arm. Keep this scope correction in a separate commit, `fix(eval): evaluate the complete one-keyhole boundary pool`, with a body recording the changed episode scope and invalidated legacy labels/results; do not alter the shared ranker/heap algorithm. The call-level requirement is `restrict_obj=tuple(boundary_objects)`, where `boundary_objects` comes from the existing boundary resolver, not all reachable objects. Make this correction before the new measurement/parity baseline and do not run an extra campaign just to test the old restriction again.
- [ ] Wrap the existing operations with optional timers, including `scorer.score_state` and the local verifier. The engine executes the same code path with timing values discarded only when explicitly disabled; all measurement clock reads must be guarded.
- [ ] On canonical state adoption, record commit/action linkage and copy the returned committed state for the statistics sidecar. Save failure prefixes and greedy setup commits. Hash existing states for pairing without performing another push.
- [ ] Preserve the pooled distinction in tests: an opener offered `("A", "B")` can first simulate B and commit B; its representative may still be A. Assert all three identities are recorded correctly and do not relabel that as a new high-level task.
- [ ] Extend one existing local-search fixture to cover the one-keyhole evaluator end to end: A and B belong to the target boundary, unrelated reachable C does not, and B supplies the opening. Assert both A/B are offered, C is excluded, the target stays fixed, and the cap is shared. Reuse the fixture for an A-then-B chain to check that allowed cross-object finishing actions remain available. Run this focused coverage before the separate object-scope commit and before measurement parity checks; no broad new test matrix is needed.
- [ ] Extend an existing exhaustion/alternative-route scenario to assert exact attempt IDs, unchanged-state blacklisting, post-change revisit linkage, and call intervals. Extend a goal-clearance scenario to assert task-kind separation and terminal prefix export. Reuse existing fixtures rather than creating a scene per field.
- [ ] Run the relevant regression files and commit only the hook changes and their focused assertions:

```bash
python -m pytest -q \
  python/tests/test_full_namo_strict_bfs.py \
  python/tests/test_full_namo_budget_and_config.py \
  python/tests/test_full_namo_goal_clearance.py \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_full_namo_policy_multi_blocker_scene.py \
  python/tests/test_best_first_sandbox_contract.py \
  python/tests/test_best_first_edge_blacklist.py \
  python/tests/test_search_measurements.py \
  python/tests/test_solvability_runner.py
git diff --check
git add python/namo/planners/opening/best_first_search.py \
  python/namo/planners/opening/best_first_region_opening.py \
  python/namo/planners/full_namo/full_namo_planner.py \
  python/namo/planners/search_measurements.py \
  python/namo/solvability_runner.py \
  python/tests/test_full_namo_strict_bfs.py \
  python/tests/test_full_namo_goal_clearance.py \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_full_namo_budget_and_config.py \
  python/tests/test_full_namo_policy_multi_blocker_scene.py \
  python/tests/test_search_measurements.py \
  python/tests/test_solvability_runner.py
git commit -m "feat(eval): record local attempts and committed Full-NAMO state" \
  -m "Observe pooled searches, exhaustion, deferral, clearance, and failed prefixes without changing search order, acceptance, state restoration, or budget accounting."
```

### Stage 3: Native region evidence and phase timers

**Files:** Modify `python/namo/cpp_bindings/rl_env.hpp`, `rl_env.cpp`, `bindings.cpp`, `python/namo/planners/__init__.py`, and the Full-NAMO snapshot call sites; extend `python/tests/test_search_measurements.py` for wrapper forwarding and a native integration case.

- [ ] Append the two optional snapshot parameters and payloads in Section 5.3. Export the existing component return value and native grid metadata only when requested.
- [ ] Add the four guarded native timers around the existing phases; preserve computation order and one call per existing operation. Keep local and global snapshot timing ownership distinct at the Python call sites.
- [ ] Forward optional data through both wrapper layers and attach cell evidence to the checkpoint for the state that actually produced that snapshot. A cached post-opening snapshot must not be attributed to a later state.
- [ ] Extend one wrapper test to assert both payloads survive, and one actual native fixture check to compare adjacency, edge objects, region labels, goal samples, goal status, and final state with optional export/timing on and off. Check disabled payloads are absent and the exported cell counts/labels match the existing components.
- [ ] Build through the documented machine environment, never by deleting a shared build directory. On Amarel, compilation happens on a compute allocation, not a login node. Run:

```bash
./build_python_bindings.sh
python -m pytest -q \
  python/tests/test_search_measurements.py \
  python/tests/test_wavefront_snapshot_semantics.py \
  python/tests/test_full_namo_goal_clearance.py \
  python/tests/test_full_namo_strict_bfs.py
git diff --check
git add python/namo/cpp_bindings/rl_env.hpp \
  python/namo/cpp_bindings/rl_env.cpp \
  python/namo/cpp_bindings/bindings.cpp \
  python/namo/planners/__init__.py \
  python/namo/planners/full_namo/full_namo_planner.py \
  python/tests/test_search_measurements.py
git commit -m "feat(eval): expose passive region evidence and snapshot timing" \
  -m "Export already-computed component cells and optional native phase durations; preserve graph semantics and forward measurement fields through both binding wrappers."
```

### Stage 4: Offline independence analysis and behavioral summaries

**Files:** Reuse/integrate `python/namo/planners/full_namo/access_audit.py`, `scripts/pipeline/audit_keyhole_access.py`, `keyhole_access_stats.py`, and `config/analysis/keyhole_access.yaml`; modify `scripts/pipeline/aggregate_multihop_solvability.py` and `compare_multihop_rankers.py`. Keep existing audit tests and add one compact `scripts/pipeline/tests/test_full_namo_statistics.py` for the new event summaries.

- [ ] Teach the audit to consume the new committed-state/graph sidecars and their failed prefixes. Prefer direct saved-state restoration; use the already implemented replay recovery only for missing legacy states. Runtime mismatches create coverage records, not negative access labels.
- [ ] Extend the reference measurement with canonical primitive identities and native joint-plug metadata, reusing its contact subset/pose logic. Do not import a stricter legacy acceptance predicate requiring unchanged poses or fewer hops.
- [ ] Generalize configured model-arm selection, seed lists, call cap, and population identity; retain the existing retrospective workflow as explicitly configured compatibility, not hard-coded defaults for this campaign.
- [ ] Add the Section 8 derivations to the existing aggregator's new-schema path. Index by `run_id`; reject duplicate complete runs with conflicting contents instead of preferring a solved row. Carry terminal technical/incomplete records into a repair report, not solve-rate/difficulty denominators.
- [ ] Use one compact synthetic timeline with pooled search, exhaustion, a different route, a post-change revisit, and final clearance to assert all related event counts and denominators. Add a renamed-region/merged-region case to prevent false route departures. Existing access-comparator tests cover retained/lost/gained/missing sets; extend them only for the new primitive sets and failed-prefix input.
- [ ] Extend the existing tests through the real analysis helpers, rather than testing Python arithmetic alone. These assertions pin early-failure censoring at the new budget and distinguish losing all old contacts from having no new contacts; reuse equivalent existing coverage where it already exists:

```python
from keyhole_access_stats import median_cost
from namo.planners.full_namo.access_audit import compare_contact_access


def test_failed_seeds_remain_in_the_median():
    rows = [
        {"solved": True, "total_calls": 12},
        {"solved": True, "total_calls": 80},
        {"solved": False, "total_calls": 3},
        {"solved": False, "total_calls": 19},
        {"solved": False, "total_calls": 9000},
    ]
    assert median_cost(rows, budget=9000) is None


def test_access_loss_and_no_access_are_different():
    result = compare_contact_access({"B": [1, 2]}, {"B": [3]})
    assert result["status"] == "measured"
    assert result["preserved"] is False
    assert result["objects"]["B"]["lost_edges"] == [1, 2]
    assert result["objects"]["B"]["gained_edges"] == [3]
    assert result["objects"]["B"]["reachable_edges_after"] == [3]
```

- [ ] Run the existing audit tests plus the single new statistics module. CLI `--help` and input-only preparation must work without starting the simulator; pure aggregation must not import it. Validate pooled one-keyhole identity, legacy object-specific source links, and target-specific deduplication under Section 1.3; never collapse by XML alone.
- [ ] Commit the stage:

```bash
python -m pytest -q \
  python/tests/test_keyhole_access_audit.py \
  scripts/pipeline/tests/test_keyhole_access_stats.py \
  scripts/pipeline/tests/test_audit_keyhole_access.py \
  scripts/pipeline/tests/test_full_namo_statistics.py
python scripts/pipeline/audit_keyhole_access.py --help
python scripts/pipeline/aggregate_multihop_solvability.py --help
git diff --check
git add python/namo/planners/full_namo/access_audit.py \
  scripts/pipeline/audit_keyhole_access.py \
  scripts/pipeline/keyhole_access_stats.py \
  scripts/pipeline/aggregate_multihop_solvability.py \
  scripts/pipeline/compare_multihop_rankers.py \
  config/analysis/keyhole_access.yaml \
  python/tests/test_keyhole_access_audit.py \
  scripts/pipeline/tests/test_keyhole_access_stats.py \
  scripts/pipeline/tests/test_audit_keyhole_access.py \
  scripts/pipeline/tests/test_full_namo_statistics.py
git commit -m "feat(analysis): derive replanning and fixed-route access statistics" \
  -m "Reuse the independence audit for saved successful and failed states; preserve five-seed censoring, coverage, multi-object identities, and explicit event denominators."
```

### Stage 5: Timing-only driver, pairing, and output integrity

**Files:** Modify `scripts/pipeline/eval_full_namo_walltime.py`, `scripts/slurm/eval_walltime.slurm`, `python/namo/solvability_runner.py`, and the existing aggregation/reporting entrypoints. Extend `scripts/pipeline/tests/test_full_namo_statistics.py` for joins and `python/tests/test_solvability_runner.py` for output behavior.

- [ ] Require `record_statistics=false`, `record_timing=true` for the final timed campaign. Retain `main`, one exclusive Icelake node, one worker, single-threaded pools, CPU-only inference, and exact CPU-model checks for every shard. Do not relax hardware checks to run candidate jobs through this entrypoint.
- [ ] Replace the old timed driver's 20,000-specific assertion/report string with a versioned new campaign configured for 9,000 Full-NAMO calls. Keep the old campaign's files and protocol intact. One-keyhole's new campaign explicitly supplies 3,000; do not silently change old registered 900-call result metadata.
- [ ] Check that every expected scene/method/seed has an untimed statistics source before scheduling a timing-only pair. Missing statistics coverage is a separate untimed run request, not permission to enable statistics on the timing job.
- [ ] Join rows and attempt timings under Section 6. Write paired rows, mismatch/coverage rows, and difficulty/template/horizon reports without overwriting either raw source. The report must work when all behavioral/timing rows are matched and clearly report incomplete coverage otherwise.
- [ ] Extend a single pairing test with equal semantic digests/different measurement flags, a changed seed, a changed call interval, and a changed attempt digest. Only the first pair is accepted. Assert no timing row is attached to an unmatched behavioral record.
- [ ] Check that model/load warmup is outside the timer, every actual search scoring/simulation remains inside it, nested timer totals are not double-counted, and all failure success-times are null. Never report an early terminal failure as a fast success.
- [ ] Run focused tests and commit:

```bash
python -m pytest -q \
  python/tests/test_search_measurements.py \
  python/tests/test_solvability_runner.py \
  scripts/pipeline/tests/test_full_namo_statistics.py
bash -n scripts/slurm/eval_walltime.slurm
python scripts/pipeline/eval_full_namo_walltime.py --help
git diff --check
git add scripts/pipeline/eval_full_namo_walltime.py \
  scripts/slurm/eval_walltime.slurm \
  python/namo/solvability_runner.py \
  scripts/pipeline/aggregate_multihop_solvability.py \
  scripts/pipeline/compare_multihop_rankers.py \
  python/tests/test_solvability_runner.py \
  scripts/pipeline/tests/test_full_namo_statistics.py
git commit -m "feat(eval): pair timing-only Icelake reruns with behavioral records" \
  -m "Enforce the new explicit budgets and unchanged hardware protocol; reject divergent deterministic pairs and retain censored failures and full pairing coverage."
```

### Stage 6: Small end-to-end parity check and handoff

This is a finite verification step, not a new candidate-mining campaign. Use existing canonical fixtures and a few existing Full-NAMO scene/seed cases exercising ordinary success, exhaustion/replanning, pooled blockers, goal clearance, and a failed committed prefix. A single case can cover several behaviors; do not manufacture dozens of new environments.

- [ ] Run the same fixed fixture inputs in all four switch combinations. Assert identical attempted-action digest, committed sequence/state digest, outcome, and total calls. Verify statistics-off produces no expanded trace/state/region files; timing-off produces no measurement durations or clock reads.
- [ ] Perform the same core parity check for one-keyhole because it shares `best_first_search.py`. Establish the required boundary-pool protocol first, then verify toggling logging does not change its target, allowed pool, action sequence, horizon, or label interpretation. Do not require equality to a superseded single-object protocol.
- [ ] Restore a successful and a failed first-opening checkpoint in the offline audit. Require a common initial reference, complete K2 object/contact/primitive evidence or a truthful explicit unavailable reason, and no changes to saved outcome/cost records.
- [ ] Validate a small cross-node deterministic pairing before launching the large two-stage workflow. The timed subset must use the final Icelake restrictions; the behavioral subset uses ordinary CPU execution with the same semantic build/configuration. A mismatch blocks that pair, not an invented replacement statistic.
- [ ] Inspect all output schemas, seed coverage, call conservation, optional-field presence, pairing status, and independence coverage. Run all relevant modified suites; do not call completion based only on a successful simulator example.
- [ ] Record actual commands, build/configuration hashes, evidence paths, and any unresolved coverage in the implementation experiment card. Commit only the handoff notes/fixture assertions changed in this stage, with message `docs(eval): record measurement parity and campaign readiness` and a body naming the verified tests and artifact paths. No broad `git add .`.

## 10. Output layout, failures, and scale

Use an output root supplied by the campaign environment. Extend the existing shard layout rather than adding a database:

```text
campaign/
  protocol.json
  candidate_statistics/
    METHOD_SEED/shard_ID/
      outcomes.jsonl
      attempts.jsonl
      checkpoints.jsonl.gz
      states/
      summary.json
  independence/
    run_observations.jsonl
    scene_labels.jsonl
    coverage.json
    recovery_requests.jsonl
    subgroup_summary.csv
    provenance.json
  timed_icelake/
    METHOD_SEED/shard_ID/
      outcomes.jsonl
      local_timing.jsonl
      summary.json
  aggregate/
    paired_runs.jsonl
    pairing_coverage.json
    pairing_mismatches.jsonl
    behavior_by_run.jsonl
    report.json
```

Capitalized directory components denote the existing method/seed/shard identifiers, not environment variables to hardcode. No statistics sidecars are written under `timed_icelake`. When both flags are off, only the minimal outcome/provenance output is required.

Buffer timing data until the planner timer has stopped; serialize and compress outside it. Untimed statistics may be written after completed local invocations/commits because no benchmark time is being collected. Store states only at committed decision boundaries and reuse graph/state references across unchanged attempts. Do not retain all speculative simulator states, all candidate pools, or images.

Use the existing exclusive/new-output conventions and atomic completion markers. A row is complete only when its referenced sidecars are finalized; a killed job or disk-write error remains incomplete and produces a repair request. Preserve completed earlier rows. Do not turn `runner_exception`, a lost socket, SIGKILL, malformed sidecar, or hash mismatch into `region_path_exhausted` or an unresolved environment.

A caught evaluation error should retain the exception class, message, problem/run ID, last completed attempt/commit IDs, and consumed calls if available. Incomplete attempts are explicitly incomplete. Do not add a catch-all inside the planner that changes its return value to success/failure for the sake of logging.

## 11. Reporting and completion checklist

- [ ] All twelve requested statistics map to fields and an offline definition in this document; independence includes reference construction, all K2 blockers, contact and primitive sets, pose/context effects, failed prefixes, five-seed witness labels, coverage, and subgroup comparisons.
- [ ] Candidate evaluation runs collect behavioral evidence with timing disabled, under the unified simulator, before freezing. Five Random seeds are complete per problem; exact one-keyhole labels retain separate certification provenance.
- [ ] One-keyhole methods evaluate every permitted boundary object for the fixed target, not only a donor object; pooled episode identities, exact difficulty/genuine-horizon evidence, and five-seed results all share that scope. Unknown pooled certification is not presented as a verified single-object label.
- [ ] Each final model/baseline run has matching untimed behavioral evidence before a timing-only rerun is joined. Do not substitute another method's or seed's behavior.
- [ ] Icelake timing-only runs rerun the full search, not just the successful physical plan, and pass semantic/digest pairing. No independence audit or heavy behavior logger is active inside those searches.
- [ ] Final tables retain failures in medians, report Random seed handling explicitly, split by difficulty and horizon/gate pattern as applicable, and show wall-template and evidence coverage. Difficulty-unresolved and audit-unavailable are separate.
- [ ] No inferred physical impossibility or general mechanical independence is claimed from a budget failure, five sampled seeds, contact preservation, or a native joint-plug flag.
- [ ] Timing tables state exact CPU model, campaign, build, configuration, and the within-campaign-only comparison rule. Non-Icelake elapsed times are not substituted into missing entries.
- [ ] New assertions are focused, existing relevant tests pass, binding/wrapper parity is verified, and all document/code changes are scoped. No generation, freeze, or cluster campaign begins without the user's execution request.

### Recommended execution order after approval

Verify the existing one-keyhole boundary-pool wiring and identify affected labels; implement and verify the logging against that scope; refresh five-seed Random results on existing candidates without timing and refresh affected exact certification/difficulty evidence before freezing; compute independence evidence offline; select/freeze the test environments; collect any missing selected model/method behavioral runs on ordinary CPUs; then submit timing-only Icelake reruns with exact identities and pair them. This collects the expensive evidence once while keeping published wall times tied to a clean, uniform measurement campaign.
