---
type: experiment
status: done
created: 2026-08-17
thread: full-namo-multihop
robot: car
commit: 92ac853
metric: exact-two-hop Full-NAMO solves and failure taxonomy, HY5U versus three random seeds, on a fixed planner
tags: [experiment, full-namo, multihop, aug9, hy5u, random-baseline, bugfix, failure-taxonomy, amarel]
---

# Exact-two-hop Full NAMO — planner fix, failure taxonomy, and three-seed random baseline

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U is the learned ranker inside a simulator-verified local region-opening search. This card composes that local solver inside Full NAMO; it does not redefine the model as a multi-hop predictor.

Direct follow-up to [EXP-2026-08-17-aug9-two-hop-full-namo](../archive/EXP-2026-08-17-aug9-two-hop-full-namo.md), which closed with three open items: one random seed only, an unexplained block of `planner_invariant_violation` failures, and no account of why 750 scenes ended in `region_path_exhausted`. This run answers all three on the same 2,531-scene population.

## Question

Three questions, in the order they were asked.

1. Are the 117 `planner_invariant_violation` failures a real bug, and how many solves does fixing it recover?
2. For the 750 `region_path_exhausted` scenes, did the greedy commit dead-end them, or did they never open anything? The planner has no top-level backtracking over committed keyholes, so this distinguishes an algorithm limitation from a scene property.
3. Does HY5U's advantage over uniform random survive more than one random seed?

## What changed in the code

**`already_accessible` was a fatal invariant, and should not have been** (`cfa23cf`). The opener returns `success=True` with `failure_reason="already_accessible"` when the target region is already reachable without any push ([best_first_region_opening.py:223](../../../python/namo/planners/opening/best_first_region_opening.py)). But `already_accessible` also sat in `_INVARIANT_TARGET_FAILURES`, and that check runs first, so the planner aborted the whole scene on a non-error. The later whitelist that explicitly tolerates `already_accessible` on success was dead code.

A naive un-abort hangs. A zero-push opening changes nothing physically, so the next iteration rebuilds an identical snapshot and picks the same target, and `full_namo_max_iterations` defaults to `None`. The fix allows a zero-push opening once per boundary, then treats a repeat as a genuine graph/opener reachability mismatch and falls back to the existing blocked-boundary reroute. It also stops clearing the blocked-boundary blacklist on a zero-push opening, since no boundary became newly openable. Two regression tests cover both paths.

**The runner now persists `iteration_trace`.** The planner always built it, but [solvability_runner.py](../../../python/namo/solvability_runner.py) wrote only the terminal `failure_kind`, so nothing in the raw output could distinguish "committed a keyhole then dead-ended" from "never opened anything". Question 2 was unanswerable from the original campaign's artifacts for this reason alone.

**Comparison metadata is no longer hardcoded** (`92ac853`). [compare_multihop_rankers.py](../../../scripts/pipeline/compare_multihop_rankers.py) asserted `"random_seed": 42` and `"simulation_budget_per_keyhole": 300` regardless of what ran. The budget is now read off the rows, which already carry `simulation_budget_limit_per_keyhole`; the seed is an explicit `--random-seed` argument defaulting to null. No computed number was ever affected — but the seed label was wrong on the s43 and s44 outputs until regenerated.

## Protocol

Identical to the archived card except for the planner fix. Same manifest, same 2,531-scene population, `hmax=2`, budget 300 per keyhole reset independently, primitive prefix `1x_car_d5_`, keyhole goal test 20 of 100 region points, raw `q` combination, discount off, dedupe and jam pruning on.

Four arms: HY5U (`/cache/home/dm1487/aquaman0/ckpts_bfix/HY5U_s2.ckpt`) at seed 42, and uniform random at seeds 42, 43, 44. Only the ranker and seed vary.

Amarel HEAD `cfa23cf`, `check_box_sync.sh` reported SAFE TO LAUNCH, and the `.so` is unchanged from the original run (`cfa23cf` is Python-only on top of `597e1be`). Every arm: 120/120 shards, `input_count` 2535 → `selected_exact_hop_count` 2531, `selection_error_count` 0, `iteration_trace` present on 2531/2531 rows.

## Result

### The fix changed the accounting and not a single solve

| arm | solved / 2531 | rate |
|---|---:|---:|
| HY5U `model_s42` | **232** | 9.17% |
| random `s42` | 193 | 7.63% |
| random `s43` | 184 | 7.27% |
| random `s44` | 201 | 7.94% |

HY5U solved 232 both before and after the fix, and `comm -3` on the solved-XML lists returns **zero differing lines** — the solved *sets* are identical, not merely the counts. Same for random s42 at 193. All 48 HY5U and 65 random `already_accessible` scenes were genuinely unsolvable under this protocol; they simply moved to other failure buckets (HY5U 48 → 42 `region_path_exhausted` + 6 `simulation_budget_exhausted`, exactly 48; random s42 65 → 57 + 8, exactly 65).

**Answer to question 1: the bug was real, the fix is correct, and it recovers nothing.** `planner_invariant_violation` drops from 48 to 0 on HY5U. The `already_accessible_repeat` guard fires on 48–65 scenes per arm with no scene looping, so the loop hazard was real and is contained. But anyone expecting the 113 scenes to convert into solves — as this orchestrator did when proposing the fix — was wrong.

### Failure kinds

| failure kind | HY5U | rand s42 | s43 | s44 |
|---|---:|---:|---:|---:|
| simulation_budget_exhausted | 1466 | 1518 | 1559 | 1537 |
| region_path_exhausted | 792 | 775 | 752 | 755 |
| goal_region_invalid | 41 | 41 | 34 | 36 |
| planner_invariant_violation | **0** | 4 | 2 | 2 |

The four surviving invariant rows are all `same_region_but_goal_unreachable`, never `already_accessible`.

`goal_region_invalid` is **seed-dependent** (41/41/34/36), which ruled out a static scene property from the start — a goal is either in free space or it is not.

**Resolved by the static probe (2026-08-17, same night).** `goal_in_free_space` is **true for all 2,535 scenes**; not one is statically bad. These are *post-push* failures: every one of the 41 has `simulation_budget_used_total > 0`, and because [full_namo_planner.py:292](../../../python/namo/planners/full_namo/full_namo_planner.py) returns on `goal_region_invalid` without recording an iteration trace, a trace of length N means the failure fired at iteration N+1 — the histogram is `{1:12, 2:23, 3:5, 5:1}`, so every one fired at iteration ≥ 2, after at least one executed push. Mechanism: a push drops an object onto the goal point and the next snapshot finds no free goal region. That fully explains the seed dependence, since different rankers push different objects to different places.

Consequence: **these 41 are not generation junk and must not be filtered out.** A candidate static proxy (`goal_clearance_m`, goal to nearest movable footprint) does not separate them — median 0.103 m against 0.112 m for the pool. There is no static filter for this class.

Separately and still true: the generator's validator checks robot region, goal region, and hop count but never `goal_in_free_space` ([generate_envs.py](../../../../mujoco_env_creator/generate_envs.py) `_runtime_validate_adjacency`). That is a real gap worth closing, but it is **not** the cause of these 41.

### `region_path_exhausted` splits three ways

The discriminator is the per-iteration simulator-call delta at each `opened_target` entry: a real opening costs at least one call, a zero-push opening costs zero.

| | HY5U | rand s42 | s43 | s44 |
|---|---:|---:|---:|---:|
| total | 792 | 775 | 752 | 755 |
| committed a real keyhole, then dead-ended | **321** | 313 | 299 | 300 |
| zero-push openings only | 11 | 11 | 11 | 11 |
| never opened anything | **460** | 451 | 442 | 444 |

**Answer to question 2: both, roughly 40/60.** About 321 scenes per arm are the greedy commit dead-ending the scene with no top-level backtracking to recover. That is a measured algorithm limitation, not a scene property, and it is the single largest actionable item in this card.

### The local opener is not giving up without trying

Among the ~460 scenes that never opened anything, per-scene classification (HY5U; other arms within ±10):

| scene class | HY5U | rand s42 | s43 | s44 |
|---|---:|---:|---:|---:|
| only `no_reachable_objects` | **133** | 133 | 133 | 133 |
| only `all_pushes_failed` | 313 | 303 | 296 | 297 |
| mixed | 14 | 15 | 13 | 14 |

Per-event, with simulator calls spent (HY5U):

| reason | 0 sims | 1–9 | 10–99 | 100+ |
|---|---:|---:|---:|---:|
| `no_reachable_objects` | 244 | 0 | 0 | 0 |
| `all_pushes_failed` | 0 | 108 | 212 | 172 |

Only the 133 `no_reachable_objects` scenes are the near-zero-call case. The other ~313 burn real simulation — 384 of 492 `all_pushes_failed` events spend at least 10 calls and 172 spend at least 100, with a median of 32 total scene calls in this group (p90 271, max 912). The prior hypothesis that these scenes fail because the local opener quits without searching is **rejected for two thirds of them**.

**The 133 `no_reachable_objects` scenes are the identical XMLs in all four arms** — intersection equals union equals 133, so they are seed- and ranker-independent. No reachable movable object exists at the boundary. This is a generation defect and those scenes are unsolvable by construction. With the 41 bad-goal scenes, roughly 174 of 2,531 (6.9%) are junk.

### HY5U versus each random seed

| | vs s42 | vs s43 | vs s44 |
|---|---:|---:|---:|
| both solved | 174 | 169 | 175 |
| HY5U only | 58 | 63 | 57 |
| random only | 19 | 15 | 26 |
| neither | 2280 | 2284 | 2273 |
| McNemar exact p | 9.78e-6 | 3.75e-8 | 8.78e-4 |
| median calls on jointly solved, HY5U / random | 3 / 10 | 3 / 8 | 3 / 13 |
| HY5U faster / tied / slower | 158/9/7 | 134/21/14 | 147/14/14 |

Solve counts at a total-scene-call cutoff:

| cutoff | HY5U | rand s42 | s43 | s44 |
|---:|---:|---:|---:|---:|
| 2 | 94 | 6 | 26 | 15 |
| 5 | 169 | 59 | 71 | 51 |
| 10 | 202 | 97 | 109 | 84 |
| 30 | 218 | 141 | 148 | 144 |
| 100 | 228 | 167 | 178 | 179 |
| 300 | 232 | 192 | 184 | 199 |

**Answer to question 3: yes.** Random's band is 184–201 (mean 192.7, spread 17), and HY5U's 232 sits well outside it. HY5U's median of **3** total simulator calls on jointly solved scenes is identical across all three seeds, against random's 8–13. The archived card's provisional claim now rests on three seeds.

## Verdict

**Accept the three-seed ranker claim. Accept the failure taxonomy. Reject the premise that the invariant bug was suppressing solves.**

HY5U orders a verified full-NAMO solution much earlier than uniform random, consistently across three seeds, and the ordering advantage is largest at tight budgets (94 vs 6–26 solves within two total calls).

The 9.17% complete-scene rate is not a ranker ceiling, and this run now says what it actually is. Of HY5U's 2,299 unsolved scenes: **159 are generation junk** (no reachable blocker at the first boundary), about 321 are the greedy commit dead-ending with no backtracking, and 1,466 are genuine local budget exhaustion. The 41 `goal_region_invalid` scenes are **not** junk — see above, they are post-push goal occlusion and stay in the pool.

**Static probe confirmation** (`scripts/pipeline/probe_static_topology.py`, one region snapshot plus `get_reachable_objects` per XML, zero simulated pushes, 4 seconds wall for all 2,535 scenes on 32 cores). It recovers the unreachable-blocker class from static geometry alone: **133 of 133**, with recall 156/156 against every scene any arm ever reported `no_reachable_objects` at iteration 1, and a 0.6% false-positive rate. It independently reproduces the 4 hop-count mismatches that the eval's own `path_length_mismatch_count` reports. The flag must gate on **boundary 0** — the only boundary the planner ever opens — since `no_reachable_blocker_any` fires on 2,517 of 2,535 scenes, the second boundary being unreachable at t=0 essentially by construction.

**Surviving pool: 2,374 scenes** (dropped 161 — 159 no reachable blocker, 4 hop mismatch, 2 no path, overlapping).

No easy/medium/hard split is reported. These complete multi-hop scenes still have no registered difficulty labels, and the project's canonical bins ([eval_common.py:35](../../../scripts/eval_common.py), hard < 0.05, med < 0.30, easy ≥ 0.30) are defined on the matched local episode's solve rate, which a composed two-hop scene does not have. Labeling them by random-trial solve rate is the agreed next step, deliberately sequenced **after** the generation defects are fixed, since a difficulty axis built on a pool that is 90% unsolved and 7% junk would describe the defects rather than the environments.

## Artifacts

Amarel root `/scratch/dm1487/multihop_aug9_hy5u/scale_20260817_planfix/`; the original `scale_20260817_0000` is untouched.

- Per-arm aggregates: `{model_s42,random_s42,random_s43,random_s44}_aggregate/`
- Paired comparisons: `comparison_s{42,43,44}/comparison.{json,md}` (regenerated at `92ac853` with correct seed metadata)
- Raw shards, with `iteration_trace` on every row: `<arm>/shard_*/`
- Jobs: 60642130 HY5U, 60642131 random s42, 60642132 s43, 60642133 s44, 60643233 s42 repair; smokes 60641980/60641981

## Incidents

**23 tasks failed on `random_s42`, repaired.** Cause was node-local `/scratch` breakage, not a launch race. All 23 landed on exactly three nodes (halk0020 ×9, halk0021 ×6, halk0028 ×8) with zero failures anywhere else across 480 tasks. Twelve died in one second with the shell's own redirect failing — `slurm_script: line 43: .../shard_0077/input.txt: No such file or directory` — one line after a `mkdir -p` that returned 0. Repaired by job 60643233 with `--exclude=halk0020,halk0021,halk0028`. Those nodes later ran other arms' tasks to COMPLETED, so they recovered.

**Amarel's `main` QOS caps submitted jobs at 500 per user**, so 4 × 240 arrays was rejected; the campaign ran as 4 × 120. `ArrayTaskThrottle` was raised to 240 mid-run, which is why the arrays ran wider than launched. Observed peak was 345 concurrent tasks (1,380 cpus), far below the 6,720 cpu cap — the job-count limit binds first at this task size.

## Next

1. **Top-level backtracking over committed keyholes.** 321 scenes per arm currently dead-end after a verified commit. This is the largest single recoverable class and the only one that is an algorithm change rather than a data fix.
2. **Fix the generator.** Diagnose why 159 scenes have no reachable blocker at boundary 0 — that is the one confirmed generation defect. Adding the missing `goal_in_free_space` check is still worth doing for future pools, but it would not have caught any scene here.
3. ~~Static per-scene probe~~ — **DONE**, see above. `scripts/pipeline/probe_static_topology.py`, surviving list at `static_probe/surviving_xmls.txt`.
4. **Then** difficulty labeling by random-trial solve rate, 30 seeds at budget 900, on the cleaned pool. Thirty seeds is the minimum that resolves the 0.05 bin edge.
