---
type: experiment
status: done
created: 2026-08-17
thread: full-namo-multihop
robot: car
commit: 597e1be
metric: exact-two-hop Full-NAMO solves and simulator efficiency, HY5U versus uniform random
tags: [experiment, full-namo, multihop, aug9, hy5u, random-baseline, best-first, hmax2, amarel]
---

# Exact-two-hop aug9 Full NAMO — generation and HY5U versus random

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U remains the learned ranker inside a simulator-verified local region-opening search; this experiment composes that local solver inside Full NAMO and does not redefine the model as a multi-hop predictor.

## Question and hypothesis

Can we generate at least 100 car scenes whose XML goal is initially two region boundaries away, then solve each boundary as an independent keyhole search while carrying only the resulting physical state into the next round? The working hypothesis was that Full NAMO could repeatedly rebuild the region graph, use HY5U to order local pushes, and solve at least 100 complete two-hop scenes more efficiently than uniform random ordering.

## Terminology — do not mix these axes

**Two-hop full scene** means that the initial shortest path from the robot region to the XML goal region crosses exactly two region boundaries. This is not the canonical local **2-push** region-opening horizon in which setup and finish pushes open one boundary.

**`hmax=2`** means each local keyhole search branch may contain at most two pushes. It does not mean two pushes for the whole Full-NAMO scene.

**Budget 300 per keyhole** means each selected boundary gets a fresh allowance of 300 simulator calls. The total cost of a scene is the sum over all keyholes attempted and may therefore exceed 300.

After a keyhole opens, its successful physical push state is retained, the wavefront and region graph are rebuilt from that new state, and a fresh local search with a fresh budget begins. Failed speculative pushes inside a local best-first search are restored and are not carried into the next candidate or keyhole.

There is no top-level backtracking over already committed keyhole solutions in this pilot. Backtracking exists only inside one local keyhole search, where simulated candidates are undone; after a keyhole is accepted, Full NAMO moves forward from that changed physical scene.

The local goal is the next region across the first blocked boundary on the current shortest region path, not the final XML goal point. The planner samples 100 fixed points in that target region and declares the keyhole open when at least 20 are reachable from the robot region.

The two local pushes may act on the same blocking object, so `hmax=2` allows a setup and finish on one object; it is a push-depth limit, not a requirement to move two different objects.

## Environment design

Generation used only the ten `mujoco_env_creator/templates/aug9_car_v3` templates: benchmarks 1–5 from set1 and set2. It reused the normal aug9 distribution machinery rather than hand-placing two nearly identical blockers: each requested layout sampled the template's normal movable-object count, object side lengths from 6–16 cm, object poses including rotation, and a car start/goal pair, then retained only pairs whose initial region path had exactly two hops.

The exact-hop filter is conditional on the template geometry. This deliberately preserves the existing wall-layout distribution, but it does not produce a balanced mixture of abstract motifs such as narrow gate, wide barrier, separate doors, clutter-assisted cut, or enclosure. Those structures appear only when the underlying template and sampled objects create them.

The generator used `--no-require-adjacent --exact-hop 2 --samples-per-pair 1`, a zero clearance-radius filter, minimum region area 0.02 m², and at most 100 layout attempts per requested seed. The final evaluator independently recomputed the initial path and rejected any generation/evaluation mismatch.

This experiment's unit is one generated `(XML, XML goal)` full-NAMO scene. It is not the usual many-episode room dataset keyed by `(XML, object, goal region)`, and it is not a room-held-out generalization benchmark. Some `_pair_000` and `_pair_001` XMLs share an obstacle layout but have different start/goal placements.

This card records only the two-hop pilot. Three- and four-hop generation were intentionally deferred until this generation, visualization, local-solving, graph-rebuild, and aggregation loop worked end to end.

## Full NAMO algorithm used

1. Build the wavefront and current region-connectivity graph, and stop if the final XML goal is already reachable.
2. Find a shortest region path from the robot's current region to the XML goal region.
3. Select only the first robot-adjacent blocked boundary on that path and define the region across it as the local keyhole goal.
4. Run simulator-verified best-first region-opening search on that boundary. HY5U or uniform random supplies only the ordering; every candidate push is checked by the simulator and restored when the search continues elsewhere.
5. If a verified push sequence opens the target region, commit its resulting physical state.
6. Rebuild the wavefront and graph from the changed scene, choose the next boundary, reset the local search and its 300-call budget, and repeat until the XML goal is reachable or no admissible path remains.

The implementation is [full_namo_planner.py](../../../python/namo/planners/full_namo/full_namo_planner.py) with [best_first_region_opening.py](../../../python/namo/planners/opening/best_first_region_opening.py). The Full-NAMO best-first integration landed in `19e9462`.

## Plan and protocol

Generate a large exact-two-hop pool on Amarel from all ten templates, independently revalidate the initial hop count, then evaluate identical scenes with the same Full-NAMO protocol under two rankers: HY5U and uniform random seed 42.

| field | value |
|---|---|
| robot/config | car, `config/namo_config_complete_skill15_car_1x.yaml` |
| local search | greedy best-first, simulator verified |
| HY5U checkpoint | `/cache/home/dm1487/aquaman0/ckpts_bfix/HY5U_s2.ckpt` |
| random arm | uniform ordering, RNG seed 42 |
| local horizon | `hmax=2` pushes per keyhole |
| primitive library | `1x_car_d5_`, candidate primitive depths 0–4 |
| simulator budget | 300 calls per keyhole, reset independently |
| keyhole goal test | at least 20 of 100 fixed target-region points reachable |
| score combination | raw `q`, discount off |
| pruning | no-op deduplication and jam-depth pruning inherited from canonical best-first evaluation |
| initial scene filter | exact shortest region-path length 2 |

The comparison changes only the ranker from `model` to `uniform`; population, seed, local horizon, primitive library, success test, budget scope, and worker count are matched. Because random has only one seed, the comparison is a quick paired check rather than a canonical multi-seed baseline.

## Run

Local tests passed before launch: all ten Full-NAMO budget/config tests, HY5U `LiveScorer` checkpoint loading, and the action-depth audit for the `d5` primitive prefix. An Amarel one-scene random smoke solved in 18 total simulator calls split as `[8, 10]` across two keyholes. Both scale arms completed all 240 evaluation shards with exit code 0.

| stage | Amarel job | code |
|---|---:|---|
| generation, 400 array tasks × 10 requested layouts | 60638907 | `cb0a6fa` generation launcher |
| HY5U Full-NAMO evaluation, 240 shards | 60639339 | `19e9462` planner + `cb0a6fa` launcher |
| random one-scene smoke | 60639798 | `bd5e53a` |
| random seed-42 evaluation, 240 shards | 60639805 | `bd5e53a` |
| paired aggregation | local | `597e1be` |

Generation requested 4,000 layouts and produced 2,535 XMLs. Independent evaluation retained 2,531 exact-two-hop scenes; four mismatched the requested initial hop count, with two recomputed as one-hop and two disconnected. The mismatch rate was 0.16%, with no selection errors.

| template | exact-two-hop scenes | HY5U solved | random solved |
|---|---:|---:|---:|
| set1/benchmark_1 | 12 | 1 | 1 |
| set1/benchmark_2 | 89 | 0 | 0 |
| set1/benchmark_3 | 146 | 0 | 0 |
| set1/benchmark_4 | 316 | 32 | 20 |
| set1/benchmark_5 | 260 | 1 | 1 |
| set2/benchmark_1 | 34 | 0 | 0 |
| set2/benchmark_2 | 192 | 0 | 0 |
| set2/benchmark_3 | 686 | 34 | 24 |
| set2/benchmark_4 | 0 | 0 | 0 |
| set2/benchmark_5 | 796 | 164 | 147 |
| **all** | **2,531** | **232** | **193** |

The yield is strongly template-dependent: set2/benchmark_4 produced no exact-two-hop XMLs, and set2/benchmark_5 supplied 31.5% of the population and 70.7% of HY5U's solves. The generated pool is therefore useful for solvability mining but not a balanced benchmark.

## Result

HY5U solved 232/2,531 complete scenes (9.17%); uniform random seed 42 solved 193/2,531 (7.63%). HY5U gained 39 solves and 1.54 percentage points. The original target of at least 100 working two-hop scenes was exceeded: 232 complete HY5U solves are available, including 197 whose successful run attempted exactly two keyholes.

Initial path length two does not force the executed run to contain exactly two keyhole attempts. Successful pushes can change the graph, so replanning can reroute or add a boundary; that is why 197 rather than all 232 solved runs used exactly two keyhole attempts.

### Paired outcomes on the same 2,531 scenes

| outcome | scenes |
|---|---:|
| both solved | 174 |
| HY5U only | 58 |
| random only | 19 |
| neither solved | 2,280 |

The paired solve advantage is statistically visible on this fixed random seed (exact McNemar `p=9.78e-6`), but it must not be generalized to the random policy distribution until more random seeds are run. The 19 random-only scenes also show that HY5U still ranks some viable local sequences too late or misses them under the 300-call cap.

### Simulator efficiency

These cutoffs use **total simulator calls per complete scene**, summed across every keyhole, and are different from the 300-call per-keyhole cap.

| total scene-call cutoff | HY5U solved | random solved |
|---:|---:|---:|
| 2 | 94 | 6 |
| 5 | 169 | 59 |
| 10 | 202 | 97 |
| 30 | 218 | 141 |
| 100 | 228 | 179 |
| 300 | 232 | 192 |
| 600 | 232 | 193 |
| 900 | 232 | 193 |

On the 174 scenes both arms solve, HY5U needs a median of 3 total scene calls versus 10 for random. HY5U is faster on 158, tied on 9, and slower on 7; the median paired difference is 6 fewer calls for HY5U. This is the cleanest result: on commonly solved scenes, HY5U usually orders a verified full-NAMO solution much earlier.

### Failure modes

| failure kind | HY5U | random |
|---|---:|---:|
| simulation budget exhausted | 1,460 | 1,510 |
| region path exhausted | 750 | 718 |
| planner invariant violation | 48 | 69 |
| invalid goal region | 41 | 41 |

Most failures reach the local 300-call ceiling without opening a required keyhole. This supports improving local ordering and generation quality before interpreting the current 9% complete-scene rate as a ceiling on Full NAMO itself.

## Verdict

**Accept the two-hop generation and Full-NAMO pilot; keep the HY5U-versus-random conclusion provisional.** We found more than the requested 100 complete two-hop solutions, verified that each boundary can be treated as a fresh local keyhole search while retaining only successful physical changes, and observed a large simulator-efficiency advantage for HY5U on jointly solved scenes. Before calling this a benchmark, balance the template mix, investigate templates with zero solves, and repeat random over multiple seeds.

No easy/medium/hard breakdown is reported because this generated full-NAMO population has no registered difficulty labels, and the project's canonical difficulty bins apply to local `(XML, object, goal region)` episodes rather than these complete multi-hop scenes.

## Artifacts

The local result root is `/common/users/dm1487/scratch_namo/multihop_aug9_hy5u/scale_20260817_0000`; the matching Amarel root is `/scratch/dm1487/multihop_aug9_hy5u/scale_20260817_0000`.

- HY5U aggregate: `aggregate/summary.json`, with raw solved/unsolved JSONL and `aggregate/solved_xmls_local.txt` for all 232 solved XMLs.
- Strict two-keyhole subset: `aggregate/solved_exactly_two_keyhole_attempts.txt` and `.jsonl`, 197 scenes.
- Random seed-42 aggregate: `random_s42_aggregate/summary.json` plus solved/unsolved JSONL.
- Paired comparison: `comparison_s42/comparison.json` and `comparison_s42/comparison.md`.
- Raw sharded outputs on Amarel: `eval/shard_*/` for HY5U and `random_s42/shard_*/` for random.
- Launchers and aggregation: [multihop_aug9_generate.slurm](../../../scripts/slurm/multihop_aug9_generate.slurm), [multihop_aug9_eval.slurm](../../../scripts/slurm/multihop_aug9_eval.slurm), [aggregate_multihop_solvability.py](../../../scripts/pipeline/aggregate_multihop_solvability.py), and [compare_multihop_rankers.py](../../../scripts/pipeline/compare_multihop_rankers.py).
- Visual review tool: `/common/home/dm1487/.codex/visualizations/2026/08/15/01a00633-2b1a-7c92-8f1f-15bdec7e2607/two-hop-wavefront-tool/two-hop-wavefront-browser-standalone.html`, with environment, wavefront, and region-graph views from the distribution-matched pilot.

## Next

1. Diagnose the zero-solve and low-solve templates separately from local budget exhaustion so generator failures are not confused with ranker failures.
2. Run at least three random seeds under the same paired protocol before making a stable ranker claim.
3. Build a balanced solved-scene bank across templates and graph structures rather than taking the naturally skewed pool as a test distribution.
4. Only after the two-hop pipeline is stable, extend exact-hop generation and Full-NAMO evaluation to initial path lengths three and four; keep `hmax` as the independent local keyhole push depth.
