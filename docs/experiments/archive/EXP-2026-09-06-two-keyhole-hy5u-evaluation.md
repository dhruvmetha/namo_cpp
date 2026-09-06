---
type: experiment
status: done
created: 2026-09-06
commit: bbdc3d18
metric: HY5U solved 38/40 complete scenes; MM 10/10, MH 8/10, HM 10/10, HH 10/10
tags: [experiment, full-namo, multihop, hy5u, same-template, interaction, medium, hard]
---
# HY5U on the controlled two-keyhole cohort

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** HY5U remains one local push ranker inside simulator-verified search. Full NAMO invokes it again after a committed K1 opening and carries the physical scene state into the K2 search.

## Hypothesis

_(user, from chat)_ Try the 40 current-physics two-keyhole scenes with HY5U and measure which complete scenes it solves.

## Plan

Freeze the exact 40-scene approval population from `EXP-2026-09-05`: ten MM, ten MH, ten HM, and ten HH scenes, with seven clean and three K2-contact scenes in each cell. The XML-list SHA-256 is `55b0d2d13918bf349ff13b598a78ef3cfc7b3caba0566d72b83532bbf7920220`; the metadata-manifest SHA-256 is `d1f3550a84446b66ce9fe8ae129afd5f377068ec21e7723aa8c3d0041f43153e`.

Use registered HY5U seed 2 checkpoint `$NAMO_SCRATCH/aquaman/round0/models/HY5U_s2/checkpoints/epoch011-val_loss0.3256.ckpt`, SHA-256 `3cf348cf7ba247f2cb143376371fc06771665793783d12e3b37bf596e0e5a854`. Run ordinary Full NAMO best-first search with model ordering, `hmax=2` per local keyhole, `1x_car_d5_` primitives, raw `q`, discount off, no-op deduplication, jam pruning, and 900 simulator calls reset per keyhole. Enable the read-only next-keyhole audit, but do not enable the rejected strict all-contact preservation gate.

First run one complete scene on rlab7 with one worker. If checkpoint loading, topology selection, Full-NAMO replanning, and artifact writing pass, use the measured runtime to size the 40-scene rlab7 run. Report complete-scene solve count and total simulator calls by ordered donor pair and by clean/contact scene type. Every scene remains in the denominator whether HY5U solves it or not.

## Run

The rlab7 one-scene smoke ran as job `272101_0` at commit `e056bbdd` and completed successfully in 2:10. It evaluated MM interaction scene `contact_0000.xml`, loaded the registered HY5U checkpoint, selected the expected exact-two-hop scene, and solved the complete goal in two simulator calls split `[1, 1]` across K1 and K2. The read-only K1 audit reported unchanged K2 object identity and pose, no lost or gained reachable K2 edges, and a path reduction from two hops to one.

The smoke's two simulator calls show that its elapsed time is dominated by process, model, and environment startup. The 40-scene run therefore uses one rlab7 allocation with ten workers, a two-hour safety limit, and the same protocol. A startup-dominated lower estimate is roughly 10 minutes for four worker waves; a pessimistic tail estimate is 45–60 minutes if several scenes consume hundreds of simulations.

The full evaluation ran as job `272102_0` at commit `bbdc3d18` on rlab7. All 40 requested real paths were selected exactly, with zero selection errors and zero path-length mismatches. The job completed in 39 seconds according to SLURM accounting. Raw per-scene rows are under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/hy5u40_v1/full/shard_0000/`; the standard aggregate is under `$NAMO_SCRATCH/eval/keyhole_mixed_context_20260905/hy5u40_v1/aggregate/`.

## Result

HY5U solved 38/40 complete two-keyhole scenes. It solved every MM, HM, and HH scene; both failures were MH. The breakdown joins raw evaluation rows to the frozen metadata manifest by `realpath`, never basename. The clean/contact split is descriptive context, not an independent difficulty label.

| ordered source pair | clean solved | contact solved | all solved | median calls among solved | maximum calls among solved |
|---|---:|---:|---:|---:|---:|
| MM | 7/7 | 3/3 | **10/10** | 3 | 9 |
| MH | 6/7 | 2/3 | **8/10** | 4 | 9 |
| HM | 7/7 | 3/3 | **10/10** | 6 | 13 |
| HH | 7/7 | 3/3 | **10/10** | 7 | 56 |
| all | **27/28** | **11/12** | **38/40** | **5.5** | **56** |

Among solved scenes, HY5U used 269 simulator calls in total: mean 7.08, median 5.5, minimum 2, and maximum 56. Complete-scene success by total simulator-call cutoff was 8/40 at two calls, 19/40 at five, 34/40 at ten, 37/40 at thirty, and 38/40 at 100; every solve had occurred by 56 calls.

| failure | context | donor opening sequence | calls used | terminal reason |
|---|---|---|---:|---|
| `approval40_v3/contact/mh/contact_0000.xml` | MH contact | `[[11,4]] → [[20,2]]` | 1 | `region_path_exhausted` |
| `scale25_b5_v1/mh/composed_0002.xml` | MH clean | `[[11,4]] → [[6,4]]` | 1 | `region_path_exhausted` |

Both failed scenes remain known-solvable under current physics through their recorded donor actions, and neither exhausted the 900-call budget. In each trace, HY5U's first-ranked K1 push passed the local opening test in one simulator call, but the intended global path did not advance from K1 to K2: the profiled K2 object and its reachable edge set stayed unchanged, while the global boundary selection remained on K1 and the exact path stayed two hops. The opener then returned `already_accessible`, the repeat guard blacklisted that boundary, and Full NAMO ended with `region_path_exhausted`. This is a local-opening/global-progress mismatch after a greedy K1 commit, not a lack of simulator budget.

The read-only K1 audit preserved the exact profiled K2 interface in 37/38 solved scenes. The one exception, MH contact scene `contact_0002.xml`, still solved in three calls split `[1,1,1]`. That counterexample confirms that exact preservation is useful diagnostic evidence but remains too strict as an acceptance gate.

## Verdict

**ACCEPT HY5U as feasible on this controlled cohort.** It solved 95% of the complete scenes and all 20 scenes with hard K1, including every HH scene, while needing at most 56 simulator calls for a solve. The two misses isolate an outer Full-NAMO commit/progress failure rather than a weak local budget. Do not restore the strict K2-interface gate: one successful contact scene legitimately changed that audit. This run has no paired random arm, so it establishes HY5U's absolute performance on these 40 scenes but does not measure its advantage over random ordering.
