---
type: experiment
status: complete
created: 2026-11-17
updated: 2026-11-17
metric: "One-keyhole frozen600 at 1 mm, the 300 two-push problems, best-first search with the depth cap at 2, 3, 4 and 5 pushes, 3000-call cap: solve rate against simulator calls, split by easy/medium/hard; no-family-local-edge seeds 1-3 against Random seeds 7000/8000/9000."
tags:
  - experiment
  - one-keyhole
  - search-depth
---
# Keyhole600 two-push problems at search depth 2 to 5

User request 2026-11-17: take every aug9_car problem that needs at least two pushes and run the best model with the search depth capped at 2, 3, 4 and 5. Run on Amarel [USER].

## Plan

Population. The 300 two-push problems of Tri-An's frozen one-keyhole 600 (manifest `bb01f360`, 1 mm, aug9_car templates), 100 easy, 100 medium and 100 hard by his exhaustive certificate. Every one has a 2-push solution, so depth 2 is enough for all of them. The question is what a deeper cap costs or gains when the problem does not need it.

Problems that need 3 or more pushes are not in this population. Tri-An's pool kept only certified 1-push and 2-push problems, and his full certification folder on Amarel is not readable to us. Our own testset_v3 sweep (5 mm) has 64 aug9 objects with no solution within 2 pushes, but counted per door, which is what the pooled search sees, only 7 of 693 aug9 test doors have none. Those 7 run separately as individual cases.

Model. HY5U_no_family_local_edge seeds 1-3 (`hy5u-no-family-local-edge-keyhole600-1mm-v1`), the best 2-push score within 5 calls of all the ablations: 57.0 against HY5U 51.7, and 57.2 against no-family 54.7 on one machine (p = 0.22, so tied with no-family). Random is uniform order, seeds 7000/8000/9000.

Search. Identical to the registered keyhole600 protocol except `hmax`: best-first, 3000 calls, mean5, raw q, discount off, no-op dedupe and jam pruning on, frozen target points and door objects, solved at 20% of target points reachable. `run_one_keyhole_frozen.py` now reads `hmax` from the arm (default 2, so registered arms record the same params as before).

Machine. Amarel `main`, our native build `cc2d2e9e`, for all four depths including depth 2. The registered depth-2 rows for this model are from rlab1, and CS and Amarel disagree on about 4% of problems, so depth 2 reruns here. Problems 373, 463 and 504 refuse to start on Amarel's build and run on arrakis, as in the registered run.

## Run

Commit `34e3fdda` (runner reads `hmax` from the arm), source fingerprint `4e57898f` on both machines. 24 arms: the model seeds 1-3 and Random seeds 7000/8000/9000, each at depth 2, 3, 4 and 5. Checkpoint sha256 `b5bf52ed`, `35d5392b`, `bc395d09` for seeds 1-3, the same files on both machines.

Amarel, binding `cc2d2e9e`: smoke on problems 550, 598, 554, 405, 333, then array 64005535 on partition `main`, 36 shards of 14 CPUs, 297 problems x 24 arms. Shards 2 and 25 were preempted after 17 and 16 minutes with one row each unfinished (`nfle_s3_h2` problem 543, `nfle_s2_h4` problem 593); job 64006908 reran those two shards from the same commit and settings, skipping rows already saved. Completed shards took 9 to 36 minutes. 7,128 rows.

arrakis, binding `fe80fb00` (the build the registered arrakis rows used): problems 373, 463 and 504, which refuse to start on Amarel's build, 72 rows, CPU scoring (`CUDA_VISIBLE_DEVICES=""`; the first attempt refused to score on GPU).

Checks: 7,200 of 7,200 rows, each arm and problem exactly once, zero failed runs, zero rejected rows, `search.hmax` equal to the arm's depth in every row, one checkpoint hash per model seed across both machines. Every unsolved run ended at the 3000-call limit; none ran out of pushes to try.

Report: `scripts/pipeline/report_keyhole_depth_sweep.py` at `b3ab2e6b`, output `eval/keyhole_depth_20261117/report.{json,md}` on CS.

## Result

Percent of runs solved within N simulator calls, three seeds pooled (seed ranges in `report.md`). Median calls counts a failed run as never solved. The last column is the share of solved runs whose plan has more than 2 pushes, although every problem has a 2-push plan.

| tier | arm | depth | @5 | @30 | @100 | @3000 | median calls | plans > 2 pushes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| easy | model | 2 | 83.3 | 94.0 | 98.7 | 100.0 | 2 | 0% |
| easy | model | 3 | 84.3 | 94.3 | 97.7 | 100.0 | 2 | 9% |
| easy | model | 4 | 85.0 | 94.3 | 97.3 | 100.0 | 2 | 10% |
| easy | model | 5 | 85.0 | 94.3 | 97.3 | 100.0 | 2 | 10% |
| easy | Random | 2 | 29.0 | 88.7 | 98.3 | 100.0 | 9 | 0% |
| easy | Random | 3 | 45.7 | 91.7 | 98.7 | 100.0 | 6 | 63% |
| easy | Random | 4 | 49.3 | 92.3 | 97.7 | 99.7 | 6 | 69% |
| easy | Random | 5 | 49.3 | 92.3 | 97.0 | 99.3 | 6 | 72% |
| medium | model | 2 | 54.7 | 87.7 | 93.3 | 99.7 | 4 | 0% |
| medium | model | 3 | 61.0 | 87.3 | 93.3 | 99.7 | 4 | 20% |
| medium | model | 4 | 61.3 | 88.7 | 94.7 | 99.7 | 4 | 21% |
| medium | model | 5 | 61.3 | 88.7 | 93.7 | 99.3 | 4 | 21% |
| medium | Random | 2 | 9.3 | 48.0 | 78.0 | 100.0 | 33.5 | 0% |
| medium | Random | 3 | 11.7 | 61.7 | 81.0 | 99.7 | 18 | 79% |
| medium | Random | 4 | 14.0 | 63.7 | 80.7 | 97.0 | 15 | 83% |
| medium | Random | 5 | 14.3 | 67.7 | 79.7 | 93.3 | 13 | 86% |
| hard | model | 2 | 30.7 | 62.0 | 70.7 | 98.0 | 14 | 0% |
| hard | model | 3 | 37.7 | 72.3 | 80.3 | 96.0 | 10 | 42% |
| hard | model | 4 | 38.7 | 73.0 | 81.7 | 93.7 | 8 | 43% |
| hard | model | 5 | 38.7 | 73.3 | 81.3 | 91.7 | 8 | 43% |
| hard | Random | 2 | 1.0 | 10.3 | 22.3 | 92.7 | 355 | 0% |
| hard | Random | 3 | 2.3 | 28.7 | 49.0 | 94.3 | 112 | 93% |
| hard | Random | 4 | 3.0 | 38.3 | 54.3 | 88.7 | 72 | 95% |
| hard | Random | 5 | 3.0 | 43.0 | 56.0 | 87.7 | 53.5 | 96% |

Paired against depth 2, same problem and seed (fewer calls / more calls, sign test; the 300 pairs are 100 problems x 3 seeds, so the p-values are optimistic):

| tier | model d3 | model d4 | model d5 | Random d3 | Random d4 | Random d5 |
|---|---|---|---|---|---|---|
| easy | 11/17, p 0.34 | 11/17, p 0.34 | 11/17, p 0.34 | 139/86, p 5e-4 | 155/71, p 2e-8 | 149/77, p 2e-6 |
| medium | 46/28, p 0.05 | 50/24, p 0.003 | 50/24, p 0.003 | 164/104, p 3e-4 | 170/95, p 5e-6 | 168/98, p 2e-5 |
| hard | 97/51, p 2e-4 | 99/48, p 3e-5 | 98/50, p 1e-4 | 207/81, p 7e-14 | 205/85, p 1e-12 | 207/84, p 4e-13 |

A deeper cap helps the model early on medium and hard problems and costs it at the end. Hard problems solved within 30 calls go from 62.0 at depth 2 to 72.3 at depth 3, and the median drops from 14 to 10 calls; depth 4 and 5 add about one point more. Problems solved by the 3000-call limit drop from 98.0 to 96.0, 93.7 and 91.7, and all of those losses ran out of budget. Easy problems do not move. Depth 3 gets nearly all of the early gain and gives up the least at the end.

The early gain comes from longer plans, not from finding the 2-push plan sooner. Every model run that needed fewer calls than at depth 2 returned a plan longer than 2 pushes: 154 of 154 at depth 3 and 159 of 159 at depth 5. For Random it is 483 of 510 at depth 3 and 515 of 524 at depth 5. At depth 3, 42% of the model's solved hard runs return a 3-push plan even though a 2-push plan exists. Rows record plan length but not the pushes, so which pushes make up those plans is not checked here. Plan length at a deep cap is the first plan found, not the shortest.

Random gains far more from depth, because it rarely picks the exact setup and finish pair: hard median 355 calls at depth 2, 112 at depth 3, 53.5 at depth 5, with 96% of its depth-5 solutions longer than 2 pushes. It also loses more at the end (hard 92.7 to 87.7, medium 100.0 to 93.3 at depth 5).

The model beats Random at every depth and tier. The gap narrows as depth grows: hard within 30 calls, 62.0 against 10.3 at depth 2 and 73.3 against 43.0 at depth 5; hard median, 14 against 355 calls at depth 2 and 8 against 53.5 at depth 5.

Depth 2 here gives 56.2 within 5 calls over all 2-push problems, against 57.0 in the registered CS run of the same checkpoints. Those are different machine families and are not compared further.

This population has no problem that needs 3 or more pushes, so it measures what a deeper cap does when the problem does not need it. It does not answer how the model does on problems that need 3 to 5 pushes.
