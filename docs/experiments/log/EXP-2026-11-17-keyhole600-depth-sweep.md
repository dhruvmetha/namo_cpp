---
type: experiment
status: running
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

## Result
