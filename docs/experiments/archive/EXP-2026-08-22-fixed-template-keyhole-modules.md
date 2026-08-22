---
type: experiment
status: done
created: 2026-08-22
commit: f40e0a7
metric: 3/3 smoke cases solved
tags: [experiment, multihop, data]
---
# Fixed-template keyhole modules

## Hypothesis

_(you, from chat)_ Because the environments use fixed wall templates, independently sampled keyholes should be composable when their blockers and push areas do not interfere; begin with simple one-push tests, then two-push tests, before scaling.

## Plan

_(Codex)_ Extract blocker-only modules from exact canonical episodes on one Aug9 wall template, first verify that one canonical one-push action still opens a stripped single-keyhole scene, then verify that two one-push donors form and solve an exact two-hop chain, and finally test a genuine local pure-two-push donor separately. Every static scene must name the intended blockers in path order, and every dynamic check uses the simulator forward from the composed XML.

## Run

_(Codex)_ Ran on `ilab3` at implementation commit `f40e0a7` with `config/namo_config_complete_skill15_car_1x.yaml`. Donors were exact canonical v3 episode keys on `set2/benchmark_5`; each candidate started from the first donor XML, removed every movable body, inserted only the selected blocker bodies, used the first donor robot pose and the final donor goal, and required the static boundary sequence to be exactly `obstacle_0_movable`, then `obstacle_1_movable` where applicable. Dynamic validation pinned 100 points from every target component at the initial state and used the canonical 20/100 opening bar after each push.

_(Codex)_ Artifacts are under `$NAMO_SCRATCH/eval/keyhole_modules_20260822/{single_1push_medium,twohop_medium_medium,single_pure2push_medium}/`, each with the accepted XML, `manifest.jsonl`, and `summary.json`. The two-hop sampler required blocker-center separation of at least 0.30 m; it examined at most 100 medium-medium pairs and stopped on the first fully replayed solution. The pure-two-push replay required the donor setup to leave its target below 20/100 before enumerating finish pushes.

## Result

| case | local donor horizon | tier | static candidates | simulator calls to accepted replay | pinned-component trace |
|---|---|---|---:|---:|---|
| stripped single keyhole | 1push | medium | 1 | 1 | `[0] → [67]` |
| composed exact two-hop | 1push + 1push | medium + medium | 13 | 2 | `[0,0] → [96,0] → [96,100]` |
| stripped single keyhole | pure 2push | medium | 1 | 177 search calls; 2 in solution | `[0] → [0] → [100]` |

_(Codex)_ The exact two-hop solution used donor blockers 0.3545 m apart. Its actions were K1 `(edge=12, depth=1)` and K2 `(edge=17, depth=3)`. After K1, the middle component crossed the bar at 96/100 while the final component remained exactly 0/100; K2 then raised the final component to 100/100. This directly demonstrates preserved future access without imposing unchanged K2 pose or contact edges.

_(Codex)_ The genuine pure-two-push control used setup `(51,1)` and finish `(55,4)`. The setup left the target at 0/100, proving it was not an accidental one-push opener, and the finish raised it to 100/100. The 177-call search cost reflects enumerating current finish actions after known donor setups and is a pilot diagnostic, not a proposed scaled labeling procedure.

_(Codex)_ An initial diagnostic incorrectly checked only the XML goal point and rejected 28/28 valid one-push donors even though a tested donor action opened 67/100 component points. The XML point remained unreachable by design; replacing that check with the canonical pinned 20/100 criterion resolved the mismatch. This is the main validation failure mode to guard against in the scaler.

## Verdict

**ACCEPT the fixed-template blocker-module approach for a measured scaling pilot.** All three requested smoke cases passed, including the key condition `[0,0] → [96,0]`: K1 opened C2 without opening C3, and the original K2 action remained executable and solved C3. This proves feasibility, not population yield; the next run must report acceptance and replay yield separately for medium-medium, medium-hard, hard-medium, and hard-hard rather than extrapolating from one solution.
