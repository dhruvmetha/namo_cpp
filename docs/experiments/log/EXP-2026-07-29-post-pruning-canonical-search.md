---
type: experiment
status: live
created: 2026-07-29
thread: rl_loop
robot: car
commit: 2d8b040
tags: [experiment, search, random-baseline, no-op-dedupe, jam-pruning, hmax2, canonical-eval]
---

# Post-pruning canonical search — learned ranker vs seeded random at hmax=2

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is a ranker that orders pushes for a simulator-verified search; this experiment measures whether it finds a verified opening with fewer simulator calls than random after the adopted search pruning.

## Hypotheses

With no-op child deduplication and jam-depth pruning enabled, the setup-only Colossus ranker will beat a three-seed random ranker in simulator calls on every easy/medium/hard tier of both canonical eval sets; allowing `hmax=2` on the 1-push set will also measure whether cheap two-push detours rescue ranker misses without changing the one-opening problem.

## Plan

Run the registered `namo_testset_v1` one-push key (1,323 episodes) and pure-two-push key (1,018 episodes) on Amarel with the same search: `hmax=2`, budget 900, `combine=q`, failure discount `conf` with `tau=0.15`, no-op dedupe on, and jam-depth pruning on. Evaluate the deterministic setup-only checkpoint once and random ordering at seeds 7000/8000/9000. Aggregate solve@{1,2,5,10,30,100,300,900} and simulator calls by easy/medium/hard/all; 1-push uses the fixed per-episode solve-rate bins (hard <0.05, medium <0.30, easy otherwise) and 2-push uses the registered fixed divisions file.

## Run

Committed implementation `2d8b040`, orchestration stamp `45899ba`. A dedicated Amarel clone avoids the existing dirty checkout. Scratch was quota-full, so the checksum-matched setup-only checkpoint and all outputs live under `/cache/home/dm1487/eval{_inputs,}/postprune_hmax2/`. The first binding build and smoke exposed that `scripts/amarel/activate.sh` changes the working directory back to the old clone; no eval ran. Rebuilding after activation with an explicit `cd` produced the dedicated-clone binding in job 59505171. Four one-scene target-box smokes (59505238–59505241) then passed for model/random × 1push/2push and recorded `hmax=2`, `dedupe_noop=true`, `prune_jam_depth=true`. The 12-scene pilots (59505343–59505346) completed in 37–38 seconds for 1push and 3:36–4:29 for 2push; production shards of 34/26 scenes therefore calibrate to roughly 2/8–10 minutes, with the three-hour limit providing ample tail margin. Post-run row-count validation caught that the 1push arrays had inherited a stale `MANIFEST` through Slurm's `--export=ALL`, limiting them to 112 of 1,323 episodes; the launcher now explicitly clears `MANIFEST` and sizes arrays by scene keys rather than episode counts, and only the missing 1push shards are rerun.

## Result

Pending.

## Verdict

Pending.
