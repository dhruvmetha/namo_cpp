---
type: experiment
status: running
created: 2026-09-14
updated: 2026-09-14
commit: pending
metric: "Full NAMO frozen400 solve rate and simulated pushes (9000 cap per scene), split by easy/medium/hard/unresolved; HY5U seeds 1-3 search vs uniform Random seeds 7000-11000."
tags:
  - experiment
  - full-namo
  - rerun
---
# Full NAMO rerun after the goal-picture fix

## Why

Every HY5U_s2 Full NAMO number from Tri-An's 2026-09-12 campaign drew the model's goal picture around the final XML goal, even on hops where the robot first had to open a different room. The fix binds the local goal once per opening (namo `e3621da3`, Sage `5e871ab`). Those numbers are invalid until rerun. [USER] Rerun all 400 frozen scenes with three HY5U seeds; no policy mode, no timing.

## Plan

Population: Tri-An's frozen400 Full NAMO manifest (sha256 `d534b2b6`), 100 easy, 100 medium, 100 hard, 100 unresolved. Arms: HY5U_s1, HY5U_s2, HY5U_s3 best-first search (shuffle seed 7000) and uniform Random with shuffle seeds 7000, 8000, 9000, 10000, 11000. Random is rerun because the current code includes the 2026-09-12 room-finding change (`e460511c`) that Tri-An's Random runs did not have.

Settings copy Tri-An's reference helper (`candidate_campaign.py:222-230`): 9000 simulated pushes shared across the whole problem, region depth 2, 5 push distances, 100 goal samples with 20 reachable, goal clearance on, Tri-An's `namo_config.yaml` (sha256 `a58e885f`, robot trajectory collision checks on) with its 1 mm `wavefront_inflation.yaml`. Untimed; per-run statistics off.

Runner: `scripts/pipeline/run_full_namo_rerun.py`, launched per scene by `scripts/slurm/full_namo_rerun.slurm` (8 arms side by side, 8 CPUs). CS smoke on scene 0: HY5U_s2 solved in 3 pushes, Random s7000 in 9, peak memory about 1 GB.

## Run

Amarel root `/scratch/dm1487/full_namo_rerun_20260914`. Tracking board: Notion "Full NAMO rerun".

## Result

Pending.
