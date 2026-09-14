---
type: experiment
status: running
created: 2026-09-14
updated: 2026-09-14
commit: 6a488f1c
metric: "Full NAMO frozen400 wall time to success on whole Intel Xeon Platinum 8358 nodes (9000-call cap, single-threaded), split by easy/medium/hard/unresolved; HY5U seeds 1-3 search after the goal-picture fix, against timed Random."
tags:
  - experiment
  - full-namo
  - timing
---
# Timed Full NAMO after the goal-picture fix

## Why

The untimed rerun ([EXP-2026-09-14-full-namo-goal-fix-rerun](EXP-2026-09-14-full-namo-goal-fix-rerun.md)) measures simulator calls. Tri-An's timed campaign (`full_namo_sim/timed/full_namo_frozen400_icelake_20260912_v1`) timed only HY5U_s2, before the fix. [USER 2026-09-14] Start timed Full NAMO runs, all 3 HY5U seeds.

## Plan

Population and search settings are the untimed rerun's: frozen400 (manifest `d534b2b6`), 9000 calls shared across the whole problem, region depth 2, 5 push distances, 100 goal samples with 20 reachable, goal clearance on, shuffle seed 7000, Tri-An's config `a58e885f` with 1 mm inflation `f8ab35e1`. Timing on, statistics off.

Timing protocol copies Tri-An's campaign: every job holds a whole `main` node (`--exclusive`), CPU `Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz` only (the 91 other icelake nodes in his `full400-platinum-node-inventory.json` are excluded), every thread pool pinned to 1, CPU scoring, one scene at a time. `scripts/pipeline/eval_full_namo_walltime.py` refuses to run unless the allocation, CPU model, clean pinned commit, binding path and artifact checksums all match `hy5u.json`. Scenes are dealt round-robin into 40 jobs per seed. Each timed row records whether it reached the same outcome in the same number of calls as the untimed rerun row for that seed and scene (`pairing`).

Arms: HY5U_s1, HY5U_s2, HY5U_s3 search, checkpoints `ac43f004`, `3cf348cf`, `c596b09b`. Tri-An's timed data puts one seed at about 34 node-hours, with single scenes up to 78 minutes.

Random: Tri-An's timed Random (5 seeds, all on Platinum 8358) ran his container build `7ff23f27` on frozen code `25c921b`; ours is the native Amarel build `cc2d2e9e` on current code. Calibration before deciding whether to rerun it: Random seed 7000 on shard 0 of 20, the same 20 scenes as his `random_s7000/shard_0000`, compared scene by scene where the call counts match.

Gate: one smoke job per arm on scene 0 must pass preflight and reproduce the untimed call count exactly.

## Run

Amarel root `/scratch/dm1487/full_namo_timed_20260914/`: `hy5u.json`, `random_calibration.json`, `campaign.env`, `logs/`. Code `6a488f1c`. Smoke jobs `61621436`-`61621439` submitted 19:03.

## Result

Pending.
