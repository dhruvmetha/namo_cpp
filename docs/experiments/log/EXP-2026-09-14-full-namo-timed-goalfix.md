---
type: experiment
status: complete
created: 2026-09-14
updated: 2026-09-15
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

Amarel root `/scratch/dm1487/full_namo_timed_20260914/`: `hy5u.json`, `random_calibration.json`, `campaign.env`, `logs/`. Code `6a488f1c`. Smoke jobs `61621436`-`61621439` (submitted 19:03) passed preflight and matched the untimed call counts: 5, 3 and 5 for HY5U seeds 1-3, 9 for Random.

Arrays `61621621`, `61621622`, `61621623` (HY5U_s1-3, 40 jobs each) and `61621624` (Random seed 7000, shard 0 of 20) started at 19:06. `main` preempted three jobs: `61621622_21` after 82 minutes, `61621623_17` after 37 and `61621623_38` after 23. I moved their partial folders to `hy5u/raw/<arm>/preempted_shard_NNNN_job<id>` and reran each shard whole as `61631494_21`, `61632093_17` and `61632682_38`. The 21 scenes the preempted jobs had saved came back from the reruns, on other nodes, with the same result and the same calls, and the 9 of them that took over 5 seconds came back within 0.3% of their first time. The last HY5U job ended 00:17 on 2026-09-15 and the Random calibration at 00:21.

`eval_full_namo_walltime.py --report` accepted all 1,200 rows (400 scenes per arm, one CPU model, one commit, one campaign config) and wrote `hy5u/report.json`. `scripts/pipeline/report_full_namo_timed.py` compares them with Tri-An's timed Random and writes `report_timed.json` and `report_timed.md`. CS copy, without the preempted partial folders: `$NAMO_SCRATCH/eval/full_namo_timed_20260914/`.

## Result

Median seconds until solved, as seed ranges. A failed run counts as infinite time, so "not reached" means most runs failed. HY5U is 3 seeds on our build. Random is Tri-An's 5 seeds on his build, and the scaled row multiplies his times by 0.867, the build ratio measured below.

| | easy | medium | hard | unresolved |
|---|---:|---:|---:|---:|
| HY5U, 3 seeds, our build | 1.0-1.1 | 2.4-3.8 | 40.2-92.0 | not reached |
| Random, 5 seeds, Tri-An's build | 1.5-2.6 | 10.5-15.0 | 175.6-464.0 | not reached |
| Random scaled to our build (estimate) | 1.3-2.2 | 9.1-13.0 | 152.2-402.2 | not reached |

Solved within 1, 5 and 60 seconds, percent of runs, seed ranges. On unresolved scenes HY5U solves at most 1% within 60 seconds and Random none.

| | easy 1 s | easy 5 s | easy 60 s | medium 1 s | medium 5 s | medium 60 s | hard 1 s | hard 5 s | hard 60 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HY5U, 3 seeds, our build | 42-49 | 83-90 | 91-95 | 8-12 | 56-73 | 86-90 | 2-7 | 31-36 | 48-53 |
| Random, 5 seeds, Tri-An's build | 19-28 | 72-87 | 92-97 | 0-4 | 16-32 | 81-89 | 0-2 | 6-15 | 13-34 |
| Random scaled to our build (estimate) | 25-34 | 74-89 | 92-97 | 0-5 | 18-35 | 82-90 | 0-3 | 7-15 | 14-36 |

Median seconds until solved per HY5U seed, easy/medium/hard: s1 1.1/3.8/92.0, s2 1.0/2.4/40.2, s3 1.1/3.1/49.7. Solved within the 9000-call cap is unchanged from the untimed rerun: HY5U 93-97/91-93/67-75/1-4, Random 93-97/88-93/56-70/0.

**The timed runs reproduce the untimed rerun exactly.** All 1,200 HY5U runs reached the same result in the same number of simulator calls as the matching untimed run.

**HY5U is faster than Random in seconds on easy, medium and hard, and no seed ranges overlap.** Median time until solved is 1.0-1.1 s against 1.5-2.6 s on easy, 2.4-3.8 s against 10.5-15.0 s on medium, and 40-92 s against 176-464 s on hard. On hard, 48-53% of HY5U runs finish within a minute; the best Random seed manages 34%. Taking 13% off Random's times for the build difference leaves every gap in place.

**Model scoring costs HY5U little time.** Scoring takes 5.6-7.2% of HY5U's time (s1 7.2%, s2 6.7%, s3 5.6%), so the untimed rerun's call savings show up in seconds too.

**Our build runs 13% faster than Tri-An's on the same CPU.** Random seed 7000 on his first 20 scenes: 19 reached the same result in the same number of calls, and on those ours took 0.867 of his time overall, 0.836-0.901 per scene (18 scenes over half a second). The 20th scene, `756175dd`, failed in both runs, after 5,686 calls in ours and 5,678 in his. I did not look into why. For scale, the untimed rerun's new-code Random rows matched Tri-An's frozen-code call counts on 1,536 of 1,716 runs.

**HY5U_s1 is again the slow seed on hard.** Its median is 92.0 s against 40.2 s and 49.7 s, and the whole seed took 34.0 hours of run time against 28.9 and 26.9.

Open decision [waiting on Dhruv]: time Random again on our build. Tri-An's five seeds took 190 node-hours on his build, which comes to about 165 on ours. Split into 40 jobs per seed like HY5U, the slowest job would run about 2.3 to 2.8 hours on our build. Until then, HY5U and Random seconds come from two builds, and the scaled row is only an estimate.

Caveats:

- Random's times come from Tri-An's container build `7ff23f27` on frozen code `25c921b`. HY5U's come from the native Amarel build `cc2d2e9e` on `6a488f1c`. Both ran on whole Platinum 8358 nodes with one thread.
- The scaled row applies one ratio, measured on 18 scenes of one Random seed, to all five seeds and 400 scenes.
- Random's five untimed runs chose the tiers, so the selection bias noted on the untimed card applies here too.
- No horizon split: every frozen400 scene starts two hops from the goal (`horizon_pattern: not_refreshed`).
