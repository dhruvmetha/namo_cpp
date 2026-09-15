---
type: experiment
status: running
created: 2026-09-15
updated: 2026-09-15
metric: "One-keyhole frozen600 at 1 mm, best-first search, 3000-call cap: simulator calls to success (untimed) and seconds to success on whole Platinum 8358 nodes (timed), split by 1-push/2-push and easy/medium/hard; HY5U with the family margin loss, local feature sampling and the contact-index embedding all removed, seeds 1-3."
tags:
  - experiment
  - ablation
  - one-keyhole
  - timing
---
# HY5U without the family loss, local feature sampling or contact-index embedding

## Why

[USER 2026-09-15] Train a model with no family loss, no local feature sampling and no edge identity, all three at once, 3 seeds on the best GPU we can get. Run inference for this model alone on Amarel, on the 1 mm one-keyhole set with 300 1-push and 300 2-push problems at 3000 simulator calls: once for simulator calls, once timed. No HY5U or Random controls get timed. Keep a Notion kanban board. "No family" means the registered no-family ablation (family margin loss off, same data), chosen by the user over dropping family boards from the training data.

On that set each change alone matched or slightly beat HY5U on two-push problems ([EXP-2026-09-14-keyhole600-ablations](EXP-2026-09-14-keyhole600-ablations.md)). This run removes all three.

## Plan

Training copies HY5U and the single ablations: `scripts/rl_loop/run_hy5u_arch_ablations_cs.sh` arm `HY5U_no_family_local_edge` sets `EGMM_LAMBDA=0 NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0 NAMO_GLOBAL_READOUT=0`. Everything else is shared: `aquaman/round0/hybrid_train_v1.h5`, `train_q2_round2.py`, 12 epochs, batch 256, `NAMO_GAMMA=0.5`, `NAMO_UNREACH_W=0.1`, grouped episode batches kept on (as in the registered no-family arm, so only the loss changes), `RANK_LAMBDA=0.1`, `LOWER_RANK_LAMBDA=0.05`, edge self-attention on, action motion off, seeds 1-3, best validation checkpoint. One 1-epoch smoke on seed 1 runs first. With local sampling and the index embedding both off, a contact's token is only the Fourier encoding of its position; `python/tests/test_action_motion.py` checks that reordering the contacts only reorders the scores.

GPUs: arrakis RTX 6000 Ada, one seed each. The only faster GPUs we can reach, ilab4's 8 RTX 5000 Blackwell, are all in use, and Amarel's GPUs are Ada or Ampere and in maintenance. Each seed gets 4 data-loader workers instead of 6, because another user's jobs hold about 18 of arrakis's 32 cores; that changes only the random data order.

Amarel is in maintenance from 2026-09-15 08:00 to 2026-09-16 23:59, so inference moved to the CS cluster [USER 06:50, chosen over queueing on Amarel].

Sim-count test: `scripts/pipeline/run_one_keyhole_frozen.py` exactly as in the keyhole600 ablations (search, 3000 calls per problem, statistics on), on the CS cluster, with the problems a node refuses run on arrakis. Results slot into the `hy5u-ablations-keyhole600-1mm-v1` table next to HY5U, no-family, no-local, no edge identity and Random.

Timed tests [USER 06:40, 06:50]: first the registered no-family model right away, then the new model. Tri-An's timed Random and geometric keyhole runs used Amarel's Platinum 8358 nodes in his container build, so they cannot be compared with CS timings; Random seeds 7000-11000 and geometric get timed again next to no-family. The best-first search already records total, simulator and scoring time when `record_timing` is on; the keyhole runner had it off. `--timed` (commit `cf64a879`) turns timing on and statistics off and runs one unit at a time in the runner process with every thread pool at 1 and CPU scoring. `scripts/slurm/one_keyhole_timed_cs.sbatch` runs one job per node on rlab3, rlab4 and ilab3, the three idle nodes with the same CPU (AMD EPYC 7352, 2 sockets of 24 cores, 16 L3 cache groups of 3 cores). Each job starts 16 processes, each pinned to the first CPU of its own cache group. The 5,400 (problem, arm) units are shuffled with seed 20260915 and dealt across the 48 processes, so every model runs on every node at every time and background load from other users falls on all models alike. Each row records host, CPU model, pinned CPU and load average. No-family rows must match the untimed keyhole600 rows' result and call count; Random rows are checked against Tri-An's Random rows, and geometric against his geometric rows.

## Run

A first version of this card planned only no-local plus no edge identity (commit `e8f2ac4d`, arm `HY5U_no_local_no_edge`). Its smoke started at 05:44 on arrakis GPU 0, which another user had just started using, so I stopped it and relaunched on GPUs 1-3 at 05:45. [USER 05:47] stopped that run about 2 minutes in, before any model trained, and asked for the three-change model instead. Both partial smoke folders stay under `$NAMO_SCRATCH/aquaman/round0/architecture_no_local_no_edge_20260915/`.

Training: worktree `ktamp/namo-train-ablation-20260915` pinned at `fc34b320`, output `$NAMO_SCRATCH/aquaman/round0/architecture_no_family_local_edge_20260915/`, launched 06:36 with `GPU_LIST="0 4 0"` because another user held GPUs 1-3. The smoke passed at 06:49 (one epoch in 13 minutes, `EGMM_LAMBDA=0.0` in the log, scorer-load check OK) and the three seeds started. Seed 3 shared GPU 0 with seed 1 and had not finished an epoch after 15 minutes while seed 2, alone on GPU 4, had; when GPU 3 freed up I stopped seed 3, moved its partial folder to `aborted_gpu0_shared/`, and restarted it at 07:06 alone on GPU 3 with the launcher's exact environment, calling `train.slurm` directly. The supervisor log will therefore list seed 3 as failed.

Timed no-family, Random and geometric: worktree `ktamp/namo-keyhole-timed-20260915` pinned at `cf64a879` with its own copy of `build_python`, output `$NAMO_SCRATCH/eval/keyhole600_timed_cs_20260915/` (`arms.json`, `rows/`, `slot_logs/`, `logs/`), source fingerprint `ff07e56f`. That fingerprint differs from the untimed keyhole600 runs' `22577f4a` only through `scripts/pipeline/eval_full_namo_walltime.py`, which the keyhole search does not use. A 6-unit smoke on arrakis and a 27-unit smoke on rlab3 (job `328354`, problems 0, 40 and 450, all 9 arms) matched the untimed no-family call counts and Tri-An's Random call counts on every compared unit; geometric took 248 calls on problem 450 against Tri-An's 249, on both boxes. Problem 40 refused to start on rlab3, as on Amarel's build. On problem 450 rlab3 took 1.28 times Tri-An's Platinum time for Random. Full jobs started 07:03 and 07:04: `328359` rlab3 (slots 16-31), `328360` rlab4 (slots 32-47), `328362` ilab3 (slots 0-15). The remote shell on ilab2 is zsh, whose arrays start at 1, so the first submit loop failed for ilab3's slot range and shifted rlab3 and rlab4 up one range; a resubmit for ilab3 with slots 32-47 (`328361`) ran 19 seconds before I cancelled it. Units in slots 32-47 finished in those 19 seconds may have been written by ilab3 and then rewritten by rlab4, and the logs `slot_32.log` to `slot_47.log` were truncated once.

## Result

### Timed no-family, Random and geometric on the CS nodes (complete 09:03)

All 5,355 timed rows saved: 595 problems for each of 9 arms. Problems 40, 277, 373, 463 and 504 refused to start on all three nodes, as on Amarel's build and rlab1, and no other unit failed. Tables from `scripts/pipeline/report_one_keyhole_timed.py`, saved to `$NAMO_SCRATCH/eval/keyhole600_timed_cs_20260915/report.{json,md}`. Ranges run over seeds: no-family 3, Random 5, geometric 1. A failed run counts as infinite time.

Median seconds until solved:

| | easy | medium | hard | all |
|---|---:|---:|---:|---:|
| 1-push, no-family | 0.55-0.57 | 0.75-0.77 | 0.84-0.96 | 0.71-0.73 |
| 1-push, Random | 0.22-0.51 | 0.88-1.18 | 1.75-2.52 | 0.82-1.03 |
| 1-push, geometric | 0.27 | 0.64 | 1.82 | 0.62 |
| 2-push, no-family | 1.43-1.59 | 1.96-2.37 | 5.57-6.37 | 1.95-2.18 |
| 2-push, Random | 2.12-3.16 | 6.05-11.60 | 64.04-108.17 | 5.93-11.93 |
| 2-push, geometric | 3.63 | 17.15 | 83.08 | 18.21 |

Solved within 1 and within 5 seconds, percent of runs:

| | 1-push 1 s | 1-push 5 s | 2-push 1 s | 2-push 5 s | 2-push hard 5 s | 2-push hard 30 s |
|---|---:|---:|---:|---:|---:|---:|
| no-family | 76.5-82.6 | 97.3-98.0 | 9.8-12.5 | 68.0-70.7 | 45.5-47.5 | 70.7-73.7 |
| Random | 48.7-58.7 | 85.9-93.3 | 4.7-8.1 | 35.0-46.1 | 6.1-11.1 | 21.2-31.3 |
| geometric | 62.4 | 77.2 | 8.4 | 35.7 | 15.2 | 26.3 |

Solved within 3000 calls: 1-push 100% for all three; 2-push no-family 98.7-99.0, Random 96.3-98.7, geometric 95.6.

**No-family is the fastest on every 2-push tier by a wide margin.** Its median time on 2-push problems is 2 seconds against 6-12 for Random and 18 for geometric, and on hard 2-push problems 6 seconds against 64-108 and 83. Within 5 seconds it solves 46-48% of hard 2-push runs; the best Random seed solves 11%.

**On 1-push problems the gap is small, and geometric's median is lowest.** Median times are 0.6-1.0 seconds for everyone, because most 1-push problems fall on the first or second push whatever the ordering. Running the network takes 33-34% of no-family's time on easy 1-push problems, against 2% for geometric's scoring, so geometric's median of 0.62 s edges out no-family's 0.71-0.73 s. On 2-push problems scoring falls to 3-4% of no-family's time. No-family still solves the most within 1 second (77-83% against 62% and 49-59%), and it is faster on hard 1-push problems (0.84-0.96 s against 1.82 and 1.75-2.52).

Checks:

- The three nodes ran at the same speed. Median seconds per simulator call, taken over runs with at least 20 calls, differ by under 5% across rlab3, rlab4 and ilab3 for every model (Random 0.237-0.250, geometric 0.281-0.294, no-family 0.342-0.354). ilab3 carried other users' load at times (load average up to 92 on 96 threads, median 17.5), rlab3 and rlab4 stayed near our own 16 processes.
- Timed no-family rows match the untimed Amarel keyhole600 rows on 570-574 of 595 problems per seed. Per seed, 3 problems changed between solved and failed, and the other 18-22 differences are call counts split evenly between more and fewer (median difference 0 or -1). Random rows match Tri-An's untimed rows on 551-561 of 595, and geometric matches his timed rows on 532 of 595. The known cross-box difference in the state restore path (`reference_crossbox_physics_identical`) fits this pattern; I did not trace individual problems. The comparison between arms is unaffected, since every arm here ran on the same nodes and build.
- 7 rows in rlab4's slots came from the 19-second duplicate job on ilab3 (same CPU model).
