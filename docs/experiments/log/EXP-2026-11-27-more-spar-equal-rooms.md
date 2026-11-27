---
type: experiment
status: running
created: 2026-11-27
updated: 2026-11-27
metric: "Frozen one-keyhole 600 at 1 mm, 3000-call cap: solved within 1, 5, 30, 100 and 3000 simulator calls and median calls, by horizon (1-push/2-push) and certificate tier (easy/medium/hard), for SPAR and MORE's PushNet trained on the same rooms with the same number of training states per room; per-problem paired fewer-vs-more call counts with a two-sided sign test."
tags:
  - experiment
  - one-keyhole
  - external-baseline
  - more
---

# SPAR against MORE on the same rooms and the same amount of data

[USER 2026-11-27] "We need them to be trained on a large quantity of rooms. Same rooms?? And same amount of data". Then, offered equal simulator calls or equal training states on the same rooms: "Which one is better", and on the recommendation of equal states with MORE given the larger simulator budget: "Do it".

Follows [EXP-2026-11-24-more-faithful](EXP-2026-11-24-more-faithful.md), where both methods had about 24.6M simulator calls but SPAR's came from 198,267 rooms and MORE's from 34,336.

## Plan

Rooms. MORE's full collection (86,272 episodes, 43.1M simulator calls) has episodes in 59,954 of its 85,689 manifest rooms; in the other 25,735 its collector found the goal already open or reachable, or no goal region. All 59,954 are SPAR training rooms (`hybrid_train_v1.h5`), with 850,406 of SPAR's 1,302,659 states. Both methods train only on these rooms.

Held-out rooms. SPAR's trainer holds out whole rooms by its own rule (`Q2DataModule.setup`: rooms shuffled with seed 0, train filled to 90% of rows). That rule runs on SPAR's filtered H5, and the rooms it holds out are left out of MORE's data too, so both train on the same room set.

Amount of data. Equal training states per room: in each training room MORE keeps as many states as SPAR has there, drawn at random (seed 0) from MORE's states in that room; if MORE has fewer, it keeps all of them and the shortfall is drawn from rooms with spare states, so the totals match exactly. MORE's states cost its whole 43.1M-call collection, SPAR's a fraction of its 24.6M, so MORE gets the larger simulator budget. A SPAR win here would also hold at equal calls; a close result would call for an equal-calls run with a new SPAR collection.

Data build. Parts 0-4 of the collection are already in `more_224.h5` (49,336 episodes). Part 5 (36,936 episodes) is rendered at 224 px by `more_build_h5.py` at `2ed6206a`, the same code, on Amarel: 148 chunks of 250 episodes. A smoke first rebuilds chunk `p00_0000` on Amarel and must match the CS part `build224/p0000.h5` byte for byte. Row subsets are written by `scripts/pipeline/subset_h5_rows.py`.

Training. SPAR: the registered best model's recipe exactly (`hy5u-no-family-local-edge-keyhole600-1mm-v1`: `train_q2_round2.py` via `train.slurm`, worktree `namo-train-ablation-20260915` at `fc34b320`, `EGMM_LAMBDA=0 NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0`, 12 epochs, last checkpoint), only the H5 changes. MORE: PushNet with MORE's recipe as in the faithful card (`train_more.py --schedule more --invalid more --epoch-samples 65384` at `2ed6206a`), with `--holdout-frac 0` because the held-out rooms are already left out of its H5. Three seeds each.

Evaluation. Same population, runner and protocol as the faithful card (`run_one_keyhole_frozen.py` at `2ed6206a`, 3000-call cap, 2 pushes): 595 problems on CS EPYC 7352, problems 40, 277, 373, 463, 504 on arrakis; 8 torch threads per worker. Arms: SPAR best-first; MORE with MORE search (protocol stop) and with our best-first. Report by horizon and tier, per-problem sign tests, against Random and against the registered full-data rows of both methods.

## Run

## Result
