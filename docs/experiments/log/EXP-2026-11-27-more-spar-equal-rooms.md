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

Training. SPAR: the registered best model's recipe exactly (`hy5u-no-family-local-edge-keyhole600-1mm-v1`: `train_q2_round2.py` via `train.slurm`, worktree `namo-train-ablation-20260915` at `fc34b320`, `EGMM_LAMBDA=0 NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0`, 12 epochs, best validation checkpoint as in [EXP-2026-09-15](EXP-2026-09-15-hy5u-no-family-local-edge.md), which was also the last there), only the H5 changes. MORE: PushNet with MORE's recipe as in the faithful card (`train_more.py --schedule more --invalid more --epoch-samples 65384` at `2ed6206a`), with `--holdout-frac 0` because the held-out rooms are already left out of its H5. Three seeds each.

Evaluation. Same population, runner and protocol as the faithful card (`run_one_keyhole_frozen.py` at `2ed6206a`, 3000-call cap, 2 pushes): 595 problems on CS EPYC 7352, problems 40, 277, 373, 463, 504 on arrakis; 8 torch threads per worker. Arms: SPAR best-first; MORE with MORE search (protocol stop) and with our best-first. Report by horizon and tier, per-problem sign tests, against Random and against the registered full-data rows of both methods.

## Run

Rooms (`select_equal_room_states.py rooms`, `select/rooms.json`). MORE has at least one state in 59,220 rooms: 734 of the 59,954 rooms with an episode saved no state. All 59,220 are SPAR rooms, and SPAR has 846,420 of its 1,302,659 states on them (`spar_eqr.h5`, 2.75 GB, sha256 `48159e67`). MORE has 3,192,809 states on them. SPAR's trainer split on this file (`spar-split`, its own `Q2DataModule` from `fc34b320`): 761,799 training states in 53,364 rooms, 84,621 held out in 5,856 rooms. MORE gets the same 761,799 states from the same 53,364 rooms.

Part 5 build on Amarel. The first smoke (`64509410`, 8 GB) was killed for memory while copying the 8.4 GB conda env into its tmpfs; at 20 GB (`64510601`) the env copy took 17 min and the chunk 12 min. The Amarel-built `p00_0000` equals CS `build224/p0000.h5` in every dataset, byte for byte (`compare_parts.py`). Full build: array `64512413`, 15 tasks × 10 CPUs on Gold 6230R, 148 chunks. `more_data_chain.sh` then copies the parts to CS, joins them (`more_merge_h5.py` at `2ed6206a`), checks the state total against the collection JSONL, draws MORE's states (`more-rows`) and writes `more_eqr_224.h5`.

Training placement. arrakis stopped answering ping and ssh at about 03:55 on 2026-11-27, before any job started there. ilab3's 8 GPUs are all allocated and our CS association runs on one node at a time (`AssocGrpNodeLimit`), and Amarel's GPU partitions had 732 jobs waiting. So all six runs train on westeros RTX 2080 Ti cards, the last-resort box, one run per GPU (tmux `eqr_train`, `run_queue.sh` + `train_eqr.sh`): SPAR seeds 1-3 on GPUs 0-2 from 04:00 and 04:12 (fp16 mixed precision, as registered; epoch 0 took 11 min), MORE seeds 1-3 on GPUs 3-5 once `more_eqr_224.h5` exists. Both methods train on the same GPU type.

SPAR training finished at 06:00, 06:15 and 06:08 for seeds 1, 2 and 3. Best validation checkpoints: seed 1 epoch 10 (val loss 0.2072), seed 2 epoch 9 (0.2176), seed 3 epoch 10 (0.2214); the last epoch was 0.2186, 0.2230 and 0.2267. These losses are on this card's held-out rooms, so they do not compare with the registered model's 0.33. MORE seeds 1-3 started at 04:27 and run about 7 minutes per epoch, so about 8.5 hours for 72 epochs.

SPAR evaluation: `eval/arms_spar.json` holds the three best checkpoints (sha256 `373c3a13`, `2fcba7d6`, `a4ca04a9`) and a control, seed 1 of the registered model (`b5bf52ed`), to check that the runner at `2ed6206a` on EPYC 7352 gives the registered rows. Job `1449811` on ilab3 (16 workers, 1 torch thread each, as the registered rows), 2,380 runs; it sat pending while ilab3 was held for another user's job planned for the next morning, and started once its limit was cut to 3 hours. The first 29 control rows match the registered rlab1 rows in solved, calls and execution digest. The job finished at 07:49 (1 h 32 min) with all 2,380 rows and no failed run, and all 595 control rows match the registered rows in solved, calls and execution digest.

SPAR on these rooms against SPAR on all its data (registered rows), per-problem median calls over three seeds, 592 problems where every arm below has all seeds: the equal-rooms model takes more calls on more problems in five of six tiers, significant on 1-push hard (fewer 16 / more 32, p = 0.029), 2-push easy (16 / 35, p = 0.011) and 2-push medium (25 / 46, p = 0.017); 2-push hard is even (38 / 42, p = 0.74). Within 5 calls, 2-push hard is 30.6 against 33.7 and 1-push hard 81.8 against 85.9. Against MORE's PushNet from the faithful card (trained on the 24.6M-call cut, 1.82M states), the equal-rooms SPAR takes fewer calls in every tier with both searches, p ≤ 0.038 everywhere except 1-push easy with best-first (11 / 5, p = 0.21); 2-push hard within 30 calls is 59.7 against 33.0. That is a preview; the comparison this card is for waits on MORE trained on these rooms.

MORE training finished at 12:07, 12:11 and 12:23 for seeds 1, 2 and 3, all 72 epochs (51, then 21 of fine-tuning), with final fine-tuning losses 0.487, 0.485 and 0.471. Each deploys `more_ft_epoch020.pt` as in the faithful card (sha256 `d0cbe6eb`, `585eb9d1`, `2d9a1017`).

MORE evaluation: `eval/arms_more.json` holds 6 arms, each seed with MORE search (protocol stop) and with best-first, in the same fields as the faithful card's PushNet arms. The faithful card's CS PushNet jobs still held ilab3 with two units left, and our CS account runs on one node at a time, so I cancelled them and reran those two units beside this eval (see that card). Job `1449930` started at 12:31 on rlab3 (EPYC 7352, another user's job using about 8 of its 96 CPUs): 9 workers × 8 torch threads, 3,570 runs (`eval/submit_more.sh`). 8 threads gave rows identical to 1 thread on ilab3; on rlab3, job `1449933` reran 4 finished units from the faithful card at 8 threads, and all 4 match the ilab3 rows in solved, total calls and calls to first solve (223, 175, 155, 179).

MORE moved to GPU at 13:25 [USER 2026-11-27: "What if we dont care about timing. Only success vs sims", then "Do it on GPUs"]. I cancelled CPU job `1449930` after about 200 rows; they stay in `eval/results_cs` and the result does not use them. GPU rows can differ from CPU rows by float rounding (the faithful card's 26 Nov check: 2 of 4 units off by one call), so every row of the MORE arms runs on GPU, SPAR's rows stay on CPU, and the comparison is success against simulator calls only. Jobs `1450001`-`1450007` on rlab3, one RTX A4000 each with 4 workers, run 7 shards of `eval/arms_more_gpu.json` (arm names end in `_gpu`) into `eval/results_gpu` (`eval/eval_gpu.sbatch`, `NVIDIA_TF32_OVERRIDE=0`). A first try with 8 workers per GPU (`1449992`-`1449998`) ran out of GPU memory on about 30 runs; those wrote no row, and the 81 runs that finished were kept. CS SLURM had no more GPUs: our account gets 8 on one node, and every other CS GPU node was full.

Network alone, no search [USER 2026-11-27, on MORE's native deploy loop: "Why do we need this? Our model has no physics change no, so the planner and the execution will be the same"; on MORE's network-only baseline: "Yes, this should be compared with SPAR-Policy". Then "Go"]. MORE's native loop (50 search iterations, execute the best push, search again) is not run here: the runner executes the state the search already simulated, so re-planning gains nothing and native only adds commitment. Both networks instead run alone through one loop, `more_stop_mode: "ppn"` in `more_vendor.py` (MORE's `ppn_main.py:380-470`): execute the network's top push at the live state, check the goal, at most 2 pushes, so at most 2 calls; a push that moves nothing is dropped and the next best is tried. SPAR-Policy (`eval/arms_spar_policy.json`, job `1450430`) feeds SPAR's ranker scores into that loop through `candidates()`, the pool best-first uses, on CPU like SPAR's other rows. MORE's (`eval/arms_more_ppn_gpu.json`, job `1450431`) runs on GPU like MORE's other rows. Check: on 1-push problems, SPAR-Policy solved at 1 call must equal SPAR best-first solved at 1 call, since both execute the same top push. Native and ppn arms queued at 14:36 were cancelled before they started.

## Result
