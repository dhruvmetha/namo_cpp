---
type: experiment
status: running
created: 2026-11-24
updated: 2026-11-24
metric: "Frozen one-keyhole 600 at 1 mm, 3000-call cap: solved within 1, 5, 30, 100 and 3000 simulator calls and median calls, split by horizon (1-push/2-push) and certificate tier (easy/medium/hard), for MORE's network and our EdgeCrossAttn (both trained on MORE's labels, loss and schedule at SPAR's simulator-call budget) under MORE's deploy search (protocol and native stopping), MORE's network-only policy (ppn) and our best-first search; against the best model (hy5u-no-family-local-edge-keyhole600-1mm-v1) and Random, with per-problem paired fewer-vs-more call counts."
tags:
  - experiment
  - one-keyhole
  - external-baseline
  - more
---
# MORE done faithfully: its search, collection and training at SPAR's data budget

[USER 2026-11-24] "We need to make sure MORE is implemented correctly okay? And we should test MORE in its own model version with its own style of search/no search or whatever, and then MORE with our architecture and still its own style of search yes?" Then: "Do a proper implementation of MORE or pull in the code and adapt, because we dont want to be running buggy shit, plus keep grounding yourself in what MORE does from its paper and codebase." And: "We have a lot of CPU cores on amarel, please use it to the fullest if needed."

This replaces [EXP-2026-11-20](EXP-2026-11-20-keyhole600-more.md), whose network was trained with five port bugs (audit `$NAMO_SCRATCH/more_audit_20261124/SPEC.md`). MORE means Huang, Guo, Boularias, Yu, ICRA 2022 (arXiv 2202.01426v3) and github.com/arc-l/more at `70402c0`.

## Plan

Code. Branch `more/faithful-port`, worktree `ktamp/namo-more-faithful-20261124`. MORE's search files (`mcts/`, `mcts_network/`, `constants.py`) are imported unmodified from the pinned clone at run time; our side is an adapter that replaces their PushState and MCTSHelper (simulator, action library, reward). Every deviation is listed with its MORE line in the branch's `DECISIONS.md` (D1-D10, S1-S4). 487 tests, including 300 differential toy runs where the vendored search and a line-by-line transliteration of MORE's code make identical simulator calls.

Collection (MORE's System 2, `mcts_main.py`). Plain UCT, 300 iterations per search, uniform random rollouts, no early stop; every tree saved as `save_mcts_data` does (one row per simulated child: label max(child.q), weight min(visits, 50)); execute `best_child_top`, search again, up to 2 pushes. Rooms: the 85,689 SPAR training rooms that still exist (`antman0` and `colossus0_1m`; the `curriculum2/xml_pool` rooms are gone everywhere), in one fixed shuffled order (manifest sha256 `e37e835d`), one episode per (room, blocking object).

Budget. SPAR's corpus cost about 24.6M simulator calls. The collection is cut at 24.6M calls in shuffled-room order, at room boundaries (`more_cut.py`). MORE's own paper used 65,384 transitions, about 375 times fewer. Cuts at 1.5M, 3M, 6M and 12M calls are learning-curve points, trained if GPU time allows.

Training (DECISIONS D9). MORE's recipe from `lifelong_trainer.py`: SGD, batch 8, CosineAnnealingWarmRestarts(T_0=50, T_mult=2) stepped per iteration in epoch 0 and per epoch after, 51 epochs, then the paper's fine-tune stage (batch 28, lr 1e-5, 21 epochs); SmoothL1(β=0.8) × visit weight; unreachable pushes as MORE's label 0 at weight 0.001 (`--invalid more`). An "epoch" is 65,384 samples, MORE's dataset size, so the optimizer steps and learning-rate curve are MORE's exactly on a 150 times larger collection. The last checkpoint is deployed, as MORE does (`README.md:81`, `ppn_main.py:334`). Two networks on the same labels: MORE's PushNet (ResNet-34 FPN, 224 px, rotated per action) and our EdgeCrossAttn (64 px, one value per edge and depth).

Arms, per network (8 arms):
- MORE search, protocol stop: one deploy search (MORE's tree policy, greedy rollouts on the network), stopped at the first verified opening, 3000-call cap.
- MORE search, native: `more_main.py` as written, MORE-50 per decision, execute, re-search; every search charged.
- MORE without search (ppn): `ppn_main.py`, execute the network's top action at the live state, up to 2 pushes.
- Our best-first search with the network as the prior.

Population and protocol. Tri-An's frozen one-keyhole 600 (manifest `bb01f360`, 1 mm), same config, target points, door objects and post-restore list as the registered best-model rows and the pilot; 2 pushes, 3000 calls, dedupe and jam pruning on, solved at 20% of target points, seed base 7000. Runner `scripts/pipeline/run_one_keyhole_frozen.py`. Machines as the pilot: CS EPYC nodes for 595 problems, arrakis for 40, 277, 373, 463, 504.

Report. `report_one_keyhole_frozen.py` by horizon and tier, versus the best model and Random; sign tests on per-problem paired calls. Network checks with `verify_more_net.py` on the 10% held-out training rooms.

## Run

Amarel smoke `64322248` failed on a missing `git` on compute nodes; fixed by reading the commit from the worktree. Collection array `64323189` (200 tasks × 8 cores, `main`, 4529a5d7), 2026-11-24 03:24. SLURM preempted tasks 48, 49 and 111 (24 collectors) after 41 minutes without requeueing them; rerun `64330348` finishes their slices from each collector's last room (records of a room in a rerun replace the killed collector's partial ones). First rerun submit `64330330` was cancelled within a minute: its collectors were started in a piped subshell the job did not wait for. Measured on 1,077 early episodes: 487 calls, 36 saved states and 208 rows per episode, 77% solved.

Room XMLs copied to CS (they were not there), all 85,689 size-checked against Amarel. H5 build (`more_build_h5.py` in parallel episode chunks, joined by `more_merge_h5.py`): a chunk built on ilab1 under SLURM is bit-identical to the same chunk built on arrakis. Training smoke: PushNet 334 samples/s at batch 8 on an RTX 6000 Ada (about 3.6 h for the full schedule), EdgeCrossAttn 380 states/s (about 2.8 h).

Collection finished: 86,272 episodes, 43.1M simulator calls, 1,819 core-hours, no environment errors. The 24 rerun collectors replayed the 27 episodes they shared with the preempted ones identically. The 24.6M-call cut is 49,336 episodes from 34,336 rooms (77.7% solved, 10.37M rows, last manifest index 48,915).

H5 build arrays `1423588` (224 px) and `1423589` (64 px), 200 episode chunks each, on ilab3 at 47 tasks at a time; ilab1 had 27 free slots and rlab7's memory was fully reserved. Our CS SLURM association runs on one node at a time and gives every job 2 CPUs. `more_224.h5`: 1,820,495 states, 49,336 episodes, 0 skipped, 21 GB, sha256 `2e2322f5`. Its cut prefixes are 112,145, 221,815, 446,152 and 887,582 states at 1.5M, 3M, 6M and 12M calls.

PushNet seeds 1-3 train on arrakis GPUs 0-2 (tmux `more_train`) from a RAM copy of `more_224.h5` whose hash matches, at 335 samples/s each, about 3.9 h. EdgeCrossAttn seeds 1-3 go to SLURM when the 64 px build is done (`launch_slurm_eca.sh`).

`more_64.h5`: the same 1,820,495 states and the same cut boundaries as the 224 px file, 5.9 GB, sha256 `be25e034`. EdgeCrossAttn seeds 1-3 are SLURM jobs `1424071`-`1424073` on ilab1 (RTX A4500), started 09:26, about 4.5 min per epoch. Both trainers leave out the same 3,393 held-out rooms, which leaves 1,639,633 states and 9,341,612 tree rows.

Evaluation placement. ilab1 and arrakis are EPYC 7313; the pilot checked only the EPYC 7352 nodes (ilab3, ilab4, rlab3, rlab4) against the best model's rlab1 rows. So the 595 problems run on one 7352 node after the training frees our one-node slot, and problems 40, 277, 373, 463 and 504 run on arrakis (`eval/arrakis_run.sh`). Arms: `eval/arms_full.json`, 24 arms = {PushNet, EdgeCrossAttn} × seeds 1-3 × {MORE protocol, MORE native, ppn, best-first}, each on its last fine-tune checkpoint (`more_ft_epoch020`).

PushNet seeds 1-3 finished at 11:42 (final fine-tune loss 0.502, 0.506, 0.485). `verify_more_net.py` on the last checkpoint, 1,000 states from the 3,393 held-out rooms (`verify/pushnet_s{1,2,3}.txt`): all three usable. Openings score 0.977, 0.992, 0.980 against 0.233, 0.255, 0.246 for non-openings; correlation with the sampled labels 0.813, 0.803, 0.807. Unreachable contacts score about the same as reachable ones (gaps +0.025, +0.033, -0.064), as expected when their weight is MORE's 0.001. `verify/verify_more_eca.py` runs the same checks for EdgeCrossAttn through `MoreEcaScorer`, the eval's own loader; on the epoch-40 checkpoint of seed 1 (200 states) openings score 0.717 against 0.308, correlation 0.646.

PushNet's 12 arms on problems 40, 277, 373, 463, 504 run on arrakis (tmux `more_train:eval_pushnet_arrakis`, 12 workers, `eval/arms_pushnet.json`). Learning-curve PushNet runs, seed 1, on arrakis GPUs 0-2 from 11:45: cuts at 1.5M, 3M and 6M calls (112,145, 221,815 and 446,152 states; 579,939, 1,143,874 and 2,294,892 training rows). Same schedule and step count as the full runs, so a smaller cut repeats its rows more often. The 12M cut and the EdgeCrossAttn cuts wait for a free GPU.
