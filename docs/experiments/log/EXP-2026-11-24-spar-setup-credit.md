---
type: experiment
status: running
created: 2026-11-24
updated: 2026-11-24
metric: "No-search (two pushes, raw-q arg-max, zero simulator calls while choosing) on held-out frozen keyhole600 problems, per arm, split by horizon and certificate tier: two-push solved; whether the first push is a useful setup (certificate lookup, and our build's own one-push finisher check); finish after one fixed certified setup (top pick opens, rank of the first opener); one-push solved after the first push as a regression control."
tags:
  - experiment
  - one-keyhole
  - spar
  - setup-credit
---
# SPAR setup credit: does labelling setups as setups change what the model tries first?

[USER 2026-11-24] Take over the SPAR two-push study on iLab: train the two matched variants past the one-epoch floor, pick the epoch on a dev set where at least one variant clearly solves some two-push scenes, then run a locked held-out test with search off. iLab only; anything beyond these jobs, or anything on Amarel, needs the user's OK.

Earlier steps (protocol in `$NAMO_SCRATCH/eval/spar_setup_target_probe_20261123.qlBvzo/PLAN.md`): one-epoch smoke on a 3,084-board cohort (train `1422188`, no-search eval `1422228`): both variants identical on 12 scenes, 1/6 one-push and 0/6 two-push. The cohort gave 11 gradient steps per epoch and both models predicted about 0.495 for every cell (val Spearman -0.05), so the result said nothing about setup credit.

## Plan

Arms. Same code (`namo-keyhole-more-20261117` at `13763e54`), architecture (HY5U no-family-local-edge settings: `EGMM_LAMBDA=0 NAMO_USE_LOCAL=0 NAMO_USE_EDGE_EMBED=0`), seed 1, split seed 0, optimiser and data order. `LOWER_RANK_LAMBDA=0` in both, so removing the setup targets does not also remove one arm's lower-rank ranking examples. The only difference is the value target of exact setup cells (stored 0.9, tried, reachable, not censored, all at chain depth 1):
- setup: `hybrid_train_v1.h5` as is (loader maps 0.9 to gamma 0.5).
- immediate: a byte copy with those cells set to 0. Built by `build_immediate_full.py` (job `1422310`): 5,592,004 cells on 185,520 of 1,302,659 boards changed; every other field verified identical (small fields by value, `ctx` by raw chunk bytes).

Training. Full data, 12 epochs (the registered count), one checkpoint kept per epoch through `train_every_epoch.py`, which runs `train_q2_round2.py` unchanged with `ModelCheckpoint(save_top_k=-1)`. H5 read from NFS, not staged to `/dev/shm` (see Run).

Population. Frozen one-keyhole 600 (manifest `bb01f360`, 1 mm), restricted to problems whose source room key does not match any training room (exact path or normalized `set/benchmark/run/env` key) and excluding the CS platform problems 40, 277, 303, 373, 463, 504, 599. Eligible: 124 two-push (35 easy, 52 medium, 37 hard) and 137 one-push. Split per horizon and tier, chosen with no model outcome: dev = a third (the 12 scenes `1422228` already looked at go to dev first, then smallest `sha256("spar-credit-split-v1:"+problem_id)`), test = the rest. Dev 87 problems (41 two-push), test 174 (83 two-push: 23 easy, 35 medium, 25 hard). `split.json` sha256 `dc02120c`, written before any full-run checkpoint existed.

Policy. Search off: at most two pushes, each the raw-q arg-max over the frozen boundary objects' reachable pushes with no simulator call while choosing (`run_greedy_commit(simulate=False)`), the episode continuing live from the pushed state; solved at 20% of the frozen target points. Same policy as `1422228`.

Epoch choice (dev, decided before seeing any dev number). For each epoch 1-12, count each arm's dev two-push solved. Pick the epoch with the highest mean of the two arms' counts, ties to fewer epochs. It must clear a bar: at least one arm solves 8 or more of the 41 dev two-push problems (about 20%). If no epoch clears it, stop and report before touching test.

Test (run once, at the chosen epoch, both arms). Per arm, by tier and horizon:
1. Two-push solved within two pushes.
2. First push a useful setup: certificate lookup of the chosen (object, edge, depth) in Tri-An's exhaustive root evidence, and our build's own check (every reachable push simulated from the arm's post-first-push state, does any open the region).
3. Finish after the same good setup: one arm-independent panel per two-push problem. Apply the lexicographically first certified setup reachable on our build, simulate every reachable push from there to get the openers, then each arm ranks that same state: does its top pick open, and the rank of its first opener.
4. One-push solved after the first push (regression control).
Paired per problem between arms (sign test on discordant pairs). One seed, so this is a pilot-sized answer, not a registered result.

## Run

Protocol files and outputs: `$NAMO_SCRATCH/eval/spar_setup_credit_full_20261123/` (`PROTOCOL_SHA256SUMS`, `EVAL_SHA256`, `SPLIT_SHA256`, `job_ids.txt`, `runs/{setup,immediate}/`, `logs/`).

All jobs on rlab1: the association allows one node at a time (`AssocGrpNodeLimit`). First smoke `1422311` died opening its `/dev/shm` H5: two ssh sessions SLURM opens at job start to save the Kerberos ticket (`saveticket ... krb5cc_slurm`) closed at 00:01:05, my last session on rlab1 ended at 00:01:18, and systemd RemoveIPC purged `/dev/shm`. Any job start on the node can do this to every other job there. Fix: NFS reads instead of `/dev/shm` staging, and a keeper ssh session held open from arrakis (`tmux` session `rlab1keep`), so RemoveIPC never sees a last logout and the DataLoader workers' semaphores survive.

- `1422315` smoke, 2 epochs on the 3,084-board pair: one checkpoint per epoch, epoch 0 val loss 3.6457 (matches `1422188`).
- `1422316` setup arm, `1422317` immediate arm: 12 epochs, RTX 4500 Ada, started 2026-11-24 00:07.
- `1422319` eval smoke on 3 dev problems with the one-epoch checkpoints.
- Dev sweep: one 6-shard job per epoch as each pair of checkpoints landed (`dev_driver.sh`, jobs in `job_ids.txt`), 2,088 rows, 0 technical errors. Training ran about 16 minutes per epoch.

Dev two-push solved, of 41 (immediate / setup): e1 12/12, e2 11/11, e3 11/17, e4 17/13, e5 15/15, e6 15/14, e7 16/13, e8 13/15, e9 15/17, e10 12/17, e11 15/16, e12 15/17. The gap changes sign from epoch to epoch, so no single epoch is evidence. Per the rule, epochs 9 and 12 tie at a mean of 16.0 and the tie goes to fewer epochs: **9 epochs** (`epoch008`; setup sha256 `b950ec0f`, immediate `e10384b2`). Bar met (setup 17 ≥ 8).

Test, launched with those two checkpoints on the 174 locked test problems:
- search off, `credit_eval.py` mode `full`, 12 shards (`test/search_off/`);
- search on [USER-approved 2026-11-24], the registered best-first protocol (`run_one_keyhole_frozen.py` at `13763e54`, prior model, mean5, raw q, discount off, dedupe and jam on, hmax 2, 3000 calls, seed base 7000) on the same 174 problems (`test/search_on/`).

## Result

Pending.
