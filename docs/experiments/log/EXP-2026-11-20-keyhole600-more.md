---
type: experiment
status: running
created: 2026-11-20
updated: 2026-11-20
metric: "Frozen one-keyhole 600 at 1 mm, 3000-call cap: solved within 5, 30, 100 and 3000 simulator calls and median calls, split by horizon (1-push/2-push) and certificate tier (easy/medium/hard), with worst-to-best seed ranges; MORE-inspired MCTS seeds 1-3 against HY5U_no_family_local_edge seeds 1-3 and Random seeds 7000-11000, plus a per-problem paired count of fewer vs more calls against the best model."
tags:
  - experiment
  - one-keyhole
  - external-baseline
  - more
---
# MORE-inspired MCTS on the frozen keyhole600

Request from the namo_main session, user-approved 2026-11-20: run the MORE baseline on the frozen keyhole600 problems on CS and compare it with the best model, HY5U_no_family_local_edge seeds 1-3 (`hy5u-no-family-local-edge-keyhole600-1mm-v1`).

Caveat that goes with every number on this card: MORE's pilot corpus is 39,052 states from 1,231 episodes, about 33x smaller than the 1.3M boards our model trained on. This is one point on MORE's learning curve, not MORE at full scale.

## Plan

Population. Tri-An's frozen one-keyhole 600 (manifest `bb01f360`, 1 mm): 300 one-push and 300 two-push problems, 100 per certificate tier each. Same problems, config (`a58e885f`, inflation `f8ab35e1`), frozen target points, door objects and post-restore list as the registered best-model rows.

Method. The MORE-inspired adaptation built in `5301b801` and later (`python/namo/planners/opening/more_mcts.py`, deploy mode): tree policy `(sum of top-3 returns + clamped prior) / visits` with no UCB term, greedy rollouts on its own network, backups discounted 0.5 per level. The prior is MORE's own ResNet-34 FPN at 224 px, trained by SmoothL1 weighted by visit count on its own plain-UCT collection (`more_pilot.h5`, 39,052 states). Arm keys `prior: more`, `planner: more`, `more_prior_scale: raw`. Search budget and success test match the best model's protocol: up to 2 pushes, 3000 simulator calls, no-op dedupe and jam pruning on, solved when 20% of the target points are reachable. Rollout pushes count as simulator calls.

Checkpoints. `$NAMO_SCRATCH/more_pilot_20260921/model_s{1,2,3}/more_epoch001.pt` (two epochs, 3.79M samples each), sha256 `13d84e73` / `5ea46a2b` / `2526175f`. All three passed `verify_more_net.py`. The rows record `checkpoint_sha256: null` for MORE arms (the runtime fingerprint hashes only `prior: model` checkpoints), so these hashes and `arms.json` are the provenance.

Code. Worktree `ktamp/namo-keyhole-more-20261117` pinned at `13763e54`, source fingerprint `533fec07`. Against the best model's runner commit `cf64a879` the diff is additive: the MORE planner, net and scorer, plus a `planner` branch in `eval_bestfirst.py` and `run_one_keyhole_frozen.py` that leaves best-first arms untouched. Binding `fe80fb00`, the same file the registered best-model rows used.

Machines. The best model's registered rows are 595 problems on rlab1 (EPYC 7413) and problems 40, 277, 373, 463, 504 on arrakis. rlab1 is fully allocated to another user's multi-day jobs, so the 595 run on the CS EPYC 7352 nodes (ilab3, ilab4, rlab3, rlab4) instead. Check from existing rows: the Sep 15 timed campaign ran the best model's three seeds on rlab3, rlab4 and ilab3, and all 1,785 of its runs match the rlab1 rows on solved and calls. The five problems that do not start in their certified state on these builds run on arrakis, as before. Problems 303 and 599 fail on every CS machine for every model; they stay in and count as failures, as in the registered report.

Report. `scripts/pipeline/report_one_keyhole_frozen.py` with the registered HY5U and best-model rows, the MORE rows and Tri-An's Random references, plus a sign test on the per-problem paired counts.

## Run

Output `$NAMO_SCRATCH/eval/keyhole600_more_pilot_20261120/` (`arms.json`, `untimed_cs.sbatch`, `arrakis_run.sh`, `results_cs/`, `results_arrakis/`, `logs/`).

Smoke: problem 40 on arrakis, all three seeds, solved in 1-3 calls, rows accepted. Problems 0, 100, 200, 300, 450, 500, 550 on ilab3 (job `1357054`, 12 workers): 14 of 21 runs saved within 4 minutes, all solved in 1-24 calls, 2 threads and 1.2-2.0 GB per worker. A first submit to rlab4 (`1357050`) never started because rlab4 had only 30 GB of memory left to allocate. MORE costs about 3.5-5.5 seconds per simulator call per core on ilab3, against about 1 second for our model, because every new state runs the ResNet-34 at 224 px on CPU over all candidate pushes. A run that reaches the 3000-call cap therefore takes 3-4.5 hours. Total cost: about 100 core-hours if MORE needs as many calls as our model (43 per run on average over the registered rows) and about 320 if it needs as many as Random (143).

Problem 550 seed 1 then used 402 calls in under 6 minutes, about 0.9 s per call, so long searches are much cheaper per call than the short ones: a run expands a few states and then spends most calls on rollouts from them. I cancelled the smoke at 00:18 with 5 runs unfinished (problem 450, all seeds; problem 200, seeds 1 and 3) so the full run would not redo them alongside it.

Full run. Our CS association allows one node at a time (`node=1`, 768G), and the partition refuses CPU requests ("Please do not specify the number of CPUs"), so all seven shards share ilab3: `--nshards 7 --shuffle-seed 20261120`, jobs `1357066` and `1357067` (shards 0-1, 12 workers each, started 00:18) and `1357074`-`1357078` (shards 2-6, 10 workers each, started 00:19), 74 workers in total, `--time=10:00:00`. A first submit spread over ilab4, rlab3 and rlab4 sat at `AssocGrpNodeLimit` and was cancelled. Problems 277, 373, 463 and 504 run on arrakis with 6 workers (`logs/arrakis_rest.log`); on ilab3 those five problems refuse to start, so each shard holding one logs a refusal and exits non-zero.

Report script: `report_one_keyhole_frozen.py` gains `--versus` (pair every family against a named family; the first one is the plots' reference), a within-100-calls table, a `MORE` label, and a colour list that cycles. Rebuilt with its default arguments it reproduces the registered best-model `report.json` byte for byte.

## Result

Pending.
