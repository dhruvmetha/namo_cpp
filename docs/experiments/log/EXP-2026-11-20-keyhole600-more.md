---
type: experiment
status: complete
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

Smoke: problem 40 on arrakis, all three seeds, solved in 1-3 calls, rows accepted. Problems 0, 100, 200, 300, 450, 500, 550 on ilab3 (job `1357054`, 12 workers): 14 of 21 runs saved within 4 minutes, all solved in 1-24 calls, 2 threads and 1.2-2.0 GB per worker. A first submit to rlab4 (`1357050`) never started because rlab4 had only 30 GB of memory left to allocate. Short MORE runs cost about 3.5-5.5 seconds per simulator call per core on ilab3, against about 1 second for our model, because every new state runs the ResNet-34 at 224 px on CPU over all candidate pushes. At that rate a run reaching the 3000-call cap would take 3-4.5 hours, but the next paragraph shows long runs cost far less per call.

Problem 550 seed 1 then used 402 calls in under 6 minutes, about 0.9 s per call, so long searches are much cheaper per call than the short ones: a run expands a few states and then spends most calls on rollouts from them. I cancelled the smoke at 00:18 with 5 runs unfinished (problem 450, all seeds; problem 200, seeds 1 and 3) so the full run would not redo them alongside it.

Full run. Our CS association allows one node at a time (`node=1`, 768G), and the partition refuses CPU requests ("Please do not specify the number of CPUs"), so all seven shards share ilab3: `--nshards 7 --shuffle-seed 20261120`, jobs `1357066` and `1357067` (shards 0-1, 12 workers each, started 00:18) and `1357074`-`1357078` (shards 2-6, 10 workers each, started 00:19), 74 workers in total, `--time=10:00:00`. A first submit spread over ilab4, rlab3 and rlab4 sat at `AssocGrpNodeLimit` and was cancelled. Problems 277, 373, 463 and 504 run on arrakis with 6 workers (`logs/arrakis_rest.log`); on ilab3 those five problems refuse to start, so each shard holding one logs a refusal and exits non-zero.

Report script: `report_one_keyhole_frozen.py` gains `--versus` (pair every family against a named family; the first one is the plots' reference), a within-100-calls table, a `MORE` label, and a colour list that cycles. Rebuilt with its default arguments it reproduces the registered best-model `report.json` byte for byte.

Finish. No shard hit the time limit, so nothing was resubmitted. The seven ilab3 jobs ran 3h39m to 4h24m. Six end `FAILED` with exit 1 and one `COMPLETED`: the six logs hold 15 failure lines between them, every one "initialized scene differs from frozen certificate", which is problems 40, 277, 373, 463 and 504 times three seeds, the refusals planned above. Arrakis finished at 00:40.

Row checks. 1,800 rows, one per seed and problem, no duplicates, no leftover `.tmp` or `.rejected` files; `results_arrakis/` holds exactly problems 40, 277, 373, 463 and 504. Every row records source `533fec07`, binding `fe80fb00`, a 3000-call cap, `hmax` 2 and seed base 7000, and none has a technical error. 1,748 runs solved, 48 hit the 3000-call cap and 4 ran out of pushes to try. One provenance gap: the three seeds of a problem share one `run_id` (600 ids over 1,800 rows), because the runtime fingerprint leaves MORE checkpoints out of it. The execution digests differ by seed and the report keeps all 1,800 runs, so nothing collided, but the arm name and folder are what tell the seeds apart.

`build_report.sh` pairs against both the best model and Random (`--versus HY5U_no_family_local_edge --versus Random`); its stdout is `report.md`. Sign tests below are two-sided exact binomial tests on the fewer-vs-more counts with ties dropped, computed from `runs.csv`.

## Result

MORE's pilot corpus is 39k states, about 33x smaller than our training data, so this is one learning-curve point, not MORE at full scale.

Verdict: the pilot MORE loses to the best model on every tier and does not beat Random. Its network does rank: the first push it tries opens a 1-push problem 40.3% of the time [39.0-42.7] against Random's 23.5% [15.0-28.7]. But by 30 calls Random has caught up, and on 2-push problems Random pulls ahead from there to the 3000-call cap. Paired problem by problem, MORE needs fewer calls than the best model on 51 problems and more on 416 (p = 3e-72). Against Random it is 280 fewer and 249 more (p = 0.19), significant only on 1-push (144 vs 95, p = 0.0018).

Rows: best model is `hy5u-no-family-local-edge-keyhole600-1mm-v1` seeds 1-3, MORE is this run's seeds 1-3, Random is Tri-An's seeds 7000-11000, HY5U (registered seeds 1-3) is context. Cells are % of runs solved, [worst-best seed].

Solved within 5 calls:

| | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 100.0 [100.0-100.0] | 92.0 [91.0-93.0] | 86.0 [83.0-88.0] | 84.0 [82.0-86.0] | 54.7 [54.0-56.0] | 32.3 [29.0-37.0] | 92.7 [92.0-93.3] | 57.0 [55.7-59.0] | 74.8 [73.8-76.2] |
| MORE pilot | 92.3 [91.0-93.0] | 67.7 [65.0-71.0] | 43.0 [38.0-47.0] | 40.0 [38.0-42.0] | 17.0 [14.0-21.0] | 1.3 [0.0-3.0] | 67.7 [65.3-69.7] | 19.4 [17.3-22.0] | 43.6 [41.3-45.8] |
| Random | 97.0 [95.0-99.0] | 66.4 [60.0-76.0] | 29.0 [15.0-40.0] | 25.6 [17.0-36.0] | 8.0 [4.0-17.0] | 1.4 [0.0-4.0] | 64.1 [58.0-69.7] | 11.7 [7.3-18.0] | 37.9 [32.7-43.8] |
| HY5U | 99.7 [99.0-100.0] | 92.7 [90.0-94.0] | 86.0 [85.0-88.0] | 77.3 [76.0-79.0] | 51.3 [48.0-53.0] | 26.3 [24.0-31.0] | 92.8 [91.3-94.0] | 51.7 [50.3-53.3] | 72.2 [70.8-73.2] |

Solved within 30 calls:

| | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 100.0 [100.0-100.0] | 97.7 [97.0-98.0] | 95.7 [95.0-97.0] | 93.3 [92.0-95.0] | 87.7 [87.0-89.0] | 63.0 [60.0-66.0] | 97.8 [97.3-98.3] | 81.3 [80.0-82.7] | 89.6 [88.7-90.2] |
| MORE pilot | 98.7 [98.0-100.0] | 87.3 [87.0-88.0] | 74.3 [73.0-76.0] | 72.0 [71.0-73.0] | 46.7 [45.0-48.0] | 13.3 [7.0-18.0] | 86.8 [86.0-87.3] | 44.0 [42.0-45.3] | 65.4 [64.0-66.3] |
| Random | 100.0 [100.0-100.0] | 99.0 [98.0-100.0] | 81.8 [72.0-90.0] | 86.0 [75.0-92.0] | 45.6 [34.0-55.0] | 9.2 [5.0-15.0] | 93.6 [90.7-96.7] | 46.9 [38.0-52.0] | 70.3 [64.3-74.3] |
| HY5U | 99.7 [99.0-100.0] | 96.0 [95.0-97.0] | 96.0 [92.0-99.0] | 94.0 [94.0-94.0] | 82.7 [81.0-86.0] | 60.0 [59.0-62.0] | 97.2 [95.3-98.7] | 78.9 [78.0-79.7] | 88.1 [87.2-88.7] |

Solved within 100 calls:

| | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 100.0 [100.0-100.0] | 99.0 [98.0-100.0] | 98.3 [97.0-99.0] | 97.7 [97.0-98.0] | 93.3 [92.0-94.0] | 70.3 [69.0-72.0] | 99.1 [98.7-99.7] | 87.1 [87.0-87.3] | 93.1 [93.0-93.3] |
| MORE pilot | 100.0 [100.0-100.0] | 92.7 [91.0-94.0] | 84.7 [84.0-86.0] | 86.0 [84.0-88.0] | 64.3 [63.0-65.0] | 27.7 [21.0-34.0] | 92.4 [91.7-93.0] | 59.3 [58.0-61.7] | 75.9 [74.8-77.2] |
| Random | 100.0 [100.0-100.0] | 99.4 [98.0-100.0] | 94.6 [93.0-97.0] | 98.4 [97.0-100.0] | 75.4 [70.0-83.0] | 23.0 [17.0-27.0] | 98.0 [97.7-98.7] | 65.6 [61.3-68.7] | 81.8 [79.5-83.5] |
| HY5U | 99.7 [99.0-100.0] | 98.3 [98.0-99.0] | 99.0 [98.0-100.0] | 97.0 [96.0-98.0] | 92.3 [91.0-93.0] | 73.0 [69.0-77.0] | 99.0 [98.3-99.3] | 87.4 [86.7-88.3] | 93.2 [93.0-93.3] |

Solved within 3000 calls:

| | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 99.0 [99.0-99.0] | 99.7 [99.0-100.0] | 98.0 [98.0-98.0] | 100.0 [100.0-100.0] | 98.9 [98.7-99.0] | 99.4 [99.3-99.5] |
| MORE pilot | 100.0 [100.0-100.0] | 99.7 [99.0-100.0] | 99.3 [99.0-100.0] | 99.0 [99.0-99.0] | 97.3 [96.0-99.0] | 87.3 [86.0-89.0] | 99.7 [99.3-100.0] | 94.6 [94.3-94.7] | 97.1 [96.8-97.3] |
| Random | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 94.2 [91.0-98.0] | 100.0 [100.0-100.0] | 98.1 [97.0-99.3] | 99.0 [98.5-99.7] |
| HY5U | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 100.0 [100.0-100.0] | 98.0 [97.0-99.0] | 100.0 [100.0-100.0] | 99.3 [99.0-99.7] | 99.7 [99.5-99.8] |

Median calls until solved, a failed run counting as infinite:

| | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 1 | 1 | 2 | 2 | 4 | 14 | 1 | 4 | 2 |
| MORE pilot | 1 | 2 | 7 | 8 | 38 | 448 | 2 | 52.5 | 8 |
| Random | 1 | 4 | 9 | 9.5 | 37 | 355 | 4 | 36 | 9 |
| HY5U | 1 | 1 | 2 | 3 | 5 | 16.5 | 1 | 5 | 2 |

Per problem, median over seeds, MORE needs fewer / the same / more calls, with the sign-test p:

| MORE against | 1p easy | 1p medium | 1p hard | 2p easy | 2p medium | 2p hard | 1p all | 2p all | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best model | 0 / 72 / 28 | 6 / 33 / 61 | 10 / 11 / 79 | 14 / 11 / 75 | 9 / 4 / 87 | 12 / 2 / 86 | 16 / 116 / 168 | 35 / 17 / 248 | 51 / 133 / 416 |
| p | 7.5e-9 | 1.5e-12 | 1.9e-14 | 3.0e-11 | 3.6e-17 | 5.9e-15 | 3.8e-33 | 1.0e-40 | 3.1e-72 |
| Random | 33 / 50 / 17 | 57 / 7 / 36 | 54 / 4 / 42 | 50 / 5 / 45 | 42 / 3 / 55 | 44 / 2 / 54 | 144 / 61 / 95 | 136 / 10 / 154 | 280 / 71 / 249 |
| p | 0.033 | 0.038 | 0.26 | 0.68 | 0.22 | 0.36 | 0.0018 | 0.32 | 0.19 |

Against Random the curves cross. At 1 call MORE is ahead on every 1-push tier (easy 72.7 vs 54.0, medium 36.7 vs 13.8, hard 11.7 vs 2.8). At 5 calls it is ahead on 1-push hard, 2-push easy and 2-push medium, level on 1-push medium and 2-push hard, and behind on 1-push easy (92.3 vs 97.0). By 30 calls Random is ahead on four of six tiers, and at 100 calls MORE leads only on 2-push hard (27.7 vs 23.0). At the cap MORE solves 87.3% of hard 2-push runs against Random's 94.2%, and its hard 2-push median is 448 calls against 355.

Failures. MORE leaves 52 of 1,800 runs unsolved on 25 problems: 38 on 2-push hard, 8 on 2-push medium, 3 on 2-push easy (problem 303) and 3 on 1-push (problem 140 once, 227 twice). The best model leaves 10 of 1,800 and Random 29 of 3,000. Problems 303 and 599 fail on every CS machine for every model while Tri-An's Random rows solve them, so they cost MORE and the best model about 1 point each on 2-push easy and 2-push hard against Random. That is far smaller than the gaps above.

Why MORE's curve flattens after the first few calls is UNVERIFIED. Hypothesis: its rollout pushes count as simulator calls and its tree policy has no exploration term, so when the first pushes fail it keeps spending calls on rollouts below the same few states instead of trying new first pushes. The statistics sidecars record the boards each run built and would test this.

Wall time is not compared: MORE ran on ilab3 and the best model's rows on rlab1, and the report counts simulator calls only.

Artifacts: `$NAMO_SCRATCH/eval/keyhole600_more_pilot_20261120/report.{json,md}`, `plots/` (`success_vs_calls_all_models.png`, `MORE_vs_HY5U_no_family_local_edge.png`), `runs.csv`. Registry row `more-pilot-keyhole600-1mm-v1`.
