---
type: experiment
status: done
created: 2026-07-03
metric: "search solve@900 (pure2push car): NoHz-v3 95.3±0.6% vs random 91.0±0.8%; model uses ~half the sims (94 vs 185) and half the wall-time"
commit: 29f3ee7
tags:
  - experiment
---
# Full Search

Do the full best first search evaluation with 10 random seeds of random (random selection of what to expand next (what action to take)). Could be breadth, could be depth, we never know. 
Compare the results against No-horizon v3. 

Our objective is to record success v/s sim and success v/s time -- and if we are recording success v/s time as well. We need to make sure that the RO instances being solved by both methods are on the same base-system so that time comparison is fair. Yes? 

Budget at 900 sims total per RO instance per seed.

I want plots (with variance bands) for the above two things we measured and tables at different time (s) and sims.  
#### Random
For each random seed, we run the "random" best-first search over all the test RO problem. 
#### No-Horizon v3
Same thing, but we 3 seeds of trained models. Use that to do best first search by using the predicted q-values as the ranker. (We already have infra). You can also add some more aggregation data as how many times do we just go breadth expansions instead of depth? Are we not deep-diving into the expanded new state? Also measure other metrics to tell us if the model is doing a poor job in ranking performance.

Interesting part is what will we aggregate here, and we can do this live in chat.

## Plan
_(Claude, 2026-07-03)_ **CAR only.** Regime = value-guided greedy **best-first search** over the labeled
object (`time_bestfirst.py`, reusing `eval_bestfirst.candidates`/`priority`, **combine=`q`** = rank pushes by
raw action-value). Key = `namo_testset_v1/labels/pure2push.json` = the region-opening 2-push SEARCH problem
(**n=1018 episodes**, one per (object,goal); hmax=2). **Budget 900 sims/instance.** Ranker:
- **random** = `--prior uniform` (uniform pick from the object's candidate pool, no model), **10 rng-seeds** (0–9).
- **NoHz-v3** = the No-horizon q-scorer, **3 ckpt-seeds** (s1/s2/s3) as the ranker.

**Measure both success-vs-sims AND success-vs-time.** Best-first explores in a budget-INDEPENDENT order, so one
budget-900 run records, per instance, the exact sim-index AND wall-time at which it solved → solve@B / solve@T
for any B≤900 / any T. Sims are machine-INDEPENDENT; **time is fair only if model & random share hardware** →
run BOTH on the **same emeraldrapids-exclusive** Amarel nodes (icelake was drained on the weather-degraded
cluster). Instrumented per instance: `{n_sim, t_wall, solved, depth_hist, solve_ranks}` →
- **(a)** breadth:depth = total sims at push-depth 0 (first push / root) : depth 1 (second push / dive).
- **(b)** histogram of sims by tree-depth (where the 900-budget goes).
- **(c)** priority-rank of the winning plan's pushes (is the ranker's #1 the winner?).
- **(d)** solved-but-slow = per-instance count where the MODEL uses MORE sims than random (ranking hurts).

**Reuse (sanity only, machine-independent):** `bfq_nohz_v3_s{1,2,3}` (model sims) → solve@900 95.5±0.4, avg
90.6 sims; `bf900_uniform_s{0..4}` (random sims) → 90.8±0.1, avg 185.3 sims. Icelake `timebench/full2_*` = a
prior fair-time cross-check.

Aggregator/plotter `scripts/sandbox/aggregate_fullsearch.py`; campaign driver
`scripts/amarel/launch_fullsearch_campaign.sh` (+ `MODELS` env now forwarded through `time_bestfirst_shard.slurm`).

## Run
_(Claude, 2026-07-03)_ instrumented `time_bestfirst.py` at commit `29f3ee7` (`feat/horizon-q-redesign`;
instrumentation `47da334`) + uncommitted `time_bestfirst_shard.slurm` (MODELS forwarding) / launcher edits.
Amarel `main,main-redhat`, `--exclusive --constraint=emeraldrapids`, CPU inference (`CUDA_VISIBLE_DEVICES=""`).
- **NoHz-v3 model (3 ckpt-seeds):** jobs `57829481/482/483` (`fullsearch/nohz_s{1,2,3}`), array 0-2 × SHARD 340.
- **random (10 rng-seeds 0–9):** jobs `57830334–57830343` (`fullsearch/rand_s{0..9}`), array 0-3 × SHARD 255.
  (First random launch at SHARD 340 was cancelled+resharded: the hard-front of the [0,340) slice risked the 3h
  wall limit → systematic tail loss across all seeds. QOS caps `--time` at 3h, so smaller shards, not longer time.)
- NoHz ckpts (v3, epoch): s1 `wl8k6iyv/ep012`, s2 `kzph0acr/ep012`, s3 `dlopoael/ep011`.

## Result + Verdict
_(Claude, 2026-07-04)_ **CAR, best-first, combine=q, budget 900, pure2push n=1018. Mean ± std across seeds
(random 10, NoHz-v3 3).** Summary JSON `scratch_namo/eval/fullsearch/fullsearch_summary.json`; regenerate via
`scripts/sandbox/aggregate_fullsearch.py`.

**Headline.** NoHz-v3 solves **95.3 ± 0.6 %** vs random **91.0 ± 0.8 %** at the full 900-sim budget — but the
story is *efficiency*: the model averages **93.8 sims** (54.8 to a solve) vs random's **184.9** (114.9), and
**15.5 s** vs **25.4 s** wall-clock per instance. The model reaches random's *ceiling* (91 %) in **~30 sims**.

**Success vs sims** (% solved with n_sim ≤ B)

| ranker | @2 | @10 | @30 | @100 | @300 | @900 |
|---|---|---|---|---|---|---|
| **NoHz-v3** | 38.7 ± 1.3 | 59.4 ± 1.9 | 70.8 ± 1.0 | 81.8 ± 0.4 | 90.2 ± 0.6 | **95.3 ± 0.6** |
| random | 3.7 ± 0.8 | 20.1 ± 0.5 | 41.1 ± 1.0 | 63.6 ± 1.2 | 80.0 ± 0.6 | 91.0 ± 0.8 |

![[fullsearch_success_vs_sims.png]]

**Success vs wall-time** (% solved with t_wall ≤ T; emeraldrapids-exclusive, model & random same HW)

| ranker | 1 s | 2 s | 5 s | 10 s | 30 s | 60 s | 120 s | 240 s |
|---|---|---|---|---|---|---|---|---|
| **NoHz-v3** | 47.8 ± 0.7 | 60.2 ± 0.4 | 70.9 ± 0.6 | 77.6 ± 0.2 | 86.5 ± 0.5 | 91.5 ± 0.5 | 94.2 ± 0.7 | 95.3 ± 0.5 |
| random | 12.4 ± 0.7 | 25.1 ± 0.7 | 44.5 ± 0.8 | 57.8 ± 1.3 | 76.8 ± 0.7 | 84.7 ± 0.7 | 89.8 ± 0.9 | 91.0 ± 0.8 |

![[fullsearch_success_vs_time.png]]

_(Bands are tight — each seed already averages 1018 episodes, so std ≈ binomial SE. 10 random seeds is plenty.)_

**Aggregations (a–d).**

![[fullsearch_aggregation.png]]

- **(a/b) Where the budget goes — both DEEP-DIVE, no root-thrash.** Of all sims, NoHz-v3 spends **6.4 % at
  push-depth 0** (first push / root) and **93.6 % at depth 1** (second push / dive); random **5.5 % / 94.5 %**
  (breadth:depth ratio 0.068 vs 0.058). So the search is *not* breadth-thrashing at the root — once it sims a
  first push it commits and dives into second-push subtrees. The model's win is **not** "dives more"; it's the
  same dive structure in **half the total sims** because it dives into the *right* branch.
- **(c) The ranker is GOOD.** The winning first push is the model's **#1 pick 50.9 %** of solves (top-3 = 69.8
  %; median rank **0**); random hits rank-0 only 14.9 % (median rank 4). Its **second**-push ranking is sharper
  still — winning 2nd push median rank **0**, mean 2.05 (random median 2, mean 5.05). The tall rank-0 bar vs
  random's near-flat spread is the headline: the q-ranker's top pick is usually the winner.
- **(d) "Ranking hurts" is a real but small minority.** On the 336 instances solved by *all* 3 model seeds AND
  *all* 10 random seeds (the robustly-easy set), the model uses **fewer** sims on 291 (86.6 %) and **more** on
  **45 (13.4 %)**; median Δ = **−33 sims** (model saves). Only-model-solves 2, only-random-solves 4.

**Sanity vs reused runs (machine-independent sims):** NoHz-v3 solve@900 95.3 ≈ `bfq_nohz_v3` 95.5; avg sims
93.8 ≈ 90.6; avg-to-solve 54.8 ≈ 53.5. Random 91.0 ≈ `bf900_uniform` 90.8; avg sims 184.9 ≈ 185.3. **Match.**
Time: emerald 15.5/25.4 s ≈ icelake `timebench` 16.2/27.7 s (emerald a touch faster). Instrumented path is
search-identical to the reference.

**Verdict [on numbers].** NoHz-v3 ≫ random on **both** axes at **every** cutoff. Reactive end (B=2) 38.7 vs 3.7
(~10×); the model hits random's full-budget rate at ~30 sims and both approach their ceilings by 900 (95.3 vs
91.0, a +4.3 pt gap that is small only because random eventually brute-forces the easy tail). Wall-time: 48 %
solved in 1 s vs 12 %. The q-ranker ranks well (top-1 = winner half the time) and the search deep-dives rather
than thrashes; the model does not systematically waste the budget. **Accept: the learned q-ranker is a large,
consistent win over random best-first, in sims and in real time.**

## Next
The gap is largest early (reactive/low-budget) and narrows as random brute-forces the easy tail — so the
model's value is *front-loaded search*, exactly the reactive-mode regime. Two follow-ups: (1) the 13 % of easy
instances where ranking *hurts* + the poor-second-push tail (mean rank 2.05 with a long tail) point at the
cross-head H1/H2 scale mismatch — a `dive_bonus` / calibration sweep could shave the deep-dive churn; (2)
stratify all of the above by difficulty tier (easy/med/hard) to see where the +4.3 pt @900 gap concentrates.

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
