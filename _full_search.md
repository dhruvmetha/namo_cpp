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

### Stratified by difficulty × horizon (easy/med/hard)
_(Claude, 2026-07-04)_ Re-aggregation of the **same** instrumented records into the canonical difficulty
tiers — 2push = per-episode `division` (`pure2push_divisions.json`, n_setups based); 1push = `solve_rate`
tertiles (`onepush_episodes.json`, hard<0.169 / med<0.533), the same bins as `_reactive_search`. Attached by
POSITIONAL join (harness writes one record per episode in `full_episodes()` order; verified 1018/1018 &
1323/1323, zero mismatch — the records' baked `tier` uses different solve_rate cuts, so it is NOT used).
Mean ± std across seeds (random 10, NoHz-v3 3). Aggregator `scripts/sandbox/agg_fullsearch_bydiff.py`. The
**`all` rows reproduce the headline exactly** (95.3/91.0 @900, 94/185 avg sims; 15.5/25.4 s avg t_wall) —
binning + timing verified against the aggregate above.

#### 2push — success vs SIMS (machine-independent, budget 900)

| difficulty | ranker | @2 | @10 | @30 | @100 | @300 | @900 | avg sims | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy | NoHz-v3 | 55.0 ± 2.1 | 77.6 ± 0.9 | 85.7 ± 0.9 | 92.9 ± 0.9 | 96.1 ± 0.5 | **98.9 ± 0.4** | 38 | 28 |
| easy | random | 6.6 ± 1.5 | 34.1 ± 2.4 | 66.3 ± 2.8 | 89.3 ± 1.3 | 98.3 ± 0.6 | **99.8 ± 0.2** | 43 | 42 |
| medium | NoHz-v3 | 40.7 ± 2.4 | 62.8 ± 3.7 | 75.4 ± 2.2 | 86.6 ± 0.5 | 94.1 ± 0.3 | **97.8 ± 0.8** | 62 | 43 |
| medium | random | 4.1 ± 0.9 | 21.0 ± 1.4 | 43.6 ± 2.4 | 69.9 ± 2.1 | 88.4 ± 1.2 | **97.0 ± 0.7** | 123 | 98 |
| hard | NoHz-v3 | 26.1 ± 1.5 | 44.0 ± 0.9 | 56.1 ± 0.4 | 69.5 ± 0.8 | 82.1 ± 1.1 | **90.2 ± 0.8** | 165 | 88 |
| hard | random | 1.6 ± 0.5 | 10.1 ± 1.0 | 22.0 ± 1.1 | 40.2 ± 1.9 | 58.9 ± 2.2 | **78.7 ± 2.1** | 344 | 197 |
| all | NoHz-v3 | 38.7 ± 1.3 | 59.4 ± 1.9 | 70.8 ± 1.0 | 81.8 ± 0.4 | 90.2 ± 0.6 | **95.3 ± 0.6** | 94 | 55 |
| all | random | 3.7 ± 0.8 | 20.1 ± 0.5 | 41.1 ± 1.0 | 63.6 ± 1.2 | 80.0 ± 0.6 | **91.0 ± 0.8** | 185 | 115 |

![[fullsearch_success_vs_sims_bydiff_2push.png]]

#### 2push — success vs TIME (emeraldrapids-exclusive, model & random same HW)

| difficulty | ranker | 1 s | 5 s | 10 s | 30 s | 60 s | 240 s | avg t_wall (s) |
|---|---|---|---|---|---|---|---|---|
| easy | NoHz-v3 | 62.9 ± 1.9 | 85.9 ± 1.4 | 89.8 ± 0.8 | 94.3 ± 0.7 | 96.9 ± 0.2 | 98.9 ± 0.4 | 7.1 |
| easy | random | 21.1 ± 1.8 | 69.0 ± 2.0 | 83.7 ± 1.9 | 96.8 ± 0.9 | 98.8 ± 0.6 | 99.8 ± 0.2 | 6.3 |
| medium | NoHz-v3 | 50.0 ± 2.2 | 74.6 ± 1.3 | 81.5 ± 0.5 | 90.5 ± 0.3 | 95.1 ± 0.5 | 97.8 ± 0.8 | 10.6 |
| medium | random | 12.6 ± 1.6 | 47.0 ± 2.2 | 62.2 ± 2.4 | 84.3 ± 1.0 | 92.0 ± 1.0 | 97.0 ± 0.7 | 17.3 |
| hard | NoHz-v3 | 35.6 ± 0.4 | 57.3 ± 1.6 | 65.6 ± 1.1 | 77.1 ± 1.0 | 84.1 ± 0.8 | 90.1 ± 0.9 | 26.3 |
| hard | random | 6.7 ± 0.4 | 25.9 ± 1.1 | 36.3 ± 1.7 | 55.7 ± 1.7 | 67.6 ± 2.1 | 78.7 ± 2.1 | 46.5 |
| all | NoHz-v3 | 47.8 ± 0.7 | 70.9 ± 0.6 | 77.6 ± 0.2 | 86.5 ± 0.5 | 91.5 ± 0.5 | 95.3 ± 0.5 | 15.5 |
| all | random | 12.4 ± 0.7 | 44.5 ± 0.8 | 57.8 ± 1.3 | 76.8 ± 0.7 | 84.7 ± 0.7 | 91.0 ± 0.8 | 25.4 |

![[fullsearch_success_vs_time_bydiff_2push.png]]

**Where the win lives — HARD.** The +4.3 pt @900 aggregate gap is **entirely a hard-tier effect**: hard
NoHz-v3 90.2 vs random 78.7 (**+11.5 pt**), while easy (98.9 vs 99.8) and medium (97.8 vs 97.0) both converge
to ~98 % by 900 — there the model's win is **efficiency, not ceiling** (easy avg 38 vs 43 sims, medium 62 vs
123). Random eventually brute-forces the easy/medium tail; on hard it can't (344 avg sims, 78.7 % ceiling).

**Time echoes it, with one honest crossover.** On **hard** the model's sim-savings convert to wall-clock: avg
**26.3 s vs 46.5 s**, @30 s 77.1 vs 55.7. But on **easy** the time curves *cross* — model avg **7.1 s** vs
random **6.3 s** — because the sim count there is already tiny and the model pays a per-sim NN-scoring overhead
random doesn't; random's cheaper sims win the wall-clock race once both are near-ceiling. So the model's *time*
edge, like its solve-rate edge, is concentrated where search is actually hard.

#### 1push — success vs SIMS (machine-independent, budget 900)
_(Claude, 2026-07-04)_ 1push best-first SEARCH (hmax=1, budget 900, n=1323) — **never run** before (only
reactive open@1 existed). Ran on iLab **rlab7** (3 NoHz-v3 ckpt-seeds + 10 random rng-seeds; s3 ckpt = shared-FS
`dlopoael/ep012`, one epoch past the 2push-s3 ep011 which wasn't synced here — same run, negligible). hmax=1 =
no dive: rank the labeled object's candidate first-pushes by q, sim in priority order until one opens the goal.
The pool per object is small (≤~35 pushes) so budget 900 is never binding — **@900 = the "some push solves"
ceiling**, **@1 = the reactive top-1 pick**.

| difficulty | ranker | @1 | @2 | @5 | @10 | @20 | @900 | avg sims | to-solve |
|---|---|---|---|---|---|---|---|---|---|
| easy | NoHz-v3 | 98.7 ± 0.4 | 99.2 ± 0.1 | 99.6 ± 0.1 | 99.8 ± 0.0 | 99.8 ± 0.0 | **99.8 ± 0.0** | 1 | 1 |
| easy | random | 72.4 ± 1.5 | 90.6 ± 0.4 | 99.0 ± 0.2 | 99.7 ± 0.1 | 99.8 ± 0.0 | **99.8 ± 0.0** | 2 | 1 |
| medium | NoHz-v3 | 94.0 ± 0.6 | 96.2 ± 0.6 | 98.3 ± 0.4 | 99.6 ± 0.1 | 99.9 ± 0.1 | **100.0 ± 0.0** | 1 | 1 |
| medium | random | 33.1 ± 2.3 | 52.7 ± 2.5 | 82.9 ± 1.2 | 96.4 ± 0.6 | 99.8 ± 0.1 | **100.0 ± 0.0** | 3 | 3 |
| hard | NoHz-v3 | 54.2 ± 0.3 | 62.0 ± 0.8 | 73.9 ± 1.0 | 82.7 ± 0.5 | 90.3 ± 0.2 | **99.2 ± 0.1** | 7 | 7 |
| hard | random | 5.9 ± 1.5 | 12.0 ± 1.7 | 27.1 ± 1.8 | 46.3 ± 2.9 | 67.6 ± 1.7 | **99.3 ± 0.1** | 20 | 19 |
| all | NoHz-v3 | 82.3 ± 0.2 | 85.8 ± 0.2 | 90.6 ± 0.4 | 94.0 ± 0.2 | 96.7 ± 0.1 | **99.7 ± 0.0** | 3 | 3 |
| all | random | 37.3 ± 0.9 | 51.9 ± 1.1 | 69.8 ± 0.8 | 80.8 ± 1.0 | 89.1 ± 0.6 | **99.7 ± 0.0** | 8 | 8 |

![[fullsearch_success_vs_sims_bydiff_1push.png]]

**1push: same ceiling, the model just finds the solver FAST.** The pool is tiny, so both rankers reach the
**same ~99.7 % @900 ceiling in every tier** (hard 99.2 vs 99.3) — given enough sims random tries every push and
finds the solver too. The model's entire win is **rank position**: solve@1 (top pick already opens the goal)
NoHz-v3 **82.3 %** vs random **37.3 %** overall, and on **hard 54.2 vs 5.9 (~9×)** — it floats the rare working
push to the top of a mostly-failing pool. Efficiency: avg **3 vs 8 sims** overall (hard 7 vs 20). **Cross-check:**
1push solve@1 reproduces `_reactive_search` open@1 almost exactly (NoHz all 82.3 = 82.3; hard 54.2 ≈ 54.3;
random all 37.3 ≈ 37.0) — the search's first sim *is* the reactive pick, validating the whole path.

**1push TIME by difficulty: PENDING a separate emeraldrapids-exclusive timed run.** rlab7 is co-tenanted (not
fenced), so its `t_wall` is not valid timing and is deliberately NOT reported here.

## Next
The gap is largest early (reactive/low-budget) and narrows as random brute-forces the easy tail — so the
model's value is *front-loaded search*, exactly the reactive-mode regime. Two follow-ups: (1) the 13 % of easy
instances where ranking *hurts* + the poor-second-push tail (mean rank 2.05 with a long tail) point at the
cross-head H1/H2 scale mismatch — a `dive_bonus` / calibration sweep could shave the deep-dive churn; (2)
~~stratify by difficulty~~ **DONE (see Stratified section above):** the +4.3 pt @900 2push gap is **entirely a
hard-tier effect** (+11.5 pt); easy/med are efficiency-only at the ceiling; 1push shares one ceiling and the
model wins purely on early rank. Remaining: a **1push emeraldrapids-exclusive timed run** to fill the one PENDING
cell (time-by-difficulty for 1push).

## Discussion
_(you ↔ Claude — ask here; I answer inline, dated `**[who YYYY-MM-DD]**`. Newest at the bottom.)_
