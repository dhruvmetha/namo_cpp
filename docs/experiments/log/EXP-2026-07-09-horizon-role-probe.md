---
type: experiment
status: live
created: 2026-07-09
commit:
metric: reactive@1/@2 by (set × difficulty), opened_at split (1-push vs 2-push solution), avg sims, avg t_wall
thread: scorer-search
tags: [experiment, horizon, hz-v3, ablation, ranker]
---
# What does the horizon input actually DO? — Hz-v3 vs NoHz-v3 vs random, H=2-first / H=1-second, on the 1-push AND 2-push sets

> Reopens the one question the [[horizon_q_HANDOFF]] arc closed on a TIE: budget/horizon-conditioning measured ≈ no-horizon (reactive 43.0 Hz vs 40.7 NoHz, within seed noise), so it was dropped. That verdict was "does it help the score." This card asks the different, mechanistic question: **what does feeding H actually change in the ranking** — and uses the 1-push set (which the reactive arc never ran on) as the clean probe, because a 1-push problem is solvable at BOTH budgets, so H=2-vs-H=1 first-push ranking is directly contrastable.

## Hypothesis
_(you, via chat 2026-07-09)_ **The horizon input changes WHICH push the model ranks first: told H=2 ("you have 2 pushes"), Hz should be willing to rank a SETUP above a direct opener; told H=1, it should commit to the opener.** On the 1-push set — where every problem IS solvable in one push — this predicts H=2 solves fewer problems at push-1 (it detours through setups) while reaching the same or higher solve@2. NoHz ignores H, so H=2==H=1 for it (the control). If the split appears for Hz and vanishes for NoHz, the horizon input is doing exactly its intended job.

## Protocol (the exact query the user specified)
Reactive forced-dive MPC (`eval_reactive_argmax.py`, object-constrained, region criterion): **push 1 = argmax Q(s0,·,H); pushes 2+ = argmax Q(s,·,H=1)**. `opened_at ∈ {1,2}` records whether the region opened at push 1 (a **1-push solution**) or push 2 (a **2-push solution**). `opened_at` IS the sim count (1 push = 1 sim). Added per-episode wall-time (`t_ep`) for the time axis.

## Arms
- **Hz-v3** (`qfull_v3_v4hq_s1`, budget_cond=True) at **--h 2** and **--h 1** (the horizon contrast).
- **NoHz-v3** (`qfull_nohz_v3_v4hq_s1`, budget_cond=False) at --h 2 and --h 1 (control — must be identical; H-invariant).
- **Random** (--prior uniform, 3 seeds) — the floor.
- Both **1-push set** (`onepush_episodes.json`, 1323 episodes) and **2-push set** (`pure2push.json`, 1018 episodes).
- Both regimes [[feedback_search_nosearch_lens]]: **reactive** (above) + **best-first search** (`eval_bestfirst.py`, combine=q, hmax2, sim-budget 30) → solve-rate + avg-sims + solve@t.

## Reporting (stratified — [[feedback_stratified_splits]])
By difficulty tier per set: 1-push binned by per-episode solve_rate via `eval_common.bin_of` (hard <0.05 / med <0.30 / easy ≥0.30); 2-push via `pure2push_divisions.json`. Sim axis = opened_at / avg-sims; time axis = avg t_wall on arrakis (single-box, fair across arms — [[feedback_wall_time_framing]]).

## Compute
arrakis, s1 ckpts rsynced from Amarel (51M each) to `/common/users/dm1487/scratch_namo/outputs/scorer/`. GPUs 3/1/2. Eval dirs: `/common/users/dm1487/scratch_namo/eval/horizon_probe/`.
⚠ Single-seed (s1) per model — near-ceiling car eval jitters ~0.3mm [[reference_eval_sim_nondeterminism]]; treat sub-2pp gaps as noise. Effect of interest here is large (~20pp), so single-seed is adequate for the mechanism; error bars deferred.

## Pilot (n=27, onepush xml[0:20], 2026-07-09) — the effect is real
| Hz-v3 first-push | opened@1 | opened@2 | reactive@1 | reactive@2 |
|---|---|---|---|---|
| **H=2** | 15 | +7 | 55.6 | 81.5 |
| **H=1** | 21 | +1 | 77.8 | 81.5 |

Told H=2, Hz demotes the direct opener below a setup on ~6/27 episodes → same reactive@2, 22pp fewer 1-push solutions. Full-set run in progress.

## Run
_(filled on completion)_

## Result
_(filled on completion)_
