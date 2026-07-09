---
type: experiment
status: done
created: 2026-07-09
commit: 3fa9b4a
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
By difficulty tier per set: 1-push binned by per-episode **solve_rate tertiles** (canonical for the results sheet — RESULTS Table 1b/6c; ≈441/tier); 2-push via `pure2push_divisions.json` (`n_setups`→division). (`eval_common.bin_of`'s fixed cuts <0.05/<0.30 are the offline-scorer convention, NOT the results-sheet binning — they disagree on 1-push.) Sim axis = opened_at / avg-sims; time axis = avg t_wall on arrakis (single-box, fair across arms — [[feedback_wall_time_framing]]).

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
2026-07-09, arrakis (GPUs 3/1/2), s1 ckpts. `run_horizon_probe.sh` (reactive) + `run_horizon_probe_bf.sh` (best-first, budget 2), agg `agg_horizon_probe.py`. Eval dirs `…/eval/horizon_probe/`. Reproduces registry: NoHz-v3 pure2 reactive 40.8 (reg 40.7), best-first 38.1 (reg 37.8), Hz-v3 reactive 45.3 (reg s1 45.6) — pipeline validated.

## Result — HORIZON IS A ROUTE KNOB, NOT A SOLVE-RATE KNOB [verdict on numbers]

**1-push set (the probe), react@1 = solved with a 1-push solution / react@2 = solved by push 2. Tiers = solve_rate tertiles (442/443/438, canonical — matches RESULTS Table 1b: NoHz react@1 53.8/94.1/98.9 ≈ 54.3/93.9/98.7).**

| arm | react@1 hard/med/easy/ALL | react@2 hard/med/easy/ALL |
|---|---|---|
| Hz-v3 · H=2 | 51.6 / 85.3 / 93.4 / **76.7** | 76.7 / 96.4 / 99.5 / 90.9 |
| Hz-v3 · H=1 | 59.0 / 95.5 / 99.3 / **84.6** | 73.1 / 96.6 / 99.5 / 89.7 |
| NoHz-v3 · H=2 | 53.8 / 94.1 / 98.9 / 82.2 | 73.1 / 96.6 / 99.5 / 89.7 |
| NoHz-v3 · H=1 | 53.8 / 94.1 / 98.9 / 82.2 | 73.1 / 96.6 / 99.5 / 89.7 |
| random (3-seed) | 5.4 / 33.9 / 73.0 / 37.3 | 20.5 / 61.5 / 91.5 / 57.7 |

- **H1 ACCEPTED (mechanism confirmed):** told H=2, Hz demotes the direct opener for a setup → **−7.9pp react@1** ALL (−7.4/−10.2/−5.9 hard/med/easy) vs H=1, at ~unchanged react@2. **NoHz H=2≡H=1 byte-identical every tier** (the control) → the Hz shift is 100% the horizon input.
- **H2 (does it help the score?) — NO, net wash:** react@2 Hz-H2 90.9 vs NoHz 89.7 = **+1.2pp**. Horizon changes *when* (1-push vs 2-push route), not *whether*.
- **Tier structure (Hz H2−H1 react@2):** easy **0.0** / med **−0.2** (the setup-detour = **pure waste**, spends +1 sim to reach the same solve) / hard **+3.6** (foresight finds 2-push paths the greedy-opener misses). Boon is hard-ONLY; and Hz-H2 hard react@2 76.7 vs NoHz 73.1 = +3.6 is horizon's one real 1-push gain.

**2-push set (canonical), both regimes:**

| arm (ALL, n=1018) | reactive@2 | best-first@2 |
|---|---|---|
| Hz-v3 | **45.3** | 35.9 |
| NoHz-v3 | 40.8 | **38.1** |
| random (3-seed) | 4.3 | 3.7 |

- **Hz wins forced-dive reactive (+4.5**; hard 28.8 vs 25.3, med 49.4 vs 40.8, easy tie) — foresight helps commit.
- **Best-first@2 NoHz edges Hz (+2.2)** — BUT this is the ONE misleading cell (see sweep below): at a 2-sim budget Hz can't express its dive, so its reluctance-to-dive costs ~2pp.

### Budget sweep — the dive-tax is a BUDGET-2 ARTIFACT; horizon is a SEARCH ACCELERATOR
`run_horizon_sweep.sh` (budget-30 run, curve derived from per-episode sims — expansion order is budget-independent). pure2, combine=q. ![[horizon_budget_curve.png]]

**Solve-rate vs sim-budget (ALL, n=1018):**

| arm | s@2 | s@5 | s@10 | s@20 | s@30 |
|---|---|---|---|---|---|
| **Hz-v3** | 35.9 | **54.4** | **65.2** | **74.8** | **79.0** |
| NoHz-v3 | 38.1 | 51.3 | 57.7 | 65.8 | 70.5 |
| random | 3.7 | 10.8 | 20.8 | 33.2 | 41.9 |

**Dive-tax (NoHz−Hz solve@B, pp) — flips sign right after budget 2:**

| tier | @2 | @5 | @10 | @20 | @30 |
|---|---|---|---|---|---|
| hard | −0.5 | −6.2 | −8.4 | **−13.5** | −11.8 |
| med | +1.5 | −4.2 | −10.0 | −8.6 | −8.5 |
| easy | +8.0 | +3.4 | −2.1 | −2.5 | −2.9 |
| ALL | +2.2 | −3.1 | −7.5 | **−9.0** | −8.5 |

**CORRECTION to the earlier "NoHz wins best-first" line:** that held ONLY at budget 2. Hz and NoHz **cross at ~budget 3**, then Hz pulls away by ~9pp (ALL) and up to **+13.5pp on hard** — because Hz's H=2 head is a genuinely better SETUP ranker (it scores first-pushes by 2-push foresight), and once search has budget to explore setups, that foresight dominates NoHz's myopic single value (which buries setups — they open nothing yet). At matched avg-sims Hz solves more (hard s@30 67.9 @6.8 sims vs NoHz 56.1 @7.1). Consistent with the registry's `s@900` Hz 97.7 > NoHz 95.9 — the sweep fills in the middle and locates the crossover. So the **only** regimes where horizon looks bad are (a) reactive best-first at the razor budget of 2 and (b) the 1-push easy/med detour; everywhere search has room, horizon wins.

**Verdict (updated by the sweep):** the horizon input does two separable things. (1) **Reactive / route:** it trades *when* you solve (setup-detour) — a wash on 1-push final solve (+1.2), a tax on easy, a boon on hard. (2) **Search:** it is a genuine **search accelerator** — its H=2 setup ranking lifts best-first by +3→+9pp (up to +13 hard) at any budget ≥~3; the "NoHz wins best-first" was a budget-2 artifact. So the earlier drop-horizon TIE was measured in the ONE regime (reactive @2 / budget-2) where horizon is neutral-to-bad; **with search budget, horizon clearly wins.** This nuances (does not overturn) the deploy pick: the prize was the ~0-sim reactive regime [[horizon_q_HANDOFF]], where NoHz-v3 stays the baseline — but if any search budget is on the table, Hz-v3 dominates.

**Caveats:** single-seed (s1) per model; near-ceiling car eval jitters ~0.3mm [[reference_eval_sim_nondeterminism]] so treat sub-2pp as noise (the ~8pp route-shift and +4pp hard-2push are well above it). Best-first at budget 2 ≈ reactive's dive space; the dive-tax widens at larger budgets (a budget sweep is the natural follow-up).
