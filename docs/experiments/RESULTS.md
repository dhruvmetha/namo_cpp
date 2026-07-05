---
status: hub
tags:
  - results
updated: 2026-07-04
---
# Results

Paper-style compilation: for each experiment, the **main table + main figure + a tight key finding**. Full
detail and verbose analysis live in each experiment's card (`_*.md`), linked per section. **Setting:** CAR
robot, testset `namo_testset_v1`, region-opening criterion. Every result is split by **difficulty
(easy/med/hard) × horizon (1push / 2push)**. ⚠ Difficulty is defined *per horizon* and is not the same scale
across them: **2push** difficulty = number of solving first-pushes (`n_setups` → `division`); **1push**
difficulty = `solve_rate` tertiles. So "hard" = *few solving setups* for 2push, *few opening pushes* for 1push
— compare within a horizon, not across.

**Contents** — 1. [Reactive vs floor](#1-reactive-control-learned-value-vs-the-random-floor) ✅ ·
2. [Best-first search vs floor](#2-best-first-search-learned-value-vs-the-random-floor) ✅ ·
3. [Step-penalty (−1/0/1)](#3-step-penalty-101-reward) ◑ softened reject · [Prior work](#prior-work-seeded-ledger)

---

## 1. Reactive control: learned value vs the random floor

The main model **NoHz-v3** (learned value, argmax setup → argmax finish) vs a **uniform-random** push of the
labelled object (no model), under the forced-dive reactive protocol. Random = 10 seeds, NoHz-v3 = 3 seeds;
mean ± std of region-opening rate (%). → card: [[_reactive_search]].

**Table 1a. 2push** — does the region open within **2 pushes** (metric = open@2, %)? Pure-2-push set, n = 1018.

| difficulty | random | **NoHz-v3** | lift (pt) |
|---|---|---|---|
| easy | 9.7 ± 1.6 | **61.2 ± 2.4** | +51.5 |
| medium | 4.4 ± 0.9 | **44.3 ± 2.9** | +40.0 |
| hard | 1.8 ± 0.6 | **27.5 ± 2.0** | +25.7 |
| *overall* | *4.7 ± 0.6* | ***42.1 ± 1.7*** | *+37.4* |

**Table 1b. 1push** — does it open within **1 push** (metric = open@1, %)? One-push set, n = 1323.

| difficulty | random | **NoHz-v3** | lift (pt) |
|---|---|---|---|
| easy | 71.7 ± 2.1 | **98.7 ± 0.4** | +27.1 |
| medium | 32.6 ± 3.1 | **93.9 ± 0.5** | +61.3 |
| hard | 6.2 ± 1.6 | **54.3 ± 0.4** | +48.0 |
| *overall* | *37.0 ± 1.1* | ***82.3 ± 0.2*** | *+45.3* |

![[react_search.png]]

**Finding.** The learned value beats the random floor in **every cell, both horizons** — ~9× on 2push (42.1
vs 4.7) and +45pt on 1push (82.3 vs 37.0). The 2push lift shrinks with difficulty (easy +51.5 → hard +25.7):
random almost never cracks hard 2push (1.8%), and the model also drops most there (27.5%) — hard 2push is
where the headroom is. The random band is tight (±0.6–3.1) because each seed averages ~1000 episodes, so its
seed-to-seed std is just the binomial SE of a proportion — verified, not noise.

---

## 2. Best-first search: learned value vs the random floor

Greedy best-first search, budget **900 sims/instance**, combine=q: NoHz-v3's predicted Q ranks expansions vs
uniform-random ordering. 10 random seeds, 3 model seeds. → card: [[_full_search]] (full per-cutoff tables,
a/b/c/d diagnostics, time-by-difficulty).

**Table 2. 2push** — best-first solve-rate within **budget 900 sims** (%), and cost (avg sims to solve).
Pure-2-push set, n = 1018.

| difficulty | random | **NoHz-v3** | avg sims: rand → NoHz |
|---|---|---|---|
| easy | 99.8 ± 0.2 | 98.9 ± 0.4 | 43 → 38 |
| medium | 97.0 ± 0.7 | 97.8 ± 0.8 | 123 → 62 |
| hard | 78.7 ± 2.1 | **90.2 ± 0.8** | 344 → 165 |
| *overall* | *91.0 ± 0.8* | ***95.3 ± 0.6*** | *185 → 94* |

**Table 2b. 1push** — best-first on the one-push set, budget 900 sims. n = 1323. On this set the pool is small
enough that best-first almost always finds *a* solver, so both methods hit the same ceiling — the win moves to
**rank and cost**, not final solve-rate.

| difficulty | solve@900: random / NoHz-v3 | avg sims: random → NoHz-v3 |
|---|---|---|
| easy | 99.8 / 99.8 | 2 → 1 |
| medium | 100.0 / 100.0 | 3 → 1 |
| hard | 99.3 / 99.2 | 20 → 7 |
| *overall* | *99.7 / 99.7* | *8 → 3* |

*1push wall-time by difficulty is pending a separate emeraldrapids-exclusive run (rlab7 was co-tenanted, so its
t_wall is not comparable) — deliberately not reported here.*

![[fullsearch_success_vs_sims.png]]

**Finding.** The aggregate gap (95.3 vs 91.0) is **modest and misleading** — the difficulty split shows the
solve-rate win is **entirely in HARD** (+11.5 pt: 90.2 vs 78.7). On easy/medium both reach ~98% by budget 900,
so there the model's value is **pure efficiency** — roughly **half the sims** (medium 62 vs 123). The ranker is
strong: the winning first push is the model's **#1 pick 50.9%** of the time (top-3 69.8%) vs random's 14.9%.
On **wall-time** (fair, emeraldrapids-exclusive) the sim-savings convert to real speed on hard (26 s vs 46 s),
but on easy the curves **cross** (7.1 s vs 6.3 s) — when problems are already trivial, the model's per-sim
NN-scoring overhead makes brute-force marginally faster. One tension flagged in the card: a heavy sims-to-solve
tail (mean 94 despite median-rank-0) traces to the H1/H2 cross-head scale mismatch → a `dive_bonus` sweep is the
natural follow-up.

On **1push** (Table 2b) the story is different: the one-push pool is small, so both methods reach ~99.7% by
budget 900 in every tier — the learned value buys no extra solve-rate, only **rank and speed**. Its first pick
already opens the goal **82.3%** vs random's **37.3%** (hard **54.2 vs 5.9**, ~9×) — and those are *exactly* the
reactive open@1 numbers in Table 1b, because **the search's first sim is the reactive pick**. It also reaches
the solution in **~⅓ the sims** (hard 7 vs 20). So 1push is the reactive ranking win re-expressed as efficiency.

---

## 3. Step-penalty (−1/0/1 reward)

**Verdict: softened reject (horizon-split).** We retrain the no-horizon q-scorer on a *signed* target
(+1 immediate-open / 0 valid-setup / −1 never-opens) and test whether it ranks pushes better for best-first
search than the incumbent 0/0.9/1. 3-way vs random and NoHz-v3, mean ± std across seeds. → card:
[[_step_penalty_]].

**Table 3. Best-first search, 2push** (combine=q, budget 900, n = 1018) — the ranking test.

| difficulty | ranker | @2 | @30 | @900 | sims-to-solve |
|---|---|---|---|---|---|
| easy | NoHz-v3 | 55.0 | 85.7 | 98.9 ± 0.4 | 28 |
| easy | step-pen | 54.9 | 87.0 | 99.2 ± 0.0 | 26 |
| medium | NoHz-v3 | 40.7 | 75.4 | 97.8 ± 0.8 | 43 |
| medium | step-pen | 39.4 | 74.5 | 97.4 ± 0.6 | 44 |
| hard | NoHz-v3 | 26.1 | 56.1 | 90.2 ± 0.8 | 88 |
| hard | step-pen | 22.6 | 54.1 | 90.7 ± 0.8 | 97 |
| *all* | *random* | *3.7* | *41.1* | *91.0 ± 0.8* | *115* |
| *all* | *NoHz-v3* | *38.7* | *70.8* | ***95.3 ± 0.6*** | *55* |
| *all* | *step-pen* | *36.9* | *70.0* | ***95.4 ± 0.5*** | *58* |

![[steppen_bestfirst_sims_2push.png]]

**Table 3b. Best-first search, 1push** (budget 900, n = 1323). solve@1 is the ranking metric; @900 ties because
the one-push pool is tiny (both eventually solve everything), so the contest is entirely front-loaded rank.

| difficulty | ranker | solve@1 | solve@900 | sims-to-solve |
|---|---|---|---|---|
| easy | NoHz-v3 | 98.7 | 99.8 | 1 |
| easy | step-pen | 98.5 | 99.8 | 1 |
| medium | NoHz-v3 | 94.0 | 100.0 | 1 |
| medium | step-pen | 94.7 | 100.0 | 1 |
| hard | NoHz-v3 | 54.2 | 99.2 | 7 |
| hard | **step-pen** | **56.7** | 99.1 | 6 |
| *all* | *random* | *37.3* | *99.7* | *8* |
| *all* | *NoHz-v3* | *82.3* | *99.7* | *3* |
| *all* | *step-pen* | ***83.3*** | *99.6* | *3* |

![[steppen_bestfirst_sims_1push.png]]

Reactive open-rate (secondary), Δ = step-pen − NoHz-v3: 2push open@2 all **−2.5** (hard −3.5); 1push open@1 all
**+1.0** (hard **+2.5**) — the same 1push-hard signal the search shows.

**Table 3c. Fair 3-way wall-time** (interleaved, sapphirerapids-exclusive; avg t_wall per instance, seconds). The
timed NoHz-v3 reproduces the full-search sim anchor bit-for-bit (0/2341 mismatch) and matches its emeraldrapids
wall-times within ~5%, so this run pools.

| horizon | random | NoHz-v3 | step-pen |
|---|---|---|---|
| 2push | 26.7 | 16.0 | 15.6 |
| 1push | 1.35 | 0.70 | 0.63 |

**Finding.** The result **splits by horizon**. On **2push** (setup ranking — the harder, primary axis) the signed
target is a **wash**: tied at the ceiling (95.4 vs 95.3) and *marginally worse* at low budgets (@2 36.9 vs 38.7)
exactly where a sharper ranker should win, at more sims (58 vs 55). On **1push** (immediate-open ranking) it earns
a small, real edge — **solve@1 83.3 vs 82.3 (+1.0 all, +2.5 hard)** and leads the whole low-budget curve, though
@900 ties. Best-first solve@1 reproduces reactive open@1 *exactly* (step-pen 83.3 = 83.3), so two independent
pipelines agree the 1push-hard gain is real. **Wall-time agrees** (Table 3c): 2push step-pen ≈ NoHz (15.6 vs
16.0 s, tie), both ≪ random (26.7 s) with the gap on hard (~27 s vs 48 s); 1push both ~½ of random — no horizon
where the signed target costs time. **Call: 0/0.9/1 stays incumbent** — it wins the harder 2push axis
and the 1push edge is small and ceiling-tied — but the hypothesis is **not cleanly false**: the signed target
sharpens the *open-now* decision, not the *setup* decision. Natural follow-up: apply −1/0/1 to the open-now (H1)
head only, keep 0/0.9/1 for setup — bank the 1push gain without the 2push wash.

---

## Prior work (seeded ledger)

Compact history, pre-loop. Detail in the [model registry](horizon_q_model_registry.md).

| Date | Experiment | Metric | Verdict |
|---|---|---|---|
| 2026-06-29 | Render speedup (`fast_scorer`) | 2019→101 ms · render-equiv 158/158 | ✅ accept, no retrain |
| 2026-06-27 | NoHorizon vs Horizon @2 | reactive 40.7 / best-first 37.8; NoHz ≥ Hz | ~ tie (NoHz ≥ Hz) |
| 2026-06-15 | M2b (+ dead-ends), 1-push | hard@1 32.86 ± 2.4 · 2-push e2e 61.9% | ✅ best 1-push model |
