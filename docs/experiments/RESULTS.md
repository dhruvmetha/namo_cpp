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
(easy/med/hard) × horizon (1push / 2push)**.

**Contents** — 1. [Reactive vs floor](#1-reactive-control-learned-value-vs-the-random-floor) ✅ ·
2. [Best-first search vs floor](#2-best-first-search-learned-value-vs-the-random-floor) ✅ ·
3. [Step-penalty (−1/0/1)](#3-step-penalty-101-reward) 🔄 · [Prior work](#prior-work-seeded-ledger)

---

## 1. Reactive control: learned value vs the random floor

The main model **NoHz-v3** (learned value, argmax setup → argmax finish) vs a **uniform-random** push of the
labelled object (no model), under the forced-dive reactive protocol. Random = 10 seeds, NoHz-v3 = 3 seeds;
mean ± std of region-opening rate (%). → card: [[_reactive_search]].

**Table 1.** Region-opening rate (%), by horizon × difficulty.

| horizon | difficulty | random | **NoHz-v3** | lift |
|---|---|---|---|---|
| 2push (open@2) | easy | 9.7 ± 1.6 | **61.2 ± 2.4** | +51.5 |
| 2push | medium | 4.4 ± 0.9 | **44.3 ± 2.9** | +40.0 |
| 2push | hard | 1.8 ± 0.6 | **27.5 ± 2.0** | +25.7 |
| 2push | **all** | 4.7 ± 0.6 | **42.1 ± 1.7** | **+37.4** |
| 1push (open@1) | easy | 71.7 ± 2.1 | **98.7 ± 0.4** | +27.1 |
| 1push | medium | 32.6 ± 3.1 | **93.9 ± 0.5** | +61.3 |
| 1push | hard | 6.2 ± 1.6 | **54.3 ± 0.4** | +48.0 |
| 1push | **all** | 37.0 ± 1.1 | **82.3 ± 0.2** | **+45.3** |

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

**Table 2.** Solve-rate @900 (%) and cost (avg sims), by difficulty (2push). *(1push in progress.)*

| difficulty | random @900 | **NoHz-v3 @900** | avg sims (rand / NoHz) |
|---|---|---|---|
| easy | 99.8 ± 0.2 | 98.9 ± 0.4 | 43 / 38 |
| medium | 97.0 ± 0.7 | 97.8 ± 0.8 | 123 / 62 |
| **hard** | 78.7 ± 2.1 | **90.2 ± 0.8** | 344 / 165 |
| **all** | 91.0 ± 0.8 | **95.3 ± 0.6** | 185 / 94 |

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

---

## 3. Step-penalty (−1/0/1 reward)

🔄 **In progress.** Retrain of NoHz on a signed target (immediate-open +1 / valid-setup 0 / never −1) vs the
current 0/0.9/1, testing whether the reshaped reward improves search **ranking**. Training done (3 seeds,
iLab); eval near-complete (sims/reactive) with 2push + 1push timing running on Amarel. → card: [[_step_penalty_]].
Results + the 3-way comparison (random / NoHz-v3 / step-penalty) land here on completion.

---

## Prior work (seeded ledger)

Compact history, pre-loop. Detail in the [model registry](horizon_q_model_registry.md).

| Date | Experiment | Metric | Verdict |
|---|---|---|---|
| 2026-06-29 | Render speedup (`fast_scorer`) | 2019→101 ms · render-equiv 158/158 | ✅ accept, no retrain |
| 2026-06-27 | NoHorizon vs Horizon @2 | reactive 40.7 / best-first 37.8; NoHz ≥ Hz | ~ tie (NoHz ≥ Hz) |
| 2026-06-15 | M2b (+ dead-ends), 1-push | hard@1 32.86 ± 2.4 · 2-push e2e 61.9% | ✅ best 1-push model |
