---
status: snapshot
tags: [experiment]
thread: scorer-search
updated: 2026-06-22
---

# NAMO Horizon-Q — Results & Design Report

> **⚠ HISTORICAL SNAPSHOT (2026-07-06): budget/horizon-conditioning was later DROPPED** (measured ≈ no-horizon, **NoHz** ahead — 40.7 vs 34.1). This dated report stands as the 2026-06-15 record; where it frames the design as budget-conditioned "Horizon-Q", read that as the **historical** line. Live model = single value/ranker (NoHz), job = first-push (setup) ranking. Current framing: [../problem_and_approach.md](../problem_and_approach.md). All numbers below stay valid as of their date.

**2026-06-15 ~07:30 ET.** All numbers verified against source files (json / journal §9–§12). Companion to the build journal.

---

## TL;DR
1. **Search is ~solved (~95%); reactive is the prize and it's broken (~24%).** And at full search the model (94.9) barely beats brute-force random (90.8) — its only real win is sim-efficiency. The whole payoff lives in **reactive** ("act without searching"), and that's what's broken.
2. **Reactive ≈ P(good setup) × P(good finish) ≈ 0.37 × 0.42.** Two weak top-1s that *multiply*. You must fix **both**.
3. **The finish is a pure *generalization* gap** (we ruled out render/state/on-policy by direct measurement); **the setup is *finishability-blind*** (a training-target flaw). The fixes are **ExIt** (diverse finish data) and the **recurrence** (relational setup label) — currently training.

---

## Part 1 — Where the numbers stand

### 1-push (ranking: `eval_scorer` hit@k on the `onepush` key; "can it rank the opener high?", no sim)
Horizon-v2, by difficulty division (`@1 / @5 / @10`):

| division | H=1 query | H=2 query |
|---|---|---|
| hard | **36.0** / 69.8 / 82.5 | 30.7 / 58.7 / 74.1 |
| med | 84.2 / 94.4 / 96.9 | 76.6 / 92.1 / 95.8 |
| easy | 98.6 / 99.8 / 100 | 95.0 / 99.1 / 100 |

Reading: **1-push opening is essentially solved on easy/med** (≥84% top-1); **hard is the frontier** (36% top-1, but 70% by top-5). The H=2 column is the same model asked "with 2 budget" — it stays close to H=1 (residual ~5pp "dilution": it slightly over-prefers setups even when a 1-push win exists).

**Ranking 2×2** (hard@1, the headline cross-model number):

| model | H=1 | H=2 |
|---|---|---|
| Horizon-v1 | 34.4 | **12.2** ← dilution bug |
| NoHorizon-v1 | 21.2 | 21.2 |
| **Horizon-v2** | **36.0** | **30.7** ← fixed by the aug |
| **NoHorizon-v2** | 31.7 | 31.7 |

The v2 data augmentation fixed Horizon's H=2 dilution (12.2→30.7) and lifted NoHorizon too. Net: Horizon's edge is the **H=1 specialization** (36.0 vs 31.7); at H=2 they tie.

### 2-push (solve-rate: `best-first` search on the `pure2push` key, object-constrained, capped 900 sims)
The full **2×2 matrix** (`s@K` = % solved within K real-sim pushes; `@2` = reactive, `@900` = full search):

| cell | rankH1 | rankH2 | **s@2** | s@10 | s@50 | s@100 | **s@900** | avg-sims |
|---|---|---|---|---|---|---|---|---|
| Horizon-v1 | 34.4 | 12.2 | 22.3 | 50.4 | 71.5 | 81.8 | 93.8 | 58.5 |
| NoHorizon-v1 | 21.2 | 21.2 | 28.7 | 46.8 | 63.1 | 71.4 | 89.0 | 85.1 |
| **Horizon-v2** | 36.0 | 30.7 | 24.2 | **55.3** | **76.3** | **82.6** | **94.9** | **54.6** |
| **NoHorizon-v2** | 31.7 | 31.7 | **32.6** | 52.0 | 67.5 | 74.0 | 91.6 | 76.7 |
| RANDOM (5-seed) | — | — | 3.3 | 19.8 | 51.0 | 63.6 | **90.8** | 113.6 |

Two regimes, two winners:
- **Reactive (@2):** NoHorizon wins in *both* data versions (28.7>22.3, 32.6>24.2). Surprising — fixing Horizon's H=2 ranking did **not** flip reactive.
- **Search (@50–900):** Horizon wins in both (+8–9pp mid-budget, 1.4–1.5× sim-efficiency).
- **The random row is the punch:** at @900, model 94.9 vs random 90.8 — only +4pp. The model's value is *sim-efficiency* (54 vs 114 avg sims), not the ceiling.

### The reactive bottleneck (oracle headroom, n=130)
Decomposing reactive@2 with the ground-truth (a1,a2)→opens map:

| condition | reactive@2 |
|---|---|
| model / model (today) | **13.8** |
| oracle finish, model setup | 36.9 |
| oracle setup, model finish | 41.5 |
| oracle / oracle | 100 |

⇒ **reactive ≈ P(top setup real) × P(top finish opens) ≈ 0.37 × 0.42.** Fixing one head alone caps you at the other (~37–42%); you need **both**, and gains compound.

---

## Part 1B — Compiled results by TRUE difficulty (multi-seed headline)

The authoritative tables, **split by true solution density** (NOT finish-density — that's R1 below), **fixed cutoffs** ([USER]: non-tertile; verified **binning-robust** — the conclusion is identical under 33% tertiles). Cells = mean±std across seeds (R2 = 2 seeds s1/s2; R3 = 3 seeds). Regenerate via `scripts/sandbox/compile_uniform.py`. Full tables: `/scratch/dm1487/eval/compiled/results_tables.md`; uniform figures: `compiled/fig_R2.png`, `compiled/fig_R3.png`.

### R1 — finish difficulty per solvable post-push state (ground truth, no model)
**88% of post-push states are DEAD** (7,551 finishable of 64,061). On the solvable ones:

| quantity | median | mean | p25 | p75 | p90 |
|---|---|---|---|---|---|
| finish density (%) | **6.9** | 15.7 | 3.0 | 18.2 | 41.7 |
| openers (numerator) | 3 | 4.4 | 2 | 6 | 10 |
| reachable a2 (denominator) | 45 | 50 | 26 | 70 | 90 |

needles: 1-opener **25%** · ≤2 41% · ≤3 54% · density≤10% **61%**. ⇒ the *problem* is needle-in-haystack; that's the structural reason reactive is hard (R1 is the *why* under R2/R3).

### R2 — 2-push solve@K by true 2-push difficulty (solving (a1,a2)/reachable (a1,a2); 2 seeds)
| tier | contender | @2 | @10 | @50 | @100 | @900 |
|---|---|---|---|---|---|---|
| EASY (254, >2%) | Horizon-v2 | 34.4±1.9 | 70.7±2.5 | 91.3±1.1 | 94.1±1.1 | 98.6±0.3 |
| | NoHorizon-v2 | **51.8±3.1** | 75.8±3.6 | 86.6±2.8 | 91.5±0.8 | 97.8±1.4 |
| | random | 8.0 | 41.7 | 80.3 | 89.3 | 98.8 |
| MED (330, 0.5–2%) | Horizon-v2 | 28.3±1.5 | 63.8±1.1 | 85.6±0.6 | 90.3±0.4 | 98.8±0.0 |
| | NoHorizon-v2 | **37.1±0.6** | 59.8±2.8 | 76.1±0.0 | 84.5±0.9 | 98.5±1.3 |
| | random | 2.9 | 20.8 | 60.1 | 75.5 | 99.0 |
| HARD (434, ≤0.5%) | Horizon-v2 | 15.7±1.0 | **38.7±1.0** | **58.9±2.4** | **68.5±2.4** | **88.8±1.5** |
| | NoHorizon-v2 | **18.9±1.0** | 37.1±2.9 | 52.4±2.1 | 59.1±3.7 | 83.4±1.3 |
| | random | 0.8 | 6.2 | 26.9 | 39.4 | 79.8 |

**Sub-problem difficulty vs ground truth** (setup = valid a1/reachable a1; finish = openers/reachable a2; lower=harder):

| tier | setup density (med) | typical setup | finish density (med) | typical finish |
|---|---|---|---|---|
| EASY | 27% | 7 / 28 | 11% | 4 / 38 |
| MED | 11% | 6 / 50 | 4% | 2 / 56 |
| HARD | **3%** | 2 / 72 | **3%** | 2 / 70 |

Reading: **NoHz wins reactive (@2) every tier; Hz wins search (@50–900), decisive on HARD** (@900 88.8 vs 83.4). Both sub-stages stiffen EASY→HARD (setup 27→3%, finish 11→3%) — HARD is a *double needle* (≈2 setups in 72, then ≈2 openers in 70), and the haystack *grows* while the needle shrinks.

### R3 — 1-push opener ranking success@k by true 1-push difficulty (openers/reachable first-pushes; 3 seeds)
| tier | contender | @1 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| EASY (636, ≥30%) | Horizon-v2 (H=1) | 98.5±0.1 | 99.8±0.1 | 99.9±0.1 | 100.0±0.0 |
| | Horizon-v2 (H=2) | 94.0±1.5 | 99.1±0.2 | 99.8±0.2 | 99.9±0.1 |
| | NoHorizon-v2 | 98.7±0.4 | 99.9±0.1 | 100.0±0.0 | 100.0±0.0 |
| | random | 63.9 | 97.5 | 99.8 | 100.0 |
| MED (354, 5–30%) | Horizon-v2 (H=1) | 81.8±2.8 | 94.2±0.2 | 97.7±0.9 | 99.3±0.4 |
| | Horizon-v2 (H=2) | 74.9±3.7 | 92.4±1.0 | 96.0±0.3 | 98.2±0.5 |
| | NoHorizon-v2 | 82.1±1.0 | 93.7±0.2 | 96.6±0.3 | 98.9±0.3 |
| | random | 15.4 | 54.9 | 77.8 | 93.7 |
| HARD (189, <5%) | Horizon-v2 (H=1) | **34.0±2.9** | 65.8±3.5 | 80.1±2.2 | 91.9±0.8 |
| | Horizon-v2 (H=2) | 27.9±2.7 | 51.5±6.3 | 66.3±6.9 | 84.3±2.9 |
| | NoHorizon-v2 | 31.9±2.9 | 58.0±4.3 | 70.4±3.7 | 85.0±2.1 |
| | random | 2.4 | 11.9 | 22.8 | 42.3 |

Reading: **the H=2 query DILUTES the direct opener** — Hz drops H=1→H=2 by ~5–7pp @1 every tier (seed-robust, paired). **NoHz ≈ Hz-at-H=1 and beats Hz-at-H=2** (HARD 31.9 vs 27.9). The cleanest proof of R2's mechanism: the horizon trades direct-opening for setup-seeking. Both crush random (HARD 34 vs 2.4 = 14×).

**Why two difficulty scales?** R2 (pair density, median ~0.7%) and R3 (push density, median ~30%) live on different scales because one is per-*pair* and one is per-*push* — so "HARD" means a different absolute threshold in each. 1-push-solvable instances are easier by construction; pure-2-push is the residue that *needs* two.

---

## Part 2 — The diagnosis (what's broken, and how we know)

### The finish (Push 2) is mushy on novel states
On the states it *trained on*, the finish head is sharp (separation **0.75**). On *novel* post-setup states (test/deploy), it collapses to **~0.30** — true openers score ~0.4 instead of ~0.9, and its top-1 finish opens only **40%**. That 40% *is* the 0.42 factor above.

### We proved it's *generalization*, not a bug — by elimination
The full **hypothesis ledger** (verdicts on numbers only):

| # | hypothesis | evidence | verdict |
|---|---|---|---|
| A | fix H=2 dilution ⇒ reactive flips | 2×2: NoHz still wins 32.6>24.2 | **REJECT** |
| B | won't-dive = cross-head *scale* mismatch | targets encode opener 1.0 > setup 0.9; a fit model dives | **REJECT** (it's the finish mush) |
| C | finish mush = *state* divergence | re-exec reproduces saved state **0.00 mm** (n=60) | **REJECT** |
| D | finish mush = goal-channel *render* mismatch | train vs deploy crops **IoU 1.0, MAE 0.0** | **REJECT** |
| E | finish mush = H5-builder vs live-builder render | stored vs live crop **MAE ~0.0005** (n=40) | **REJECT** |
| F | finish mush = on-policy / which-setup shift | coll-setup 0.344 vs model-setup 0.287 → **Δ0.057** | **REJECT** (policy *minor*) |
| **G** | **finish mush = pure GENERALIZATION** | by elimination (C–F dead) + 0.75 vs 0.30 | **ACCEPT** |
| **H** | **setup head finishability-BLIND** | 33% of top setups *dead*; targets flat {0.9,1.0} | **ACCEPT** |
| I | pre-ExIt "max Q₁ vs opener-count" linchpin | artifact of mushiness, vanishes for a perfect finish | **REJECT** |

**Two surviving roots:**
- **G — the finish doesn't generalize.** It memorized a narrow finish set (~58k scenes, 4:1 fail-skewed) and doesn't transfer. *Not* a render/physics/which-state bug — those were each measured dead.
- **H — the setup is finishability-blind.** We trained Push 1 with a flat "is this a setup? → 0.9" label, never "how *finishable* is it." So it can't prefer a 15-opener setup over a 1-opener needle — and a *third* of its top picks lead to states with **no** finish at all.

---

## Part 3 — Design choices, and why

**Why a budget-conditioned horizon-Q at all?** The task is *finite-horizon* (≤K pushes), and the optimal value of a finite-horizon problem is genuinely *budget-dependent* (`V*_H ≠ V*_{H-1}`): a setup is great with 2 pushes left, useless with 1. A plain stationary `Q(s,a)` solves the *wrong* (infinite-horizon) Bellman equation. So `Q(s,a,H)` is the principled object. *Caveat the data showed:* its edge over plain no-horizon is **modest and regime-dependent** (2×2), and we never built the actual recurrence — only flat supervised heads.

**Why ExIt for the finish (G)?** A generalization failure from a narrow training set has exactly one fix: more, more-varied, correctly-labeled examples. ExIt manufactures them using our one superpower — a **perfect simulator**: step to a post-setup state, *try all ~40 finishes in sim*, record which open. *Honest:* we measured (F) that collecting from the model's *own* states barely matters (0.057) — so the real lever is **data diversity**, not "on-policy" cleverness.

**Why the recurrence for the setup (H)?** Replace the flat 0.9 setup label with a *relational* one: `setup value = γ × (the model's best finish-confidence at the state it leads to)`. Now Push 1 learns to steer toward easy-to-finish states, and the search dives by construction (γ<1 forces child > parent). It needs a *good* finish first → **ExIt before recurrence**. *Design detail:* the aggregate must be robust (`mean_top_k`, not raw `max`, which is fluke-dominated).

**Why drop the cascade / H2-on-finish ideas?** They're *deploy band-aids* (the cascade trades the search ceiling for +15pp reactive; it doesn't raise the ceiling). [USER]: fix the core, not patch deploy.

**Why run NoHorizon too?** [USER] — to measure the horizon's *actual* contribution. The ExIt finish retrain is the clean apples-to-apples; the recurrence is intrinsically a horizon mechanism, so **Horizon+recurrence vs NoHorizon+ExIt** is the measurement of the horizon's value-add.

---

## Part 4 — What ran overnight, what's running, next gates

- **P1 (done):** diverse opener-rich finish collection — **24,347 rows** (21,381 opener-bearing, 2,966 dead), `--setups valid`, exhaustively sim-labeled.
- **P2 (running):** finish retrain, **Horizon-v3 + NoHorizon-v3** (the ablation), v3 mix with the narrow postpush *replaced* by the ExIt data. At ~epoch 2; converges ~ep8 (hours out).
- **GATE (P2):** does novel-state finish separation climb **0.30 → 0.6+**? That's the go/no-go for the whole direction.
- **P3 (next):** on the *fixed* finish, check calibration — does `mean_top_k Q_finish` predict actual finish-success? (the real linchpin, replacing the rejected pre-ExIt one).
- **P4 (held until P2 gate passes):** recurrence retrain (Horizon-only) — relabel setups by `γ·mean_top_k Q_finish`, retrain. Gate: setup top-1 ↑, reactive@2 ↑.

**Target:** with both factors lifted toward ~70–80%, reactive@2 → ~**50–64%** (from ~15–25%). That's the prize.
