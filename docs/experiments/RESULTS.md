---
status: hub
tags:
  - results
updated: 2026-07-16
---
# Results — DAgger curriculum training framework

The maintained approach. A clean seed, then rounds of **{ generate fresh scenes → screen with the current model → keep its mistakes → label → accumulate → retrain → eval }**, laddering **1-push → 2-push → …** (Ant-Man → Beast → …). Each stage's model also produces the *next* stage's dataset for free (screened-dead scenes become the multi-push bank).

**Setting:** CAR robot, testset `namo_testset_v1`, region-opening criterion. Every result split by **difficulty (easy/med/hard)**. Difficulty is defined *per horizon* (compare within a horizon, not across): **1-push** = `solve_rate` fixed cuts (hard < 0.05 / med 0.05–0.30 / easy ≥ 0.30); **2-push** = number of solving first-pushes (`n_setups`). Full per-experiment detail lives in each card under `log/`.

**Archive:** the pre-curriculum line — horizon-Q / NoHz single-ranker, RL-only self-imitation, horizon-role probe, prior-work ledger — is preserved verbatim in [archive/RESULTS_pre_dagger_horizonq_2026-07.md](archive/RESULTS_pre_dagger_horizonq_2026-07.md).

---

## 1-push ladder (Ant-Man)  ✅ CLOSED (saturated ~39 hard@1, beats exhaustive NoHz-v3)

Clean 50k seed, then five DAgger rounds screening fresh scenes with antman_{r-1} (best-first, budget 300), keeping not-top-5 mistakes. Full table, execution notes, open questions → [EXP-2026-07-14 card](log/EXP-2026-07-14-region-opening-curriculum-marvel.md).

| model | rows | easy@1 | med@1 | **hard@1** | all@1 | hard@20 |
|---|---|---|---|---|---|---|
| antman-0 (seed) | 50,000 | 92.7 | 63.7 | **23.0** | 72.7 | 85.8 |
| antman-2 | 90,700 | 95.6 | 73.2 | **28.4** | 78.1 | 91.2 |
| antman-3 | 120,657 | 96.3 | 77.2 | **32.8** | 80.4 | 91.7 |
| antman-4 | 151,218 | 96.7 | 81.2 | **39.2** | 82.9 | 94.6 |
| antman-5 (449k, undersized — noise) | 167,655 | 97.1 | 78.9 | **42.6** | 82.9 | 92.6 |
| **antman-5-redo** (737k, 3-seed) | 178,364 | 97.3 | 81.0 | **39.1±1.4** | 83.1 | — |
| _NoHz-v3 baseline_ (exhaustive) | — | 97.9 | 81.5 | **30.9** | 82.3 | — |
| random | — | 62.6 | 19.2 | **1.5** | ~39.4 | 37.7 |

**Finding.** hard@1 **23.0 → ~39** over five rounds (~26× random), then **PLATEAU**; gains all at low k (better *ordering*, sim verifies for free). Beats the exhaustive **NoHz-v3** baseline by **+8.2 hard@1** (39.1 vs 30.9) with cheap sampled data. The old antman-5 (449k) 42.6-hard/78.9-med was undersized-run noise — the full-scale 3-seed redo lands at 39.1 with med restored to 81.0.

**Round-5 redo + 3-arm control [2026-07-16] settled both open questions:** at 178k rows hard@1 saturates ~40, and **DAgger targeting buys nothing over volume** — mistakes (+1.8 over base), random iid (+3.2), difficulty-matched iid (+3.3) all TIE (base 37.3±1.0; each arm = base + same 27k rows, only selection differs). The lift is pure volume; composition is irrelevant. **⇒ 1-push CLOSED; the lever for more hard performance is 2-push structure.** Free byproduct: **72,521 labeled 2-push (Beast) episodes** + ~865k unlabeled leads. Details → [EXP-2026-07-14 card](log/EXP-2026-07-14-region-opening-curriculum-marvel.md).

## 2-push ladder (Beast)  ✅ ROUND-1 CLOSED — champion **beast-1-c081** [2026-07-19, single seed]

The same ~190k antman-solvable scenes, relabeled at depth-2 with the censored "forward-ness" grammar (opener=1 / verified setup=exact 0.9 / unknown ≤0.9 / no-2-push ≤0.81) and trained with the -c recipe (censored HL-Gauss + rank-aux-over-ceilings + unreachable-floor). Full arc, ablations, and the label grammar → [EXP-2026-07-14 card](log/EXP-2026-07-14-region-opening-curriculum-marvel.md).

| model | recipe/data | e/m/**hard**/all@1 (1push) | 2push solve/avg sims/@30 | **2push@2 (perfect play)** | hardh2 s2s |
|---|---|---|---|---|---|
| antman-5 | zeros, 0% rich | 96.7/82.4/**40.7**/83.5 | 91.4/154/60.7 | — | 15.2 |
| antman-5c | ceilings, 0% rich | 98.3/85.3/**39.7**/85.1 | 91.5/138/61.4 | 26.3 | 12.4 |
| beast-0 g0.3 | zeros + depth2 flood | 89.0/65.3/**25.0**/71.6 | 94.9/120/60.2 | — | — |
| beast-0c | -c recipe, 25% rich | 98.1/85.3/**45.6**/85.9 | 93.7/117/65.9 | 27.0 | 10.3 |
| beast-1-c09 | 100% rich, strict ≤0.9 fails | 97.3/83.8/**38.7**/84.0 | 94.5/104/67.6 | — | 11.3 |
| **beast-1-c081** | **100% rich, posterior ≤0.81 fails** | 97.9/86.9/**48.5**/86.8 | **95.1/93/69.4** | **32.4** | **8.5** |
| random | — | 62.8/15.4/**2.5**/38.4 | 89.9/194/37.6 | 3.1 | 25.9 |

**Findings.** (1) **The labels were the lever, not the data**: identical 2-push sims labeled as zeros collapsed 1-push (beast-0), as loose ceilings wasted the data (c09 lost to a model with ¼ the rich labels), as verified-values+posterior-bounds produced the best model on every axis. (2) **Posterior ceilings beat strict** (+2.8 all@1 / +9.8 hard@1 on identical data): write the tightest bound the posterior supports (~96%-dead after a failed top-15 sweep ⇒ ≤0.81), not the tightest bound proven. (3) **Label density is load-bearing** (full/30/15/8 labels per board → hard@1 45.6/40.2/35.3/25.5) — don't sample setups. (4) **Perfect-play @2 is the sharpest discriminator** (26-27 for everything before, 32.4 for the champion; 10× random) — adopt as a standing column. (5) **The hard 1-push tier is a depth-1 artifact**: given depth-2, even random solves 98.5% (via dense setup routes); models beat random only under ~30-sim budgets — tight-budget ordering is the whole game. (6) Collection economics settled by pilot (n=2.4M): sweep with antman-5c at k=15 → 97.7% retention at 30% cost; the sweep ranker changes cost only, never answers (99.86% agreement).

**Plots:** [arc curves (antman-5 → beast-0c → beast-1-c081, 3 suites)](plots/beast_round1_arc_curves.png) · [ceiling A/B + density ablation](plots/beast_round1_ab_density.png) · [hard-tier depth-2 experiment](plots/hard1p_depth2_experiment.png) · [beast-0a-era curves](plots/beast0a_curves.png).

**Caveats:** single seed throughout (confirm-seed pending — the diagnostics below make this more urgent); 23,639 bonus episodes in round-1 (12.3% of the set — same rooms, extra blocking objects; agent-verified: ledger reconciles to the row, zero test leakage by exact-path + content-diff, and the extras are *harder* than the targets (19% vs 99.9% opener-rate), so they toughen rather than inflate; 17,461 original targets went unrecovered — open item).

### Post-round-1 diagnostics (2026-07-20) — model forensics + the round-2 data recipe

Four studies reusing round-1 artifacts, no new collection. Full detail → [EXP-2026-07-14 card](log/EXP-2026-07-14-region-opening-curriculum-marvel.md) "Post-round-1 diagnostics"; reports at `curriculum2/beast/round1/{analysis/beast1_c081,mix_arms}/`.

1. **What the champion learned** (`analysis/beast1_c081/report.md`): **openers mastered** (score median 0.973 vs 1.0 target, AUC 0.963 vs dead) but **setups NOT** (0.583 vs 0.9 target — collapsed toward dead; the core defect behind the 32.4 @2 ceiling). **Forward-ness is real** — board-max score separates opener-exists from 2push-only at **AUC 0.892**. Hard-1push misses are "bad pushes scored high" (opener still ~0.893; 100% of out-rankers are proven non-openers), i.e. weak fine-ordering among high-scorers. vs antman-5c: **net +37 outright 2push solves (34 hard)**.

2. **Finish-board gates → round-2 data recipe** (`mix_arms/REPORT.md`, on-disk labels only): **Gate 2 (masking) PASS → exhaustive finish recollection (~28k wh) CANCELLED** (k15 finish labels ≥ dense: @2 34.7 vs 29.4). **Gate 1: finish boards belong in round-2, ADDED to full roots in k15 form** — M2-k15 = new best 2-push (@2 32.4→**34.7**, @30 69.4→72.4, avg 93→89); **never substitute** (M1 = roots-swapped-out lost everywhere). Distinct from the density ablation (that was ROOT setup density; finish ceilings want k15-sparse).

3. **k-policy for round-2 sweeps**: finish-layer base rate is **74/26 live/dead** (not the 54/46 previously claimed); at that prior NO k reaches 95% posterior-dead offline (needs recall ≥0.982; champion LB recall@30 = 0.90). Recipe: **k≈20 + ≥0.95 early-exit, ~0.85 ceilings, 1–2k exhaustive-finish calibration batch first**. Tension: c081's 0.81 ceilings still won round-1 empirically — so aggressive-cheap-negatives, not calibrated-96%-dead, may be the real reason.

4. **Extras attribution — beast-1-c081-noextra** (`eval/noextra_results.md`): champion recipe on the H5 minus the extras (170,772 rows) → 1p med/hard/all@1 83.4/45.1/85.3, 2p solve/@2/@30 95.4/28.0/69.3. **Extras mildly load-bearing — keep them** (dropping ~21k of the harder episodes cost ~3 hard@1 / ~4 @2, left solve/@30 flat; the volume-is-the-lift pattern). **Seed nuance:** champion's 48.5 hard@1 looks like a HIGH seed — noextra (45.1) + M2 arms (43.6/43.1) cluster at 43–45 → confirm-seed before locking round-2.

**Round-2 recipe (specified):** dead-bank (true-2push-only) + fresh scenes → exhaustive ROOT sweeps (dense) + champion-ordered finish sweeps (k≈20, ≥0.95 early-exit, ~0.85 ceilings + calibration) → train on dense roots + k15 finish boards ADDED; target the setup-anchor gap; scene-selection stays dumb (targeting ≈ volume, proven). **Open:** confirm-seed of champion/M2-k15; dead-bank (Phase B) collection; registry row for beast-1-c081 (with @2); the 17,461 unrecovered round-1 targets.

## 2026-07-21 — Round-2 exhaust-on-miss slice: arms A/B (card EXP-2026-07-14)

24,168 dead-bank scenes collected with the new exhaust-on-miss labeler (commit 5fd5259) → `beast2_all.h5` 1,039,341 rows → armA (uniform) / armB (balanced 50/50 root-vs-postpush). Deploy 1p@1 armA 97.9/83.8/43.1 all 85.0 · armB 97.3/81.0/**49.5** all 84.7 (B hard@1 = best of beast line, champion 48.5). 2p armA solve **96.7** avg sims **81.6** @30 **71.4** (all best-ever) · armB 96.3/94.5/67.4. hardh2 99.0/8.0 · 99.5/9.3. Dead-bank GT: setup-vs-dead AUC 0.913/0.921 (r1-clean 0.876); **true recall@20 67.0→90.6%**. Testset setup-vs-dead wall UNMOVED (0.72-0.73) ⇒ wall = eval-set property, not model/data. H6: exposure ratio trades 1p-hard vs 2p-search; no dominant arm. Single seed; H3/H5 pending. Bank false-dead measured: 14.5% of "dead" scenes have root openers.

## 2026-07-21 (PM) — The 2×2: ceiling×exposure on corrected round-2 data (card EXP-2026-07-14)

Label corrections [USER]: dead cells are ceilings (root 0.81=γ², finish 0.9=γ — proof covers only depth≤2), sparse top-5-hit boards dropped. Identical-data twins (859,766 rows). **Ceiling BINDS in the dead-heavy regime:** ceil>hard on every 2p axis, both arms — solve +2.7/+3.1, avg sims −27/−49 (clean-pair edge tripled), @2 +5.7/+13.1, @30 +4.3/+11.9. Mechanism measured: ordering is label-invariant, MAGNITUDES are not (hard opener median 0.11–0.19 vs ceil 0.63–0.75) → cross-board priority queue misorders. **armB_ceil = new 2p front-runner: 97.2 solve / 78.6 avg sims / 69.7 @30, 1p 86.4 all@1.** Canonical finish-GT minted (testset_gt.h5, 982 scenes): recall@20 94.6–97.7 for ALL models (the 67→90.6 gap was dead-bank-specific); canonical root wall 0.75–0.80 for everyone, unmoved by round-2 data. Single seed/cell; deltas 5–10× guardrails.
