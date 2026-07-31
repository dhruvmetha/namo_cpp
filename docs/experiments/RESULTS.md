---
status: hub
tags:
  - results
updated: 2026-07-23
---
# Results — DAgger curriculum training framework

The maintained approach. A clean seed, then rounds of **{ generate fresh scenes → screen with the current model → keep its mistakes → label → accumulate → retrain → eval }**, laddering **1-push → 2-push → …** (Ant-Man → Beast → …). Each stage's model also produces the *next* stage's dataset for free (screened-dead scenes become the multi-push bank).

**Setting:** CAR robot, testset `namo_testset_v1`, region-opening criterion. Every result split by **difficulty (easy/med/hard)**. Difficulty is defined *per horizon* (compare within a horizon, not across): **1-push** = `solve_rate` fixed cuts (hard < 0.05 / med 0.05–0.30 / easy ≥ 0.30); **2-push** = exhaustive-GT setup density with the same fixed percentage cuts, with 37 build-version-unmatched roots reported as unknown. Full per-experiment detail lives in each card under `log/`.

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

## 2026-07-21 — d20 dose test: +20% dead added to 2c-A-ceil (card EXP-2026-07-21-colossus)

Does adding dead data past the 2c ablation keep helping? Took **beast-2c-A-ceil** (192,822 positive rows) and appended **20% dead, 50/50 non-dup**: 19,282 dead roots + 19,282 dead finishes sampled from `beast2_exh_ceil.h5` → **beast2c_d20_ceil** (231,386 rows). Same -c recipe, ceilings at build (root 0.81, finish 0.9), single seed. ckpt `round2/models/beast2c_d20_ceil/checkpoints/epoch010-val_loss1.7072.ckpt`.

| model | 1p e/m/h/all@1 | 2p solve/avg-sims/@2/@30 |
|---|---|---|
| beast-2c-A-ceil (0% dead) | 97.1/84.1/**35.3**/83.4 | 95.7/50.9/24.3/70.5 |
| **beast-2c-d20 (20% dead)** | 97.4/80.8/**39.7**/83.2 | 96.5/54.6/**26.6**/69.1 |

**Dead helps, measured two ways.** (1) Deploy: 1p **hard@1 +4.4** (35.3→39.7), 2p @2 +2.3, solve +0.8 — small cost at med@1 (−3.3) and @30 (−1.4); the dose sharpens the hard tier + tight-budget ordering at a mild mid-tier/deep-search cost. (2) Dead-bank GT AUC: opener-vs-dead **0.859→0.940 (+0.081)**, setup-vs-dead 0.851→0.906, dead-cell score median 0.196→0.022 (pushed down). A ~19-episode solve move AND an independent 0.08 AUC jump point the same way. **AUC→top-1 gap stays large** (0.94 vs 39.7 hard@1) — pooled separation good, per-board hard top-1 not yet — which is what MORE dead volume (colossus) is meant to close. **Supply wall:** used 19,282 of 19,448 available dead roots (166 left) → scaling past 20% needs new collection (colossus).

## 2026-07-22 — Colossus-0 top-20 + 20%-audit labeling smoke (card EXP-2026-07-21-colossus)

On the same 100 XMLs as the exhaustive baseline, root rejection + capped finishes + random exhaustive audits used **53,226 vs 173,902 simulator trials (−69.4%, 3.27× fewer)** while recovering all **15/15** shared genuine-2push roots and all 65/65 direct roots; old dead roots split into proven-dead or censored, never false-dead. Label H5: 269 exact setups + 269 exact finishers, 0 false exact-zero reachable cells, unknown parents/finishes masked. Finish winners: rank1 222 / rank2–5 40 / rank6–20 7 / audited >20 0. Locked training selector yields **63 positive-mistake + 1,354 negative rows per 100 XMLs**, so positive mistakes bottleneck the 166,666+33,334 Colossus block at a point estimate ~265k XMLs. No-op rows 562 and rank1-only finish rows 222 are excluded. This validates scalable labeling, not model improvement; d20+200k training/eval is next.

## 2026-07-22 — Push-depth-aware corrected final-pose head: three-seed negative result

Fresh baseline/corrected models used the same 178,364 Antman-5c ceiling-labeled boards and differed only in whether the shared post-attention action head received the corrected image-aligned final pose `(final_x, final_y, sin(final_yaw), cos(final_yaw))`. Three paired seeds trained from scratch on six identical `rlab3` A4000s; canonical prediction-only 1-push evaluation used the same 1,323 episode identities for every checkpoint and ran no push simulations.

| 1push tier | baseline mean @1 | corrected mean @1 | delta | baseline mean @5 | corrected mean @5 | delta |
|---|---:|---:|---:|---:|---:|---:|
| easy | 97.1% | 97.6% | +0.5 pp | 99.6% | 99.7% | +0.1 pp |
| med | 82.3% | 80.4% | -1.9 pp | 95.0% | 95.4% | +0.4 pp |
| hard | 40.4% | 37.2% | **-3.1 pp** | 72.9% | 73.0% | +0.1 pp |

| 1push tier | baseline wrong contact | corrected wrong contact | delta | baseline right-contact/wrong-depth | corrected right-contact/wrong-depth | delta |
|---|---:|---:|---:|---:|---:|---:|
| easy | 1.9% | 1.3% | -0.6 pp | 1.0% | 1.2% | +0.2 pp |
| med | 14.0% | 14.8% | +0.8 pp | 3.6% | 4.8% | +1.1 pp |
| hard | 53.6% | 54.6% | **+1.0 pp** | 6.1% | 8.1% | **+2.1 pp** |

**Verdict: failure to replicate.** Hard exact @1 fell in every seed (`-3.9/-2.0/-3.5` points), mean hard wrong-contact rose, and medium breached the two-point guardrail in two seeds. Mean @5 stayed flat, indicating that the treatment mainly worsened precise top-1 ordering rather than removing valid pushes from the shortlist. The earlier legacy-motion one-seed `+10.8` hard@1 result did not survive the corrected representation plus proper seed replication. Keep the original Antman-5c head; no D20 or 2-push simulator evaluation was run. Full per-seed tables and artifacts → [EXP-2026-07-22 card](log/EXP-2026-07-22-push-depth-aware-ranker.md).

## 2026-07-23 — Correct crop-relative motion and Fourier+depth identity: three-seed negative result

The final repair uses exactly the intended image-aligned relative push `(2dx/0.5m, 2dy/0.5m, dtheta/pi)`. Plain sends those three numbers through the post-attention action MLP; sharp adds eight-band Fourier features plus a learned five-depth identity. Both reuse the matched fresh Antman baseline seeds and the same 1,323 canonical prediction-versus-GT episodes; no push simulations ran.

| 1push tier | baseline @1 | plain @1 | sharp @1 | baseline @5 | plain @5 | sharp @5 |
|---|---:|---:|---:|---:|---:|---:|
| easy | 97.1% | 97.7% | 97.1% | 99.6% | 99.9% | 99.8% |
| med | 82.3% | 82.0% | 81.4% | 95.0% | 95.2% | 95.6% |
| hard | **40.4%** | **39.9%** | **40.5%** | 72.9% | **75.3%** | **76.8%** |

**Verdict: reject both for top-1 ranking.** Plain improved hard @1 in 0/3 seeds (mean -0.5 points); sharp improved 1/3 (mean +0.1), failed a medium guardrail, and did not reduce hard wrong-contact errors. Both improve the hard shortlist at @5 (+2.5/+3.9), but the research goal is to order the correct push first, not merely keep it nearby. This closes additive post-attention motion fusion; a future action-grounding experiment must change the interaction itself, such as action-conditioned attention or scene sampling at the proposed final footprint. Full per-seed and mechanism tables → [EXP-2026-07-22 card](log/EXP-2026-07-22-push-depth-aware-ranker.md).
## 2026-07-23 — Exact-value ranking loss pilot (card EXP-2026-07-22-exact-value-ranking-loss)

Generalizing the listwise auxiliary from exact `1.0` openers to every exact tier fixed the missing pure-2push-root supervision on unchanged d20 data. The intended mechanism moved strongly on exhaustive held-out roots: setup-vs-dead AUC **0.9063→0.9252**, setup hit@1 **55.0→64.5**, hit@5 **83.9→88.1**, and mean first-setup rank **4.01→3.44**.

| model | 1p easy/med/hard/all@1 | 1p hard@5 | 2p @2/@5/@10/@30 | 2p avg sims all |
|---|---|---:|---|---:|
| d20 baseline | 97.4/80.8/39.7/83.2 | 71.6 | 26.6/40.9/53.3/69.1 | 83.7 |
| exact-value rank seed1 | 96.8/82.9/40.2/83.7 | 66.2 | **29.7/46.8/58.7/72.3** | **80.9** |

**Mixed verdict; not promoted.** Tight-budget 2push improved at every tier and passed the primary bar, but hard-1push@5 regressed **5.4 points**, failing the pre-registered no-tier-regression guard. The finish-opener diagnostic also shows the likely tradeoff: AUC `0.9400→0.9339` and hit@1 `45.9→44.8`, while hit@5 and recall@20 were flat-to-up. Any follow-up must use paired fresh baseline+treatment seeds, not treatment-only repeats.

### 2026-07-23 update — paired 3-seed confirmation FLIPS the verdict to WIN

The single-seed "regression" was eval-sim noise. Paired 3 control (opener-only, `LOWER_RANK_LAMBDA=0`) vs 3 treatment (`opener 0.10 + lower-exact 0.05`) seeds, same code/data/recipe, mean [min,max]:

| metric | control | treatment | Δ |
|---|---|---|---:|
| 1p hard solve@5 | 70.9 [68.1,73.0] | 70.6 [68.6,73.5] | −0.3 |
| 1p all solve@1 | 84.2 | 83.8 | −0.5 |
| 2p all solve@5 | 41.6 [41.0,42.2] | 46.0 [44.5,47.8] | **+4.4** |
| 2p all solve@10 | 51.4 | 57.3 | **+5.9** |
| 2p all avg sims | 89.5 | 79.6 | **−9.9** |
| 2p hard solve@5 | 30.5 [29.1,31.8] | 37.0 [35.3,39.4] | **+6.5** |
| 2p hard solve@10 | 38.3 | 46.5 | **+8.2** |
| 2p hard avg sims | 151.0 | 132.4 | **−18.6** |

**WIN — recommend default.** 2push improves on every tier/budget with hard-2push seed ranges NON-overlapping; 1push flat within a ~5pt per-arm seed spread. The v1/v2/v3 single-seed hard-1push dips were noise. Loss `opener 0.10 + lower-exact 0.05` recommended as the default ranking-aux; Colossus flip staged for user (live-run timing).

## 2026-07-24 — Failure-discount best-first ADOPTED (card EXP-2026-07-24-failure-discount-search)

Per-board credibility `w` demoted by verified failed sims on that board (root frozen, floor ε, lazy stale-reinsert). Frozen ranker, search-only change. Testset pure2push 1018, budget 900, paired vs the reused static column.

| arm | solve@900 | sims-to-solve | @30 | hard solve / s2s |
|---|--:|--:|--:|---|
| static | 97.5 | 46.0 | 74.4 | 94.1 / 74.3 |
| fitted-g | 98.0 | 30.0 | 81.6 | 95.4 / 50.5 |
| **conf τ=0.15** | **98.1** | **27.8** | **82.6** | **95.7 / 47.0** |
| random static | 89.9 | 115.6 | 37.6 | 76.5 / 179.3 |

**ADOPT `--discount conf --tau 0.15`** — sims-to-solve −40% overall with solve@900 up on every tier. Fitted g remains the calibration reference. User hypothesis (fitted > γ) NOT confirmed at equal calibration — the measurement was the value, not the functional form. Cap-1 ablation confirms graded return is load-bearing for solve: bench, never bury.

## 2026-07-25 — Search depth vs label horizon: the ranker's value is horizon-local (card EXP-2026-07-25-search-depth-horizon)

{model, random} × hmax {2,3,4}, discount OFF, 180 paired episodes (60/tier), budget 900. Random = 3 seeds. hmax=2 controls reused (both reproduce their parent rows exactly). Model on ilab, random on Amarel; `eval_bestfirst.py` md5-matched across boxes, no C++ delta.

| hard tier | @5 | @30 | @100 | solve@900 |
|---|--:|--:|--:|--:|
| model h2 | 31.7 | 60.0 | 68.3 | **98.3** |
| model h3 | 40.0 | 63.3 | 76.7 | 85.0 |
| model h4 | 36.7 | 58.3 | 71.7 | 78.3 |
| random h2 | 4.4 | 16.7 | 37.8 | 73.9 ±6.8 |
| random h3 | 6.1 | 35.0 | 62.2 | **87.8** ±1.6 |
| random h4 | 10.0 | 48.9 | 67.8 | 85.6 ±2.1 |

**Depth helps RANDOM, not the model.** Random climbs monotonically (hard @30 16.7→35.0→48.9); the model does not (60.0→63.3→58.3). **At hmax≥3 the ranker falls below random on solve@900** (hard 85.0 vs 87.8; 78.3 vs 85.6); its @30 margin collapses 3.6×→1.8×→1.19×. Model still dominates tight budgets at every depth (hard @5 36.7 vs 10.0 at h4). All failures were budget exhaustion, ZERO queue exhaustion — a ranking failure, not a combinatorial wall.

**Cause (verified):** training nodes are `{root, depth2}` only — no supervision past one push. Plus 48.1% of supervised cells carry only a one-sided ceiling and **0.0% of exact cells are 0**, so there is no downward gradient anywhere on the ranked action space. Dead finish boards are only 46.1% full sweeps, so the 0.81 ceiling is "tried nearly all and failed", not proof.

## 2026-07-26 — Deploy sigmoid squashes the trained value scale; ordering unaffected (card EXP-2026-07-26-scorer-scale-and-combine-mode)

The scorer trains an unbounded HL-Gauss value in [0,1] (`train_q2_rankaux.py:158`, no sigmoid anywhere in the trainer), but `live_scorer.py:184` applies a sigmoid at inference, folding it into [0.5, 0.7311]. Measured over 361,755 ceiling-model candidates: sigmoid path [0.5025, 0.7291] (median 0.5380) vs `--raw` path [0.0098, 0.9899] (median 0.1525); hard model raw median 0.0496 over 412,720 candidates. The trained opener/setup gap (1.0 vs 0.9) is served as 0.731 vs 0.711 — a 0.10 gap crushed to 0.02.

Confirmed on the 2-push test set (1018 episodes, `--combine q --discount off`, hmax 2, budget 30) — raw and sigmoid are identical cell-for-cell, as monotonicity requires:

| model | tier | solve@30 | avg sims | median sims-to-solve |
|---|---|--:|--:|--:|
| ceiling | easy | 84.9 | 9.82 | 3 |
| ceiling | medium | 80.9 | 10.06 | 3 |
| ceiling | hard | 59.6 | 16.01 | 3 |
| hard model | easy | 81.1 | 11.63 | 4 |
| hard model | medium | 80.2 | 11.25 | 4 |
| hard model | hard | 60.1 | 15.89 | 4 |

**Every ordering-only result (top-1, hit@k, rank-of-first-good-push) is immune to this — a sigmoid is monotone, verified to max abs error 6.1e-08.** What IS affected is every deploy consumer of magnitude: the `blend` combine, the `conf` failure discount, and `free_strike_q`. Under the squashed scale, the adopted `(1-q)^0.15` discount ranges only 0.821-0.901 (within 8% of a flat 0.87 multiplier); under raw values it would range ~0.50-0.99. **Open question, not a conclusion:** the adopted `conf tau=0.15` win (sims-to-solve 46→27.8, card [EXP-2026-07-24](log/EXP-2026-07-24-failure-discount-search.md)) may be substantially a failure-counting effect rather than a confidence effect — the discriminating `--discount gamma --gamma 0.87` control has not been run. `--free-strike-q` defaults to 2.0 but sigmoid `q` never exceeds 0.7311, so that allowance never fires at default settings. Full derivation → [EXP-2026-07-26 card](log/EXP-2026-07-26-scorer-scale-and-combine-mode.md).

## 2026-07-26 — `--combine q` beats the default `--combine blend` on every tier (same card)

Same 1018-episode 2-push test set, hmax 2, budget 30, both models (ceiling, hard), both discount settings. `blend` (current default) = `0.5*q + 0.5*V` (V = board mean top-5 q); `q` = raw action score alone.

| arm | tier | solve% blend → q | avg sims blend → q |
| --- | --- | --- | --- |
| ceiling conf tau=0.15 | easy | 90.3 → 93.3 | 9.28 → 8.20 |
| ceiling conf tau=0.15 | medium | 83.9 → 87.5 | 10.12 → 9.19 |
| ceiling conf tau=0.15 | hard | 66.8 → 70.6 | 15.48 → 14.61 |
| ceiling off | easy | 80.3 → 84.9 | 11.10 → 9.82 |
| ceiling off | medium | 79.0 → 80.9 | 10.96 → 10.06 |
| ceiling off | hard | 55.8 → 59.6 | 16.80 → 16.01 |
| hard conf tau=0.15 | easy | 84.5 → 89.9 | 11.42 → 9.66 |
| hard conf tau=0.15 | medium | 79.0 → 84.1 | 11.49 → 10.58 |
| hard conf tau=0.15 | hard | 59.3 → 64.4 | 16.64 → 15.43 |
| hard off | easy | 73.5 → 81.1 | 13.40 → 11.63 |
| hard off | medium | 75.6 → 80.2 | 12.19 → 11.25 |
| hard off | hard | 55.8 → 60.1 | 17.07 → 15.89 |

**12 of 12 cells improve on both solve rate and sims** — both models, both discount settings, every tier. Single seed, one test set. **Hypothesis, not a conclusion:** `V` mixes two boards' score scales when comparing across boards, and dropping it removes that distortion — consistent with the sigmoid finding above (the scorer's scale is weakest exactly at cross-board magnitude comparison). Worth considering a deploy-default change from `blend` to `q`; **not adopted here** — user's call. Full table + design → [EXP-2026-07-26 card](log/EXP-2026-07-26-scorer-scale-and-combine-mode.md).

## 2026-07-29/30 — Post-pruning canonical search beats seeded random on every fixed tier

The setup-only ranker and a three-seed random ranker used identical Amarel search settings on both registered test sets: `hmax=2`, budget 900, `combine=q`, confidence discount τ=0.15, no-op dedupe on, and jam-depth pruning on. Random values are mean ± sample standard deviation. “Tight” is solve@1 for 1push and solve@2 for pure-2push; `s2s` is average simulator calls among solved episodes.

The 2push rows use the finalized 35-root exhaustive-GT fill and fixed setup-density tiers: hard <5% (142), medium 5–30% (488), easy ≥30% (385), and two explicitly unknown. Search-ineligible queue-exhausted episodes are excluded by the eval registry, leaving 1,322 1push and 1,017 2push episodes.

![Exact per-episode success versus simulator calls for the learned ranker and three-seed random baseline, split by fixed difficulty and horizon.](plots/postprune_hmax2_gt_tiers/success_vs_sims_both_horizons.png)

| horizon | tier | model tight | random tight | model @30 | random @30 | model @900 | random @900 | model s2s | random s2s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1push | easy | 97.7 | 60.1±2.3 | 100.0 | 100.0±0.0 | 100.0 | 100.0±0.0 | 1.0 | 1.8±0.1 |
| 1push | medium | 84.6 | 12.7±0.5 | 99.5 | 97.9±0.5 | 100.0 | 100.0±0.0 | 1.6 | 6.4±0.2 |
| 1push | hard | 39.7 | 3.3±1.2 | 96.6 | 80.1±2.1 | 100.0 | 100.0±0.0 | 4.6 | 23.0±1.0 |
| 1push | all | 84.6 | 36.2±1.5 | 99.3 | 96.2±0.3 | 100.0 | 100.0±0.0 | 1.8 | 6.6±0.3 |
| 2push | easy | 44.4 | 6.3±2.1 | 96.1 | 73.4±2.8 | 100.0 | 100.0±0.0 | 8.4 | 28.6±1.1 |
| 2push | medium | 33.0 | 0.9±0.6 | 84.6 | 31.6±2.4 | 99.8 | 98.8±0.4 | 24.0 | 99.7±2.2 |
| 2push | hard | 9.2 | 0.0±0.0 | 53.5 | 8.9±1.5 | 90.8 | 74.4±2.3 | 68.5 | 287.4±39.4 |
| 2push | unknown | 50.0 | 0.0±0.0 | 100.0 | 50.0±0.0 | 100.0 | 100.0±0.0 | 3.5 | 26.7±1.8 |
| 2push | all | 34.0 | 2.8±1.0 | 84.7 | 44.3±2.1 | 98.6 | 95.9±0.5 | 23.7 | 91.7±4.4 |

**WIN.** The learned ordering is dramatically better where simulator calls are scarce and remains better on every fixed difficulty tier. Exhaustive-GT hard 2push reaches 53.5% by 30 calls versus random's 8.9±1.5%, while solved episodes need 68.5 calls versus 287.4±39.4. Hard 1push reaches 83.3% versus 25.7±6.0% by five calls. Both methods eventually saturate on 1push, but the learned ranker reaches the verified opening far sooner. Full solve@{1,2,5,10,30,100,300,900} tables and original run audit → [experiment card](archive/EXP-2026-07-29-post-pruning-canonical-search.md).

The GT-hard tails were then extended from 900 calls to natural queue exhaustion under a 10,000-call cap, preserving each random episode's original seed index. Learned rises from 129/142 = 90.8% at 900 to 137/142 = 96.5%; random finishes at 135/142, 137/142, and 137/142 = 96.0±0.8%. None hit the cap, so neither ordering can reach 100% under unchanged search. Learned remains the speed winner: it reaches 50/75/90/95% success in 22/118/631/2,261 calls versus random-mean 358/916/2,446/6,997 calls, or 16.3×/7.8×/3.9×/3.1× fewer calls.

![Equal-budget exhaustive-GT hard-2push learned and three-seed random tails.](plots/postprune_hmax2_gt_tiers/success_vs_sims_2push_hard_tail.png)

Tail run, seed-stability audit, and queue-exhaustion results → [experiment card](log/EXP-2026-07-30-random-hard2push-search-tail.md).
