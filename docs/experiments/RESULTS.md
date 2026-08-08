---
status: hub
tags:
  - results
updated: 2026-08-08
---
# Results — DAgger curriculum training framework

The maintained approach. A clean seed, then rounds of **{ generate fresh scenes → screen with the current model → keep its mistakes → label → accumulate → retrain → eval }**, laddering **1-push → 2-push → …** (Ant-Man → Beast → …). Each stage's model also produces the *next* stage's dataset for free (screened-dead scenes become the multi-push bank).

**Setting:** CAR robot, testset `namo_testset_v1`, region-opening criterion. Every result split by **difficulty (easy/med/hard)**. Difficulty is defined *per horizon* (compare within a horizon, not across): **1-push** = `solve_rate` fixed cuts (hard < 0.05 / med 0.05–0.30 / easy ≥ 0.30); **2-push** = exhaustive-GT setup density with the same fixed percentage cuts, with the two unfinished GT roots reported as unknown. Full per-experiment detail lives in each card under `log/`.

**Archive:** the pre-curriculum line — horizon-Q / NoHz single-ranker, RL-only self-imitation, horizon-role probe, prior-work ledger — is preserved verbatim in [archive/RESULTS_pre_dagger_horizonq_2026-07.md](archive/RESULTS_pre_dagger_horizonq_2026-07.md).

---

## 2026-08-07/08 — Loss anatomy: ordering supervision dominates labels and architecture (cards [EXP-2026-08-02 DC](log/EXP-2026-08-02-bootstrap-value-loop.md), [EXP-2026-08-08 arjuna](log/EXP-2026-08-08-arjuna-hard-labels.md))

Eight arms × 3 seeds, one canonical protocol (1322 1push@hmax2 + 1012 2push, budget 900, `combine=q`, discount off, dedupe+jam on), zero unmatched episodes anywhere.

| arm | 2p h@2 | 2p h@5 | 2p h@30 | 2p h@900 | 1p h@1 | 1p h@5 |
|---|--:|--:|--:|--:|--:|--:|
| random (3 seeds) | 0.0 | 1.7 | 11.2 | 70.1 | 3.3 | 28.1 |
| θ₀ deployed | 9.5 | 22.6 | 50.4 | **92.0** | 39.7 | **82.4** |
| arm A (45k guessed cells) | 13.6 | 27.7 | 54.5 | 91.5 | 39.2 | 79.9 |
| arm A, ranking aux OFF | 3.7 | 10.7 | 31.1 | 82.9 | 15.4 | 57.3 |
| arm A, λ_lower 0.10 | 13.6 | 27.0 | 55.5 | 91.3 | 37.6 | 79.9 |
| Bfix (591k guessed cells) | 13.4 | 28.9 | 50.4 | 87.1 | **41.8** | **83.5** |
| Bfix, ranking aux OFF | 5.8 | 11.2 | 28.2 | 81.5 | 15.2 | 58.1 |
| **ANG** = arm A + guess-exclusion | **14.6** | **30.7** | 53.1 | **91.8** | 39.0 | 81.0 |
| BNG = Bfix + guess-exclusion | **14.6** | **32.1** | **55.7** | 88.6 | 38.4 | 81.4 |
| ARJ = arjuna-0, 141k exact **zeros** | 14.1 | 27.7 | 54.7 | 91.0 | **42.5** | 80.9 |
| AJ2 = arjuna-0 **v2**, 8.4M exact zeros (47.6%) | — | 26.8 | 53.3 | 90.0 | 38.1 | 78.8 |
| AJ2NR = v2, ranking aux OFF | — | 20.0 | 40.6 | 87.8 | 29.4 | 73.5 |

**Headline: the listwise ranking auxiliary carries ~half of deployed performance, and it had never been swept.** Removing it costs hard-2p@5 27.7→10.7 and 1p-hard@1 39.2→15.4, at both label doses, every band non-overlapping. For contrast the depth-token architecture moved ±3–4 pts (rejected) and a 12× label dose moved ±4–5. It is also the **only** consumer of the ceilings, which are 46–48% of supervised cells — regression can penalise a bound but never order it.

**Why regression can't carry it: 96–100% of all training targets sit ≥ 0.8, and zero of 143,705 exact facts is a 0.** The regression term fits a near-constant. Not a focal-loss problem — there are no negatives to down-weight.

**λ_lower saturates at 0.05.** 0→0.05 is +24 pts on hard-2p@2; 0.05→0.10 is nothing. Question closed.

**Guessed cells were being enforced as CERTAIN tiers.** Bootstrapped targets are continuous floats and the aux treats each distinct value as a known tier: θ₀ 2 tiers per batch, arm A 27, Bfix **593** — asserting 4th-decimal model noise as ground truth, at full weight (the half-weight only reaches the regression), and costing ~300× the loop iterations. Barring them (`NAMO_RANK_EXCLUDE_GUESS=1`, still competitors) gives **+3.0 hard-2p@5 on arm A, +3.2 on Bfix**, both non-overlapping, plus **4× faster training** (20 → 5 min/epoch).

**Arm B rejected, with the sharper reading.** Its 546,035 extra cells are *exactly* the children whose sweeps were censored — the one population whose answer is unrecoverable from this data. Guessing there cost 4–5 pts of hard-2p reach. Not "more labels hurt" but **"labels on the unknowable hurt."** Arm A's population is fully resolved; it succeeded partly by accident of which cells the H5 recorded.

**Floor hypothesis FALSIFIED on its own pre-registered test.** arjuna-0 wrote the project's first exact zeros (141,581 proven-dead cells, zero new sims). Pre-registered: *"if V5 doesn't move, the theory is wrong regardless of solve rates."* **V5 = 0.533 vs arm A's 0.543 — unmoved.** The floor bought precision, not comparability: best 1push-hard@1 in the line (**42.5** vs θ₀ 39.7), best setup hit@1 (24.6), best V6 (0.790), but 2-push flat.

**Floor hypothesis re-tested at 72× the dose — still falsified, and the null is now solid.** v1's zeros reached only 1.4% of the 8.29M bounded cells (a join limit: 94% sit on d20-base rows with no child stored), so its null was dismissible. v2 needs no linkage — every bounded cell becomes an exact 0, taking zeros from 0.66% to **47.6%** of supervised cells. `AJ2`'s hard-2p@5 band **[24.8, 28.5]** contains arm A (27.7), ARJ (27.7) and Bfix (28.9); BNG (32.1) sits above the whole band. **Giving the regression a real zero to predict is not what the ranker was missing.**

**But the aux and the labels turn out to be SUBSTITUTES — the session's most useful number.** Re-run the ablation under honest labels and most of the aux's apparent dominance evaporates:

| labels | aux off | aux on | aux's marginal value |
|---|--:|--:|--:|
| bootstrap guesses (`BfixNR`→`Bfix`) | 11.2 | 28.9 | **+17.7** |
| hard floor (`AJ2NR`→`AJ2`) | 19.9 | 26.8 | **+6.9** |

Hard labels nearly **double** the no-ranking model (11.2 → 19.9 hard-2p@5; 15.2 → 29.4 1p-hard@1) and cut the aux's contribution by ~60%. So "ordering supervision carries half of deployed performance" is true *only while the labels are degenerate* — with 96–100% of targets ≥0.8 the regression has nothing to learn and the aux carries the model alone. The aux still wins by ~7 pts with non-overlapping bands, so it is not merely a workaround; and its contribution is **uniform across difficulty** (+7.1/+7.0/+6.8 at 2push easy/medium/hard@5), not concentrated in the hard tail.

**One caveat, pre-registered before the run:** zeroing *child* bounded cells is provably correct under hmax=2, but zeroing *root* ones asserts "not a setup" for ~5.5M cells whose children were never resolved. Flat-to-slightly-down is exactly what "real information on one half, false labels on the other" predicts. The split arm (child→0, root→masked) is the outstanding test.

**AUC: the label route to cross-board comparability is now CLOSED, and the F2 collapse turns out to be a label artefact.** Seven rank-on arms across three label regimes — bootstrap guesses, a 1.4% floor, a 47.6% floor — all sit at **V5 = 0.527–0.543**; `AJ2`'s 0.543 [0.529, 0.562] overlaps BNG and ARJ. A 72× dose does not move it. Meanwhile `AJ2NR` posts **the highest V5 ever measured, 0.642**, so the aux *actively suppresses* cross-board comparability by ~0.10 — as its `log_softmax(dim=1)` shift-invariance predicts. Two further reads: removing the aux used to destroy finish separation (F2 0.902 → 0.735) but under hard labels **it does not** (0.882 → 0.877, overlapping), so the aux was compensating for labels in which finish and setup were nearly the same number — that is the mechanism behind the substitution result. And `AJ2` posts the best **V4 = 0.900** while V5 stays flat: it beats the *typical* dead cell better than any model we have, but not each dead board's *maximum*, which with ~75 cells per board is an extreme order statistic. **Cross-board weakness is exactly that gap.** ⚠ Honest gap: under hard labels the AUC panel does *not* explain the +6.9 deploy gain — V1/V2/F2/setup@1 are flat or worse with the aux, and only **V6 board triage** (0.760 vs 0.734) is a non-overlapping gain. Hypothesis, not conclusion.

**Cross-board comparability is a LOSS-STRUCTURE problem, measured not inferred.** The aux is `log_softmax(dim=1)` over one board — shift-invariant per board, so nothing keeps boards on a common scale. Measured: within-board spread 0.516 (off) → 0.666 (on), spread across board maxima 0.227 → 0.204, and **dead-board maxima inflated 0.625 → 0.720**. V5 is exactly "setup cell vs dead board max". No label change touches this; it needs a signal spanning boards.

**Deploy candidate: `ANG`** — hard-2p@5 **30.7 vs θ₀ 22.6 (+8.1)** with reach intact (91.8 vs 92.0). Not yet promoted; pending user call.

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

The setup-only Colossus ranker (`d20_plus_setup_only_splitloss`, epoch 11, SHA256 `3a43f5ea5fe5e553abbb1bb099f657699dda82cc2b08e079bd6a54677fc2c2b6`) and a three-seed random ranker used identical Amarel search settings on both registered test sets: `hmax=2`, budget 900, `combine=q`, confidence discount τ=0.15, no-op dedupe on, and jam-depth pruning on. Random values are mean ± sample standard deviation. “Tight” is solve@1 for 1push and solve@2 for pure-2push; `s2s` is average simulator calls among solved episodes.

The 2push rows use the finalized 35-root exhaustive-GT fill and fixed setup-density tiers: hard <5% (137), medium 5–30% (488), easy ≥30% (385), and two explicitly unknown. The registry excludes two shared search-queue failures and four sampled-manifest episodes whose exhaustive-GT roots contain zero genuine setups, leaving 1,322 1push and 1,012 2push episodes.

![Exact per-episode success versus simulator calls for the learned ranker and three-seed random baseline, split by fixed difficulty and horizon.](plots/postprune_hmax2_gt_tiers/success_vs_sims_both_horizons.png)

| horizon | tier | model tight | random tight | model @30 | random @30 | model @900 | random @900 | model s2s | random s2s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1push | easy | 97.7 | 60.1±2.3 | 100.0 | 100.0±0.0 | 100.0 | 100.0±0.0 | 1.0 | 1.8±0.1 |
| 1push | medium | 84.6 | 12.7±0.5 | 99.5 | 97.9±0.5 | 100.0 | 100.0±0.0 | 1.6 | 6.4±0.2 |
| 1push | hard | 39.7 | 3.3±1.2 | 96.6 | 80.1±2.1 | 100.0 | 100.0±0.0 | 4.6 | 23.0±1.0 |
| 1push | all | 84.6 | 36.2±1.5 | 99.3 | 96.2±0.3 | 100.0 | 100.0±0.0 | 1.8 | 6.6±0.3 |
| 2push | easy | 44.4 | 6.3±2.1 | 96.1 | 73.4±2.8 | 100.0 | 100.0±0.0 | 8.4 | 28.6±1.1 |
| 2push | medium | 33.0 | 0.9±0.6 | 84.6 | 31.6±2.4 | 99.8 | 98.8±0.4 | 24.0 | 99.7±2.2 |
| 2push | hard | 9.5 | 0.0±0.0 | 55.5 | 9.2±1.5 | 94.2 | 76.9±2.8 | 68.5 | 285.6±36.4 |
| 2push | unknown | 50.0 | 0.0±0.0 | 100.0 | 50.0±0.0 | 100.0 | 100.0±0.0 | 3.5 | 26.7±1.8 |
| 2push | all | 34.2 | 2.9±1.0 | 85.1 | 44.5±2.1 | 99.1 | 96.3±0.5 | 23.7 | 91.5±3.9 |

**WIN.** The learned ordering is dramatically better where simulator calls are scarce and remains better on every fixed difficulty tier. Exhaustive-GT hard 2push reaches 55.5% by 30 calls versus random's 9.2±1.5%, while solved episodes need 68.5 calls versus 285.6±36.4. Hard 1push reaches 83.3% versus 25.7±6.0% by five calls. Both methods eventually saturate on 1push, but the learned ranker reaches the verified opening far sooner. Full solve@{1,2,5,10,30,100,300,900} tables and original run audit → [experiment card](archive/EXP-2026-07-29-post-pruning-canonical-search.md).

The GT-hard tails were then extended from 900 calls to natural queue exhaustion under a 10,000-call cap, preserving each random episode's original seed index. Learned rises from 129/137 = 94.2% at 900 to 137/137 = **100%** at 3,831 calls; random finishes at 135/137, 136/137, and 137/137 = 99.3±0.7%. Learned remains the speed winner: it reaches 50/75/90/95/99% success in 20/100/336/1,071/2,507 calls versus random-mean 330/871/1,728/3,456/8,286 calls, or 16.5×/8.7×/5.1×/3.2×/3.3× fewer calls. The random mean never reaches 100% under the unchanged search.

![Equal-budget exhaustive-GT hard-2push learned and three-seed random tails.](plots/postprune_hmax2_gt_tiers/success_vs_sims_2push_hard_tail.png)

Tail run, seed-stability audit, and queue-exhaustion results → [experiment card](log/EXP-2026-07-30-random-hard2push-search-tail.md).

**Wall-clock status: MEASURED 2026-08-06 — see the wall-clock section below.** The forecast that stood here (2.6–2.8× wall from the 3.2× call advantage) was not confirmed as stated: measured wall speedup is **1.77× on 1push and 1.93× on 2push** by mean time-to-solve, and the forecast's anchor statistic (95% success on hard 2push) is unreachable under this budget-900 protocol by either arm, so it could not be checked at all. The forecast's *direction* was right — wall gain is smaller than call gain — but the conversion is worse than assumed. Full checkpoint/config/artifact record → [Colossus card](log/EXP-2026-07-21-colossus-data-scaleup.md).

## 2026-08-02 — Failure audit: hard setups are sparse, then the flat queue under-probes the correct board

This is a read-only diagnosis of the existing 2-push run, so no model/search policy changed and no 1-push evaluation was rerun.

The full registered 2-push population contains 488 medium and 137 hard episodes. Medium solves 487/488 by 900 calls with 24 episodes costing over 100; hard solves 129/137 with 34 over 100, but all 137 solve by 3,831 calls. On current usable exhaustive GT, medium setup hit@1/@5 is 55.1/80.7% and finisher hit@1/@5 is 65.6/87.7%; hard setup is only 21.2/44.9% while finisher is 53.4/85.6%. Hard setup rank correlates with search cost and genuine hard setups are sparse—median one per episode versus eight on medium.

Full clean no-discount traces cover all 24 expensive-medium episodes and all eight hard budget-900 failures. The hard traces do not usually miss the setup: all seven usable-GT cases expose and pop a correct setup board, but their median finisher rank is 33, the correct board receives only 19 probes across 13 visits, and only 2/7 correct finishers are popped. Across traces, 88.6% of medium and 94.4% of hard calls go to child boards; each wrong root can inject roughly 70–80 new actions into the same flat heap, while training's listwise loss only compares actions inside one board.

**Diagnosis:** sparse setup ordering is the population weakness; multiplicative branch flooding, missing cross-board liveness supervision, and rare finisher ranks 14–57 create the catastrophic tail. Fixed 2/3/5-strike γ patience is rejected on the hard panel because it benches correct and dead boards alike. Next tests are a two-level scheduler with increasing probe tranches, the narrow one-step-live board head, and targeted exploratory DAgger; use all 58 episodes over 100 calls plus matched cheap controls before full canonical evaluation. Nineteen hard episodes currently have no genuine setup in the canonical H5 despite live success, so exact hard rank metrics use 118/137 and the disagreement remains a separate integrity audit. Full evidence, tables, artifacts, and caveats → [failure-audit card](log/EXP-2026-08-02-2push-failure-audit.md).

## 2026-08-02 — Five-depth local attention improves hard 2push shortlist recall but regresses the single ranker

The seed-1 depth-token treatment expands each contact into five motion-grounded depth tokens and applies one local self-attention head across those five depths. It adds 157,044 parameters (+3.6%) to the deployed 4.40M model and trains on the identical 257,409-row `d20+setup-only` H5. Canonical evaluation is a paired clean architecture read: all 1,322 1push and 1,012 genuine-2push episodes, `hmax=2`, budget 900, `combine=q`, discount off, no-op dedupe and jam-depth pruning.

| horizon | tier | tight control → depth-token | solve@5 control → depth-token | solve@30 control → depth-token | solve@900 control → depth-token | avg calls control → depth-token |
|---|---|---:|---:|---:|---:|---:|
| 1push | easy | 97.7 → 98.3 | 99.9 → 99.9 | 100.0 → 100.0 | 100.0 → 100.0 | 1.1 → 1.0 |
| 1push | medium | 84.6 → 84.3 | 97.9 → 97.6 | 99.8 → 100.0 | 100.0 → 100.0 | 1.8 → 1.4 |
| 1push | hard | **39.7 → 33.8** | **82.4 → 77.5** | 96.6 → 95.1 | 100.0 → 100.0 | **7.6 → 10.1** |
| 2push | easy | 44.4 → 44.7 | 67.3 → 69.9 | 94.3 → 93.0 | 100.0 → 100.0 | 9.3 → 10.9 |
| 2push | medium | **32.8 → 30.3** | **57.4 → 51.0** | 80.1 → 81.8 | 99.6 → 99.2 | **38.1 → 46.6** |
| 2push | hard | **9.5 → 13.9** | **22.6 → 29.2** | **50.4 → 54.0** | 92.0 → 92.7 | 149.5 → 150.7 |

**REJECT as the next deployed ranker; do not run seeds 2–3.** The intended mechanism exists—hard exhaustive-GT setup hit@5 rises 44.9→55.1 and hard 2push solve@5 rises 22.6→29.2—but it is not a global win. Hard 1push solve@1 falls 5.9 points, medium 2push solve@5 falls 6.4, and weighted medium+hard 2push calls rise 62.52→69.42 (+11.0%) instead of falling ≥10%. The useful takeaway is narrow: local depth attention can promote sparse hard setups, but the loss/data composition must preserve contact ordering and medium finish ranking before this mechanism is deployable. Full tables, offline diagnosis, checkpoint, and exact aggregate/raw artifact paths → [depth-token card](log/EXP-2026-08-02-depth-token-push-motion.md) and the [model/evaluation artifact registry](horizon_q_model_registry.md).

## 2026-08-06 — Wall-clock: the learned ordering wins in seconds too, on every tier but easy-1push (card EXP-2026-07-21-colossus)

The first measured success-vs-time result. Same deployed setup-only ranker (`d20_plus_setup_only_splitloss` epoch011) and three-seed random baseline, full registered populations (1,322 1push + 1,012 2push), `hmax=2`, budget 900, `combine=q`, **discount off**, no-op dedupe and jam-depth pruning on. Every shard ran `--exclusive --constraint=icelake` on Amarel `main-redhat` — one task per whole 64-core node, single-threaded, model scoring on **CPU** (no GPU at deploy). Times are seconds per episode; comparable only within this campaign.

| horizon | tier | n | model @1s | random @1s | model @5s | random @5s | model @30s | random @30s | model mean s | random mean s | wall speedup |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1push | easy | 697 | 99.1 | 96.7±0.6 | 100.0 | 100.0±0.1 | 100.0 | 100.0±0.0 | 0.4 | 0.3±0.0 | **0.85×** |
| 1push | medium | 421 | 92.2 | 70.9±0.8 | 99.3 | 95.4±0.5 | 99.8 | 99.4±0.4 | 0.8 | 1.6±0.2 | 2.06× |
| 1push | hard | 204 | 62.3 | 35.9±2.7 | 91.7 | 78.1±2.8 | 99.0 | 97.0±1.3 | 2.6 | 5.4±1.0 | 2.09× |
| 1push | all | 1322 | 91.2 | 79.1±0.4 | 98.5 | 95.2±0.4 | 99.8 | 99.3±0.3 | 0.9 | 1.5±0.2 | **1.77×** |
| 2push | easy | 385 | 37.9 | 24.8±0.7 | 86.2 | 75.0±1.0 | 99.7 | 98.4±0.9 | 3.0 | 4.7±0.4 | 1.53× |
| 2push | medium | 488 | 34.6 | 10.6±0.7 | 72.1 | 40.3±0.8 | 92.6 | 79.2±2.3 | 9.3 | 19.4±0.8 | 2.08× |
| 2push | hard | 137 | 16.1 | 3.6±0.8 | 42.3 | 13.8±1.3 | 75.9 | 35.0±0.8 | 19.6 | 42.9±2.4 | 2.19× |
| 2push | all | 1012 | 33.4 | 15.0±0.3 | 73.5 | 49.9±0.6 | 93.1 | 80.6±1.2 | 8.2 | 15.8±0.1 | **1.93×** |

![Verified success versus wall-clock seconds for the learned ranker and three-seed random baseline, split by fixed difficulty and horizon.](plots/walltime_hmax2/success_vs_time_both_horizons.png)

**WIN, with one honest exception.** The advantage survives the conversion to seconds on seven of eight rows: hard 2push reaches 42.3% within five seconds against random's 13.8±1.3%, and 75.9% vs 35.0±0.8% within thirty. **The exception is easy 1push, where the model is 15% SLOWER** (0.40 s vs 0.34 s): both arms are at 100% by five seconds, so there is no ordering left to win, and the model simply pays its ranking overhead for nothing. Informed ordering does not pay when the problem is already trivial — report this rather than hide it behind the aggregate.

**The wall gain is consistently smaller than the call gain** (1push 2.96× calls → 1.77× wall; 2push 3.03× → 1.93×), for two measured reasons. (1) Ranking overhead: render + NN forward costs the model **17% of wall on 1push but only 6% on 2push** — inference matters least exactly where search is most expensive. (2) **Simulator calls are not a uniform-cost unit**: the model's calls cost more than random's (2push 0.234 vs 0.165 s/sim). Cause verified by direct measurement — sim time scales near-linearly with push depth (0.087 / 0.170 / 0.252 / 0.331 / 0.397 s for depths 0–4, 4.6× end to end), while failure is *not* the driver (failed 0.228 s vs clean 0.241 s, i.e. jams do not terminate early). Consequence for the paper: **a sim-count axis mildly flatters whichever method pushes less**, so both axes belong in the results.

**The speedup depends strongly on which statistic you quote.** On hard 2push, mean time-to-solve gives 2.19×, but time to resolve the first half of the population gives **7.89× (7.2 s vs 57.1±—s)** — averages are dominated by the expensive tail where both arms are slow. Report the curve, not one ratio.

**⚠ What the 1push rows actually measure [USER caught this].** The canonical protocol runs `hmax=2` on BOTH horizons, so a "1push" episode may be closed by a setup+finish chain — the rows above are **time to open the region by any route within two pushes**, NOT time to find the single opening push. The two arms take very different routes, so this is not a cosmetic distinction: the model closes 1push episodes in ONE push **93.1%** of the time (easy 98.9 / medium 93.8 / hard 72.1), while random manages only **56.9%** (easy 79.3 / medium 38.5 / **hard 18.1**) — i.e. 82% of random's hard-1push solves are 2-push chains it stumbled into, not opening pushes it ranked. This re-derives the beast-round-1 finding that the hard 1-push tier is a depth-1 artifact, and it is why random stays competitive on easy 1push: depth is doing its work for it. An isolated depth-1 ranking comparison would need an `hmax=1` arm; not run (declined 2026-08-06 [USER]). Same figure confirms population integrity in the other direction: **0% of pure-2push episodes were closed in one push** by either arm, on every tier.

![Share of episodes closed in one push versus two, by arm, tier, and horizon.](plots/walltime_hmax2/plan_depth_share.png)

**Measurement integrity.** The timed search IS the canonical search: `solve_scene` was instrumented in place (`t_wall`/`t_sim`/`t_score`/`n_score`) and the pre-existing `scripts/sandbox/time_bestfirst.py` fork was retired — it kept a private copy of the loop that predated `dedupe_noop` and `prune_jam_depth`, so it had been timing a different, slower search. Determinism cross-check against the registered `deploy-nodiscount-hmax2-v1` control: **1,321/1,322 1push episodes (99.92%) and 999/1,012 2push episodes (98.72%)** reproduce sims and solved exactly; the differences are the documented ~0.3 mm warmstart jitter, amplified at depth 2 where one flip changes the branch. Node-pooling validity: arms ran on separate (pinned, exclusive) nodes rather than per-episode interleaved, so seconds/sim was checked across the three random seeds at matched shard index — arm-level standard error ≈ ±3%, far below the effects above, and node assignment overlaps between arms. Per-episode interleaving remains the strictly cleaner design and is the standing limitation of this campaign. Raw rows `$NAMO_SCRATCH/eval/walltime_hmax2/v1/{model,random_s{7000,8000,9000}}_{1push,2push}/`.
