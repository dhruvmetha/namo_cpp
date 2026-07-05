---
status: ref
tags: [experiment]
updated: 2026-06-25
---

# Horizon-Q — SEARCH-FIRST REDESIGN Journal

> **Started 2026-06-23.** The PIVOT journal. The build journal
> ([horizon_q_build_journal.md](horizon_q_build_journal.md)) records the value-classification line (v2/v3/v4, the 2×2
> matrix, ExIt). THIS journal is the reframe that came out of the 2026-06-22/23 design session: **the model is a
> SEARCH HEURISTIC whose job is to MINIMIZE SIMS, not a value-classifier.** Every design decision below runs as
> **Hypothesis → Evidence (from experiments) → Verdict (accept/reject ON NUMBERS ONLY)** [[feedback_journal_attribute_decisions]].
> Model paths: [horizon_q_model_registry.md](horizon_q_model_registry.md) (read, never glob). Empirical state
> (v2/v3 numbers): build journal §9.

---

## 0. WHY THE PIVOT (read this first)

We kept asking "is the value calibrated / does the horizon help." Wrong frame. Re-grounded in the code this session:
- **The model is a RANKER, not a simulator.** Code-verified (`eval_m3.py:53` `rank_first_pushes_h2` = ZERO sims; the SIM executes the push (`env.step`) and the SIM checks connectivity (`goal_open_pts`)). Reactive@2 = **2 real sims**. The model only ORDERS the ~50 reachable pushes so the sim executes the fewest. It never predicts effects or checks connectivity. (Earlier "asking the net to BE the simulator" framing was WRONG — [USER] caught it.)
- **The perfect deterministic sim is a FREE EXACT VERIFIER.** So the model never needs calibrated P(open) or to predict outcomes — only correct ORDER. The sim is the calibrator. This frees all capacity for ranking the needle.
- **The objective is E[sims-to-solve], and it decomposes:**
  > **sims-to-solve ≈ rank(true setup) + rank(true finish | that setup)**
Reactive@2 = the **rank-1 corner** (setup rank 1 AND finish rank 1) = "how good are the predictions if we fully trust the model." Search = pay sims to allow rank > 1. So reactive isn't the product — it's the rank-1 special case; the product is **model + search, minimize sims**.

**Consequence:** train the model AS a sims-minimizing search heuristic — a RANKER, on the search's own on-policy data, with the setup value tied to the finish via a bootstrap. This is AlphaZero's *net-improvement* half (distill search into the net) while DELETING its *search-at-deploy* half (we want ~0 deploy sims). That inversion is why our problem is HARDER than AlphaZero, not easier, despite the trivial H=2 depth: AlphaZero always keeps the sim; we're trying to be the net alone on the sim's single hardest, sparsest step (the finish needle).

---

## 1. THE SIMS DECOMPOSITION — the central evidence [2026-06-23, exhaustive pairmap, canonical 1018]
`scripts/sandbox/` ad-hoc on `/scratch/dm1487/eval/exhaustive_pairmap_pure2.pkl` (`pairmap[(xml,obj)][a1][a2]=opens?`). Expected sims, sampling w/o replacement, `E[trials-to-first-success] = (N+1)/(m+1)`. Per-tier via `pure2push_divisions.json`.

**Where the 31.5 naive sims go, removed one chunk at a time (oracle ladder):**
| regime | sims | chunk removed | saves |
|---|---|---|---|
| L0 naive (random setup-search + random finish) | 31.5 | — | — |
| L1 + perfect SETUP ranker (find a solver in 1) | 18.7 | setup-search | 12.8 |
| L2 + D1 findability (best-finish setup) | 12.5 | setup-choice | 6.2 |
| L3 + D2 finish ranker (opener at rank 1) | **2.0** | finish-search | 10.5 |

**Standalone headroom per decision, per tier (vs naive):**
| decision | removes | overall | easy | med | **hard** |
|---|---|---|---|---|---|
| **D2 finish ranker** | find the opener | **~16** | 15.4 | 15.8 | **18.6** |
| setup ranker (existence) | find a solver | 12.8 | 3.4 | 9.0 | **23.0** |
| **D1 findability** | which solver | 6.2 | 10.4 | 7.7 | **1.9** |

**Reads (all on NUMBERS):**
1. **D2 dominates and SUBSUMES D1** — a perfect finish makes finish-sims=1 from ANY solving setup, so setup-CHOICE (D1) becomes moot. D1 is 68/48/**10**% of D2 across easy/med/hard.
2. **Hard is a DOUBLE needle, cracked only by RANKING.** On hard, setup-search (24) + finish-search (20) is the whole cost; findability (D1) saves ~2. D2 saves **18.6** on hard — the ONLY finish lever that touches the frontier.
3. **Setup ranking is a big SEPARATE chunk (12.8, hard 23)** — but the model is ALREADY good at setup EXISTENCE (H2 setup AUC 0.93), so most is realized; the residual is setup **top-1** (37% real), which matters for REACTIVE (the weaker multiplicative link), not search-sims.
4. **Caveat (honest):** these are ORACLE ceilings (perfect rankers). Realized prize = current-model → oracle, smaller, needs a scoring pass. For D2 the realizable gap is bounded by `top1-finish-opens` (0.55 v2 → 0.60 v3 → ~1.0 oracle).

---

## 2. DESIGN-DECISION LEDGER (Hypothesis → Evidence → Verdict; accept/reject on numbers only)

### D1 — Findability setup target (setup value = finish DENSITY at s1, not flat 0.9)
- **Hypothesis:** ranking setups by how findable their finish is (density / Monte-Carlo over sampled a2) cuts sims — the search dives into the cheap-to-finish setup instead of a needle-finish one.
- **Evidence:** pairmap headroom saves **10.4/7.7/1.9** (easy/med/hard); 87% of episodes have setup-choice (median 6 solvers). BUT collapses on hard (best-finish setup STILL 17.7 sims — double needle), and a perfect finish (D2) makes it MOOT (D1/D2 = 68/48/10%). MC note: the existing 2-push trees already hold ~30 sampled a2/setup ⇒ density is a free 30-sample MC estimate, but high-variance on needles (30 uniform misses a 1/45 opener ~half the time) → importance- sample by the finish head (= the recurrence limit) to reduce variance.
- **Verdict: ⚖ PARTIAL / DEPRIORITIZED.** Real easy/med sim-saver, free from existing data — but SUBSUMED by D2 and dead on the hard frontier. Do NOT build as a standalone relabel; its value is absorbed by D2 + D3.

### D2 — Finish ranker (InfoNCE over reachable a2, hard negatives, on ExIt s1) — **THE PRIORITY**
- **Hypothesis:** training the finish head to RANK the opener above the plausible impostors (masked multi-positive InfoNCE over the reachable set, self-mining hard negatives) lowers rank-of-first-opener ⇒ fewer finish-sims (every tier incl hard) + higher finish-top1 (reactive).
- **Evidence (ceiling, pre-build):** pairmap — D2 saves **15–19 sims on EVERY tier incl hard (18.6)**; dominant standalone (~16); the only lever that cracks hard. Learnability LEANS plausible (finish train-sep 0.75 sharp on SEEN states ⇒ not aliasing/representation-wall; depth + reachability already solved; residual = contact-EDGE selection) but UNPROVEN on novel s1.
- **Spec:** `L_finish = L_class + λ·L_rank`, `L_rank = -log(Σ_{openers} e^{z/τ} / Σ_{reachable} e^{z/τ})`; dead s1 keep BCE-to-0; `sample_k=0` on finish rows; KEEP L_class (Decision 3 needs calibrated cross-state V_finish). Code: `classifier_module.py` loss + `scorer_data.py` masks; NO arch change. Data = `v4_hq_exit_finish_v4` (existing). Train `v4r` = v4 mix + ranking loss, 3 seeds, Hz+NoHz, clean A/B vs v4. Full spec in this session's transcript / build below.
- **Verdict: ⏳ PENDING (spec'd, highest-priority build).** Pre-registered ACCEPT iff: finish rank-of-opener ↓ AND top1-finish-opens > v4 (3 seeds, fixed-s1) AND hard-tier finish-sims ↓ AND `V_finish=mean_top_k Q` solvable-vs-dead AUC ≥ v4 (calibration guard for D3). REJECT iff no rank improvement (⇒ needle is aliased, not mis-ranked → stop).

### D3 — Recurrence (setup target = γ·mean_top_k Q_finish(s1), bootstrap off the frozen finish)
- **Hypothesis:** bootstrapping the setup value off the finish head (a) makes the setup finishability-aware = D1's gift but LEARNABLE, and (b) ties H1/H2 to one scale so the search DIVES (the won't-dive fix).
- **Evidence:** dive-fix already MEASURED = **+11.2pp** (Hz forced-dive reactive 38.5 − best-first@2 27.3, combine=q, 3-seed v2; NoHz only +3.3 — it already dives). Setup-findability part = D1's 6.2 (easy/med). REQUIRES a calibrated finish first (bootstrap off mush = garbage) ⇒ gated on D2.
- **Verdict: ⏳ PENDING (after D2).** Dive-fix COMPONENT ✅ ACCEPTED on numbers (+11pp). Full recurrence build pending; ACCEPT iff setup top-1 ↑ AND reactive@2 ↑ AND best-first dives without the forced-dive hack.

### D4 — Search-in-the-loop / ExIt data (train on the on-policy s1 the model actually visits)
- **Hypothesis:** the finish gap is a DEPLOY DISTRIBUTION SHIFT (trained on collection-setup s1, queried on model-setup s1); training on the s1 the model visits fixes it — static data can't.
- **Evidence:** gen gap MEASURED (finish train-sep 0.75 → test 0.30; disentangled ⅔ deploy-shift / ⅓ scene-gen). ExIt v3 lifted finish-sep 0.31→0.385, top1-finish-opens 0.55→0.60 (+0.05 ~3σ REAL but MODEST; the 0.6 gate NOT cleared). v4 (scale ExIt 24k→47k + dead-s1 coverage) training now; error bars overnight (eval-chains 57031227–236).
- **Verdict: ✅ ACCEPTED-MODEST + ⏳ ENABLER.** Real but small lever alone (data autopsy: finish was 3.6% of the mix, too easy). It's the DATA half of D2 — the ranking loss (D2) on the on-policy ExIt distribution (D4) is the real test.

### D5 — Search diversity / soft exploration term (PUCT-style or ε / ensemble)
- **Hypothesis:** a soft exploration term recovers needles that greedy best-first buries within budget.
- **Evidence:** 92% of failures recoverable by RANDOM ordering (the deterministic greedy steers AWAY from findable needles on its blind spots). BUT this is a SEARCH-regime (high-budget) lever — reactive can't explore (no sims), and search already ~95% (barely above random@900 91%). Note: with a perfect-info sim we DON'T need PUCT's visit-count re-sampling (that solves noisy-leaf-value, which we don't have); only the exploration/diversity term is relevant.
- **Verdict: ⏸ DEFERRED.** Real but wrong regime — optimizes search (not the reactive/sims prize) and is partly dissolved by fixing the finish (D2) upstream.

---

## 3. BUILD ORDER (what the decomposition implies)
1. **D2 (finish ranker) + D4 (ExIt data, already running)** — the dominant, hard-cracking pair. D2 = the loss, D4 = the on-policy data. THE priority.
2. **D3 (recurrence)** — after D2 (bootstrap needs a calibrated finish); buys setup-findability learnably + reactive parity (the measured +11pp dive-fix, without the deploy hack).
3. **SKIP D1** as a standalone relabel (subsumed by D2). **DEFER D5** (search regime).
- **Reactive note:** reactive@2 = P(setup top-1 real) × P(finish top-1 opens), MULTIPLICATIVE (oracle headroom 0.37×0.42). D2 lifts the finish factor; the setup factor (37%, the WEAKER link) needs setup-side sharpening too ("sharpen BOTH top-1s"). D3 makes the setup finishability-aware; a setup ranking loss (the setup analog of D2) is the open lever for the setup top-1.

## 4. OPEN MEASUREMENTS (zero/low compute, do before/with the build)
- **D2 realized prize (mirror measurement):** score the CURRENT finish head's rank-of-first-opener on HARD s1 (given GT setup) vs perfect(1)/random — converts the 18.6 oracle ceiling into the realized current→perfect gap. ~30 min CPU.
- **Setup top-1 realized gap:** same idea on the setup side (current setup head's rank of first solving setup vs oracle).

## 5. EMPIRICAL STATE (current, from the build journal — DO NOT duplicate, pointer only)
Reactive@2 vs best-first@2 (combine=q, region, n=1018): **Hz-v2 38.5±2.1 / 27.3±2.2 (dive tax +11.2)**, NoHz-v2 38.2±3.0 / 34.9±2.6 (+3.3), Hz-v3 45.6/36.1 (1s), NoHz-v3 40.7/38.0 (1s), random ~4–5. Hz≈NoHz reactive at v2 (TIED; single-seed "Horizon wins" was noise). v3/v4 seed error bars training overnight. Full table + correction: build journal §9 "REACTIVE@2 FORCED-DIVE MULTI-SEED" + registry "Horizon-v3/NoHorizon-v3".

## 6. HOW TO RESUME
Read §0 (the reframe) → §1 (the sims decomposition = the evidence) → §2 (decision verdicts) → §3 (build order). The priority build is **D2** (finish ranker spec in §2/D2). Gate everything on **avg-sims-to-solve** (not just solve-rate), 3 seeds, Hz+NoHz. Pre-register predictions; fill verdicts on numbers only.
