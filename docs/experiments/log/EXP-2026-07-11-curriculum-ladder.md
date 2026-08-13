---
status: superseded
thread: rl_loop
robot: car
updated: 2026-08-12
---

# EXP-2026-07-11 — Curriculum ladder: exhaustive 1-push → self-collected clean 2-push

One model, built up like a curriculum. Learn a rock-solid **1-push** pusher on **complete** data; use it to collect **clean 2-push** data (try every first move, let the model finish); keep feeding **harder** scenes so it's forced to get good at the hard ones. No ranking objective — the **data** carries the ranking.

## Why this experiment (the diagnosis that motivated it)

Q2 (the γ^k value ranker, [EXP-2026-07-10](EXP-2026-07-10-exit-search-loop.md)) failed on **data**, not objective — verified three ways:
- **NoHz and Q2 are the same recipe.** Same `hl_gauss` value head, same target (opener=1 / setup=0.9 / dead=0), same no-horizon, same direct-score deploy. The ONLY material difference is the training data. So the setup-#1 gap (NoHz 50.9% vs Q2 21.6%) is a data-quality story, not architecture/objective.
- **The noise is false-negative setups, and it's structural.** At depth-1 a setup is indistinguishable from a dead-end (neither opens in one push), so rung-1 stamps every setup `0`. Rung-2 only *rescues* (→0.9) the setups its **Q1-steered** beam expands — but Q1 is an *opens-now* detector with no signal for setup selection, so it steers the setup search blind → many true setups never get a finisher → stay `0`.
- **Calibration ruled the objective out.** `pos_weight=6` lifted setups out of the dead band **in-sample** (pred 0.16→0.57) but held-out setup AUC stayed **~0.46 (near-chance)**. The head *can* represent setup-value; the labels won't let it generalize. ⇒ **fix the data, not the loss.**

## The bar (definition of winning)

Every bucket **easy / med / hard** × horizon **1push / 2push**: **≥95% success in far fewer sims than random.** Reported **with-search AND reactive (no-search)**, difficulty per-episode, held out by room.

## Design decisions (locked by USER)

- **NO ranking objective, ever** — the data carries ranking. (Adding one would confound "did the data get better?")
- **Exhaustive 1-push collection** (`region_sample_k: 0` → all reachable first-pushes, not a 25-sample).
- **Depth-2 = exhaustive first-ply + guided finish** (near-oracle finisher). Cost O(N·k), not O(N²) — the finish-sim budget is *reallocated* from "20 blind samples on 15 blind setups" to "a few guided finishes on all setups."
- **One model does open-now (depth-1) AND finish (depth-2 second push)** — same policy. Depth-2's finisher-misses feed back as depth-1 training data → both improve together.
- **Harder scenes via rejection sampling** (keep the hard tail) — the interim hardness knob.
- **pure2push exhaustive testset = finish diagnostic, EVAL ONLY** (never train; canonical held-out).

## Stages (plain)

- **Stage 1 — rock-solid 1-push model.** Collect exhaustive 1-push → train one value model (opens-now) → test all buckets vs the bar + grade its finishing on the exhaustive 2-push testset.
- **Stage 1b — make it harder (rejection sampling).** Drop easy scenes / keep the hard tail (and the ones it still fails) → retrain until the **hard** bucket clears the bar.
- **Stage 2 — collect clean 2-push.** Try **every** first push (1 sim each) → from each new state let the 1-push model pick the finisher (guided) → a first push is a **setup** iff a finish opens. Full first-ply coverage ⇒ no missed setups. Finisher-misses → new hard depth-1 examples.
- **Stage 3 — train the ≤2-push model** on the clean data (same value head, now also ranks setups) → test vs bar + random.
- **Stage 4 — climb.** Rejection-sample harder → recollect with the better model → retrain. Better model → cleaner+harder data → better model.

## Hypotheses [USER to confirm/edit]

- **H1** — With complete 1-push labels + rejection-sampled hard data, one value model clears **≥95% all buckets on 1push well before random**, with **no ranking loss**.
- **H2** — Exhaustive-first + guided-finish cuts the false-negative-setup rate **sharply** vs the Q1-steered beam, at **comparable sim cost**.
- **H3** — The same clean data lets the ≤2-push model **beat NoHz's setup-#1 (50.9%)** — because the wall was data, and we removed it.

## Plan / Run / Result

**Binning note:** 1push difficulty = **tertile** on solve_rate (the results-sheet convention; matches exit_q2_headline 447/435/441), NOT `eval_common` fixed cuts. See the 1push-binning-mismatch note.

**Stage 1 — baseline (DONE, 2026-07-11).** Incumbent NoHz-v3 (`qfull_nohz_v3_v4hq_s1/.../epoch012`) vs the bar, tertile binning. Result: **the bar is already met on easy + medium 1push by the incumbent — the entire gap is the HARD bucket, on both the opener and the finish.**

| 1push bucket | NoHz react @1 | clears 95% at | random 95% at | verdict |
|---|--:|--:|--:|---|
| easy | 98% | @1 | ~@3-5 | clears hugely |
| medium | 95% | @2 | ~@8-10 | clears, ~5× fewer sims |
| **hard** | **54%** | ~@35-40 | ~@80-90 | clears but only ~2× ahead (not "far"); @1 weak |

Finish sub-task (from true setup, exhaustive pure2push, ~58/bucket): strong within a few tries (beats random 4-7×, 95% by @20-30) but **reactive @1 weak everywhere: 70 / 62 / 53%** (easy/med/hard).

Two consequences:
1. **The whole game is the HARD bucket** (opener @1 54% + only 2× ahead; finish @1 53%). Stage-1b rejection-sampling-hard is aimed exactly here.
2. **Stage-2 design input:** finish @1 is only 53% on hard and reaches 95% only by @20-30 ⇒ the guided finish must be **top-~20-30, not top-1** — else we re-inject the false-negatives we're removing.

Paths: `/common/users/dm1487/scratch_namo/eval/curriculum_ladder_s1/`.

**Stage 1 collection + H5 (DONE, 2026-07-11).** Exhaustive depth-1 (`region_sample_k:0`) on 28,269 rooms — 7 one-per-node SLURM jobs (had to fight iLab `unlimited`'s no-CPU-reservation quirk with `--nodelist` + disjoint manifest chunks), 21-65 min each. Merged H5 `curriculum/rung1_exhaustive.h5` = **52,255 episodes**, complete opener labels:
- MASK band **13.35% → 3.21%** (residual = physically-blacklisted deep cells on stuck edges, not sampling holes).
- opener-among-reachable-tried **23.5% → 32.5%** (the sampled rate was biased LOW — trying everything reveals more openers); tried cells 29.2 → 63.7/ep.
- (52,255 vs old sampled 29,891 rows — different run/coverage; flagged, not blocking.)

**Stuck-cell relabel (2026-07-11).** The residual 3.2% MASK is entirely deep pushes (depth-2 1.3% → depth-4 10.3%; depths 0-1 = 0%): contacts where the object physically jams at a shallow depth, so deeper pushes are blacklisted (they'd re-jam at the same spot). A jammed deep push lands at the same state as its jam-point, so its correct value = the jam-point's value. Verified: **100% of the 110,582 stuck cells have a DEAD (0) jam-point, 0% an opener** (physical: jamming is anti-correlated with opening) → relabeling stuck → 0 (dead) is safe, no false negatives. Produced `rung1_exhaustive_stuck0.h5` (MASK band 3.21% → 0.00%, opener 31.4%).

**Stage 1 train+eval — A/B (RUNNING).** Depth-1 model = hl_gauss value head (NoHz-compatible), `Q2_POS_WEIGHT=1` (no reweighting). Two arms for a clean A/B: **MASK** (`depth1_v1`, on `rung1_exhaustive.h5`) vs **stuck→dead** (`depth1_stuck0_v1`, on `rung1_exhaustive_stuck0.h5`, canonical). Eval both: 1push @k-by-bucket (search+reactive, tertile) vs random + finish transfer diagnostic, vs NoHz baseline. Headline question: **does complete data move the HARD bucket** (NoHz hard @1 = 54%), and does stuck-labeling help (esp. reactive)?

**Stage 1 RESULT + KEY FINDING (2026-07-11) — the opener is NOT the data wall.** depth1_v1 (MASK arm) trained clean but lands *uniformly below* NoHz on every bucket (1push @1: easy 88.8/med 73.6/hard 32.4 vs NoHz 98.4/94.7/53.7; finish @1 41 vs 62). **This is a confound, not a result:**
- **Wrong room pool** — the reused ExIt manifest is `exit_pool/v1/{aug9,feb}_car`, **geometry-disjoint from `namo_testset_v1`**; NoHz trained on `car_envs/v3` that the testset is *held out from* → NoHz has home-field, depth1_v1 has none. (Orchestrator miss: reused the manifest without checking lineage vs the testset — the verify-before-bet lesson.)
- **5× less data** — 52,255 eps (exit_pool) vs NoHz's **252,805** eps.
- **Decisive:** NoHz's own `v4_hq_m2b_scorer` data is **100%-exhaustive** 1-push `f_grid` (every reachable cell labeled, 0 holes) on the v3 lineage, same ~29% opener rate. ⇒ **exhaustive openers can't beat NoHz — it already has them.** Opener completeness is a solved wall.

**Implication (pivot, PROPOSED — pending USER):** the wall is where we diagnosed — **setups** (Stage 2; NoHz has no clean setup data) + **hard-example coverage** (NoHz hard @1 = 54% is too-few-hard-examples, i.e. the rejection-sampling lever, NOT incompleteness). Both run on the EXISTING v3 data (no opener re-collection): (1) rejection-sample hard v3 rooms → retrain → does hard @1 beat 54%; (2) Stage 2 clean setups on v3. Drop exit_pool; move all collection to the v3 `car_envs` lineage (testset home turf). Stuck0 A/B still finishes (isolates the relabel effect, reusable for v3 setup collection).

_(run log appended below as stages complete)_

## Status reconciliation (2026-08-12)

**Marked `superseded`** — [EXP-2026-07-12-opener-curriculum-loop](EXP-2026-07-12-opener-curriculum-loop.md) declares `supersedes:` this card in its own frontmatter ("findings kept; this is the clean plan"). Stage 1 completed and produced a real finding: the apparent data wall was a room-pool lineage confound (`exit_pool` vs the testset's v3 lineage), not data incompleteness. **Dangling:** Stage 2 (clean setups on `exit_pool`) never ran — dropped for the v3-lineage pivot. Canonical record of this line: [EXP-2026-07-14](EXP-2026-07-14-region-opening-curriculum-marvel.md).
