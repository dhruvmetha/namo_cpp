---
type: experiment
status: done
created: 2026-07-24
thread: rl_loop
robot: car
parent: EXP-2026-07-21-colossus-data-scaleup
commit: a4a55d0
tags: [experiment, search, best-first, failure-discount, dirt, suppression]
---

# Failure-discount best-first — demote a board's siblings when its candidates keep failing

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model stays a frozen ranker; this experiment changes only the SEARCH's trust in it: a per-board credibility multiplied into candidate priorities, demoted by verified failures. Motivated and quantified by the queue-suppression trace in the parent card ([EXP-2026-07-21](EXP-2026-07-21-colossus-data-scaleup.md) §"Queue-suppression trace").

## The one sentence

The static best-first queue lets children of wrong roots flood above the true setup (measured 63–77% of wasted sims); tax every post-push board's surviving candidates by a credibility `w` that drops with each verified failed finish on that board, and win back most of that waste with zero retraining.

## Why (evidence, from the parent card's trace — 1018×2 episodes, 0 validation mismatches)

Wasted-sims attribution on labeled-setup solves: placement 7–8% / **suppression 63% (setup-only), 72% (d20)** — hard tier 66%/77% — / finish 20–28%. The labeled setup's median initial root rank is 6–9 (in-pool 100%), but each simmed wrong root injects ~15–25 raw-q children above it (hard pops_child p90 = 221/418). Upper bound if suppression were eliminated: labeled-setup solve cost ~60→~22 sims (all), ~98→~33 (hard).

## Hypotheses

_(user, 2026-07-24)_ The fitted survival curve g(k) will beat the fixed-γ demotion — fitted g is the better way to set the penalty.

_(Claude, falsifiable refinements)_ (1) Any sane demotion (γ=0.65) recovers the bulk of the suppression term because the waste is heavy-tailed; (2) fitted g adds a real but smaller increment over γ, mostly on hard/tight budgets where curve shape matters; (3) random-with-discount improves MORE in relative sims than the ranker (its floods are worst), shrinking but not closing the ranker's margin.

## Design (locked with user — the full derivation lives in chat 2026-07-24)

- Per-board `(n, k_failed, w)`; effective priority `q(a)·w(board)`; root board w=1 always (single root board — nothing to demote against; root failures spawn children, they are not evidence). Only post-push (finish) boards are taxed.
- Modes: `off` (default, bit-identical to today), `gamma` (w×=γ per failure, γ=0.65 = Bayes anchor from measured finish hit@1≈0.5 at prior≈0.5), `fitted` (w=g_table[k] from the survival fit), `conf` (w×=(1−q_failed)^τ hook; raw 1−q is ~10× too aggressive per the same Bayes check — implemented, not a primary arm).
- Lazy stale-reinsert on pop (w monotone-decreasing → correct); floor ε=1e-3 (reallocation, never pruning — completeness kept).
- Scalability invariants (why this shape): w = P(subtree pays off), strikes only at leaves, interior layers COMPOSE child w's (3push-ready without per-layer fitting); g is fitted from the search's own board-lifetime logs (survival analysis with censoring) so it refreshes per checkpoint with zero exhaustive labeling; random baseline gets the identical machinery (its g = closed-form (n−k)/n Bayes — no model involved).
- Lineage: DIRT's demotion-on-selection (own lab) with verifier-evidence-weighted demotion; MCTS/PUCT backup in the deterministic perfect-verifier limit; Koopman Bayesian search theory is the math. Verified reading list: [queue.md](../../queue.md) §"Adapting the search when the heuristic misleads". Web sweep found NO published match for the integrated mechanism (learned ranker + perfect verifier + sibling demotion on failure).

## Plan

1. **Implement + gate:** flag-gated change in `scripts/sandbox/eval_bestfirst.py` (worktree); `--discount off` must be bit-identical to the unmodified script on smoke episodes before anything runs at scale.
2. **g fitting (no new collection):** instrumented static-search sweep on ~1.5–2k TRAIN-side 2push episodes (never testset — leakage-clean), ranker + random arms in parallel SLURM arrays; per-board lifetime logs → Kaplan-Meier g(k) on child boards. Validations: g(1)≈0.65 Bayes anchor; shape vs best-fit γ^k; random empirical vs analytic (n−k)/n.
3. **The grid (testset, pure2push 1018, budget 900, paired):** static column REUSED from existing evals (setup-only 97.5 solve / 46.0 sims; random 89.9 / 194). New runs = the discount column only: ranker×{gamma, fitted} + random×{analytic}. Report by tier at budgets {2,5,10,30,900} + sims-to-solve. [USER 2026-07-24: launch authorized on gate-pass.]
4. **Decision bar (pre-registered):** adopt discount as deploy search if sims-to-solve improves ≥20% on med/hard with no solve@900 regression on any tier. Fitted-vs-γ verdict = the user hypothesis test. Prediction to check: suppression component of labeled-setup solves shrinks toward the ~place+1+finish floor (~22 all / ~33 hard); some hard solve@900 upside (10 of setup-only's 25 unsolved never reached their labeled setup).
5. On adoption: rerun the hard1p_h2 and (later) timed-campaign suites under the new search; the parent card's "beat random" claims must be re-stated within the discount column (baseline symmetry).

## Run

**Phase 1–2 launched (2026-07-24):** Opus agent in isolated worktree building the flag-gated search + lifetime logging, then the two-arm train-side lifetime sweep (SLURM `unlimited`, no --time) and the survival fit. Outputs → `round3/eval/gfit/{ranker,random}/`, g_table JSON + fit script. Testset discount column fires on gate-pass + g delivery.

**g fit DONE (2026-07-24, clean array 188502 after a false-alarm incident — see parent-card memory `squeue-ssh-banner-bug`).** 1616 train-side pure-2push episodes (validset_r2, never testset), 34,385 ranker child boards (63.6% censored, KM handles). Implementation verified LEAF-IDENTICAL at `--discount off` before any run. Fit outputs `round3/eval/gfit/v2/fit/{g_table.json, g_table_random.json, g_fit_report.json}`; code synced to main `scripts/sandbox/{eval_bestfirst.py, fit_g_survival.py, build_gfit_episodes.py}`.

- **The measured curve is FAR steeper than the Bayes anchor and non-geometric:** g(1)=**0.284** (anchor predicted 0.65), g(2)=0.168, g(5)=0.085, plateau ~0.02-0.03 by k≥8 (tail low-confidence, single-digit events past k≈9). Best-fit γ=0.773 misfits BOTH ends (0.77 vs 0.28 at k=1; 0.006 vs 0.018 at k=20) — the true discount is a sharp 1-2-strike drop then flat, which a single γ cannot express. Mechanism: **g0 = 0.63%** — over 99% of post-setup child boards are dead, so a top-ranked miss is highly diagnostic; the 0.65 anchor wrongly assumed a ~50% alive prior.
- **Random arm needs its OWN empirical curve** (g(1)=0.754 — an early random miss carries little signal), and the analytic (n−k)/n is falsified by the same dead-board mass (predicted 0.978 at k=1). Design deviation from plan: random×discount uses `g_table_random.json`, not the analytic form.
- Pre-eval read on the [USER] hypothesis: shape-misfit of γ is mechanistic support for fitted>γ; the eval decides. γ arm runs at 0.28 (the measured first-strike), not the falsified 0.65.

**Discount column LAUNCHED:** ilab array `188650` (96 tasks): ranker×fitted, ranker×γ0.28, random×fitted-random on testset pure2push 1018, budget 900, canonical shards. Static column reused (setup-only 97.5/46.0, random 89.9/194).

## Post-adoption trace — WHERE the remaining cost sits (job 188958+188998, full 1018, discount=fitted)

Re-ran the attribution tracer under the adopted search (tracer ported to the discount modes; same episodes/harness as the static trace). Mean sims on labeled-setup solves, static → discount:

| tier | place | supp | finish | total | shares (place/supp/finish) |
|---|--:|--:|--:|--:|---|
| easy | 2.6 → 2.5 | 12.2 → **1.3** | 10.9 → 10.0 | 26.8 → **14.9** | 10/46/41 → 17/9/68 |
| med | 3.8 → 3.6 | 27.2 → **2.1** | 12.1 → 21.7 | 44.1 → **28.5** | 9/62/27 → 13/8/76 |
| hard | 7.2 → 11.0 | 65.0 → **8.8** | 25.0 → 48.2 | 98.3 → **69.0** | 7/66/25 → 16/13/70 |
| all | 4.8 → 6.2 | 37.6 → **4.4** | 16.5 → 29.1 | 59.9 → **40.7** | 8/63/28 → 15/11/72 |

**Mechanism confirmed:** suppression collapsed 8.5× (all 37.6→4.4; hard 65.0→8.8), exactly the targeted term. Per-wrong-root cost is now ~1.8 sims (push + one probe) vs ~17 before. Placement (r0) is unchanged by construction — the ranker is untouched (r0 mean identical: easy 9.3 / med 9.7 / hard 17.6).

**Unpredicted side effect — the finish term GREW** (all 16.5→29.1; hard 25.0→48.2) and is now the dominant cost (70-76% on med/hard). Hypothesis (UNVERIFIED): the discount also benches the CORRECT setup's board when its first finish misses, so live boards whose winner is not rank-1 pay repeated round-trips through the root list before being revisited. Same mechanism as the measured @5 dip, larger than expected. Note the totals still fall on every tier (hard −30%), and the eval-level s2s (50.5 hard) is lower than this subset's 69.0 because ~58% of solves route through UNLABELED first pushes not counted here.

**Correction to an earlier claim in this card's chat:** the post-fix bottleneck is NOT placement (16%) — it is finish ranking after the correct setup. An arithmetic estimate that assumed finish held at ~16 sims was wrong; only measurement settled it.

**Next experiments this motivates (cheap, one sweep each, not yet run):** (a) `conf` mode — demote proportional to the FAILED candidate's score (a low-scored miss should barely dent a board), already implemented; (b) asymmetric patience — exempt or floor the board reached via a high-V setup; (c) the cap-1-then-requeue ablation to separate first-pass benching from graded revisits.

## Result — grid complete (188650, testset pure2push 1018, budget 900, paired vs reused static column)

| arm | solve@900 | s2s | @2 | @5 | @10 | @30 | hard: solve / s2s / @30 |
|---|--:|--:|--:|--:|--:|--:|---|
| ranker static | 97.5 | 46.0 | 33.9 | 51.6 | 61.6 | 74.4 | 94.1 / 74.3 / 60.1 |
| **ranker fitted-g** | **98.0** | **30.0** | 33.9 | 49.7 | **64.4** | **81.6** | **95.4 / 50.5 / 68.5** |
| ranker γ=0.28 | 98.0 | 30.2 | 33.9 | 49.7 | 64.4 | 81.6 | 95.7 / 52.2 / 68.5 |
| random static | 89.9 | 115.6 | 3.1 | 9.6 | 19.4 | 37.6 | 76.5 / 179.3 / 22.6 |
| random fitted-g | 93.6 | 98.3 | 2.6 | 8.2 | 17.5 | 39.5 | 84.9 / 166.3 / 22.1 |

**Verdict 1 — ADOPT (pre-registered bar PASSED).** Discount cuts ranker sims-to-solve **−35% overall** (46.0→30.0; easy −50%, med −34%, hard −32%) with solve@900 UP on every tier (hard 94.1→95.4) and @30 +7.2. The one cost: tight-budget @5 dips −1.9 (hard −3.7) — the predicted delay of deep winners on live boards after early strikes; @10 already net-positive. Deltas far exceed eval-sim jitter.

**Verdict 2 — [USER] hypothesis (fitted > γ): NOT confirmed at equal calibration.** Fitted and γ=0.28 are a tie (s2s 30.0 vs 30.2; identical @k; tier diffs inside noise). The refined truth: the MEASUREMENT was the value — γ's winning value (0.28) came from the fitted curve and is 2.3× steeper than the falsified theory anchor (0.65, untested arm). Ship **fitted** anyway: no extra knob, self-refits from run logs per checkpoint.

**Verdict 3 — baseline symmetry holds, margin intact.** Random also gains (solve 89.9→93.6, hard +8.4; s2s −15%) — as predicted it benefits, and MORE in solve-rate. But the ranker's margin within the discount column stays decisive: 3.3× fewer sims-to-solve (30 vs 98), 6× @5 (49.7 vs 8.2).

**Adopted deploy search: best-first + fitted-g failure discount** (`--discount fitted --gtable round3/eval/gfit/v2/fit/g_table.json`). Follow-ups queued for the next round: rerun hard1p_h2 + reactive under the new search; re-state parent-card headline numbers in the discount column; timed campaign on the adopted configuration; τ/tempered-conf arm only if the @5 dip matters for a deploy target. Eval outputs `round3/eval/discount_2push/*`; sbatch `run_discount_2push.sbatch`. Plot: `round3/eval/plots/success_vs_sims_discount.png` (anytime curves, all + hard, 5 arms).
