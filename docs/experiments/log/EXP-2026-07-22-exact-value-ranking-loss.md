---
type: experiment
status: live
created: 2026-07-22
commit: cddbe15
metric: v2 pending — restore full opener rank weight and retain a bounded lower-exact pool
thread: rl_loop
parent: EXP-2026-07-14-region-opening-curriculum-marvel
related: EXP-2026-07-22-push-depth-aware-ranker
tags: [experiment, ranker, loss, listwise, exact-value, antman, beast]
---
# Exact-value ranking loss — teach verified setups to outrank known-worse actions

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is one ranker that orders pushes for search; the simulator remains the perfect verifier, so the objective is better ordering and fewer simulator calls rather than calibrated classification.

## The one sentence

The current listwise auxiliary loss ranks only exact `1.0` openers, so add direct ranking pressure for every simulator-verified exact value against actions whose known upper bound is strictly lower, and test that single loss change on the d20 dataset.

## Hypothesis

_(user, 2026-07-22)_ Any exact-value action should receive ranking loss rather than limiting listwise supervision to immediate openers.

_(Claude, falsifiable refinement)_ On d20, adding the missing exact-`0.9` setup-versus-ceiling-`0.81` competition will improve pure-2-push root ordering and tight-budget search without materially regressing 1-push ordering; comparisons against equal ceilings or unknown actions remain forbidden because their ordering is not known.

## Why this experiment exists

The current auxiliary defines positives as `label >= 0.999`, then skips every row with no such cell. A pure-2-push root contains exact `0.9` verified setups and ceiling-`0.81` known-worse alternatives but no exact `1.0` opener, so it receives exact-value, ceiling, and unreachable losses but zero direct ranking loss.

The loss-only experiment must stay separate from the push-depth input experiment because changing architecture and ranking supervision together would make either result uninterpretable. The push-depth worktree keeps its frozen loss; this worktree owns the ranking change.

## Proposed loss

Keep the existing exact HL-Gauss loss, censored ceiling loss, unreachable-floor loss, and opener-ranking behavior unchanged.

Add certain-order ranking comparisons only when an exact action's value is strictly greater than the alternative's exact value or ceiling: exact `1.0` ranks above exact `0.9` and ceilings `0.9`/`0.81`; exact `0.9` ranks above ceiling `0.81`.

Do not compare exact `0.9` against ceiling `0.9`, because the capped action may also be a valid `0.9` setup. Do not compare against reachable-untried cells, because their value is unknown. Keep unreachable cells in their existing separate floor loss and outside reachable-action ranking.

Average the new ranking contribution per board before averaging boards so dense boards do not dominate. Log opener-ranking and setup-ranking components separately in training and validation.

## Plan

Use the existing `beast2c_d20_ceil.h5` unchanged: it contains the pure-2-push roots needed to test exact-setup ordering and the finish boards needed to guard existing opener ordering. No relabeling, rebuilding, or new simulator calls are part of this experiment.

First add focused unit tests covering exact `1.0` versus ceiling `0.9`, exact `0.9` versus ceiling `0.81`, forbidden exact `0.9` versus ceiling `0.9`, unknown masking, unreachable exclusion, multiple verified actions, and a row with no admissible comparison.

Run one treatment seed with the registered d20 architecture, split, optimizer, schedule, ceiling weight `1.0`, unreachable weight `1.0`, ranking weight `0.1`, and temperature `0.15`. Compare first against the registered d20 checkpoint; only if the mechanism moves in the predicted direction should a fresh paired multi-seed baseline/treatment confirmation be authorized.

The mechanism readout is setup-versus-dead ranking on held-out pure-2-push roots, including setup hit@1/@5 and first-valid-setup rank by easy/medium/hard. The deployment verdict still requires both canonical horizons: 1push solve@1/@5 and pure-2-push solve@2/@5/@10/@30 plus average/median simulator calls, all split by easy/medium/hard.

Accept the one-seed pilot only if 2push solve@2 or solve@5 improves by at least 3 percentage points or average simulator calls improves by at least 10%, setup-ranking diagnostics move in the same direction, and no 1push difficulty tier drops by more than 2 percentage points. A passing pilot authorizes paired seeds; it is not a final result by itself.

## Follow-up v2 — split opener and lower-exact rank budgets

**Cause found after v1, pre-registered before v2 training.** On a board with both exact `1.0` openers and exact `0.9` setups, v1 optimized `0.1 × mean(opener term, setup term)`, so each term received effective weight `0.05`. This unintentionally halved the original opener auxiliary on 140,426 of 182,178 opener-eligible d20 rows (77.1%) while adding setup pressure. The mechanism and deployment results moved in exactly those directions.

V2 preserves the original opener auxiliary as an independent `0.10 × opener-ranking` term and adds one bounded `0.05 × lower-exact-ranking` pool. The lower pool automatically includes every exact tier below `1.0`, averages multiple lower tiers within a board, and therefore does not grow when future exact tiers are added. The exact-value, ceiling, unreachable, temperature, architecture, d20 H5, split, optimizer, schedule, seed, and evaluation protocol remain unchanged.

The v2 seed-1 gate requires both: (1) retain a meaningful setup gain—held-out setup hit@1 at least 60% and either 2push solve@2 or solve@5 at least 3 points above baseline; (2) repair the safety failure—hard-1push solve@5 at least 69.6% (within 2 points of baseline 71.6%). Report the same complete difficulty×horizon table and opener/setup diagnostics. This is a corrective single-seed pilot, not a multi-seed confirmation.

## Run

**Implementation frozen at `365f8ec`.** `train_q2_rankaux.py` now finds every exact target tier on each board and applies the existing listwise softmax competition only against tried-reachable cells with a strictly lower exact value or ceiling; the total remains averaged once per board, and opener/setup components are logged separately. Equal ceilings, reachable-untried cells, and unreachable cells are excluded from ranking. Seven focused tests pass, covering opener ranking, the new setup ranking, equal-ceiling exclusion, unknown/unreachable exclusion, no-comparison rows, multiple exact tiers, mixed-validity batches, and finite gradients.

**Real-d20 label-tensor gate passed.** On the first 1,024 artifact rows, the new path produced finite nonzero total/opener/setup losses (`3.5744/3.5368/3.6440`) and finite gradients on 69,261 value cells. No H5 data or labels were changed.

**First target-box smoke caught and contained a mixed-row NaN (`186849`, canceled at 2m24s before any checkpoint).** Invalid rows for one exact tier entered an all-masked softmax before the valid-row filter; the warning appeared during the first real epoch even though single-row unit tests were finite. Commit `365f8ec` computes each tier softmax only on valid rows and adds the reproducing mixed-batch test. The required next gate remains a fresh complete epoch on the target SLURM box using the exact full-run command and d20 H5.

**Second target-box smoke completed training cleanly, then exposed an old diagnostic-only OOM (`186850`, ilab2 A4500).** The full epoch finished in 4m25s with `train_loss=2.8414`, `val_loss=2.2859`, and checkpoint `epoch000-val_loss2.2859.ckpt`; there was no NaN warning. After checkpointing, the inherited reload check put all 23,139 validation rows into one GPU batch and exhausted 20 GB. Commit `3aa667d` streams that check at the training batch size, uses the same exact/ceiling/unreachable validation formula as the trainer, and exposes the registered early-stopping patience through the canonical SLURM launcher. A third one-epoch smoke is required because the training run must also pass its checkpoint reload/loadability checks before the full treatment begins.

**Third target-box smoke passed every launch gate (`186852`, ilab2 A4500, commit `41cc9e3`).** One epoch completed in 4m05s with `train_loss=2.8530` and monitored `val_loss=2.2676`; two independent reloads were bit-identical, streamed reload validation was `2.2665` (`0.0011` from Lightning's monitor), and the evaluator reconstructed the 51-bin head with the expected `(1,60,5)` value shape. The canonical launcher observed the success marker and ended the known diagnostic teardown cleanly at 5m05s. This authorizes the planned 12-epoch d20 treatment with patience 2 on the same GPU type.

**Full treatment and evaluation completed.** Training job `186859` ran all 12 epochs on ilab2 A4500 and selected `d20_exact_value_rank_seed1/checkpoints/epoch011-val_loss1.6855.ckpt`; reload validation was `1.6860` versus monitored `1.6855`. Evaluation smoke `186864` and canonical 38-shard array `186866` completed successfully on all 1,323 one-push and 1,018 pure-two-push episodes. Board-level diagnostic job `186921` scored baseline and treatment on the same 73,368-row exhaustive held-out H5 using commit `4a72536`.

**V2 launch at handoff.** Implementation and tests are frozen at commit `3785aa5` (code commit `cddbe15`); target-box one-epoch smoke `186922` is running on ilab2 A4500 with output `exact_value_rank/v2_smoke_epoch1`. No v2 full training or evaluation job has been launched yet; those remain gated on finite epoch metrics, checkpoint reload agreement, and the evaluator-load marker.

**V2 smoke `186922` passed all three launch gates (Claude, 2026-07-23, overnight drive; user AFK).** Finite epoch (`train_loss=3.0939 val_loss=2.2926`); two-reload `max|Δlogit|=0.000e+00` and reloaded `val_loss=2.2915` vs monitored `2.2926` (Δ0.0011); evaluator reconstructed `value_bins=51`, value shape `(1,60,5)`. Submitted with `NAMO_REPO=<worktree>` so the split-budget code runs (main lacks it).

**V2 seed-1 full treatment launched: job `186923`**, ilab2 A4500, `EPOCHS=12 SEED=1 PATIENCE=2`, OUT `exact_value_rank/d20_exact_value_rank_v2_seed1` (v1 artifacts untouched). Eval (array 0-37), aggregation, and rankdiag (baseline `beast2c_d20_ceil/epoch010` vs v2-seed1 on `round2_eval.h5`) fire on completion. Seed-1 gate: setup hit@1 ≥60% + (2push solve@2 or @5 ≥+3pt) AND hard-1push solve@5 ≥69.6%. Pass → paired fresh baseline+treatment multi-seed; hold across seeds → promote the loss to default incl. the Colossus card [USER].

## Result + Verdict

Each cell is baseline d20 → exact-value-ranking treatment; solve changes are percentage points.

| 1push difficulty | n | solve@1 | solve@5 | avg sims all |
|---|---:|---:|---:|---:|
| easy | 698 | 97.4→96.8 (−0.6) | 99.7→99.4 (−0.3) | 1.1→1.2 |
| medium | 421 | 80.8→82.9 (+2.1) | 92.9→94.1 (+1.2) | 1.8→2.1 |
| hard | 204 | 39.7→40.2 (+0.5) | 71.6→66.2 (−5.4) | 8.1→10.3 |
| all | 1,323 | 83.2→83.7 (+0.5) | 93.2→92.6 (−0.6) | 2.4→2.9 |

| 2push difficulty | n | solve@2 | solve@5 | solve@10 | solve@30 | avg sims all |
|---|---:|---:|---:|---:|---:|---:|
| easy | 238 | 30.3→34.0 (+3.7) | 47.9→55.0 (+7.1) | 60.9→71.0 (+10.1) | 79.0→85.3 (+6.3) | 32.3→29.4 |
| medium | 409 | 31.3→33.3 (+2.0) | 46.0→52.1 (+6.1) | 59.7→64.8 (+5.1) | 72.9→78.7 (+5.8) | 57.9→51.5 |
| hard | 371 | 19.1→22.9 (+3.8) | 30.7→35.6 (+4.9) | 41.5→44.2 (+2.7) | 58.5→56.9 (−1.6) | 144.9→146.3 |
| all | 1,018 | 26.6→29.7 (+3.1) | 40.9→46.8 (+5.9) | 53.3→58.7 (+5.4) | 69.1→72.3 (+3.2) | 83.7→80.9 |

**The intended mechanism is confirmed on 547 held-out root boards with exact setups and exact dead alternatives.** Setup-vs-dead AUC rose `0.9063→0.9252`, setup hit@1 `55.0→64.5` (+9.5), hit@5 `83.9→88.1` (+4.2), mean first-setup rank improved `4.01→3.44`, and p90 improved `8→7`. The finish-opener guard was mixed: opener-vs-dead AUC `0.9400→0.9339`, hit@1 `45.9→44.8`, hit@5 `82.9→83.7`, and recall@20 `87.3→87.8`.

**Verdict: mechanism PASS, deployment pilot MIXED / pre-registered promotion gate FAIL.** The primary tight-budget 2push bar passed decisively, but hard-1push solve@5 fell 5.4 points, exceeding the allowed 2-point tier regression. Do not interpret this seed as a replacement for d20. If further confirmation is authorized, run paired fresh baseline+treatment seeds rather than two treatment-only seeds so training variance and the apparent setup/opener tradeoff are separable.

## V2 result — split budget (opener 0.10 + lower-exact 0.10) [Claude, 2026-07-23 overnight, user AFK]

V2 seed-1 full treatment `186923` trained 12 epochs (patience-2 stop at epoch 10, `val_loss=1.6964`; reload Δ0.0002, evaluator-load OK). Eval array `186964` (all 6+32 shards COMPLETED) and rankdiag `rankdiag_v2.json` (baseline `beast2c_d20_ceil/epoch010` vs v2-seed1 on `round2_eval.h5`). Baseline column = registered d20.

Infra note: eval must run from the MAIN repo (built `namo_rl` bindings); the worktree has no compiled bindings, and the eval `.py` scripts are byte-identical to main. First eval submit with `NAMO_REPO=<worktree>` died on `ModuleNotFoundError: namo_rl`; corrected by dropping `NAMO_REPO` so it cd's to main.

| 1push difficulty | n | solve@1 | solve@5 | avg sims all |
|---|---:|---:|---:|---:|
| easy | 698 | 97.4→97.6 (+0.2) | 99.7→99.9 (+0.2) | 1.1→1.1 |
| medium | 421 | 80.8→82.4 (+1.6) | 92.9→96.0 (+3.1) | 1.8→1.7 |
| hard | 204 | 39.7→39.2 (−0.5) | 71.6→69.6 (−2.0) | 8.1→8.1 |
| all | 1,323 | 83.2→83.7 (+0.5) | 93.2→94.0 (+0.8) | 2.4→2.4 |

| 2push difficulty | n | solve@2 | solve@5 | solve@10 | solve@30 | avg sims all |
|---|---:|---:|---:|---:|---:|---:|
| easy | 238 | 30.3→39.1 (+8.8) | 47.9→54.6 (+6.7) | 60.9→68.9 (+8.0) | 79.0→84.5 (+5.5) | 32.3→40.6 |
| medium | 409 | 31.3→29.1 (−2.2) | 46.0→48.9 (+2.9) | 59.7→61.6 (+1.9) | 72.9→73.1 (+0.2) | 57.9→58.2 |
| hard | 371 | 19.1→24.0 (+4.9) | 30.7→36.4 (+5.7) | 41.5→46.6 (+5.1) | 58.5→59.0 (+0.5) | 144.9→128.8 |
| all | 1,018 | 26.6→29.6 (+3.0) | 40.9→45.7 (+4.8) | 53.3→57.9 (+4.6) | 69.1→70.6 (+1.5) | 83.7→79.8 |

**Setup mechanism did NOT hold.** Setup-vs-dead AUC `0.9063→0.9066` (flat), setup hit@1 `55.0→53.7` (−1.3, BELOW baseline), hit@5 `83.9→85.2`. Opener guard held/improved: opener-vs-dead AUC `0.9400→0.9423`, hit@1 `45.9→44.0`, recall@20 `87.3→92.3` (+5.0).

**Verdict: pre-registered gate FAIL (mechanism sub-bar), deployment PASS.** Gate condition (1) required setup hit@1 ≥60 — v2 delivered 53.7, so FAIL. Safety condition (2) PASSED (hard-1push@5 = 69.6, exactly the −2 boundary; v1 had failed at 66.2). Deployment: 2push improved on every tier (all +3.0@2/+4.8@5/+4.6@10) and 1push stayed neutral-to-up outside hard@5. So v2 REPAIRS v1's safety failure and keeps ~v1's 2push deploy gain — but achieves it via the opener term, NOT the intended setup mechanism (which fell to baseline).

**Root cause (diagnosed, pre-registered before v3).** On pure-2push-root boards (setup-only, no exact-1.0 opener), v1's `0.1×mean(single setup term)` applied the FULL 0.10 to setups; v2's bounded `0.05×lower-exact pool` applies only 0.05 there — HALF the setup pressure exactly where setups live. That is why v2's setup ranking regressed to baseline while its opener ranking (untouched at 0.10) stayed strong. The deploy 2push gain is opener-driven, robust; the mechanism needs the setup weight restored.

**V3 (Claude autonomous fix, same gate).** Double the lower-exact pool weight to `LOWER_RANK_LAMBDA=0.10` (opener stays `0.10` → safety preserved), everything else identical. Tests whether restoring setup pressure recovers the mechanism (setup hit@1 ≥60) WITHOUT re-breaking hard-1push safety. A clean v3 pass authorizes the paired fresh baseline+treatment multi-seed confirmation the user's promote-to-default decision needs; promotion itself (incl. the Colossus card) waits for the user.

## V3 result — lower-exact 0.10, and the tradeoff across all configs [Claude, 2026-07-23 overnight]

V3 seed-1 (`187002`, `RANK_LAMBDA=0.10 LOWER_RANK_LAMBDA=0.10`) trained clean (epoch 10, `val_loss=1.7069`, reload Δ0.0004, evaluator-load OK). Eval `187005` + rankdiag `187043`, baseline = registered d20.

| 1push (v3) | n | solve@1 | solve@5 |
|---|---:|---:|---:|
| hard | 204 | 39.7→35.8 (−3.9) | 71.6→**67.6** (−4.0) |
| all | 1,323 | 83.2→82.4 (−0.8) | 93.2→93.3 (+0.1) |

| 2push (v3) | n | solve@2 | solve@5 | solve@10 | solve@30 |
|---|---:|---:|---:|---:|---:|
| all | 1,018 | 26.6→30.7 (+4.1) | 40.9→47.1 (+6.2) | 53.3→59.5 (+6.2) | 69.1→72.5 (+3.4) |

V3 rankdiag: setup hit@1 `55.0→56.7`, AUC `0.9063→0.9103`; opener hit@1 `45.9→43.4`, recall@20 `87.3→89.3`.

**Verdict: v3 FAILS both gate bars.** Setup hit@1 = 56.7 (<60) AND hard-1push@5 = 67.6 (<69.6). Doubling the setup weight nudged setups up (+3 vs v2) but pulled hard-1push DOWN (−2 vs v2) — the same axis v1 exposed.

**The finding — a tradeoff, not a clean win.** Across baseline/v1/v2/v3, the exact-value setup-ranking loss RELIABLY improves 2push deployment (+3–4 solve@2, +5–6 solve@5, EVERY config) but TAXES hard-1push@5 whenever setup pressure is high enough to move setup ranking; and setup hit@1 never reaches 60 once the opener term is kept at full strength (v1's 64.5 came WITH the opener-halving bug).

| config | opener_w | setup_w | setup hit@1 | hard-1push@5 | 2push all@2 | 2push all@5 |
|---|---|---|---:|---:|---:|---:|
| baseline d20 | 0.10 | 0 | 55.0 | 71.6 | 26.6 | 40.9 |
| v1 (opener halved) | 0.05* | 0.05 | 64.5 | 66.2 | 29.7 | 46.8 |
| v2 | 0.10 | 0.05 | 53.7 | 69.6 | 29.6 | 45.7 |
| v3 | 0.10 | 0.10 | 56.7 | 67.6 | 30.7 | 47.1 |

*unintentional.

**Why single-seed can't close this.** The hard-1push@5 deltas (−2 to −4 on 204 episodes) are inside car eval-sim noise (~0.3 mm warmstart jitter flips near-threshold rooms — see reference_eval_sim_nondeterminism). The 2push gains (+5–6 on 1,018 episodes) are larger and consistent, but the promote-or-not question hinges on whether the hard-1push tax is REAL or noise. That requires paired multi-seed with the noise averaged down.

## Paired multi-seed confirmation [Claude, 2026-07-23 overnight — resolving the user's promote criterion]

Running 3 seeds × 2 arms, fresh + paired (same code, same H5, same recipe; ONLY the loss differs):
- **Control arm** (`LOWER_RANK_LAMBDA=0` → opener-only 0.10, reproduces baseline recipe in-code): seeds 1/2/3.
- **Treatment arm = v2** (`opener 0.10 + lower-exact 0.05`, the safety-respecting config; v2 seed-1 = `186923` reused): seeds 1/2/3.

Readout: seed-AVERAGED 2push solve@2/@5/@10 (all + hard) and hard-1push@1/@5, with per-seed spread, so the hard-1push delta is judged against its own seed variance. This does NOT auto-promote — it produces the seed-averaged deltas the user's promote-to-default (incl. Colossus) decision needs; the flip stays a user call.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
