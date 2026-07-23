---
type: experiment
status: complete
created: 2026-07-22
commit: 4a72536
metric: setup ranking confirmed and tight-budget 2push improved, but hard-1push@5 regressed 5.4pp; mixed pilot, no automatic promotion
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

## Run

**Implementation frozen at `365f8ec`.** `train_q2_rankaux.py` now finds every exact target tier on each board and applies the existing listwise softmax competition only against tried-reachable cells with a strictly lower exact value or ceiling; the total remains averaged once per board, and opener/setup components are logged separately. Equal ceilings, reachable-untried cells, and unreachable cells are excluded from ranking. Seven focused tests pass, covering opener ranking, the new setup ranking, equal-ceiling exclusion, unknown/unreachable exclusion, no-comparison rows, multiple exact tiers, mixed-validity batches, and finite gradients.

**Real-d20 label-tensor gate passed.** On the first 1,024 artifact rows, the new path produced finite nonzero total/opener/setup losses (`3.5744/3.5368/3.6440`) and finite gradients on 69,261 value cells. No H5 data or labels were changed.

**First target-box smoke caught and contained a mixed-row NaN (`186849`, canceled at 2m24s before any checkpoint).** Invalid rows for one exact tier entered an all-masked softmax before the valid-row filter; the warning appeared during the first real epoch even though single-row unit tests were finite. Commit `365f8ec` computes each tier softmax only on valid rows and adds the reproducing mixed-batch test. The required next gate remains a fresh complete epoch on the target SLURM box using the exact full-run command and d20 H5.

**Second target-box smoke completed training cleanly, then exposed an old diagnostic-only OOM (`186850`, ilab2 A4500).** The full epoch finished in 4m25s with `train_loss=2.8414`, `val_loss=2.2859`, and checkpoint `epoch000-val_loss2.2859.ckpt`; there was no NaN warning. After checkpointing, the inherited reload check put all 23,139 validation rows into one GPU batch and exhausted 20 GB. Commit `3aa667d` streams that check at the training batch size, uses the same exact/ceiling/unreachable validation formula as the trainer, and exposes the registered early-stopping patience through the canonical SLURM launcher. A third one-epoch smoke is required because the training run must also pass its checkpoint reload/loadability checks before the full treatment begins.

**Third target-box smoke passed every launch gate (`186852`, ilab2 A4500, commit `41cc9e3`).** One epoch completed in 4m05s with `train_loss=2.8530` and monitored `val_loss=2.2676`; two independent reloads were bit-identical, streamed reload validation was `2.2665` (`0.0011` from Lightning's monitor), and the evaluator reconstructed the 51-bin head with the expected `(1,60,5)` value shape. The canonical launcher observed the success marker and ended the known diagnostic teardown cleanly at 5m05s. This authorizes the planned 12-epoch d20 treatment with patience 2 on the same GPU type.

**Full treatment and evaluation completed.** Training job `186859` ran all 12 epochs on ilab2 A4500 and selected `d20_exact_value_rank_seed1/checkpoints/epoch011-val_loss1.6855.ckpt`; reload validation was `1.6860` versus monitored `1.6855`. Evaluation smoke `186864` and canonical 38-shard array `186866` completed successfully on all 1,323 one-push and 1,018 pure-two-push episodes. Board-level diagnostic job `186921` scored baseline and treatment on the same 73,368-row exhaustive held-out H5 using commit `4a72536`.

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

## Discussion

_(you ↔ Claude — newest at the bottom.)_
