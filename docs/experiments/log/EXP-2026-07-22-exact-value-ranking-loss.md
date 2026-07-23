---
type: experiment
status: live
created: 2026-07-22
commit: 9b8f5ba
metric: pending loss-only d20 A/B; report 1push solve@1/@5 and 2push solve@2/@5/@10/@30 plus sims-to-solve by easy/med/hard
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

**Implementation frozen at `9b8f5ba`.** `train_q2_rankaux.py` now finds every exact target tier on each board and applies the existing listwise softmax competition only against tried-reachable cells with a strictly lower exact value or ceiling; the total remains averaged once per board, and opener/setup components are logged separately. Equal ceilings, reachable-untried cells, and unreachable cells are excluded from ranking. Six focused tests pass, covering opener ranking, the new setup ranking, equal-ceiling exclusion, unknown/unreachable exclusion, no-comparison rows, multiple exact tiers, and finite gradients.

**Real-d20 label-tensor gate passed.** On the first 1,024 artifact rows, the new path produced finite nonzero total/opener/setup losses (`3.5744/3.5368/3.6440`) and finite gradients on 69,261 value cells. No H5 data or labels were changed.

No training job launched yet. The required next gate is one complete epoch on the target SLURM box using the exact full-run command and d20 H5.

## Result + Verdict

Pending.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
