---
type: experiment
status: idea
created: 2026-07-22
commit:
metric: Antman-5c edge-vs-depth failure decomposition; 1push solve@1/@5; 2push solve@2/@5/@10/@30 and sims-to-solve, all by easy/med/hard
thread: rl_loop
parent: EXP-2026-07-14-region-opening-curriculum-marvel
related: EXP-2026-07-12-depth-geometric-grounding
tags: [experiment, ranker, architecture, push-depth, action-grounding, antman, beast]
---
# Push-depth-aware ranker — represent all 60×5 complete pushes

**⛔ Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** The model is one ranker that orders pushes for search; the simulator remains the perfect verifier. This card is the current Antman/Beast-line successor to the unrun pre-curriculum analysis in [EXP-2026-07-12](EXP-2026-07-12-depth-geometric-grounding.md).

## The one sentence

Measure whether Antman-5c is choosing the wrong contact or the wrong push depth, then test a cheap action-aware head that scores each of the 300 `(contact, depth)` pushes using that push's nominal motion instead of treating the five depths as anonymous output slots.

## Hypothesis

_(you, via chat 2026-07-22)_ Giving the ranker an explicit representation of each complete `(contact, depth)` push will improve ordering because the current model represents only 60 contact tokens and leaves depth as five ungrounded output channels.

_(Claude, falsifiable refinement)_ The likely gain is small on 1-push if Antman-5c repeats M2b's mostly-wrong-contact failure pattern, but larger on 2-push search because direct openers favor deeper pushes while useful setup pushes favor shallow pushes; the expected benefit is fewer simulator calls, not a new verifier or a higher brute-force ceiling.

## Known facts before the run

The current `EdgeCrossAttn` produces one vector per contact, shape `(B,60,D)`, after four edge cross/self-attention blocks. Its shared head emits `5×51` logits from that one vector and reshapes them to `(B,60,5,51)`; depth has no input feature, positional encoding, or motion description.

The older M2b diagnostic found, on hard 1-push episodes, 29.6% success, 6.9% correct-contact/wrong-depth, and 63.5% wrong-contact; 90% of its misses were therefore contact errors. This result does not establish Antman-5c's failure split.

Antman-5c's current aggregate reports 1-push easy/med/hard/all solve@1 of 98.3/85.3/39.7/85.1 and 2-push solve 91.5%, 138 average simulator calls over all episodes, 26.3%@2, and 61.4%@30. Its saved aggregate does not record the predicted `(edge,depth)`, so it cannot answer the edge-versus-depth question.

The existing `scripts/eval_scorer.py` already implements the correct three-way diagnostic and aggregation, including per-episode matching, true per-episode difficulty, depth accuracy conditional on a solving edge, and top-1 depth histograms. Its legacy pre-rendered test H5 inputs are absent on both the CS estate and Amarel, so the logic must be reused through the current live canonical-test loader rather than copied into a new evaluator.

## Plan

### Phase 0 — measure Antman-5c before changing the model

Extend `scripts/eval_scorer.py` in place with a live-canonical mode that iterates `onepush_episodes.json` by the full episode identity `(xml, object_id, goal region)`, obtains Antman-5c's deployment-realistic reachable action scores through the existing `BeamPlanner`/live scorer, and sends those scores through the existing diagnostic aggregation unchanged.

For each episode, classify the top-ranked reachable action as `success` when its exact `(edge,depth)` is valid, `right_edge_wrong_depth` when its edge has some valid depth, and `wrong_edge` otherwise. Also report depth accuracy conditional on a solving edge, the top-1 depth histogram, rank of the first valid action, and success/edge hit@1/@5/@10/@20.

Hard gates are 1,323 canonical 1-push episodes with the existing fixed-cut composition easy/med/hard = 698/421/204, zero duplicate episode identities, every evaluated episode containing at least one valid reachable action, and difficulty taken from that episode's own `solve_rate`. Do not key, deduplicate, or bin by XML alone.

Phase-0 decision: if wrong-depth is at least 20% of Antman-5c's misses or conditional depth accuracy is below 80% on any tier, an Antman-5c architecture A/B is directly motivated. If not, Antman becomes only the plumbing/1-push guardrail and the main architecture test moves to setup-labeled Beast data, where the depth hypothesis actually lives.

### Phase 1 — add a cheap 60×5 action-aware head

Keep the scene encoder and all 60-contact attention blocks unchanged. Immediately after `edge_norm`, expand `(B,60,D)` to `(B,60,5,D)`, encode each action's nominal pre-simulation primitive motion, combine it with the contact vector, and apply one shared head that emits 51 HL-Gauss bins per complete action.

The first treatment is nominal motion grounding, not merely a learned depth ID: use the shape-specific primitive's pre-simulation action parameters after verifying their coordinate semantics against live generated goals. The data and live scorer must construct the same `(60,5,feature)` tensor; simulated post-push poses are forbidden because they would leak the verifier.

Do not send all 300 actions through the transformer in the first treatment. Full 300-token attention is a later ceiling test only if the cheap post-attention action head helps but leaves evidence that different depths need separate scene attention.

### Phase 2 — controlled training A/B

Train baseline and depth-aware Antman-5c with identical 178,364 boards, ceiling loss, listwise rank auxiliary loss, room-grouped split, seed, optimizer, and schedule; only the action head and its nominal-motion input may differ. Train from scratch for the clean architecture comparison, with optional encoder warm-start recorded separately rather than mixed into the main A/B.

If the Phase-0 gate says Antman depth is secondary, use a one-seed Antman smoke only to prove training and live loading, then run the meaningful one-seed A/B on the same stable setup-labeled Beast dataset and recipe. Three seeds are authorized only after the one-seed mechanism and deployment gates pass.

### Evaluation and verdict

Report both horizons and every difficulty tier. For 1-push, report solve@1/@5 plus the full edge-versus-depth decomposition. For 2-push, report solve@2/@5/@10/@30 and average/median simulator calls to solution; use the canonical pure-2-push difficulty divisions. Wall-time comparisons require the same hardware and an interleaved baseline, so simulator calls are the first architecture verdict.

Accept the depth-aware head only if it preserves 1-push performance within 2 percentage points on every tier, improves 2-push average simulator calls by at least 10%, and improves solve@2 or solve@5 by at least 3 percentage points with a matching mechanism shift toward the valid setup depth. Reject it if those search gains do not appear, even if validation loss improves.

## Run

_(pending; no code change, commit, training, or evaluation has been launched.)_

## Result + Verdict

_(pending.)_

## Next

Run Phase 0 first. The measured Antman-5c failure split decides whether Antman receives the full architecture A/B or serves only as the 1-push guardrail before the setup-labeled Beast test.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
