---
type: experiment
status: live
created: 2026-07-22
commit: c86a86a
metric: Antman-5c 1push edge-vs-depth diagnostic complete; architecture A/B pending; final 1push solve@1/@5 and 2push solve@2/@5/@10/@30 plus sims-to-solve by easy/med/hard
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

**Phase-0 implementation (2026-07-22).** Commit `ddb18d2` extends `scripts/eval_scorer.py` in place with a zero-push live-canonical mode, adds the three-way category tests, and adds the CS `unlimited` launcher `scripts/slurm/eval_scorer_live.slurm`. Local compile + focused tests pass (2/2).

**CS smoke (job 186711, 2026-07-22).** The exact one-episode live path passed on `ilab1` in 15 seconds total and 0.367 seconds for scoring: zero `env.step` calls, one row written, no valid cells missing from the live candidate pool, and the labeled push ranked first. At that measured rate the full 1,323-episode diagnostic is approximately eight minutes of scoring, so one GPU job is sufficient.

**Canonical Phase-0 run (job 186712, commit `c131c8e`, 2026-07-22).** One `unlimited` GPU on `ilab1`; checkpoint `antman5c/checkpoints/epoch018-val_loss0.6276.ckpt`; 1,323/1,323 episodes completed in 2m38s with exit code 0 and 0.110 seconds/episode steady-state. Artifacts: `/common/users/dm1487/scratch_namo/eval/push_depth/full/antman5c_depth_diag.{json,jsonl}` and log `logs/a5cdepth_full_186712.out`.

**Phase-1 implementation frozen (2026-07-22).** NAMO commit `75c8c11` and Sage commit `09acfe3` implement the cheap treatment exactly as planned: the existing 60 contact tokens complete attention unchanged, then each token is expanded to its five candidate depths, combined with that candidate's normalized nominal `(dx,dy,dtheta)`, and scored by one shared 51-bin value head. `NAMO_ACTION_MOTION=0` retains the original Antman-5c architecture and checkpoint keys; `NAMO_ACTION_MOTION=1` selects the treatment.

The existing Antman-5c H5 already stores `contact_px`, whose ordered rectangle samples recover the target object's current axes, size family, and yaw. The loader therefore constructs the exact active primitive table from the pinned `1x_car_d5_motion_primitives_15_{square,wide,tall}.dat` files without XML lookup or simulator calls. Focused tests cover all three shape families and rotation, the original Antman-5c checkpoint still loads through `eval_scorer`, and an actual H5 batch completes forward/loss/backward with finite values. Parameter counts are closely matched: baseline 4,397,055 versus treatment 4,395,891.

**Future-data requirement.** Main-tree commit `d602a97` updates the Colossus-0 card to require direct `(N,60,5,3)` `action_motion` storage, current per-board object pose/size, primitive database identity/hash, PKL→NPZ→H5 preservation, and a 300-action alignment gate. Historical Antman data remains usable through exact reconstruction, but new post-push data must not depend on reopening the initial XML.

**First target-box smoke attempt (jobs `186716`/`186717`).** Both jobs exited before model construction because `env.ilab.sh` replaced the explicitly exported Sage worktree path with the main Sage checkout, whose `EdgeCrossAttn` does not yet have `action_motion_dim`; no epoch, checkpoint, or GPU training ran. Commit `780b365` preserves an explicit `SAGE_REPO` across environment activation, and an exact launcher-environment import check resolves `EdgeCrossAttn` to Sage commit `09acfe3` with the expected constructor.

**Worker-start retry (jobs `186718`/`186719`).** Both correct models reached Lightning setup on `rlab2`, then the CS multiprocessing `spawn` workers lost their startup semaphore and the Python parents exited before the first batch. The dead polling shells were canceled after 59 seconds; neither job trained an epoch or wrote a checkpoint. Commit `c86a86a` makes the canonical launcher surface an early trainer death instead of polling its zombie indefinitely.

**Matched one-epoch smoke PASS (jobs `186720`/`186721`).** Baseline and treatment ran concurrently on two `rlab2` A100s with `num_workers=0`, each completing all 178,364 rows, checkpoint save/reload, and deployment-loader shape checks. Baseline completed in 4m34s with epoch-0 train/validation loss 1.6200/1.0502; treatment completed in 4m37s with 1.5943/1.2034. Both reloads had zero logit delta and both produced `(1,60,5,51)` logits and `(1,60,5)` values. The near-identical runtime makes 20 epochs approximately 92 minutes before final checks, so the full pair uses the proven worker-free path with a four-hour limit; the postcheck subset is diagnostic-only and does not alter training or the full validation monitor.

## Result + Verdict

All count and identity gates passed: easy/med/hard = 698/421/204, no valid ground-truth cells were missing from any live candidate pool, and the evaluator made zero push simulations.

| 1push tier | n | exact GT hit@1 | right contact, wrong depth | wrong contact | wrong-depth share of misses | depth accuracy given right contact |
|---|---:|---:|---:|---:|---:|---:|
| easy | 698 | 98.9% (690) | 0.4% (3) | 0.7% (5) | 37.5% (3/8) | 99.6% |
| med | 421 | 86.0% (362) | 2.9% (12) | 11.2% (47) | 20.3% (12/59) | 96.8% |
| hard | 204 | 41.2% (84) | 7.4% (15) | 51.5% (105) | 12.5% (15/120) | 84.8% |
| all | 1,323 | 85.9% (1,136) | 2.3% (30) | 11.9% (157) | 16.0% (30/187) | 97.4% |

| 1push tier | exact GT hit@1 | @5 | @10 | @20 | right-contact@1 |
|---|---:|---:|---:|---:|---:|
| easy | 98.9% | 99.9% | 99.9% | 100.0% | 99.3% |
| med | 86.0% | 95.5% | 98.6% | 99.8% | 88.8% |
| hard | 41.2% | 81.9% | 90.7% | 96.1% | 48.5% |

These are prediction-versus-saved-GT hits, not replayed-physics open rates. The existing Antman-5c deployment row remains easy/med/hard open@1 = 98.3/85.3/39.7; its evaluator executes the selected push and tests the live opening criterion, while this diagnostic intentionally performs no push and asks whether the predicted action is in the saved valid set.

**Phase-0 verdict: PASS the pre-registered architecture-test gate, with a narrow mechanism claim.** Medium has 20.3% right-contact/wrong-depth among misses, meeting the ≥20% gate; easy also crosses it but has only eight misses, while no tier breaches the <80% conditional-depth-accuracy gate. Depth is therefore a measurable error source worth representing explicitly, but it is not Antman-5c's main hard-tier defect: 105/120 hard misses (87.5%) choose the wrong contact. Expect the Antman A/B primarily to serve as a clean 1-push mechanism test and guardrail; the larger intended payoff remains ordering setup depths in 2-push search.

This Phase-0 diagnostic covers only canonical 1-push ground-truth comparison. No 2-push search evaluation was run because that would require forward simulations; both horizons remain mandatory for the eventual trained architecture verdict.

## Next

Run matched one-epoch baseline/treatment target-box smokes on CS `unlimited`, use the measured runtime to size the full jobs, then train the controlled Antman-5c pair. Evaluate both canonical 1-push and 2-push tiers before accepting or rejecting the architecture.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
