---
type: experiment
status: live
created: 2026-07-22
commit: c86a86a
metric: One-seed 1push A/B complete; depth-aware hard exact hit@1 47.1% vs 36.3% (+10.8 pp), easy/med preserved; 2push simulator verdict deferred
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

**Full one-seed A/B launched (jobs `186722`/`186723`, 2026-07-22).** NAMO launch commit `fd17abe`, Sage commit `09acfe3`; two concurrent `rlab2` A100s; baseline/treatment respectively; exact Antman-5c H5; seed 1; room split seed 0; 20 epochs; batch 256; learning rate `3e-4`; rank auxiliary weight/temperature `0.1/0.15`; `num_workers=0`; four-hour limit. Both jobs entered `RUNNING` on the intended node; outputs are under `/common/users/dm1487/scratch_namo/curriculum2/push_depth/antman5c_ab/full_seed1/{baseline,treatment}`.

**Treatment live-eval preflight (jobs `186745`–`186748`, 2026-07-22).** The first three one-episode attempts were useful zero-data failures: the isolated NAMO worktree lacked compiled bindings, the isolated Sage worktree lacked the main checkout's `fast_scorer` visualizer interface, and the selected epoch-4 best checkpoint was deleted by normal Lightning best-checkpoint rotation before its job began. NAMO commits `231ce96`/`4556bd4` preserve the Sage override and resolve shared bindings; Sage commit `912e276` adds the byte-equivalent fast-render interface; subsequent smokes use an immutable checkpoint copy. Job `186748` then passed on `rlab3` in 16 seconds: one canonical episode scored, the valid `(contact,depth)` ranked first, zero valid cells were missing, and no push simulation ran.

**Deferred best-checkpoint resolution smoke PASS (job `186749`).** Commit `9e22239` lets the live-eval launcher resolve the exact best-checkpoint path printed by a completed training log, avoiding Lightning filename rotation. The exact no-`CKPT` path completed one treatment episode on `rlab3` in 17 seconds with the valid action ranked first and zero missing valid cells. The full baseline/treatment prediction jobs may therefore be submitted with `afterok` dependencies on both training jobs and will start without manual polling only after both trainers succeed.

**Automatic prediction-only comparison queued (jobs `186751`/`186752`).** Baseline and treatment full-canonical evaluators are pending on `afterok:186722:186723`, request one `rlab3` A4000 each with a one-hour limit, and resolve their respective best checkpoint from the completed training log. Outputs are `/common/users/dm1487/scratch_namo/eval/push_depth/action_head_full_seed1/{baseline,treatment}.{json,jsonl}`. SLURM reports both dependencies unfulfilled as intended; neither evaluation can start before both trainers complete successfully.

**Full one-seed A/B training PASS (jobs `186722`/`186723`).** Baseline and treatment completed all 20 epochs with exit code 0 in 1h25m15s and 1h24m50s. Baseline selected `epoch012-val_loss0.6324.ckpt` (SHA-256 `bf8c1c1dac2faf44cb0cdfe74e3057b20ff0e3007da8b9d072ab9d5c508026a4`); treatment selected `epoch019-val_loss0.6371.ckpt` (SHA-256 `9f10d93828537f78f26f91df083845fc9ffca4a6488d0afe5ebbede290714d12`). Both checkpoints had zero two-reload logit delta and passed the deployment loader with `(1,60,5,51)` logits and `(1,60,5)` values. The separate pooled postcheck loss is not the Lightning validation monitor because it uses different mask normalization, so it is excluded from checkpoint selection and comparison.

**Automatic prediction-only comparison PASS (jobs `186751`/`186752`).** The dependency chain selected the exact checkpoints above and completed both canonical 1-push evaluations on `rlab3` with exit code 0 in 3m41s/3m42s. Each output has 1,323 unique episode identities in identical order, easy/med/hard = 698/421/204, zero missing valid actions, and `mode=live_canonical`; the evaluator contains no `env.step` call, so these jobs made predictions and compared them with saved ground truth without simulating pushes.

**Staged evaluation scope (user, 2026-07-22).** After training, run only the prediction-based canonical 1-push baseline/treatment comparison now. The 2-push simulator evaluation and final cross-horizon acceptance verdict are explicitly deferred.

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

### One-seed architecture A/B — prediction-only 1-push

The controlled comparison below trains both models from scratch on the same 178,364 Antman boards and changes only whether the head receives each complete push's nominal motion.

| 1push tier | n | baseline exact @1 | depth-aware exact @1 | delta | baseline exact @5 | depth-aware exact @5 | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 698 | 97.9% (683) | 98.1% (685) | +0.2 pp | 100.0% (698) | 99.9% (697) | -0.1 pp |
| med | 421 | 82.7% (348) | 82.7% (348) | 0.0 pp | 96.0% (404) | 95.5% (402) | -0.5 pp |
| hard | 204 | 36.3% (74) | 47.1% (96) | **+10.8 pp** | 76.0% (155) | 79.4% (162) | **+3.4 pp** |

| 1push tier | baseline right-contact/wrong-depth | depth-aware right-contact/wrong-depth | delta | baseline wrong-contact | depth-aware wrong-contact | delta |
|---|---:|---:|---:|---:|---:|---:|
| easy | 0.6% (4) | 0.9% (6) | +0.3 pp | 1.6% (11) | 1.0% (7) | -0.6 pp |
| med | 2.4% (10) | 3.3% (14) | +0.9 pp | 15.0% (63) | 14.0% (59) | -1.0 pp |
| hard | 5.4% (11) | 3.9% (8) | **-1.5 pp** | 58.3% (119) | 49.0% (100) | **-9.3 pp** |

**Interim 1-push verdict: promising, not final.** The depth-aware head stays within the pre-registered two-point guardrail on every 1-push tier and materially improves the hard tier: exact hit@1 rises by 22 episodes / 10.8 points, hit@5 rises by seven episodes / 3.4 points, and both wrong-contact and right-contact/wrong-depth errors fall. It does not improve every cell—medium/easy @5 regress by two/one episodes and right-contact/wrong-depth rises slightly there—so this one-seed result supports the mechanism but is not a universal win.

The pre-registered architecture decision still requires 2-push search cost and solve@k. Per the staged user scope, no 2-push simulator evaluation was launched, so the treatment is neither accepted nor rejected yet.

## Next

Stop here under the staged user scope. Do not launch the 2-push simulator evaluation until explicitly requested; when authorized, compare solve@2/@5/@10/@30 and simulator calls by easy/medium/hard before making the final architecture decision.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
