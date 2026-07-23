---
type: experiment
status: live
created: 2026-07-22
commit: 17b4df3
metric: Correct crop-relative-motion three-seed repair pending; absolute-final-yaw treatment failed hard exact @1 40.4→37.2 (-3.1 pp)
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

### Phase 3 — corrected representation and fresh three-seed confirmation

_(user, 2026-07-22)_ Correct the action feature to the image-aligned final pose, then train three fresh corrected treatments and three fresh baselines simultaneously on six cluster GPUs.

The crop generator translates the target object to the image center but does not rotate the crop: world X/Y remain the image axes, world Y increases with image row, and the 0.5 m crop spans ±0.25 m around the object. The corrected feature for each complete push is therefore `(2*world_dx/0.5m, 2*world_dy/0.5m, sin(theta+dtheta), cos(theta+dtheta))`: the proposed final center in the same `[-1,1]` coordinates as the image/contact positions plus the proposed final yaw without an angle discontinuity.

This correction remains a cheap late-fusion treatment. The image and 60 contact tokens complete scene/contact attention first; the model then expands to 60×5 actions, adds each action's four-number final-pose embedding, and applies the shared value head. Action-specific attention or a rasterized final footprint is a separate architecture experiment and is not mixed into this test.

Preserve the legacy three-number `(world_dx/0.5m, world_dy/0.5m, dtheta/pi)` path when loading its existing checkpoint, but make every newly trained treatment use the corrected four-number feature. Focused gates must verify all three shape families, crop-coordinate equality, wrapped-angle-safe final orientation, an actual H5 forward/backward, and successful legacy-checkpoint forward loading before cluster smoke.

Train a fresh `3 seeds × {baseline, corrected}` A/B: seeds 1/2/3, room split seed 0, the same 178,364-row H5, 20 epochs, batch 256, learning rate `3e-4`, rank auxiliary weight/temperature `0.1/0.15`, and `num_workers=0`. Run all six jobs concurrently on six identical GPUs so neither paired deltas nor seed variance are mixed with hardware changes.

After training, run the same zero-simulation canonical 1-push comparison for all six checkpoints and report paired baseline→corrected deltas by easy/medium/hard. Strong confirmation means hard exact hit@1 improves in all three seeds, the three-seed mean hard wrong-contact rate falls, and easy/medium exact hit@1 stay within the existing two-point guardrail in every seed. Two positive hard seeds with a positive mean is partial confirmation; one or zero positive hard seeds is a failure to replicate. Right-contact/wrong-depth remains a reported mechanism diagnostic rather than a gate because the legacy seed-1 gain was dominated by fewer wrong contacts.

This fresh A/B does not authorize 2-push simulator evaluation; the final cross-horizon architecture verdict remains deferred.

### Phase 4 — repair the motion semantics and repeat three treatment seeds

_(user, 2026-07-22)_ The Phase-3 treatment mistakenly injected absolute final yaw even though the intended signal is the push's motion relative to the centered object; correct that representation and train three seeds.

For every `(contact, depth)` action, use exactly `(2*world_dx/0.5m, 2*world_dy/0.5m, dtheta/pi)`. The first two values are the proposed center displacement in the crop's image-aligned `[-1,1]` coordinates, so a displacement to a crop boundary has magnitude one; the third is the primitive's relative rotation, not the object's absolute yaw and not `theta+dtheta`.

Keep the cheap late-fusion location and the rest of the Antman-5c recipe unchanged. Tag new checkpoints with the named encoding `crop_relative` because this corrected feature and the historical legacy feature are both three numbers; untagged three-number checkpoints must continue to load as legacy, while untagged four-number checkpoints continue to load as Phase-3 final pose.

Train treatment seeds 1/2/3 concurrently on three identical `rlab3` A4000 GPUs using the same 178,364-row Antman-5c H5, room split seed 0, 20 epochs, batch 256, learning rate `3e-4`, rank auxiliary weight/temperature `0.1/0.15`, and `num_workers=0`. Reuse the fresh Phase-3 baseline seeds because they were trained moments earlier with exactly this data, recipe, seed set, and hardware; retraining identical controls would add no new comparison information.

Before the full launch, require focused unit tests, an actual H5 forward/backward, preservation of legacy/final-pose checkpoint loading, and one full-epoch target-node smoke that saves a `crop_relative`-tagged checkpoint and reloads it through the deployment scorer. Then run the same zero-simulation canonical 1-push evaluation for all three treatments and the same paired three-seed comparator against the existing baseline JSONL identities. No D20, rasterized footprint treatment, 2-push search, or push simulation is authorized in this phase.

Also test a separate `crop_relative_sharp` treatment requested after this repair was specified. It applies eight-band Fourier features independently to normalized `(dx,dy,dtheta)` and adds a learned five-way depth embedding before the same additive late fusion. This mirrors the existing contact encoder's Fourier position plus learned edge identity without adding 300-token attention; it adds only 9,600 parameters to the small action projection in the controlled one-block construction. Train seeds 1/2/3 with the same recipe and hardware alongside the three plain-relative seeds, and report it as a separate arm against the same saved baselines.

## Run

**Phase-0 implementation (2026-07-22).** Commit `ddb18d2` extends `scripts/eval_scorer.py` in place with a zero-push live-canonical mode, adds the three-way category tests, and adds the CS `unlimited` launcher `scripts/slurm/eval_scorer_live.slurm`. Local compile + focused tests pass (2/2).

**CS smoke (job 186711, 2026-07-22).** The exact one-episode live path passed on `ilab1` in 15 seconds total and 0.367 seconds for scoring: zero `env.step` calls, one row written, no valid cells missing from the live candidate pool, and the labeled push ranked first. At that measured rate the full 1,323-episode diagnostic is approximately eight minutes of scoring, so one GPU job is sufficient.

**Canonical Phase-0 run (job 186712, commit `c131c8e`, 2026-07-22).** One `unlimited` GPU on `ilab1`; checkpoint `antman5c/checkpoints/epoch018-val_loss0.6276.ckpt`; 1,323/1,323 episodes completed in 2m38s with exit code 0 and 0.110 seconds/episode steady-state. Artifacts: `/common/users/dm1487/scratch_namo/eval/push_depth/full/antman5c_depth_diag.{json,jsonl}` and log `logs/a5cdepth_full_186712.out`.

**Legacy Phase-1 implementation frozen (2026-07-22).** NAMO commit `75c8c11` and Sage commit `09acfe3` implement the first cheap treatment: the existing 60 contact tokens complete attention unchanged, then each token is expanded to its five candidate depths, combined with that candidate's normalized nominal `(dx,dy,dtheta)`, and scored by one shared 51-bin value head. `NAMO_ACTION_MOTION=0` retains the original Antman-5c architecture and checkpoint keys; the historical `NAMO_ACTION_MOTION=1` run used this legacy three-number treatment.

**Corrected final-pose implementation frozen (commit `17b4df3`, 2026-07-22).** New treatments use the four-number image-aligned final pose from Phase 3; checkpoint-inferred feature dimensionality keeps the legacy three-number model evaluable. Eight focused tests pass, including all shape families, exact crop-coordinate equality, angle-wrap continuity, and scorer diagnostics. An actual Antman H5 row completes corrected `(1,60,5,51)` forward/backward with finite logits, and the legacy treatment checkpoint reloads and forwards with its inferred three-number feature. Parameter counts remain closely controlled: baseline 4,397,055 versus corrected treatment 4,396,083.

**Matched corrected target-node smoke PASS (jobs `186783`/`186784`).** Baseline and corrected treatment ran concurrently on separate `rlab3` A4000s with the exact full-run launcher, full 178,364-row epoch, seed 1, and `num_workers=0`; both completed with exit code 0 in 8m07s/8m06s. Baseline train/validation loss was 1.6363/1.0435 and corrected was 1.6461/1.4175. Both selected an epoch-0 checkpoint, had zero two-reload logit delta, and passed the deployment loader with `(1,60,5,51)` logits and `(1,60,5)` values. The measured rate calibrates 20 epochs at roughly 2.5–3.5 hours under six-job shared-NFS contention, so the full jobs use an eight-hour limit.

**Fresh corrected three-seed A/B launched simultaneously (jobs `186785`–`186790`).** All six jobs entered `RUNNING` together on six separate identical `rlab3` A4000s from pre-run card commit `5ee2880`. Seed 1 baseline/corrected are `186785`/`186786`, seed 2 are `186787`/`186788`, and seed 3 are `186789`/`186790`; every job uses the exact Phase-3 recipe and writes under `/common/users/dm1487/scratch_namo/curriculum2/push_depth/final_pose_3seed/full/seed{1,2,3}/{baseline,corrected}`.

**Automatic prediction-only evaluations queued (jobs `186791`–`186796`).** Each evaluator has an `afterok` dependency on its matching training job, resolves that run's printed best checkpoint, and writes `/common/users/dm1487/scratch_namo/eval/push_depth/final_pose_3seed/seed{1,2,3}/{baseline,corrected}.{json,jsonl}`. These are the same canonical 1-push saved-ground-truth comparisons as the one-seed result; no 2-push or forward-simulation job is queued.

**Three-seed verdict tooling frozen.** `scripts/compare_scorer_seeds.py` consumes exactly three paired `eval_scorer.py` outputs, verifies identical canonical JSONL episode identities across every input, rejects missing valid actions, reports exact hit@1/@5 plus wrong-contact and right-contact/wrong-depth by seed and tier, and applies the pre-registered strong/partial/failure gate without redefining episode matching or difficulty bins.

**Automatic verdict queued (job `186804`).** One lightweight CPU job depends `afterok` on all six prediction jobs and will write `/common/users/dm1487/scratch_namo/eval/push_depth/final_pose_3seed/comparison.md`; the full training→evaluation→paired-comparison chain now requires no active polling.

**Fresh corrected three-seed execution PASS (jobs `186785`–`186796`, verdict job `186804`).** All six 20-epoch trainers completed with exit code 0 in 2h21m–2h30m on separate identical `rlab3` A4000s; all six canonical prediction-only evaluators completed with exit code 0 in 3m31s–3m38s; the paired comparator completed in three seconds. Best checkpoints by seed were baseline `epoch017-val_loss0.6282`, `epoch015-val_loss0.6587`, `epoch013-val_loss0.6413` and corrected `epoch018-val_loss0.6620`, `epoch015-val_loss0.6604`, `epoch017-val_loss0.6444`. Every output used the same 1,323 canonical identities in the same order, easy/med/hard = 698/421/204, with zero valid actions missing from the candidate pool. No 2-push or forward-simulation job ran.

The existing Antman-5c H5 already stores `contact_px`, whose ordered rectangle samples recover the target object's current axes, size family, and yaw. The loader therefore constructs the exact active primitive table from the pinned `1x_car_d5_motion_primitives_15_{square,wide,tall}.dat` files without XML lookup or simulator calls. Focused tests cover all three shape families and rotation, the original Antman-5c checkpoint still loads through `eval_scorer`, and an actual H5 batch completes forward/loss/backward with finite values. The legacy three-number treatment's parameter count was also closely matched to baseline: 4,395,891 versus 4,397,055.

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

### One-seed legacy-motion architecture A/B — prediction-only 1-push

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

**Interim legacy-v1 1-push verdict: promising, not final.** The depth-aware head stays within the pre-registered two-point guardrail on every 1-push tier and materially improves the hard tier: exact hit@1 rises by 22 episodes / 10.8 points, hit@5 rises by seven episodes / 3.4 points, and both wrong-contact and right-contact/wrong-depth errors fall. It does not improve every cell—medium/easy @5 regress by two/one episodes and right-contact/wrong-depth rises slightly there—so this one-seed result supports the mechanism but is not a universal win.

### Corrected final-pose three-seed A/B — prediction-only 1-push

| tier | seed | n | baseline @1 | corrected @1 | delta | baseline @5 | corrected @5 | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 1 | 698 | 98.0% | 97.0% | -1.0 pp | 99.7% | 99.9% | +0.2 pp |
| easy | 2 | 698 | 96.1% | 97.7% | +1.6 pp | 99.3% | 99.7% | +0.4 pp |
| easy | 3 | 698 | 97.1% | 98.0% | +0.9 pp | 99.9% | 99.6% | -0.3 pp |
| easy | mean | 698 | 97.1% | 97.6% | +0.5 pp | 99.6% | 99.7% | +0.1 pp |
| med | 1 | 421 | 84.3% | 81.7% | -2.6 pp | 94.8% | 96.2% | +1.4 pp |
| med | 2 | 421 | 81.7% | 77.2% | -4.5 pp | 94.1% | 94.3% | +0.2 pp |
| med | 3 | 421 | 81.0% | 82.4% | +1.4 pp | 96.2% | 95.7% | -0.5 pp |
| med | mean | 421 | 82.3% | 80.4% | -1.9 pp | 95.0% | 95.4% | +0.4 pp |
| hard | 1 | 204 | 39.7% | 35.8% | -3.9 pp | 72.5% | 72.5% | 0.0 pp |
| hard | 2 | 204 | 39.7% | 37.7% | -2.0 pp | 69.6% | 72.5% | +2.9 pp |
| hard | 3 | 204 | 41.7% | 38.2% | -3.5 pp | 76.5% | 74.0% | -2.5 pp |
| hard | mean | 204 | 40.4% | 37.2% | **-3.1 pp** | 72.9% | 73.0% | +0.1 pp |

| tier | seed | baseline wrong contact | corrected wrong contact | delta | baseline right-contact/wrong-depth | corrected right-contact/wrong-depth | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 1 | 1.3% | 1.4% | +0.1 pp | 0.7% | 1.6% | +0.9 pp |
| easy | 2 | 2.3% | 1.0% | -1.3 pp | 1.6% | 1.3% | -0.3 pp |
| easy | 3 | 2.1% | 1.4% | -0.7 pp | 0.7% | 0.6% | -0.1 pp |
| easy | mean | 1.9% | 1.3% | -0.6 pp | 1.0% | 1.2% | +0.2 pp |
| med | 1 | 13.5% | 13.8% | +0.3 pp | 2.1% | 4.5% | +2.4 pp |
| med | 2 | 14.7% | 17.3% | +2.6 pp | 3.6% | 5.5% | +1.9 pp |
| med | 3 | 13.8% | 13.3% | -0.5 pp | 5.2% | 4.3% | -0.9 pp |
| med | mean | 14.0% | 14.8% | +0.8 pp | 3.6% | 4.8% | +1.1 pp |
| hard | 1 | 54.4% | 56.4% | +2.0 pp | 5.9% | 7.8% | +1.9 pp |
| hard | 2 | 53.4% | 54.4% | +1.0 pp | 6.9% | 7.8% | +0.9 pp |
| hard | 3 | 52.9% | 52.9% | 0.0 pp | 5.4% | 8.8% | +3.4 pp |
| hard | mean | 53.6% | 54.6% | **+1.0 pp** | 6.1% | 8.1% | **+2.1 pp** |

**Final staged verdict: FAILURE TO REPLICATE.** Hard exact hit@1 fell in all three seeds, by 3.9/2.0/3.5 points, while mean hard wrong-contact rose by 1.0 point and mean hard right-contact/wrong-depth rose by 2.1 points. The easy/medium guardrail also failed because medium fell by 2.6 and 4.5 points in seeds 1 and 2. Exact @5 was essentially flat on average in every tier, so the corrected representation usually leaves a valid push nearby but makes top-1 ordering worse. The earlier legacy one-seed gain is therefore not reliable evidence for this corrected late-fusion design; this experiment changes both the representation and the seed cohort, so it cannot isolate whether that earlier gain came from the legacy encoding or seed noise. Per the staged scope, this is a complete rejection on the canonical Antman 1-push test only; no claim is made about D20/setup data or 2-push search.

## Next

Keep the original Antman-5c head for now and close this corrected late-fusion arm. Do not run D20 or 2-push evaluation for this treatment. If action grounding is revisited, open a separate experiment for a materially different interaction—HACMan-style concatenation/gating, action-conditioned attention, or sampling scene features at the proposed final footprint—rather than extending this failed additive late-fusion test.

## Discussion

_(you ↔ Claude — newest at the bottom.)_
