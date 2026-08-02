---
type: experiment
status: done
created: 2026-08-02
commit: none; retrospective analysis of existing artifacts
metric: Hard tail reaches correct setup boards but buries finishers at ranks 14-57; flat cross-board scheduling is the main residual mechanism
thread: region_opening
robot: car
parent: EXP-2026-07-29-post-pruning-canonical-search
related: [EXP-2026-07-24-failure-discount-search, EXP-2026-08-02-board-live-head, EXP-2026-08-02-depth-token-push-motion]
tags: [experiment, diagnosis, search, ranking, 2push, hard-tail, cross-board]
---
# Why expensive medium and hard 2-push search still fails

**Read [docs/problem_and_approach.md](../../problem_and_approach.md) first.** This audit diagnoses a ranker inside a simulator-verified search; a low-ranked correct push is delayed, not declared impossible.

## Question

Why do the expensive medium 2-push episodes and the hard budget failures consume hundreds of simulator calls: is the correct setup absent from the model's top ranks, does search fail to revisit it, is the finisher buried after the setup, or is simulator execution itself failing?

## Scope and identity

All population joins use the exact episode key `(realpath(xml), target object, goal region)`, never `xml` alone; difficulty comes from the registered per-episode exhaustive-GT divisions.

The canonical 2-push search view contains 1,012 episodes: 385 easy, 488 medium, 137 hard, and two unknown. This audit concentrates on medium and hard because all easy episodes solve by the 900-call budget and the expensive tail lives in the other two tiers.

No 1-push evaluation was rerun because this is a read-only diagnosis of existing 2-push search rather than a model or search-policy change; the paired canonical 1-push result remains in [RESULTS.md](../RESULTS.md).

The deployed search source is `/common/users/dm1487/scratch_namo/eval/postprune_hmax2/raw/model_2push/shard_*.jsonl`, its aggregate cross-check is `/common/users/dm1487/scratch_namo/eval/postprune_hmax2/final35/agg_model.json`, the tier registry is `/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/pure2push_gt_divisions_search_eval.json`, and the exhaustive action/tree truth is `/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/testset_gt_plus35.h5`.

## Method

The analysis has two evidence levels that must not be conflated.

First, the full canonical population was screened for solve cost and exhaustive-GT setup/finisher rank. A genuine setup is an exact root `value_target=0.9`; its child board is live when that board contains an exact finishing opener.

Second, full clean traces were collected with confidence discount disabled for all 24 medium episodes whose deployed cost exceeded 100 calls and all eight hard episodes that failed at the 900-call budget. The trace key is `(realpath(xml), object_id, region)` and one visit means one contiguous run of pops from a child board.

The 24 medium traces are the entire expensive-medium tail, 24/488 = 4.9%. The eight hard traces are all budget-900 failures, 8/137 = 5.8% of hard, but only 8/34 = 23.5% of hard episodes costing over 100 calls; 26 expensive-but-solved hard episodes were not traced.

Therefore the completed 32 traces are a worst-tail mechanism probe, not a representative sample of medium or hard. The correct future expensive panel is all 58 episodes costing over 100 calls—24 medium plus 34 hard—together with matched cheap controls.

## Full-population result

| tier | episodes | solved @900 | median calls | p90 calls | cost >100 |
|---|---:|---:|---:|---:|---:|
| medium | 488 | 487 (99.8%) | 5 | 57 | 24 |
| hard | 137 | 129 (94.2%) | 20 | about 316 | 34 |

All 137 hard episodes solve under the unchanged search by 3,831 calls. The residual problem is therefore inefficient ordering under a useful budget, not missing candidates or intrinsic unsolvability.

| tier | usable exhaustive GT | setup hit@1 | setup hit@5 | finisher hit@1 | finisher hit@5 |
|---|---:|---:|---:|---:|---:|
| medium | 488/488 | 55.1% | 80.7% | 65.6% | 87.7% |
| hard | 118/137 | 21.2% | 44.9% | 53.4% | 85.6% |

Setup rank correlates with live search cost: Spearman 0.626 on medium and 0.455 on hard. Hard episodes also have far fewer genuine setups—median one, p75 two, p90 three—than medium, whose median is eight, so one root-ordering error is much harder to recover from.

The population-level first bottleneck is sparse hard setup ordering. The expensive residual tail adds a different failure: after search reaches a correct setup board, its finisher can be far down that board's list.

## What the 32 clean traces show

| trace statistic | expensive medium, n=24 | hard @900 failures, n=8 |
|---|---:|---:|
| usable exhaustive GT | 24 | 7 |
| solved without confidence discount | 22 | 3 |
| setup rank median | 8 | 3 |
| setup hit@1 / @5 | 25.0 / 41.7% | 14.3 / 57.1% |
| finisher rank median | 9.5 | 33 |
| finisher hit@1 / @5 | 12.5 / 41.7% | 14.3 / 28.6% |
| preferred setup exposed | 23/24 | 7/7 |
| preferred board popped | 21/24 | 7/7 |
| preferred finisher actually popped | 11/24 | 2/7 |
| median root / child pops | 25.5 / 114 | 39 / 844 |
| median distinct child boards popped | 18.5 | 34 |
| median probes on preferred board, if popped | 7 | 19 |
| median visits to preferred board, if popped | 2 | 13 |

Expensive medium is mixed: 14/24 have the best genuine setup below rank five, 14/24 have the preferred board's finisher below rank five, and 8/24 have both problems. Search usually finds a real setup, but spreads its child budget across a median 18.5 boards and actually pops the preferred finisher in fewer than half the episodes.

The worst hard failures are more specific. Every usable-GT trace exposes and pops a genuine setup board, but the unsolved cases' finishers rank 14–57; four such cases reach the correct setup at calls 3, 19, 57, and 331, then give that board only 47, 7, 13, and 19 probes while its best finisher ranks 50, 14, 50, and 57.

This rejects the simple story that search becomes stuck on one post-setup state and never switches. It switches and revisits too diffusely: the correct hard board is revisited a median 13 times, yet the scheduler cannot decide that it deserves a deep enough block of finish probes.

The detailed machine-readable analysis is `/common/users/dm1487/scratch_namo/eval/failure_audit_20260802/off_control_gt_trace_analysis.json`; the selected episode keys are `medium_gt100_key.json`, `hard900_key.json`, and `tail32_key.json` in the same directory.

## Why the flat queue amplifies ranking errors

The deployed implementation is one global heap in [`eval_bestfirst.py`](../../../scripts/sandbox/eval_bestfirst.py). Root pushes enter with their own action score; simulating one root push can create a child board with roughly 70–80 reachable finish candidates; under `combine=q`, every child candidate is then ranked only by its action score on the child board.

The child starts with board weight `w0=1`, and the parent setup/path score is discarded. A root setup at rank ten is therefore not approximately ten simulator calls away: the nine earlier roots can inject hundreds of child actions above it.

| trace tier | root candidates | spawned child boards | child candidates | candidates per child board |
|---|---:|---:|---:|---:|
| medium | 2,110 | 645 | 49,865 | 77.3 |
| hard | 720 | 265 | 18,950 | 71.5 |

The resulting work is overwhelmingly post-setup: 5,426/6,127 = 88.6% of medium trace pops and 5,074/5,375 = 94.4% of hard trace pops occur on child boards.

The training objective does not directly supervise this deployment comparison. The listwise loss in [`train_q2_rankaux.py`](../../../scripts/rl_loop/train_q2_rankaux.py) ranks the 300 actions inside each H5 row independently; deployment compares maxima from many different boards in one heap. The distributional value loss and one-sided ceilings remain appropriate, but they do not guarantee that scores are comparable across boards.

This is the central train/deploy mismatch: action ordering within a board is learned directly, while board selection across the search frontier is an accidental consequence of action-score magnitudes.

## Confidence discount and fixed patience

Removing confidence discount solves three of the eight deployed hard misses, so the discount contributes to this selected tail. That does not justify removing it globally: the prior full canonical experiment improved average cost and slightly improved the solve ceiling.

| clean 32-case search | medium solved / mean calls | hard solved / mean calls |
|---|---:|---:|
| no discount | 22/24 / 256.5 | 3/8 / 671.9 |
| demote every 2 misses | 22/24 / 259.4 | 2/8 / 892.4 |
| demote every 3 misses | 22/24 / 239.0 | 0/8 / 900.0 |
| demote every 5 misses | 22/24 / 241.9 | 2/8 / 824.6 |

The tested rule “allow K misses, then multiply the board by γ” is rejected on the hard tail. It benches live and dead boards using the same evidence and can demote the correct board before reaching a rank-14–57 finisher.

The inference sigmoid is monotone, so it cannot change q-only action ordering, but it compresses an approximately `[0.01,0.99]` HL-Gauss expectation into `[0.50,0.73]`; confidence-dependent magnitude logic should therefore not be interpreted as calibrated confidence without a separate validation.

## Simulator and branch waste

Exact no-motion outcomes account for 1,719/6,127 = 28.1% of expensive-medium simulations and 936/5,375 = 17.4% of hard simulations. Every explicit controller failure in these traces leaves both robot and object unmoved.

This supports a feasibility/collision auxiliary target, but it is secondary on the hard tail: about 82.8% of hard calls execute and move the object yet still do not open the region.

No-op child spawning and same-edge deeper-jam expansion were already disabled, so the old duplicate/no-op bug is not the main remaining mechanism. Child boards collapse by roughly 35–40% when clustered only by target-object pose at 5 mm/5 degrees, but pose similarity does not prove search-state equivalence; deduplication is an unverified hypothesis, not a recommendation.

## Exhaustive-GT integrity limit

Nineteen of the 137 registered hard episodes have zero genuine setups in the current canonical H5 despite succeeding in live search, leaving 118 hard episodes usable for exact setup/finisher ranks. One source exhaustive artifact even contains a verified two-push chain that disappears in the later H5 rerun.

The population rank percentages are therefore reported only on the 118 usable hard episodes, and individual-case attribution requires live replay before calling one H5 action the definitive winner. Resolving these 19 episodes is a measurement-integrity task separate from improving the ranker.

## Diagnosis, ranked

1. Sparse hard setup ordering is the full-population weakness.
2. Each wrong setup causes multiplicative branch amplification by injecting roughly 70–80 child actions.
3. Search has no learned cross-board liveness signal and discards the parent setup/path score under `combine=q`.
4. Rare finisher ranks of 14–57 dominate the remaining hard budget failures after the correct setup is reached.
5. Collision and no-motion waste consume 17–28% of calls but do not explain most hard failures.
6. Current exhaustive GT disagrees with live search on 19 hard episodes and limits exact attribution.
7. Near-duplicate child expansion may add waste, but safe equivalence is not yet established.

## Consequences for the next experiments

**Search first:** replace the flat heap with a two-level scheduler: a root/setup queue, a board queue, and an action queue inside each board. Give boards increasing probe tranches such as 2, 4, 8, 16, and 32, guarantee continued root exploration, and guarantee later board revisits; yielding a board must not permanently γ-bury it.

**Cross-board supervision:** proceed with the narrow [`one_step_live` child-board head](EXP-2026-08-02-board-live-head.md): verified live when a moved child contains an immediate finish, verified dead only after exhaustive checking, and censored/unknown masked. The existing deployed H5 already supplies 22,271 verified-live and 19,282 verified-dead moved children.

**Data:** use targeted exploratory DAgger rather than pure top-k imitation. Mix model top-k, diverse/random actions, uncertainty or disagreement, and a fixed exhaustive-audit fraction; prioritize setup rank above 5/10, finisher rank above 10/20, and episodes costing over 100 calls; never turn a capped miss into a hard negative.

**Architecture:** continue the five-depth local-attention A/B as an orthogonal test of wrong-depth ordering, but do not expect added capacity alone to fix the cross-board scheduling mismatch.

**Evaluation:** use the complete 58-case expensive panel plus matched cheap controls for mechanism checks, then require the full registered 1-push and 2-push success-versus-simulator-call curves before promotion.

## Verdict

The residual 2-push tail is not one failure and is not primarily lack of solvability. Across hard episodes, genuine setups are sparse and often ranked late; after a correct setup is reached, the flat global queue spreads most of its budget across many child boards and the remaining catastrophic misses have finishers buried at ranks 14–57.

The highest-value intervention is board-aware search allocation plus explicit cross-board liveness supervision. Targeted exploratory data and the depth-token architecture are useful follow-ups; indiscriminate data/model scaling under the unchanged within-board objective is not the first lever.
