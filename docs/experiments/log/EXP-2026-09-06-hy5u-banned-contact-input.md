---
type: experiment
status: idea
created: 2026-09-06
updated: 2026-09-06
thread: ranker-as-policy
robot: car
commit:
metric:
tags: [experiment, hy5u, policy, blacklist, todo]
---

# HY5U with a banned-contact input

## Hypothesis

User-proposed idea, recorded from the 2026-09-06 discussion: give HY5U an explicit indicator of banned contacts so it can change the ordering of the remaining pushes and find useful alternatives. Test whether this improves over unchanged HY5U with the same external blacklist. A different contact or direction is useful only if it improves opening performance; novelty alone is not the objective.

## Status and scope

Deferred TODO, pinned on the experiment dashboard. The user requested a card for later, not implementation or a run. Use existing simulated training data with synthetic contact masks; no real-robot data is needed. Preserve the current execution blacklist and its clearing behavior. HY5U remains one ranker with no horizon or budget conditioning.

## Plan

Proposed design, to finalize before launch:

1. Fine-tune from HY5U with a binary banned/available embedding added to each contact token before inter-contact attention. Keep banned tokens visible as context, while excluding their actions from execution at every push distance. An empty mask should initially reproduce the original model, for example by zero-initializing the added embedding.
2. Generate synthetic contact masks on existing simulated training examples. Include empty masks and several nonempty mask sizes, sampled independently of action labels. Do not select masks specifically to leave a known winner. Preserve the existing room-level splits and episode identity; never treat untried actions as failures.
3. Apply the existing ranking supervision to remaining eligible, labeled actions. Preserve HY5U's existing auxiliary supervision unless an explicit control changes it. Rows with no eligible ranking comparisons contribute no such ranking term. Finalize and record the exact loss treatment and mask distribution before training.
4. Compare three arms: unchanged HY5U with external masks; HY5U fine-tuned with the same masked ranking examples but no mask input; and HY5U fine-tuned with the explicit mask input. All arms receive identical candidate exclusions. The second arm separates the effect of extra training from the information supplied by the new input. Include uniform random ordering under the same exclusions as the project baseline.
5. First check paired masked-state ordering and empty-mask behavior. Then evaluate simulated policy rollouts, holding the local target fixed and using the same ban/update rules and push allowance in every arm. Compare opening rates at matched push counts and repeated no-op attempts. Report easy/medium/hard separately for both canonical 1push and 2push populations, including empty-pool stops. Any grouped-boundary evaluation is a separate extension, not evidence supplied by the single-object benchmark.
6. Check canonical search behavior as a regression check before adopting the checkpoint as a replacement search ranker. Read the evaluated-artifact and eval-set registries before evaluation, reuse controls only when the full protocol matches, and register completed canonical artifacts.

Synthetic-mask construction requires no new simulator calls. Rollout evaluation does. Random masks encode unavailable choices, not physical failure evidence: this experiment cannot establish that the model understands jams. Failure-derived simulated masks would be a later experiment requiring actual jam/no-op outcomes; a push that does not open can still be a useful setup.

## Existing implementation anchors

- Contact representation and attention: `../sage_learning/src/model/dit/edge_crossattn.py`, `EdgeCrossAttn.forward`. The optional reachability embedding illustrates where another per-contact input could enter; banned and unreachable remain distinct inputs.
- Existing training loader: `python/namo/rl_loop/sage_ext/q2_dataset.py`, `Q2ValueDataset.__getitem__`, currently exposes context, contact coordinates, reachability, sampled targets and loss masks, but no blacklist history. Extend the existing training path after resolving the exact HY5U recipe from the registry.
- Runtime candidate exclusions: `python/namo/planners/opening/best_first_region_opening.py`, `_BlacklistedEdgeFilter`. Preserve the external hard exclusion even when the model receives the mask.
- Policy decision: `python/namo/planners/opening/best_first_search.py`, `run_reactive`. The new mask must reach the scorer as well as candidate filtering; filtering alone leaves the network unaware of bans.

## Run

Not started. Before launching: finalize commands, masks, losses, checkpoints and acceptance thresholds; follow the compute-resources and scaled-run skills; commit the implementation and run plan.

## Result + Verdict

No results. The hypothesis is untested. Adoption requires better ordering/opening performance than external masking alone, with acceptable empty-mask behavior; learning to ignore the input is a possible outcome.

## Next

- [ ] Resume when policy recovery training is prioritized; inspect the exact HY5U training recipe and implement the smallest controlled comparison above.

## Discussion

User constraints from 2026-09-06: no real-robot data; keep the current blacklist behavior; record this as a TODO for later.
