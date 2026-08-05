---
status: live
tags: [paper, icra27, region-opening, ranking, search]
updated: 2026-08-02
---

# ICRA 2027 paper plan — learned ranking for physics-verified region opening

> This is the living paper brief: the claim, the evidence needed to support it, the comparisons, and the boundary between what we have shown and what we must not overclaim. Canonical problem framing lives in [problem_and_approach.md](problem_and_approach.md); verified results live in [experiments/RESULTS.md](experiments/RESULTS.md); evaluated artifacts live in the [model registry](experiments/horizon_q_model_registry.md) and [evaluation-set registry](experiments/eval_set_registry.md).

## One-sentence thesis

Given a local NAMO keyhole and a fixed library of controller-executable pushes, learning the ordering of those pushes lets physics-verified search find an opening in far fewer simulator calls than uninformed or geometric ordering.

## Exact problem scope

An episode is `(robot region, neighbouring goal region, one blocking object)`. The blocking object and target region are given. The task is to push that same object until at least 20 of 100 fixed samples in the goal region become robot-reachable.

A 1-push episode opens after one primitive. A 2-push episode requires a setup push followed by a finishing push on the same object. This is one local region transition, not multi-region navigation, object selection, or full NAMO.

The research objective is not calibrated opening probability and not minimum path length. It is the solve-rate-versus-simulator-calls curve: use a learned search heuristic to reach the first simulator-verified opening sooner.

## Method in one paragraph

At each board, the system enumerates the reachable contact-by-depth push primitives, scores them jointly with a scene-conditioned ranker, and inserts them into a best-first search. The controller executes the selected primitive in MuJoCo, the wavefront verifier checks the 20/100 opening condition, and failed pushes create post-push boards whose finishing actions are ranked by the same model. Search currently stops at depth two for the canonical 1push/2push evaluation.

## Intended contributions

1. **Problem/interface:** formulate controller-grounded keyhole opening as learning to rank executable push skills for minimum physics-verifier cost.
2. **Structured ranker:** score contact and push-depth choices using local visual evidence, whole-scene context, and interaction among candidate contacts.
3. **Learning under incomplete search:** combine distributional value learning, one-sided ceiling supervision for censored actions, and direct listwise ranking only where the label order is known.
4. **Physics-verified evaluation:** measure success versus simulator calls on held-out 1push and genuine 2push episodes, split by easy/medium/hard, with exhaustive ground truth used only as an evaluation luxury.
5. **Failure analysis:** explain whether residual expensive cases arise from setup ranking, finisher ranking, post-setup board allocation, controller failures, or action-library coverage.

## What is and is not claimed about action primitives

The current action library contains up to 60 perimeter contacts × 5 push durations. Reachability removes contacts the robot cannot approach. A primitive specifies what the controller attempts—contact and duration—not the resulting object pose. Contact dynamics, rotation, early stopping, collision, and jamming remain outcomes of controller execution in physics.

Motion primitives are not claimed as novel. Discrete actions are not claimed to be universally better than continuous actions. The paper adopts a controller-grounded skill interface so every search method receives the same meaningful, executable choices and every attempted choice has an exact physics outcome.

The exact numbers 60 and 5 are engineering hyperparameters. They must not receive a retrospective theoretical story. They should be selected or defended through the saturation study below.

**Paper wording:** “We adopt a controller-grounded discrete action representation; we do not claim it is superior to continuous control. We validate that its selected resolution preserves the solution coverage of substantially denser and continuously sampled action sets while keeping the search branching factor manageable.”

## Primitive-resolution saturation study

### Question

Does the current 60×5 library capture nearly all openings available to the same controller under a denser or continuously parameterized contact-and-duration space?

### Population

Use all immediate-neighbour episodes extracted from the held-out canonical room pool, keyed by `(room, blocking object, goal region)`. Do not select only `onepush` or `pure2push`: those populations were defined using the current primitive library. Report current-1push, current-2push, currently-unsolved, and no-reachable-contact strata separately.

### Nested action families

| family | contacts | duration resolution/range | purpose |
|---|---:|---|---|
| `G0` | current 60 | current five | existing system |
| `G-contact` | `G0` plus interstitial contacts | current range | test missed contact locations |
| `G-duration` | current contacts | current durations plus half-steps | test missed push lengths |
| `G-dense` | both refinements | same maximum range | dense discrete reference |
| `G-long` | dense contacts | longer maximum range | test range separately from resolution |
| `G-continuous` | continuous face coordinate | continuous duration | randomized space-filling sanity check |

Longer pushes must be analyzed separately from finer duration spacing: the first changes the physical range and can collapse a current 2-push problem into one long push, while the second tests discretization within the existing range.

### Execution protocol

Use a coverage-only parametric action `(face, position along face, push duration)` that calls the existing controller and verifier. First prove that the parameter values corresponding to `G0` reproduce a fixed panel of current 60×5 outcomes exactly. Then evaluate 1-push coverage and the three missing 2-push combinations: existing setup→new finisher, new setup→existing finisher, and new setup→new finisher. Reuse registered existing→existing ground truth rather than repeating it.

### Measurements

- Fraction of topology-defined episodes solved by each action family at depth one and depth two.
- Retention `solved(G0) / solved(G-dense ∪ G-continuous)`.
- Marginal rescues from contact refinement, duration refinement, and longer range.
- Current 2-push episodes that become 1-push under longer actions.
- Reachable candidates and finite search-tree size per board.
- Robustness of successful actions to small contact and duration perturbations.
- Simulator calls plus controller ticks, push distance, and same-hardware wall time when duration ranges differ.

### Decision rule

Before running the full audit, preregister the acceptable coverage loss and choose the smallest action family on the coverage-versus-branching Pareto frontier. A provisional bar is at least 98% dense-reference retention overall and at least 95% within every fixed tier; freeze or revise these thresholds before seeing the result.

### Interpretation

If 60×5 lies at the saturation knee, retain it and state that the discretization is empirically adequate. If a denser family rescues a material fraction, change the library or narrow the claim to ranking the fixed library. If successful actions form broad neighbouring clusters, a later coarse-to-fine representation may be worthwhile; do not add that hierarchy to this paper unless it improves the primary solve-versus-cost result.

## Core canonical comparison

All canonical methods must receive the identical episode, candidate library, controller, verifier, depth limit, no-op deduplication, jam pruning, and simulator budget. A search-policy change must be applied symmetrically to every ranking prior or reported as a separate ablation.

| method | scientific question | current status |
|---|---|---|
| Uniform-random ranker + best-first | Does informed ordering help at all? | Complete, three seeds, registered |
| Geometric keyhole ranker + best-first | Does learning beat classical geometry? | Geometric strategy exists; canonical adapter and evaluation missing |
| Independent learned action scorer + best-first | Does joint candidate reasoning matter? | Missing |
| Immediate-opening classifier + best-first | Why value setups instead of predicting only direct openings? | Missing loss/label ablation |
| Full learned ranker + best-first | Proposed method | Complete for current deployed checkpoint; clean controls registered |
| Exhaustive-GT oracle | What is the minimum possible search cost and remaining headroom? | Evaluation-only diagnostic available on canonical GT coverage |

Random alone is not an adequate paper comparison. The minimum defensible submission adds the geometric ranker and a simple learned scorer under the identical search interface.

## External comparison position

There is currently no executed external-method comparison. Existing systems do not expose the same decision interface: classical and modern NAMO systems commonly choose objects and robot trajectories, output continuous controls, optimize path/contact costs, or solve full navigation rather than rank a fixed push library by simulator calls to local opening.

The closest classical comparison is a faithfully documented Stilman-style geometric keyhole manipulation search adapted to the identical candidate library and verifier. It should be presented as an adapted external method only if the implementation-to-paper mapping is explicit; otherwise call it a geometric baseline inspired by classical keyhole search.

Yao et al.’s local learned NAMO policy and SVG-MPPI are close task-level related work but not drop-in canonical baselines because their actions, objectives, and accounting units differ. Bench-Push Maze is the strongest optional external validation surface: embed the local opener in a full navigation stack and compare on Bench-Push’s task metrics against its supplied baselines. Keep this in a separate external-validation table, never mix its control timesteps or wall time with the canonical simulator-call axis.

## Evaluation protocol

- Report every result by horizon (`1push`, `2push`) and fixed difficulty (`easy`, `medium`, `hard`).
- Primary plot: cumulative solve rate versus physics simulator calls.
- Report solve@tight budgets, solve@30, solve@900, and simulator calls among solved episodes.
- Random is a seed mean with sample standard deviation; learned models require paired seeds for architecture or loss claims.
- Wall time is compared only on identical pinned hardware; controller ticks and physical push distance accompany experiments that change primitive duration.
- Offline AUC and hit@k are diagnostics, not substitutes for physics-verified solve-versus-cost curves.
- Exhaustive ground truth remains evaluation-only; training uses sampled/model-guided experience and preserves unknown or right-censored actions rather than stamping false negatives.

## Evidence already available

The registered canonical evaluation already shows that the deployed learned ordering beats three-seed random ordering across every fixed 1push/2push difficulty tier under the same search. The strongest result is tight-budget efficiency, especially on hard 2push; the detailed frozen numbers and plots are in [RESULTS.md](experiments/RESULTS.md) and their exact artifacts are in the [model registry](experiments/horizon_q_model_registry.md).

The exhaustive-GT ranking panel, hard-tail search, clean no-discount controls, failure audit, and depth-token negative result already support the core diagnosis: the learned heuristic amortizes search, residual failures are mainly ranking/allocation failures rather than absence of a shallow solution, and architectural novelty must be justified by paired end-to-end search rather than validation loss alone.

## Claims we can defend now

- The scoped local problem is the classical NAMO keyhole subproblem for one specified blocker and one neighbouring region transition.
- A fixed controller-grounded primitive library plus physics verifier gives a well-defined search problem.
- Learned ordering finds verified solutions in substantially fewer simulator calls than uniform-random ordering across the canonical tiers.
- Setup pushes require future-aware value/ranking supervision because they do not immediately satisfy the opening condition.
- Exhaustive ground truth is feasible as a held-out diagnostic but not as the scalable training recipe.

## Claims not yet supported

- Discrete primitives are better than continuous control.
- The current 60×5 resolution is sufficient for continuous region opening.
- The method solves full NAMO, chooses the blocking object, or navigates through multiple regions.
- State-of-the-art performance against external task-level NAMO systems.
- Real-robot region-opening performance.
- A new model architecture is responsible for the full gain without matched simple learned and architectural baselines.

## Submission-critical work

### P0 — required before a defensible submission

1. Adapt and run the geometric keyhole ranker on the full canonical 1push and 2push sets under the clean identical protocol.
2. Run the primitive-resolution saturation pilot, calibrate its cost, freeze the population and decision rule, then complete the paired audit.
3. Train/evaluate an independent learned scorer or similarly strong simple learned baseline on the same data and search.
4. Freeze the primary search policy and report the learned/random/geometric comparison under that one policy; keep confidence discount or bounded-patience variants as symmetric search ablations.
5. Convert the failure audit into a compact mechanism table tied to proposed remedies.

### P1 — strongly strengthens the paper

1. Add the immediate-opening-label ablation to isolate why setup-value learning matters.
2. Complete identical-hardware wall-time measurements for the final methods.
3. Add a Bench-Push Maze transfer experiment or another genuinely shared external task.
4. Add controller/action perturbation robustness and representative qualitative videos.

## Planned figures and tables

1. **Problem figure:** room topology → robot region, neighbouring goal region, one blocker → 1push or setup→finish opening.
2. **Method figure:** scene/contact encoding → joint ranker → best-first physics verification → post-push board.
3. **Headline curves:** success versus simulator calls, 1push and 2push, each split easy/medium/hard.
4. **Baseline table:** random, geometry, simple learned, full model, oracle.
5. **Action-space figure:** opening coverage versus reachable branching factor for coarse, current, dense, and continuous action families.
6. **Failure table:** root setup rank, conditional finisher rank, board revisit/allocation, jam/no-op, and no-GT-chain cases.
7. **Optional external table:** shared-task Bench-Push metrics, separate from canonical search-cost results.

## Paper outline

1. Introduction: physics can verify a push, but verification is expensive; the missing capability is ordering.
2. Related work: NAMO keyholes, physics-guided manipulation search, learned search heuristics, object-centric push action representations.
3. Problem formulation: local region opening, primitive action library, success verifier, simulator-call objective.
4. Method: structured candidate encoding, distributional/censored/listwise learning, best-first verified search.
5. Experimental setup: canonical held-out episodes, fixed tiers, baselines, artifact protocol, primitive saturation.
6. Results: solve-versus-cost, ranking diagnostics, ablations, action coverage, failure mechanisms.
7. External validation or limitations: what differs from full NAMO and continuous-control systems.
8. Conclusion: learned ordering turns expensive physics search into a small number of targeted verification calls.

## Reviewer-facing summary

“This work does not propose a new push primitive or claim that discretization dominates continuous control. It studies a specific planning bottleneck: when a robot already has a library of executable contact skills and physics is an accurate but expensive verifier, which skill should search try first? We validate the action-library resolution against denser and continuously sampled alternatives, hold the controller and verifier fixed across methods, and show that learned ordering—not a privileged action set—produces the simulator-efficiency gain.”
