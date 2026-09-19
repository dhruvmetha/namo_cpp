# PAVE external baselines: MORE and Interactive-FAR

Saved September 11, 2026, to continue this discussion in another chat. Repository: `/common/home/dm1487/robotics_research/ktamp/namo`. Checkout at handoff: `7fb501226cc4eeb2e44b45862371d0e294179be0`. This file records the recommendation and prior audit evidence. No baseline implementation, dependency installation, training, or evaluation was performed for this handoff.

## Recommendation

**Pursue both, with MORE first and Interactive-FAR's local solver second.** They answer different questions and would strengthen different claims in the paper. This is a recommendation to develop and pilot the comparisons; neither has a completed external-method result in the current manuscript plan.

| Comparison | Question it answers | Scope |
|---|---|---|
| MORE-inspired adaptation | Does PAVE reach a verified opening more efficiently than another method that learns push values from search? | Adapted learned search and return supervision on our local task. |
| Interactive-FAR local solver | Is PAVE competitive with a published geometric manipulation planner for opening a blocked passage? | Its local hybrid A* interaction search, adapted to a shared task and execution test. |

Keep the existing random and geometry comparisons and architecture/loss controls. If only one external baseline is feasible, prioritize MORE because it addresses the learned-search comparison directly and has an inspectable implementation. Interactive-FAR is especially useful for a broader claim of competitive NAMO planning efficiency. A narrow claim about improving action ordering over random does not by itself require either complete external system.

## Task and claims to preserve

Read [problem_and_approach.md](problem_and_approach.md) before reasoning about implementation or results. PAVE is a single ranker/search heuristic that orders simulator attempts. An episode fixes the robot region, the neighboring goal region, and the one blocking object between them. A two-push solution is a setup and finish on that same object that merge the same pair of regions. The ranker is not horizon-conditioned.

Use the existing opening verifier and the same initial goal-region samples for every method. The canonical problem document specifies success when at least 20% of the fixed target samples become reachable after a simulated push. Moving the object to a proposed pose does not establish execution success.

The current geometry baseline ranks our candidate endpoints by virtual target-region reachability, then verifies attempted pushes in physics. It does not reproduce a published classical planner or Interactive-FAR. Its equal-score queue policy also differs from model/random ordering: geometry prefers deeper chains. Preserve that qualification when citing existing results; a new score-only comparison needs a common tie policy. See [the manuscript baseline audits](icra27_paper.md#geometry-implementation-to-paper-audit-2026-09-06) and [the implemented ranker and queue](../python/namo/planners/opening/best_first_search.py).

## MORE adaptation

The primary reference is Huang, Guo, Boularias, and Yu, *Interleaving Monte Carlo Tree Search and Self-Supervised Learning for Object Retrieval in Clutter*, ICRA 2022. The method learns push values from search returns and uses learned values to guide subsequent search. See [the paper, Sections IV-B–D](https://arxiv.org/html/2202.01426v2) and [author code](https://github.com/arc-l/more). The existing [learned-method audit](icra27_paper.md#learned-method-baseline-audit-2026-09-06) records public revision `70402c001c30908e7a70b93f8de3c31abdd26fdb`, including branch selection, return backups, visits, transition caching, and return regression.

Preserve learned priors, sampled-return backups, visit counts, guided branch selection, and guided rollouts. Adapt the action representation to our reachable contact/duration library and replace grasp-specific reward and termination with verified region opening. Retain the same episode, controller, depth cap, no-op rule, and jam pruning as the matched PAVE experiment. Call the result a **MORE-inspired adaptation**, with each substitution documented.

Count actual physics attempts, including rollout attempts. Stop at the first verified opening wherever search finds it. Reuse saved states and observations so solution logging does not replay pushes or lose a success at the budget boundary. MCTS iterations are not simulator calls.

Training needs sampled returns and visitation evidence with action identities. The existing Q2 board labels are not automatically an MCTS training record. Whether older raw search records contain recoverable inputs remains unaudited. Do not invent visit counts, treat unknown actions as failures, or enumerate every push for training. Use bounded sampled experience from training rooms, separate acquisition cost from deployment cost, and tune on held-out development rooms.

Reusing PAVE's encoder is a reasonable controlled adaptation if stated explicitly. Training a regressor on PAVE's existing labels and placing it in the unchanged best-first queue would test a narrower learning control. An optional shared-queue scorer comparison can isolate ranking quality, alongside the adapted search comparison.

## Interactive-FAR adaptation

The primary reference is He et al., *Interactive-FAR: Interactive, Fast and Adaptable Routing for Navigation Among Movable Obstacles in Complex Unknown Environments*. [Section VI and Figure 5](https://arxiv.org/html/2404.07447#S6) describe local hybrid A* search, stable-pushing state transitions, and contact switching. Its polygon-width heuristic uses the minimum separation of parallel supporting lines; this differs from our endpoint reachable-fraction score. The extracted primary text is available in [papers/interactive-far/paper.txt](../papers/interactive-far/paper.txt).

The earlier read-only audit found that the [public repository](https://github.com/Bottle101/Interactive-FAR) provided a README with ROS Noetic/Docker demo instructions. The source and local solver interface inside the image were not inspected. Recheck availability before estimating integration effort; the audit did not establish that the solver is unavailable.

Compare its local interaction solver on the same supplied blocker and region pair. Preserve stable-pushing transitions and contact switching, then execute and verify its proposed motion through a documented adapter. Forcing its search onto PAVE's contact/duration library would change the method substantially. Transplanting only the width heuristic would be an inspired heuristic baseline, not an evaluation of Interactive-FAR's solver.

The local solver supports a local practical-efficiency claim. Integrating the complete Interactive-FAR navigation system becomes relevant if the manuscript claims superiority over full NAMO systems. Demonstrating PAVE inside a full NAMO stack does not alone require that larger comparison.

## Evaluation and implementation sequence

Use identical hardware for wall-clock comparisons. Include relevant preprocessing, local planning, inference, state management, execution verification, and replanning. Report success versus budget and time to the first verified opening. Do not place published Interactive-FAR path-search timing beside PAVE's simulator-inclusive planning time as if the measurements cover the same work.

Simulator-call comparisons are meaningful when methods share the same physics-attempt interface, as intended for the MORE adaptation. Interactive-FAR's analytic search expansions or short control segments are not equivalent to PAVE push simulations. Report those counts separately. Keep one-/two-push and easy/medium/hard labels as strata defined by PAVE's evaluation library; they are not universal difficulty measures or caps on Interactive-FAR's control segments.

1. Read the [model artifact registry](experiments/horizon_q_model_registry.md) and [evaluation-set registry](experiments/eval_set_registry.md) before choosing a comparison protocol. Reuse completed artifacts only when the full protocol matches.
2. Recheck MORE's source and specify the adapted search, reward, training records, and accounting. Inspect one- and two-push traces before a small held-out pilot. Advance based on a correct, competently trained implementation, regardless of whether the preliminary result favors PAVE.
3. Check Interactive-FAR's solver accessibility and define the execution adapter, contact-switch handling, and local start/goal mapping. Resolve these before committing to a full campaign.
4. Run small pilots before full evaluations. Follow the project's data-pipeline, compute, and scaled-run skills when the corresponding work begins; commit before runs. These audits have not established runtimes or resource requirements.
5. Keep the main ranking curves. For the discussed eight-page paper, a compact external-baseline table can report the two comparisons, with adaptation details elsewhere. Do not collapse the required difficulty and one-/two-push breakdowns into an aggregate-only result.

## Suggested prompt for the next chat

> Read `docs/problem_and_approach.md`, then `docs/pave_external_baselines_handoff_2026-09-11.md` and the linked baseline audits. Continue planning the MORE-inspired and Interactive-FAR local-solver comparisons, prioritizing MORE. First verify current implementation status, available source/interfaces, and reusable training records. Produce a concrete adaptation and pilot plan grounded in the code. Preserve the one-object local-opening task, faithful method mechanisms, and fair execution/cost accounting. No external baseline results have been established by this handoff.
