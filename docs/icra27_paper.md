---
status: live
tags: [paper, icra27, region-opening, ranking, search]
updated: 2026-09-06
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
| Geometry-inspired region-reachability ranker + best-first | Does learning beat this endpoint-geometry ordering? | Complete, corrected full campaign registered August 29; not an external-method reproduction; geometry-specific tie-breaking documented below |
| Independent-contact scorer + best-first (architecture ablation) | Does joint candidate reasoning matter? | Complete, three seeds; [independent-contact control](experiments/log/EXP-2026-08-31-hy5u-icra-ablations.md); not a separate external method |
| Immediate-opening classifier + best-first | Why value setups instead of predicting only direct openings? | Missing loss/label ablation |
| Full learned ranker + best-first | Proposed method | Complete for current deployed checkpoint; clean controls registered |
| Exhaustive-GT oracle | What is the minimum possible search cost and remaining headroom? | Evaluation-only diagnostic available on canonical GT coverage |

Random and corrected geometry provide completed non-ablation comparisons. The independent-contact experiment supplies the architecture control for joint candidate reasoning; a separate learned-method comparison needs a distinct specification rather than another copy of that ablation.

## External comparison position

There is currently no executed external-method comparison. Existing systems do not expose the same decision interface: classical and modern NAMO systems commonly choose objects and robot trajectories, output continuous controls, optimize path/contact costs, or solve full navigation rather than rank a fixed push library by simulator calls to local opening.

The implemented baseline is **geometry-inspired region-reachability ranking**, not a reproduction of Stilman and Kuffner's planner. The audit below establishes the shared local connectivity problem and the different scoring and search mechanisms. Its measured performance supports a comparison against this baseline, not a claim that HY5U outperforms the published classical planner.

### Geometry implementation-to-paper audit (2026-09-06)

Scope: the registered `geometric-region-walltime-4000-v3` evaluation at commit `c0413c3`, traced through the current evaluator and canonical search. The C++ scoring file has no changes since that evaluated commit; the geometry-specific queue tie rule is also present at that commit. This is an audit of the evaluated local baseline, not the separate full-NAMO planner.

**Implemented score.** [rank_geometric_pushes](../python/namo/planners/opening/best_first_search.py) enumerates the same `PrimitiveGoalStrategy` candidates as the other priors, restricted to the labeled object and reachable contact edges. It sends each requested object pose `(x,y,yaw)` to [evaluate_primitive_region_scores](../src/wavefront/wavefront_planner.cpp). That function removes the object's current footprint from a robot-inflated occupancy grid, rasterizes its inflated footprint at the requested endpoint, runs eight-connected BFS from the robot's current position, and returns the fraction of fixed target samples reachable within the verifier's goal tolerance. It retains the trapped-start clearing rule. No push simulation runs during scoring.

**What that approximation omits.** The virtual score keeps the robot at its pre-push position and all other objects at their current poses. It neither rolls out the controller nor checks a swept manipulation trajectory. It does not separately reject object endpoints that overlap obstacles; overlapping occupancy is simply combined in the virtual grid. Consequently, a high score does not establish that the requested endpoint is physically attainable or that the robot will occupy a useful post-push position. This is a mechanism-level limitation, not a measured attribution of the baseline's failures.

**Executed search and verification.** [solve_scene and _queue_key](../python/namo/planners/opening/best_first_search.py) maintain a global best-first queue with `combine=q`, discount off, `hmax=2`, budget 4000, no-op deduplication and jam-depth pruning for the registered campaign. Each popped action restores its parent state, executes `env.step`, then tests actual reachability. Unsuccessful changed states generate freshly scored child candidates on the same object. [goal_open_pts](../scripts/sandbox/eval_m3.py) accepts at least `ceil(0.2*N)` of the initially fixed target samples, with up to 100 samples. Geometry requires nonempty samples; the published comparison uses the registered common population.

**Queue-policy qualification.** On exactly equal scores, geometry prefers the candidate with more already executed pushes in its chain, then insertion order. Model and random use insertion order directly. At `hmax=2`, this favors a potential second push over a root push, without knowing whether it will finish. The completed campaign therefore compares the geometric score plus this tie rule against the other registered priors. It is not a strictly score-only substitution. A future strictly symmetric ranking comparison must apply one tie policy to every arm; existing artifacts retain their original protocol.

The primary reference is Stilman and Kuffner, [Navigation Among Movable Obstacles: Real-Time Reasoning in Complex Environments (2005)](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2005_3/stilman_michael_2005_3.pdf), Sections 5–7 and Appendix A. The following mapping is our interpretation of that paper and the code above.

| Component | Published method | Evaluated NAMO baseline |
|---|---|---|
| Local problem | `Manip-Search(C1,C2,O)` connects two free-space components by moving one object (§6.2). | Same local connectivity objective; object and target region are supplied. |
| Manipulation actions | Reachable grasps and directional robot motions with force-based push/pull interaction (§5). | Our fixed car push library and controller; no grasp/pull action interface. |
| Local search | Bounded breadth-first action search from sampled contacts, selecting minimum-work connecting motion (§6.2). | Best-first endpoint-fraction priority; first verified opening; simulator-call budget. |
| Global heuristic | Relaxed A* selects an object/region pair using distance and estimated work; recursive planning backtracks (§7). | No global object/region selection in this evaluated local arm. |
| Connectivity | Dynamic C-space grid connectivity test (Appendix A). | Related grid machinery, but fraction-based endpoint scoring and a fixed-sample post-simulation verifier. |

The paper does not specify our endpoint-fraction ranker. In particular, its global relaxed-path heuristic must not be conflated with our local push score. Shared problem structure does not establish algorithm reproduction.

The later artificial-constraints method anticipates future manipulation paths to constrain earlier object choices and placements under monotone assumptions ([IJRR 2008](https://journals.sagepub.com/doi/10.1177/0278364908098457)). Its [WAFR 2006 precursor](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2006_2/stilman_michael_2006_2.pdf), Sections 5–6, explicitly reverses object-motion ordering and accumulates transit/transfer swept-volume constraints. Those mechanisms are absent from the evaluated baseline; this experiment cannot be labeled an artificial-constraints reproduction either.

**Legacy naming trap.** `--prior geometric_transport` uses [evaluate_primitive_priorities](../src/wavefront/wavefront_planner.cpp): remove the blocker, choose one BFS path to the XML goal, then rank endpoints in six bins by path obstruction and collision category. It is a different, superseded proxy and also not a reproduction. Within canonical best-first, `geometric` and `geometric_region` select the corrected fraction score. Separately, [RegionOpeningPlanner's goal-strategy dispatch](../python/namo/planners/opening/region_opening.py) still maps the strategy strings `geometric` and `geometric_transport` to `GeometricTransportStrategy`; a strategy name alone does not identify the evaluated method.

**Paper wording.** "We compare against a geometry-inspired ranker that orders the shared push library by virtual target-region reachability and verifies attempted pushes in physics. It shares the classical keyhole connectivity objective but does not reproduce the classical manipulation search." A future adapted classical baseline would require an explicit implementation of the chosen paper's local search, with the substituted actions, controller, verifier, objective and cost accounting documented. Renaming the existing score is insufficient.

### Existing region BFS versus Manip-Search (2026-09-06)

Code line references in this audit describe the pre-refactor source at `0891e923`. The replay fix below records the subsequent implementation change.

**Conclusion:** the existing `RegionOpeningPlanner` already supplies the local systematic-search architecture. Do not implement another BFS. It is a primitive-based adaptation, with evaluation and cost-accounting issues to resolve before a new canonical comparison. This planner is separate from the endpoint-geometry prior audited above. `FullNAMOPlanner` also exposes it as `full_namo_local_search=region_bfs`.

The reference's local routine searches motions after a reachable grasp, connects the two components, and selects minimum work (§6.2); work means force times distance (§2). Its action model allows directional push/pull forces at a contact (§5). The paper gives a routine-level description, not detailed Manip-Search pseudocode defining every queue tie, bound or duplicate rule. We should document those choices as ours rather than attribute them to the authors. Source: [Stilman and Kuffner 2005](https://www.ri.cmu.edu/pub_files/pub4/stilman_michael_2005_3/stilman_michael_2005_3.pdf).

| Mechanism | Existing implementation | Audit finding |
|---|---|---|
| Same-object local search | `region_opening.py:_search_with_chaining_bfs` (line 2108) keeps `object_id` fixed through all frontier expansions. | Present. The public wrapper can try several boundary objects, so canonical evaluation must select the episode's labeled object explicitly. |
| Search over actual motions | `_search_bfs` constructs an action with its real contact/depth identifiers, restores the parent state, and calls `env.step` (lines 2818–2889). | Present. Unlike the geometric score, this searches physically reached states. |
| Reachable contacts and continuation | Goals and reachable edges are regenerated at each saved state (lines 2263–2331); unsuccessful admissible states become children (line 3114). | Present. Children may use a different contact on the same object. |
| Systematic ordering | Chain depth increases from one upward (line 2224). With primitive scores zero, `_sort_candidates_sync` orders by requested push duration, then contact index (line 60). `cost_first` sorts the next frontier by accumulated proxy cost (line 2499). | Layered search over complete push skills, not uniform-cost search over all chains. Each child board is swept before the next child board. |
| Primitive interface | `PrimitiveGoalStrategy.generate_goals` uses the canonical car library; the controller relocates the robot to contact and runs a continuous push for the requested duration (`namo_push_controller.cpp`, lines 470 and 573). | Deliberate substitution for grasped directional actions. No pulling or persistent-grasp action branching. Keep this common interface for comparisons with HY5U. |
| Opening test | `_validate_opening` supports pinned samples and the 20% reachable-fraction rule (line 3158). | Present, with our intentionally different success definition. No extra geometric endpoint heuristic is required to make BFS a local manipulation search. |
| Cost | `_compute_chain_cost` sums requested duration levels and adds `region_chain_link_cost` once for a multi-push chain (line 2616; default surcharge zero). | No physical-work measurement. Retaining the cheapest discovered solution does not prove minimum physical work or global minimum proxy cost under caps. |

**Minimum-cost qualification.** The outer search retains only the lowest proxy-cost solutions it has found, but solution caps and layer ordering can stop it earlier. For example, a one-push solution with requested duration five can be returned under cap one before a two-push solution with durations one plus one is explored. That is a structural example, not a measured episode. Neither `cost_first` nor the method name establishes Dijkstra-style cost ordering. If the experiment measures calls to the first opening, minimum-work optimization is unnecessary, but the paper must state that objective substitution.

**Concrete budget/reporting defect.** Ordinary successes trigger observation replay: one extra simulation for a one-push result (lines 2398–2426), or replay of the full chain through `_collect_chain_observations` (lines 2041–2088 and 2592). These calls consume the shared `PushAttemptBudget` but do not increment the local `skill_call_counter`. A success on the last allowed search call can therefore be followed by replay budget exhaustion; the public `search` handler returns `success=False` (lines 1032–1056). Confirmed with an isolated mocked-success check through the real outer BFS and public `search`: one verified search success with budget one returns `simulation_budget_exhausted` when observation replay requests another call. No physics evaluation ran. The special root-opener rejection mode skips the one-push replay but is not a general fix for two-push evaluation. Capture the verified solution and its observations during search, preserve success before any optional replay, and count every simulator call consistently before running a budgeted baseline.

**Replay fix (2026-09-06):** the budget/reporting defect above was reproduced and then fixed. Search nodes now retain the original pre/post observations, reachable-object lists and per-push collision metadata. Both one-push and multi-push results use parent-link reconstruction with no simulator calls. The replay helpers and special one-push replay branch were removed. First-solution tests cover successful one- and two-push searches at exactly the simulator budget, including observation provenance and collision aggregation. Historical replay timing fields remain zero for serialized-record compatibility. This changes BFS simulator costs; historical results retain their original protocol.

**Historical continuation-policy difference (fixed below).** BFS rejected a nonopening child when `collision_object` is present or `stuck` is true, even if the state changed (lines 2955–2978 and 3114). Canonical best-first can expand such changed states; it uses failure information to prune longer pushes from the same parent/contact. Benign pushed-object wall contact is not by itself the BFS exclusion condition, and verified openings are accepted even with collision/stuck metadata. Optional BFS no-op pruning considers target-object translation/yaw with default thresholds 0.01 m and 0.05 rad, whereas best-first's default no-op check compares both robot and target pose at 1e-6 tolerance. These are different candidate trees. Their effect on solve rates has not been measured by this audit; they must be aligned for a search-order-only comparison or explicitly retained as algorithm differences.

**Pruning alignment (2026-09-06):** BFS now uses best-first's `_unmoved` helper with no-op pruning on by default: both robot and target poses must remain within 1e-6. The old object-only thresholds were removed from the planner and collection configurations. Both searches use `_record_jam_depth`, driven by nonempty `failure_reason`, to prune equal/longer pushes from that parent/contact; collision or stuck metadata alone no longer prunes candidates. A moved failed push remains eligible for continuation. Ten controlled cases compare actual BFS and best-first root attempts and child states, including robot-only motion, sub-tolerance movement, failed moving pushes and benign wall contact. Explicit pruning opt-outs remain available; matching assumes the default enabled rules. Collection-time solution caps, beam limits and sampling still require explicit settings if a baseline is run.

**Configuration rather than missing machinery.** For a systematic baseline, use the primitive strategy, one fixed episode, pinned target points, `region_selection_strategy=cost_first`, the intended chain limit, and no sampling, beam truncation, label-mode finish caps, ML phases or collection restarts. Set first-solution stopping for a first-opening cost metric. `region_object_skip` can exclude other boundary objects; alternatively the internal same-object routine already accepts `object_id`. The existing object/frontier logic, simulator budget type, and verifier can be reused. Replay-related success accounting and default continuation/jam/no-op pruning are now aligned. Reproducing physical-work optimization and the original grasp/pull action model would be a different, larger experiment; the global planner and artificial-constraints machinery are not missing pieces of this local baseline.

Yao et al.’s local learned NAMO policy and SVG-MPPI are close task-level related work but not drop-in canonical baselines because their actions, objectives, and accounting units differ. Bench-Push Maze is the strongest optional external validation surface: embed the local opener in a full navigation stack and compare on Bench-Push’s task metrics against its supplied baselines. Keep this in a separate external-validation table, never mix its control timesteps or wall time with the canonical simulator-call axis.

**Baseline priority:** a full BFS campaign is optional. HY5U versus random within the same search directly tests learned ordering; geometry supplies the hand-written ordering comparison. BFS changes the search schedule and would test the secondary question of whether systematic search is competitive. It does not supply a reproduced external-method result. Reuse the existing matched-hardware timing comparisons for the main efficiency evidence; do not add a BFS run merely to enlarge the baseline table.

### Learned-method baseline audit (2026-09-06)

**Recommendation: pilot one task-adapted MORE comparison.** This is the strongest actionable candidate found in this focused audit, based on its physics-based push search, learned search returns, and inspectable implementation. It is not a drop-in ranker or a reproduced NAMO result. A full campaign is premature until the adaptation and its training records pass a small pilot. Keep the completed random, geometry, and architecture/loss controls; keep BFS optional. This recommendation supersedes the preliminary preference for Bejjani or Kim–Shimanuki as the next implementation.

The scientific question is whether our treatment of incomplete search evidence and our best-first deployment reach the first verified opening sooner than another learned planning approach. Random and geometry cannot answer that question; a separate learned approach can. An external comparison matters for this broader claim, rather than being necessary merely to establish an improvement over random. Nothing in this audit establishes that the candidate will be competitive or that our method will win.

| Candidate and primary evidence | What the published method contributes | Fit and decision for this project |
|---|---|---|
| [MORE, Huang et al., ICRA 2022](https://arxiv.org/html/2202.01426v2), Sections IV-B–D; [author code](https://github.com/arc-l/more) | A push-value network learns from MCTS returns; learned values guide subsequent physics search. | Best candidate to pilot as an adapted planning method. Preserve search-derived supervision and branch updates. A regressor using our labels in our unchanged queue would not establish a MORE comparison. |
| [SAVE, Hamrick et al., ICLR 2020](https://arxiv.org/html/1912.02807v2), Section 3 | Combines Bellman learning with distillation of searched action values. Search uses learned Q priors and a learned leaf value. | Strong algorithmic alternative, especially for limited search budgets, but needs transition batches, stored searched Q vectors, and a new learner/search loop. Defer behind MORE's inspectable robotics implementation. No author implementation located in the primary pages checked. |
| [Bejjani et al., Humanoids 2018](https://arxiv.org/html/1803.08100v2), Sections IV–VI | Learns discounted remaining-plan returns, suppresses overvalued unchosen actions, then refines Q with DQN. Q guides sampled receding-horizon rollouts. | Relevant physics-planning precedent. Full adaptation requires online transition learning and rollout planning. Borrowing only demonstration supervision omits central mechanisms. No author implementation located through the paper and [author publications page](https://www.wissambejjani.com/publications). |
| [Kim and Shimanuki, CoRL 2019/PMLR 2020](https://proceedings.mlr.press/v100/kim20a/kim20a.pdf), Sections 4–5 and Appendix 3 | Regresses negative remaining plan length with a demonstrated-action large-margin term; relational predicates and a geometric heuristic guide abstract-edge search. | The published decision selects objects/placement regions; ours already fixes object and target region and must select contact/duration. Its manipulation-occlusion predicates require different motion queries. Borrowing the loss is a useful adapted supervision control, not reproduction of the relational GTAMP method. No author implementation located through the paper and [author page](https://beomjoonkim.github.io/). |
| [PIGINet, RSS 2023](https://piginet.github.io/) | Ranks complete symbolic task plans for motion refinement; the project page links code. | Defer. Our second push's reachable contacts depend on the simulated first push. A sequence adaptation needs explicit action semantics and sampled sequence evidence; scoring a single push with our labels removes the plan-level mechanism. |

The shortlist was checked against full method sections for MORE, SAVE, Bejjani, and Kim–Shimanuki; PIGINet was screened against its primary project description. The older broad research notes are candidate lists, not verified implementation mappings. Code availability statements above mean “not located in this audit,” not “does not exist.”

**What we already implemented.** [The production ablation launcher](../scripts/ilab/hy5u_ablations_train.slurm) selects `train_q2_round2.py`, setup value 0.5 and unreachable regression weight 0.1. [The weighted module](../python/namo/rl_loop/sage_ext/weighted_module.py) separates exact distributional regression from censored upper-bound loss. [The ranking loss](../scripts/rl_loop/train_q2_rankaux.py) compares exact tiers against lower known bounds; [the round-2 module](../scripts/rl_loop/train_q2_round2.py) adds family comparisons and unreachable regression. This is different from assuming every undemonstrated action is worse than the demonstrated action. The completed regression-only and independent-contact controls are recorded in the [HY5U ablation card](experiments/log/EXP-2026-08-31-hy5u-icra-ablations.md).

There is also an existing [successful-trajectory dataset adapter](../python/namo/rl_loop/sage_ext/rl_dataset.py): `pi` trains masked behavioral cloning on chosen actions from solved rows; `v` regresses a chosen action's Monte Carlo return without bootstrapping. [train_gen.py](../python/namo/rl_loop/train_gen.py) wires both learners. Therefore “implement behavior cloning” is not a newly discovered missing capability. This audit establishes code availability, not a matching canonical evaluated artifact for those older learners.

**MORE implementation evidence.** Audited public revision: `70402c001c30908e7a70b93f8de3c31abdd26fdb`. In [mcts_network/nodes.py](https://github.com/arc-l/more/blob/70402c001c30908e7a70b93f8de3c31abdd26fdb/mcts_network/nodes.py), `best_child` combines the learned action prior with the best sampled returns and divides by visit count; `backpropagate` discounts returns and increments visits; rollouts select the highest predicted action. [push.py](https://github.com/arc-l/more/blob/70402c001c30908e7a70b93f8de3c31abdd26fdb/mcts_network/push.py) caches simulated transitions and obtains rewards from grasp estimates. [search.py](https://github.com/arc-l/more/blob/70402c001c30908e7a70b93f8de3c31abdd26fdb/mcts_network/search.py) uses iteration/early-stop conditions that need replacement for our first-opening metric. [lifelong_trainer.py](https://github.com/arc-l/more/blob/70402c001c30908e7a70b93f8de3c31abdd26fdb/lifelong_trainer.py) uses Smooth L1 regression. These are concrete differences from [our global best-first queue](../python/namo/planners/opening/best_first_search.py), even if both receive the same push library.

**Required adaptation, before any full evaluation:**

1. Keep our one-object, one-region-pair task, controller, reachable action library, fixed target samples, depth cap, no-op rule and jam-depth pruning. Replace grasp termination with the existing exact opening verifier and stop immediately on any verified solution, including one found during a rollout. Count actual physics attempts, not MCTS iterations. Reuse saved states; solution logging must not simulate the plan again.
2. Preserve MORE's learned prior, sampled-return backups, visit counts, and guided branch selection. Replace its image/push representation with an explicitly documented contact/duration scorer that observes robot and goal-region context. Its fixed-distance tabletop output cannot represent our full library unchanged. The adapted reward must remove grasp-specific shaping and use verified region opening. These changes require the name “MORE-inspired adaptation.”
3. Train on bounded sampled search experience from training rooms: retain action identity, sampled returns and visitation evidence. The current [Q2 loader](../python/namo/rl_loop/sage_ext/q2_dataset.py) supplies board labels/masks, not MCTS returns/counts or next-state batches. Existing raw search records might supply some inputs, but their recoverability has not been audited. Do not invent counts or convert censored cells into exact failed actions. A failed sampled rollout can have return zero without proving the action is a dead end.
4. Separate training acquisition cost from deployment efficiency. Match training-room access and total acquisition simulator budget, record any pretrained components, and tune on held-out development rooms. The pilot must verify that both methods receive usable evidence; identical numeric budget caps alone do not guarantee a competent adaptation. If a new teacher search supplies additional training evidence, expose that difference and compare learning curves before claiming equal-data learning efficiency.
5. Evaluate time to first opening on identical hardware, including scoring, state management and all physics attempts; report simulator-call curves separately and always split easy/medium/hard by 1push/2push. Use the canonical registry protocol. Inspect one- and two-push search traces first, then a small held-out pilot. Advance on correct accounting and a competently trained method, regardless of whether the preliminary result favors us.

The smallest useful implementation is the task-adapted search and supervised return learner. Reusing our encoder would isolate learning/search differences but would explicitly omit MORE's architecture. An optional comparison of both scorers in the same best-first queue would isolate ranking quality; that result alone must not be labeled an evaluation of the full external planner. No new learner, data build, or evaluation was launched for this audit.

## Evaluation protocol

- Report every result by horizon (`1push`, `2push`) and fixed difficulty (`easy`, `medium`, `hard`).
- Efficiency objective: elapsed planning time to the first verified opening, including scoring and attempted pushes; compare only on matched hardware. Report cumulative solve rate versus simulator calls alongside time, using calls as the cross-box proxy.
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

1. Use the completed geometry-inspired campaign with its audited scope and tie policy. Any reproduced/adapted classical-method claim requires a separately specified implementation; the current score does not supply it.
2. Run the primitive-resolution saturation pilot, calibrate its cost, freeze the population and decision rule, then complete the paired audit.
3. If retaining a broader learned-planning comparison claim, pilot the MORE-inspired adaptation specified in the learned-method audit above before a full campaign. The independent-contact architecture control is complete and does not justify duplicate training.
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
4. **Baseline table:** random, geometry, full model; adapted MORE only after a completed evaluation. Put independent-contact/loss controls in the ablation table and oracle diagnostics separately.
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
