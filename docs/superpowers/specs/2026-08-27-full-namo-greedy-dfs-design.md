# Full NAMO Greedy DFS Design

Date: 2026-08-27

Status: approved for implementation

## Goal

Add a whole-problem `greedy_dfs` execution mode to `FullNAMOPlanner` that repeatedly rebuilds the current region graph, chooses the current boundary blocker, ranks its reachable push candidates, simulates one highest-priority moving push, commits that resulting simulator state, and continues from the child without retaining or revisiting the parent.

## Problem statement

The current whole-problem path calls `plan_from_xml()` and runs branching `full_namo` search. The existing `reactive` rule is reachable only through `solve_boundary_from_xml()`, so the robot CLI requires `--hold-region-target`; that rule remains tied to one persisted boundary and may advance more than one simulated push before returning. Neither path expresses the requested simulator policy: recompute the global high-level plan after every committed simulated push with no state backtracking.

## Scope

This mode remains inside the canonical single region-opening problem: the robot region and goal region are immediate neighbours across one movable blocker, and the goal is to find a short push sequence on that blocker that merges the regions. The global graph is rebuilt after each committed push so labels, reachability, and the selected immediate boundary always come from the committed simulator state rather than from durable target matching.

The deployed scorer remains the existing horizon-independent ranker. `greedy_dfs` must not add horizon-conditioned scoring or a new model interface.

## Command and routing

The robot CLI adds `greedy_dfs` to the execution-mode vocabulary. `--exec-mode greedy_dfs` is valid only with `--algorithm full_namo`, `--local-search best_first`, and without `--hold-region-target` or `--active-target`. It is forwarded through `NAMOPlanningService.plan_from_xml()` as a dedicated whole-problem algorithm parameter rather than through `solve_boundary_from_xml()`.

Existing behavior remains closed and unchanged: an omitted execution mode runs ordinary unheld full search, while explicit `search` and `reactive` remain the held-boundary comparison modes already defined by the real-robot study.

The startup banner, diagnostics configuration, and trial metadata must report `exec mode: greedy_dfs` so an arg-max rollout cannot be filed as full search.

## Planner algorithm

`FullNAMOPlanner` owns the rollout because it already implements the shared loop for goal reachability, region snapshots, shortest region paths, immediate-boundary selection, state installation, and final action aggregation.

At each committed depth, the planner performs the following sequence:

1. Test whether the robot goal is reachable in the current simulator state; if so, return the committed action chain as success.
2. Rebuild the region snapshot and shortest admissible path from the robot region to the goal region.
3. Select the immediate boundary `path[1]` and its movable blocker set from this snapshot.
4. Build the same reachable candidate pool used by canonical model/uniform best-first: only reachable blocker objects, reachable edges, valid primitive goals, and the existing model or seeded-uniform priorities.
5. Simulate the highest-priority live candidate exactly once.
6. If the push changes the state, append the action, commit the resulting state, discard all parent-state search data, and restart at step 1 with a fresh global graph.
7. If the push produces no state change, keep the current state, blacklist that `(edge, depth)` candidate, apply the canonical same/deeper jam pruning for that edge, and try the next-ranked live candidate. These rejected simulations consume simulator budget but not committed DFS depth.

There is no heap of child states, sibling-state expansion, or restoration to an earlier moving state. A push that moves and then jams is committed because it produced a new irreversible child; its failure metadata is recorded, but the parent is not restored. State-local no-op and jam blacklists are discarded after a moving push because the next committed state has a new candidate pool.

If every live candidate for the selected boundary is rejected without movement, the boundary is marked unavailable in the unchanged state and the existing high-level path logic may select another admissible boundary. If no admissible boundary remains, the rollout fails without revisiting any previously committed state.

## Limits and accounting

The canonical best-first maximum push count remains the committed rollout-depth limit; rejected no-op/jam simulations do not consume that depth. Every call to `env.step()` consumes the existing simulation budget, including rejected candidates.

The rollout terminates with distinct reasons for goal reached, committed-depth exhausted, simulation-budget exhausted, no admissible region path, no live moving candidate, and invariant failures already recognized by `FullNAMOPlanner`.

Existing planning wall-clock measurement continues to surround the complete `plan_from_xml()` call. Ranker warmup remains recorded separately and excluded. Total simulator counts include every greedy candidate verification, including state-local candidates rejected by jam/no-op filtering.

`FullNAMOStats` and the iteration trace record the effective execution mode, committed depth, selected object/edge/depth, whether a simulation was rejected or committed, its failure reason, cumulative simulator usage, and the freshly selected region path at each committed state. The returned action sequence contains only moving committed actions in order.

## Real-robot execution

The planner produces the complete greedy simulator rollout before physical execution. Existing robot-side MPC suffix verification remains unchanged: after each real push, the runtime may verify the remaining suffix against the camera-observed state and fall back to a fresh plan when the suffix no longer matches. `greedy_dfs` itself uses no held-region target and persists no boundary identity across physical observations.

## API boundaries

The shared candidate ranking, primitive generation, reachability filtering, `_unmoved` test, and jam-depth rule remain single-source in the best-first opening module. A focused one-step chooser exposes the result needed by `FullNAMOPlanner`: chosen action, resulting state, simulator attempts consumed, rejection trace, and terminal reason when no moving candidate exists.

`FullNAMOPlanner` controls graph recomputation and committed-depth semantics. The one-step chooser must not select region paths, persist boundary labels, or recursively continue from a moving child.

The robot-control bridge only validates and forwards the mode. It does not reproduce planner logic.

## Tests

Backend tests must first fail for and then pin these behaviors: the highest-ranked moving push is committed; the global snapshot/path is recomputed after each committed push; no-op arg-max candidates are blacklisted and the next candidate is tried from the unchanged state; same/deeper jam continuations are pruned; moved-then-jammed states are committed; no parent state is restored after a moving push; the rollout stops when the goal becomes reachable; depth and simulation budgets are accounted separately; and ordinary branching search remains unchanged.

Robot-control tests must pin that `greedy_dfs` is accepted only on the unheld `full_namo`/`best_first` path, reaches `plan_from_xml()` under a dedicated algorithm parameter, is rejected with held-target options, appears in the startup/diagnostic metadata, and does not silently route to the existing held `reactive` implementation.

An end-to-end simulation regression on a deterministic two-push fixture must assert the exact committed action sequence, graph-rebuild count, simulator count, success outcome, and absence of parent-state branching before this mode is used on hardware.

## Non-goals

This change does not alter model weights, scoring features, physics, region-opening success thresholds, candidate primitives, ordinary best-first search, held-boundary reactive behavior, physical suffix verification, or final navigation.
