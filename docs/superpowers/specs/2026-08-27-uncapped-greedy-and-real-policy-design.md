# Uncapped Greedy DFS and Real Greedy Policy Design

Date: 2026-08-27

Status: approved for implementation

## Goal

Keep the existing whole-simulation `greedy_dfs` arm, remove its artificial
two-committed-push rollout limit, and add a distinct `greedy_policy` arm that
executes exactly one policy-selected push on the real robot before observing
the camera and selecting again.

The two modes must remain separately named in commands, logs, configuration,
diagnostics, and result paths. Neither mode uses held-boundary target matching,
branches to sibling simulator states, or restores a committed simulator
parent.

## Mode 1: uncapped whole-simulation `greedy_dfs`

`greedy_dfs` retains its current planning semantics: it constructs a complete
greedy rollout in MuJoCo before the robot executes any push. At every committed
simulator state it rebuilds the region graph, selects the immediate boundary,
ranks the current candidate actions, commits the highest-ranked action that
moves, and continues from that child.

The rollout will no longer use `best_first_hmax` as a maximum number of
committed greedy decisions. `best_first_hmax` continues to define the local
candidate-action horizon exposed to the ranker; for the current experiment it
remains 2. This separates the action vocabulary from the number of policy
decisions needed to reach the goal.

The uncapped rollout terminates when any existing semantic or resource terminal
condition occurs:

- the robot goal becomes reachable in the committed simulator state;
- no admissible region path remains;
- no live moving candidate remains at the selected boundaries;
- the configured simulator-attempt budget is exhausted;
- an existing graph, goal, or state invariant fails; or
- an explicitly configured full-NAMO iteration limit is reached.

There is no fallback committed-push limit and no replacement arbitrary depth
constant. Rejected no-op and jam simulations continue to consume simulator
budget without becoming committed decisions.

## Mode 2: camera-closed-loop `greedy_policy`

`greedy_policy` is a new, separate unheld Full NAMO execution mode. One planner
call performs one policy decision:

1. Build the region graph from the latest camera-derived simulator scene.
2. If the final goal is already reachable, transition to the existing real
   navigation-to-goal behavior without returning a push.
3. Select the immediate boundary on the new shortest admissible region path.
4. Rank the current reachable push candidates with the configured model or
   uniform prior.
5. Simulate candidates in rank order until one changes the simulator state,
   applying the existing state-local no-op and jam blacklist rules.
6. Return exactly that one moving push as an executable policy step.
7. Execute it once on the real robot.
8. Observe the new camera state, discard the old graph and boundary identity,
   and begin a fresh policy decision at step 1.

The mode never holds or coordinate-matches a previous region target. It does
not verify or reuse a simulated suffix because a policy step has no suffix.
After every successful physical push the robot planner must clear pending-chain
state and force a fresh camera-based plan.

There is no physical two-push cap. The real loop terminates when the live goal
is reachable and final navigation completes, the policy has no executable
moving action, physical execution fails under the existing failure/replan
rules, connectivity is lost under existing runtime rules, or the operator uses
the existing emergency stop/quit controls.

## API and command routing

The backend execution-mode vocabulary becomes `search`, `greedy_dfs`, and
`greedy_policy`. Both greedy modes require `full_namo` with `best_first` and are
invalid with `--hold-region-target` or `--active-target`.

The robot CLI forwards `--exec-mode greedy_policy` through the unheld
`plan_from_xml()` path. It must not route through `solve_boundary_from_xml()` or
the existing held-target `reactive` implementation. Startup banners and saved
configuration must state the exact mode.

For deterministic model-prior best-first and for any run with edge shuffling
disabled, an empty result is final for that observed state. The runtime must not
silently sweep additional shuffle seeds inside one formal trial. Exceptions
remain eligible for the existing retry behavior because they produced no
planner result.

## Results and metrics

The arms use distinct result directories, including `model_greedy_dfs` and
`model_greedy_policy` for model-prior runs.

For `greedy_dfs`, one fresh-search record covers the complete uncapped simulator
rollout. For `greedy_policy`, each camera-based policy decision creates one
fresh-search record. Planning wall time and simulator counts include graph
construction, ranking, and all simulated candidate attempts for that decision.
Model warmup remains recorded once, separately, and excluded from measured
planning wall time.

Each successful `greedy_policy` planning record returns one subgoal and records
the selected object, edge, primitive depth, simulations used, and a
`policy_step_ready` outcome. Empty decisions record their concrete failure kind.
The real-run summary aggregates all decisions and records every attempted and
successful physical push. Real success still requires final camera-confirmed
navigation to within the configured goal tolerance.

## Failure handling and safety

An uncapped simulator rollout cannot cause unbounded physical motion because it
must find a complete simulator rollout before `greedy_dfs` starts execution.
Its configured simulator budget remains authoritative.

`greedy_policy` deliberately permits more than two successful physical pushes,
but it does not bypass any robot safety mechanism. Serial stop on shutdown,
offline detection, physical push-stuck handling, failed-push blacklisting,
emergency stop, and final goal-distance validation remain unchanged. A policy
decision that cannot find a moving action aborts the real trial instead of
printing success or repeating the same deterministic decision under another
seed.

## Tests

Backend tests must first fail for and then pin:

- `greedy_dfs` can commit more than `best_first_hmax` policy decisions while
  `best_first_hmax` still configures the local candidate action set;
- the uncapped rollout terminates through goal reachability and simulator
  budget exhaustion rather than committed depth;
- `greedy_policy` returns exactly one highest-ranked moving action even when
  the resulting simulator child does not yet make the final goal reachable;
- `greedy_policy` rebuilds from the supplied state on each independent call;
- no-op/jam filtering and simulator accounting remain exact; and
- ordinary search and whole-rollout `greedy_dfs` behavior remain separate.

Robot-control tests must first fail for and then pin:

- CLI validation and forwarding of the distinct `greedy_policy` mode;
- rejection of held-target combinations;
- exactly one returned push per policy decision;
- no suffix verification after a successful physical policy step;
- a fresh camera-based planning call before the next physical push;
- deterministic empty results do not trigger an internal seed sweep;
- telemetry distinguishes policy decisions, failures, and final navigation;
  and
- existing search, reactive, and `greedy_dfs` execution paths do not change.

An integration test must run a deterministic fixture requiring more than two
greedy decisions, prove that uncapped `greedy_dfs` reaches it in simulation,
and prove that repeated `greedy_policy` calls produce the same action sequence
one action at a time from successive committed states.

## Non-goals

This change does not alter scorer weights or features, candidate primitive
generation, MuJoCo physics, region connectivity thresholds, held-target
reactive behavior, ordinary branching search, final navigation tolerances, or
the formal trial-to-seed mapping.
