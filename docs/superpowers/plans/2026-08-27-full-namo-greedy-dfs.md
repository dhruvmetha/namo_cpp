# Full NAMO Greedy DFS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an unheld `greedy_dfs` mode that commits one highest-ranked moving simulator action, rebuilds the full region graph from that child state, and never backtracks to a previously committed state.

**Architecture:** `namo_cpp` adds a one-state greedy commit primitive to the existing best-first module, exposes it through `BestFirstRegionOpeningPlanner`, and integrates committed intermediate states into `FullNAMOPlanner`. `robot_control` validates and forwards `--exec-mode greedy_dfs` through `plan_from_xml`; it does not reproduce simulator planning logic.

**Tech Stack:** Python 3.12, pytest, `namo_rl` Python bindings, existing `NAMOPlanningService`, and the robot-control runtime.

**Engineering Standards:** Follow `plan-coding-standards`: preserve canonical candidate ranking and simulation accounting through reuse; add docstrings to new public behavior; use repository naming; keep backend and robot routing commits separate; report contextual terminal reasons and trace fields; introduce no environment-specific constants or secrets; and require focused, regression, and end-to-end simulation tests before completion.

---

## File map

- `namo_cpp/python/namo/planners/opening/best_first_search.py` owns one-state candidate ranking, no-op blacklisting, jam-depth pruning, and one moving-state commit.
- `namo_cpp/python/namo/planners/opening/best_first_region_opening.py` adapts one greedy commit to the existing boundary target, success predicate, budget, and `PlannerResult` contract.
- `namo_cpp/python/namo/planners/full_namo/full_namo_planner.py` owns committed depth, full graph recomputation, terminal outcomes, action aggregation, and trace metrics.
- `namo_cpp/python/tests/test_greedy_commit.py` pins state-local filtering and one-child semantics without physics.
- `namo_cpp/python/tests/test_full_namo_greedy_dfs.py` pins global graph rebuilds, committed-state semantics, depth, budget, and search non-regression.
- `robot_control/src/robot_control/planner/search_config.py` owns the closed CLI/runtime vocabulary and forwarding contract.
- `robot_control/scripts/run_namo.py` exposes and documents `--exec-mode greedy_dfs`.
- `robot_control/tests/test_exec_mode_routing.py` and `robot_control/tests/test_local_search_config.py` pin valid/invalid routing and metadata.
- `robot_control/real_exp/README.md` records the runnable command and clarifies that this is simulator DFS, not held reactive execution.

## Stage 1: Add the one-state greedy commit primitive

**Files:**
- Modify: `python/namo/planners/opening/best_first_search.py`
- Create: `python/tests/test_greedy_commit.py`

- [ ] **Step 1: Write failing tests for one-child commitment and state-local invalid filtering**

Create fakes following `python/tests/test_reactive_jam_guards.py` and tests with these exact assertions:

```python
def test_greedy_commit_blacklists_noops_then_commits_the_first_moving_candidate():
    env = _Env(moves_on={(1, 0)})
    result = run_greedy_commit(
        _planner(), env, GOAL_M, XML, env.get_full_state(),
        h=2, sim_budget=20, prior="model", agg="mean5", combine="q",
        rng=np.random.default_rng(0), restrict_obj=OBJ,
        is_open=lambda _env: False,
    )
    assert env.stepped == [(0, 0), (1, 0)]
    assert result.action is not None
    assert (result.action.edge_idx, result.action.depth) == (1, 0)
    assert result.simulations_used == 2
    assert result.end == "committed"


def test_greedy_commit_prunes_same_and_deeper_depths_after_a_noop_jam():
    env = _Env(failure_reason="OBJECT_STUCK")
    result = _run_commit(env)
    assert env.stepped == [(edge, 0) for edge in EDGES]
    assert result.action is None
    assert result.end == "exhausted"


def test_greedy_commit_keeps_a_state_that_moved_before_reporting_a_jam():
    env = _Env(moves_on={(0, 0)}, failure_reason="OBJECT_STUCK")
    result = _run_commit(env)
    assert result.action is not None
    assert result.resulting_state["pose"][0] == 1.0
    assert result.rejections == []


def test_greedy_commit_never_simulates_a_sibling_after_a_moving_child():
    env = _Env(moves_on={(0, 0), (1, 0)})
    result = _run_commit(env)
    assert env.stepped == [(0, 0)]
    assert result.simulations_used == 1


def test_greedy_commit_counts_rejections_against_the_simulation_budget():
    env = _Env()
    result = _run_commit(env, sim_budget=2)
    assert result.action is None
    assert result.simulations_used == 2
    assert result.end == "budget"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp
set -a && . env.robotlearning.sh && set +a
PYTHONPATH="$PWD/build_python:$PWD/python" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest python/tests/test_greedy_commit.py -v
```

Expected: collection fails because `run_greedy_commit` and `GreedyCommitResult` do not exist.

- [ ] **Step 3: Implement the minimal one-state primitive**

Add a result type and a function beside `run_reactive`:

```python
@dataclass
class GreedyCommitResult:
    action: Optional[namo_rl.Action]
    resulting_state: Any
    simulations_used: int
    opened: bool
    end: str
    rejections: List[Dict[str, Any]]


def run_greedy_commit(
    planner, env, goal, xml, state, h, sim_budget, prior, agg, combine, rng,
    restrict_obj=None, is_open=lambda e: e.is_robot_goal_reachable(), raw=True,
    dedupe_noop=True, prune_jam_depth=True, region_samples=None,
) -> GreedyCommitResult:
    """Commit the first moving arg-max candidate from one simulator state.

    Invalid no-op candidates are filtered at the unchanged state. The first
    moving child is returned immediately, so no sibling child can be explored.
    """
    pool, value, _grid = candidates(
        planner, env, goal, xml, state, h, prior, agg, rng,
        restrict_obj=restrict_obj, raw=raw, region_samples=region_samples,
    )
    banned = set()
    jam_at = {}
    rejections = []
    simulations = 0
    while simulations < sim_budget:
        live = _state_local_live_candidates(pool, banned, jam_at, prune_jam_depth)
        if not live:
            return GreedyCommitResult(None, state, simulations, False, "exhausted", rejections)
        obj, goal_spec, _score = max(
            live, key=lambda candidate: priority(candidate[2], value, combine)
        )
        env.set_full_state(state)
        before = env.get_observation() if dedupe_noop else None
        step_result = env.step(make_action(obj, goal_spec))
        simulations += 1
        opened = bool(is_open(env))
        action = make_action(obj, goal_spec)
        if not dedupe_noop or not _unmoved(before, env.get_observation(), obj):
            return GreedyCommitResult(
                action, env.get_full_state(), simulations, opened, "opened" if opened else "committed", rejections
            )
        edge = int(goal_spec.edge_idx)
        depth = int(goal_spec.depth)
        banned.add((edge, depth))
        _record_state_local_jam(jam_at, goal_spec, step_result, prune_jam_depth)
        rejections.append({"edge_idx": edge, "depth": depth, "reason": "no_state_change"})
    return GreedyCommitResult(None, state, simulations, False, "budget", rejections)
```

Extract `_state_local_live_candidates()` and `_record_state_local_jam()` from the existing reactive loop and call them from both `run_reactive()` and `run_greedy_commit()`. Preserve reactive’s existing rule that a no-op consumes one reactive attempt; the new helper changes filtering implementation only, not held-reactive depth semantics.

- [ ] **Step 4: Run focused and reactive-regression tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  python/tests/test_greedy_commit.py \
  python/tests/test_reactive_jam_guards.py \
  python/tests/test_reactive_search_first_choice_parity.py -v
```

Expected: all tests pass, including unchanged held-reactive ordering and jam behavior.

- [ ] **Step 5: Commit Stage 1**

```bash
git add python/namo/planners/opening/best_first_search.py python/tests/test_greedy_commit.py
git commit -m "feat: add one-state greedy push commit"
```

## Stage 2: Integrate committed children into FullNAMOPlanner

**Files:**
- Modify: `python/namo/planners/opening/best_first_region_opening.py`
- Modify: `python/namo/planners/full_namo/full_namo_planner.py`
- Create: `python/tests/test_full_namo_greedy_dfs.py`
- Modify: `python/tests/test_full_namo_budget_and_config.py`

- [ ] **Step 1: Write failing tests for global graph rebuild and terminal semantics**

Use the fake snapshots and opener pattern from `test_full_namo_strict_bfs.py` and pin:

```python
def test_greedy_dfs_rebuilds_the_global_graph_after_each_committed_push(monkeypatch):
    env = FakeEnv()
    opener = FakeGreedyOpener([
        committed_result("state-1", edge=3),
        committed_result("opened", edge=7, opened=True),
    ])
    planner = make_greedy_planner(monkeypatch, env, opener, max_pushes=2)
    snapshots = iter([snapshot_for("first"), snapshot_for("second")])
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: next(snapshots))
    result = planner.search(GOAL)
    assert result.success is True
    assert opener.targets == ["first", "second"]
    assert [action.edge_idx for action in result.action_sequence] == [3, 7]
    assert result.algorithm_stats["greedy_committed_pushes"] == 2


def test_greedy_dfs_does_not_restore_a_parent_after_a_committed_dead_end(monkeypatch):
    env = FakeEnv()
    opener = FakeGreedyOpener([committed_result("dead-child", edge=3)])
    planner = make_greedy_planner(monkeypatch, env, opener, max_pushes=1)
    result = planner.search(GOAL)
    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "greedy_depth_exhausted"
    assert env.state_history.count("baseline") == 0
    assert env.current_state == "dead-child"


def test_greedy_dfs_reselects_at_the_same_state_when_a_boundary_has_no_moving_candidate(monkeypatch):
    opener = FakeGreedyOpener([
        exhausted_result("a"),
        committed_result("opened", edge=4, opened=True),
    ])
    planner = make_greedy_planner(monkeypatch, FakeEnv(), opener, max_pushes=1)
    result = planner.search(GOAL)
    assert result.success is True
    assert opener.targets == ["a", "b"]


def test_greedy_dfs_requires_best_first():
    with pytest.raises(ValueError, match="best_first"):
        FullNAMOPlanner(FakeEnv(), PlannerConfig(algorithm_params={"full_namo_exec_mode": "greedy_dfs"}))


def test_ordinary_full_namo_search_does_not_enter_greedy_commit(monkeypatch):
    planner = make_search_planner(monkeypatch)
    result = planner.search(GOAL)
    assert result.success is True
    assert planner.region_opener.greedy_calls == 0
```

- [ ] **Step 2: Run the Full NAMO focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_full_namo_budget_and_config.py -v
```

Expected: failures report the missing `full_namo_exec_mode`, greedy opener entry point, and committed-state branch.

- [ ] **Step 3: Add the boundary-adapter entry point**

Add `BestFirstRegionOpeningPlanner.greedy_commit(robot_goal, target_neighbor, **opener_kwargs)` using the same boundary snapshot, blocker restriction, fixed region samples, opening predicate, scorer, and `PushAttemptBudget` as `search()`. Call `run_greedy_commit()` with the remaining budget, consume `result.simulations_used`, and return a `PlannerResult` whose action sequence contains only the moving committed action.

Use these explicit result semantics:

```python
failure_reason = {
    "opened": "success",
    "committed": "greedy_step_committed",
    "budget": "simulation_budget_exhausted",
    "exhausted": "all_pushes_failed",
}[commit.end]
```

Set `success=commit.opened`, retain `resulting_state` for both `opened` and `committed`, set `boundary_exhausted=True` only for `end == "exhausted"`, and add `greedy_commit` plus the rejection rows to `algorithm_stats`.

- [ ] **Step 4: Add FullNAMOPlanner execution-mode and committed-depth control**

Add named constants and validation:

```python
FULL_NAMO_EXEC_MODES = ("search", "greedy_dfs")
DEFAULT_FULL_NAMO_EXEC_MODE = "search"

self.exec_mode = str(algo_params.get("full_namo_exec_mode", DEFAULT_FULL_NAMO_EXEC_MODE))
if self.exec_mode not in FULL_NAMO_EXEC_MODES:
    raise ValueError(f"Unknown full_namo_exec_mode {self.exec_mode!r}")
if self.exec_mode == "greedy_dfs" and self.local_search != "best_first":
    raise ValueError("full_namo_exec_mode='greedy_dfs' requires full_namo_local_search='best_first'")
self.greedy_max_pushes = int(algo_params.get("best_first_hmax", CANONICAL_BEST_FIRST_HMAX))
```

Before selecting another boundary, fail with `failure_kind="greedy_depth_exhausted"` when the goal is still unreachable and `stats.total_pushes >= greedy_max_pushes`.

Choose the opener entry point without duplicating the high-level loop:

```python
if self.exec_mode == "greedy_dfs":
    result = opener.greedy_commit(robot_goal, target_neighbor=target, **opener_kwargs)
else:
    result = opener.search(robot_goal, target_neighbor=target, **opener_kwargs)
```

Before ordinary success/failure handling, recognize a returned committed action, install its resulting state, append exactly one action, clear state-dependent blocked boundaries, record `greedy_step_committed` or `greedy_step_opened`, increment the iteration, and continue to the top-level reachability/graph rebuild. Never retain a simulator parent state in this branch.

- [ ] **Step 5: Extend stats and trace without changing existing fields**

Add `exec_mode`, `greedy_committed_pushes`, `greedy_rejected_simulations`, and per-step object/edge/depth/end/rejection rows to `algorithm_stats` and `iteration_trace`. Keep `total_attempted_pushes` sourced from the opener budget so rejected candidates remain counted.

- [ ] **Step 6: Run backend focused and regression suites and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/build_python:$PWD/python" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  python/tests/test_greedy_commit.py \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_full_namo_strict_bfs.py \
  python/tests/test_full_namo_budget_and_config.py \
  python/tests/test_best_first_protocol_defaults.py \
  python/tests/test_boundary_mode_routing.py \
  python/tests/test_reactive_jam_guards.py \
  python/tests/test_reactive_search_first_choice_parity.py -v
```

Expected: all selected tests pass with ordinary search and held reactive unchanged.

- [ ] **Step 7: Commit Stage 2**

```bash
git add python/namo/planners/opening/best_first_region_opening.py \
  python/namo/planners/full_namo/full_namo_planner.py \
  python/tests/test_full_namo_greedy_dfs.py \
  python/tests/test_full_namo_budget_and_config.py
git commit -m "feat: add Full NAMO greedy DFS execution"
```

## Stage 3: Route greedy DFS through robot_control

**Files:**
- Modify: `src/robot_control/planner/search_config.py`
- Modify: `src/robot_control/planner/__init__.py`
- Modify: `scripts/run_namo.py`
- Modify: `tests/test_exec_mode_routing.py`
- Modify: `tests/test_local_search_config.py`

- [ ] **Step 1: Write failing routing tests**

Add `MODE_GREEDY_DFS = "greedy_dfs"` and pin the complete routing table:

```python
def test_greedy_dfs_is_an_unheld_full_namo_mode():
    config = _greedy_dfs()
    check_search_reaches_planner(
        BEST_FIRST_ALGORITHM, PRIMITIVE_STRATEGY, config,
        held_boundary=False, exec_mode_named=True,
    )
    assert config.as_planner_kwargs()["full_namo_exec_mode"] == MODE_GREEDY_DFS


def test_greedy_dfs_refuses_held_target_state():
    with pytest.raises(ValueError, match="without --hold-region-target"):
        check_search_reaches_planner(
            BEST_FIRST_ALGORITHM, PRIMITIVE_STRATEGY, _greedy_dfs(),
            held_boundary=True, exec_mode_named=True,
        )


def test_greedy_dfs_refuses_region_bfs_and_non_full_namo():
    with pytest.raises(ValueError, match="best_first"):
        LocalSearchConfig(exec_mode=MODE_GREEDY_DFS)
    with pytest.raises(ValueError, match="full_namo"):
        check_search_reaches_planner(
            SWEEP_ALGORITHM, PRIMITIVE_STRATEGY,
            _greedy_dfs(), held_boundary=False, exec_mode_named=True,
        )


def test_greedy_dfs_banner_and_metadata_name_the_effective_mode():
    config = _greedy_dfs()
    line = describe_effective_search(
        BEST_FIRST_ALGORITHM, PRIMITIVE_STRATEGY, config, held_boundary=False
    )
    assert "exec mode: greedy_dfs" in line
```

Extend `test_args_mapping_round_trips()` to assert the forwarded key:

```python
assert cfg.as_planner_kwargs()["full_namo_exec_mode"] == "greedy_dfs"
```

- [ ] **Step 2: Run robot routing tests and verify RED**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
PYTHONPATH="../namo_cpp/build_python:src" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_exec_mode_routing.py tests/test_local_search_config.py -v
```

Expected: failures report that `greedy_dfs` is outside the vocabulary and is not forwarded.

- [ ] **Step 3: Implement the closed routing contract**

Add:

```python
GREEDY_DFS_EXEC_MODE = "greedy_dfs"
EXEC_MODE_CHOICES = (SEARCH_EXEC_MODE, REACTIVE_EXEC_MODE, GREEDY_DFS_EXEC_MODE)
```

Add `uses_greedy_dfs`, require `best_first` for reactive and greedy DFS, and emit `full_namo_exec_mode="greedy_dfs"` only from `as_planner_kwargs()` for that mode. Update `check_search_reaches_planner()` so greedy DFS is allowed only when unheld and aimed at `full_namo`; keep the existing named search/reactive held requirement unchanged. Update `describe_effective_search()` so the unheld banner reports greedy DFS rather than the default search mode.

Update `--exec-mode` help to state that `greedy_dfs` commits one moving simulator child, rebuilds the full graph, never backtracks, and must not be combined with held-target flags.

- [ ] **Step 4: Run focused robot tests and verify GREEN**

Run:

```bash
PYTHONPATH="../namo_cpp/build_python:src" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_exec_mode_routing.py \
  tests/test_local_search_config.py \
  tests/test_search_flag_routing.py \
  tests/test_held_mode_planner_options.py -v
```

Expected: all routing tests pass and every invalid combination fails before robot startup.

- [ ] **Step 5: Commit Stage 3 in robot_control**

```bash
git add src/robot_control/planner/search_config.py src/robot_control/planner/__init__.py \
  scripts/run_namo.py tests/test_exec_mode_routing.py tests/test_local_search_config.py
git commit -m "feat: route Full NAMO greedy DFS mode"
```

## Stage 4: Document and verify the runnable experiment mode

**Files:**
- Modify: `robot_control/real_exp/README.md`
- Modify: `namo_cpp/docs/superpowers/specs/2026-08-27-full-namo-greedy-dfs-design.md` only if implementation names differ from the approved contract.

- [ ] **Step 1: Add the formal launch example**

Document the `hard_004` model command delta:

```bash
--algorithm full_namo \
--local-search best_first \
--best-first-prior model \
--scorer-ckpt /home/dhruv/projects_dhruv/namo/ranking/models/HY5U_s2.ckpt \
--exec-mode greedy_dfs \
--goal 11.0 67.6 \
--no-shuffle-edges --max-chain-depth 2 --shuffle-seed 0
```

State explicitly that this command omits `--hold-region-target`, performs the complete committed rollout in simulation before robot motion, retains canonical jam/no-op filtering, records all rejected simulator calls, and leaves physical MPC suffix verification unchanged.

- [ ] **Step 2: Run cross-repository simulation verification**

Run the backend focused suite from Stage 2, the robot routing suite from Stage 3, then a plan-only captured-scene regression:

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
set -a && . ../namo_cpp/env.robotlearning.sh && set +a
export NAMO_REPO=/home/dhruv/projects_dhruv/namo/namo_cpp
PYTHONPATH="$NAMO_REPO/build_python:src" /home/dhruv/miniconda3/envs/namo312/bin/python -u scripts/run_namo.py \
  --sim-xml real_exp/results/formal_v1/hmax2/hard_004/model_search/trial1/scene_before.xml \
  --robot-model car --algorithm full_namo --local-search best_first \
  --best-first-prior model \
  --scorer-ckpt /home/dhruv/projects_dhruv/namo/ranking/models/HY5U_s2.ckpt \
  --exec-mode greedy_dfs --goal 11.0 67.6 --no-shuffle-edges \
  --max-chain-depth 2 --shuffle-seed 0 \
  --diag-path real_exp/results/verification/hmax2/hard_004/model_greedy_dfs \
  --run-name captured_scene_seed0
```

Expected: the banner says `exec mode: greedy_dfs`; the result contains at most two committed moving actions; every committed action is followed by another top-level region snapshot unless the goal becomes reachable; simulator counts include invalid candidates; and no real serial device is opened in `--sim-xml` mode.

- [ ] **Step 3: Run broad regressions**

Backend:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp
PYTHONPATH="$PWD/build_python:$PWD/python" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest python/tests -q
```

Robot control:

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
PYTHONPATH="../namo_cpp/build_python:src" /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest tests -q
```

Expected: both suites pass with no new warnings or skips attributable to greedy DFS.

- [ ] **Step 4: Commit documentation and verification contract**

```bash
git add real_exp/README.md
git commit -m "docs: add greedy DFS real experiment command"
```

- [ ] **Step 5: Final repository checks**

Run in each repository:

```bash
git status --short
git log -5 --oneline
git diff --check HEAD~1..HEAD
```

Expected: only pre-existing unrelated robot-control changes remain; the backend worktree is clean; each feature commit is scoped to its stage; and no test or verification artifact is committed.
