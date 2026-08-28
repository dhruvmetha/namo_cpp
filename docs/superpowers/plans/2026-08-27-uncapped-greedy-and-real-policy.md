# Uncapped Greedy DFS and Real Greedy Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the committed-push ceiling from whole-simulation `greedy_dfs`, run that arm on `hard_004`, then add a distinct `greedy_policy` mode that executes one policy-selected real push per fresh camera observation.

**Architecture:** FullNAMOPlanner continues to own graph selection and the shared one-step `greedy_commit` primitive. `greedy_dfs` repeatedly consumes that primitive inside one simulator plan until a semantic/budget terminal condition, while `greedy_policy` returns after the first moving commit. Robot control routes the new mode through unheld `plan_from_xml()` and deliberately clears single-step chain state after physical success so the next decision starts from the new camera scene.

**Tech Stack:** Python 3.12, pytest, MuJoCo/namo_cpp planning service, robot_control diagnostics and real-camera runtime.

**Engineering Standards:** Follow `plan-coding-standards`. Keep the shared greedy commit path single-source, preserve repository naming and docstring style, avoid a replacement magic depth constant, emit explicit mode/outcome logs, keep environment paths in the existing shell/environment configuration, and end each code stage with a focused commit after behavior-driven tests. Tests are intentionally minimal: one backend regression per new behavior and focused robot routing/closed-loop regressions only.

---

## File map

- `python/namo/planners/full_namo/full_namo_planner.py`: backend execution-mode routing and termination semantics.
- `python/tests/test_full_namo_greedy_dfs.py`: minimal backend regressions for uncapped rollout and one-step policy return.
- `src/robot_control/planner/search_config.py`: CLI-to-backend mode vocabulary, validation, banner, and forwarding.
- `src/robot_control/planner/__init__.py`: public export for the new mode constant.
- `src/robot_control/planner/namo_planner.py`: deterministic retry stop and fresh-plan-after-physical-policy-step behavior.
- `scripts/run_namo.py`: CLI help text for the two distinct unheld modes.
- `tests/test_local_search_config.py`: focused `greedy_policy` routing validation.
- `tests/test_namo_planner_chain_reuse.py`: focused proof that a policy step never verifies a suffix and plans again from the next observation.
- `real_exp/README.md`: separate formal commands/result arms and removal of the old rollout-cap statement.
- `real_exp/METRICS.md`: per-decision accounting for real greedy policy.

### Task 1: Remove the whole-simulation committed-push cap

**Files:**
- Modify: `python/tests/test_full_namo_greedy_dfs.py`
- Modify: `python/namo/planners/full_namo/full_namo_planner.py`

- [ ] **Step 1: Write the failing uncapped-rollout regression**

Extend `FakeEnv.is_robot_goal_reachable()` so an explicit final state is reachable, then replace the depth-exhaustion assertion with one test whose `best_first_hmax` remains 2 but whose rollout needs three committed decisions:

```python
def test_greedy_dfs_is_not_capped_by_candidate_hmax(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([
        _result("goal", state="state-1", edge=3),
        _result("goal", state="state-2", edge=7),
        _result("goal", state="opened", edge=11, opened=True),
    ])
    planner = _planner(monkeypatch, env, opener, max_pushes=2)
    monkeypatch.setattr(
        planner, "_compute_region_snapshot", lambda: _snapshot(["goal"], "goal")
    )

    result = planner.search(GOAL)

    assert result.success is True
    assert [action.edge_idx for action in result.action_sequence] == [3, 7, 11]
    assert result.algorithm_stats["greedy_committed_pushes"] == 3
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/python
PYTHONPATH=.:../build_python /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_full_namo_greedy_dfs.py::test_greedy_dfs_is_not_capped_by_candidate_hmax -q
```

Expected: FAIL because the current planner returns `greedy_depth_exhausted` after two commits.

- [ ] **Step 3: Remove only the committed-depth check**

In `FullNAMOPlanner.__init__`, stop deriving `greedy_max_pushes` from `best_first_hmax` and remove its greedy-only validation. In `search()`, delete the `greedy_depth_exhausted` branch. Keep `best_first_hmax` untouched in `BestFirstRegionOpeningPlanner`; it still defines the candidate action depths passed into `run_greedy_commit`.

The resulting loop must remain:

```python
while True:
    if self.max_iterations is not None and iteration > self.max_iterations:
        return self._failure_result(..., failure_kind="max_iterations_exceeded")
    if self.env.is_robot_goal_reachable():
        return self._success_result(start_time, actions, region_openings)
    # graph/path selection and greedy_commit continue here
```

No new depth constant or implicit fallback cap is permitted.

- [ ] **Step 4: Run focused backend tests and verify GREEN**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/python
PYTHONPATH=.:../build_python /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_full_namo_greedy_dfs.py tests/test_best_first_protocol_defaults.py -q
```

Expected: all selected tests pass, including the three-commit/hmax-2 case.

- [ ] **Step 5: Commit the uncapped backend stage**

```bash
git add python/namo/planners/full_namo/full_namo_planner.py \
        python/tests/test_full_namo_greedy_dfs.py
git commit -m "feat: remove greedy DFS rollout depth cap"
```

### Task 2: Run uncapped whole-simulation greedy DFS on `hard_004`

**Files:**
- Preserve pilot artifacts under: `real_exp/results/pilots/hmax2/hard_004/model_greedy_dfs/`
- Write formal attempt under: `real_exp/results/formal_v1/hmax2/hard_004/model_greedy_dfs/trial1/`

- [ ] **Step 1: Verify the live build and runtime ownership**

Confirm camera port 5556 is reachable, no `run_namo.py`/`check_build.py` process is active, and `/dev/ttyACM0` has no stale owner. Run the existing `check_build.py` command for `hard_004` and require PASS or explicit human acceptance of MARGINAL before motion.

- [ ] **Step 2: Preserve the capped formal artifact**

Move the current capped `trial1` directory to a timestamped, unambiguous pilot path. Do not delete or rewrite its JSONL records; it is provenance for the old two-commit behavior.

- [ ] **Step 3: Launch the uncapped old arm in real mode**

Use the existing model checkpoint, seed 0, one deterministic planning attempt, and the same action horizon:

```bash
--algorithm full_namo \
--local-search best_first \
--best-first-prior model \
--exec-mode greedy_dfs \
--best-first-hmax 2 \
--max-chain-depth 2 \
--max-planning-retries 1 \
--shuffle-seed 0 \
--goal 11.0 67.6
```

Expected: planning may commit more than two simulator decisions. The robot moves only if the complete uncapped simulator rollout reaches the goal; otherwise the run records a concrete failure and shuts down without motion.

- [ ] **Step 4: Validate the saved run**

Inspect `summary.json`, `plans.jsonl`, `pushes.jsonl`, `push_phases.jsonl`, scene captures, and recordings. Confirm the exact mode, seed, warmup exclusion, planning wall time, simulator count, physical push count, outcome, final distance, and clean shutdown before proceeding to the new mode.

### Task 3: Add one-step backend `greedy_policy`

**Files:**
- Modify: `python/tests/test_full_namo_greedy_dfs.py`
- Modify: `python/namo/planners/full_namo/full_namo_planner.py`

- [ ] **Step 1: Write the failing one-step policy regression**

Add one focused test using a moving but not-yet-opening result:

```python
def test_greedy_policy_returns_one_moving_step_before_goal_opens(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([_result("goal", state="state-1", edge=3)])
    planner = _planner(monkeypatch, env, opener, mode="greedy_policy")
    monkeypatch.setattr(
        planner, "_compute_region_snapshot", lambda: _snapshot(["goal"], "goal")
    )

    result = planner.search(GOAL)

    assert result.success is True
    assert [action.edge_idx for action in result.action_sequence] == [3]
    assert result.algorithm_stats["exec_mode"] == "greedy_policy"
    assert result.algorithm_stats["policy_outcome"] == "policy_step_ready"
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/python
PYTHONPATH=.:../build_python /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_full_namo_greedy_dfs.py::test_greedy_policy_returns_one_moving_step_before_goal_opens -q
```

Expected: FAIL because `greedy_policy` is not a recognized Full NAMO execution mode.

- [ ] **Step 3: Generalize the shared greedy-mode branch**

Add `greedy_policy` to `FULL_NAMO_EXEC_MODES`, define a single internal predicate for modes that consume `greedy_commit`, and keep the existing action/state validation single-source. After one moving action is appended, return an executable success result immediately for `greedy_policy`; continue the simulator loop for `greedy_dfs`.

The policy-specific branch should be equivalent to:

```python
if self.exec_mode == "greedy_policy":
    self._record_iteration_trace({**context, "outcome": "policy_step_ready", ...})
    return self._success_result(
        start_time,
        actions,
        region_openings,
        extra_stats={"policy_outcome": "policy_step_ready"},
    )
```

Update `_success_result` with an optional `extra_stats` mapping rather than duplicating the result constructor. Its docstring must explain that success means an executable planner result; real goal success remains runtime/camera validated.

- [ ] **Step 4: Run focused backend tests and verify GREEN**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/python
PYTHONPATH=.:../build_python /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_full_namo_greedy_dfs.py tests/test_best_first_protocol_defaults.py -q
```

Expected: all selected tests pass; ordinary search still calls `search`, whole `greedy_dfs` can exceed two decisions, and `greedy_policy` returns one action.

- [ ] **Step 5: Commit the backend policy mode**

```bash
git add python/namo/planners/full_namo/full_namo_planner.py \
        python/tests/test_full_namo_greedy_dfs.py
git commit -m "feat: add one-step Full NAMO greedy policy"
```

### Task 4: Route `greedy_policy` through robot control and force fresh observation replanning

**Files:**
- Modify: `tests/test_local_search_config.py`
- Modify: `tests/test_namo_planner_chain_reuse.py`
- Modify: `src/robot_control/planner/search_config.py`
- Modify: `src/robot_control/planner/__init__.py`
- Modify: `src/robot_control/planner/namo_planner.py`
- Modify: `scripts/run_namo.py`

- [ ] **Step 1: Write the failing routing test**

Add one test proving the distinct unheld backend key:

```python
def test_greedy_policy_forwards_whole_problem_mode():
    cfg = LocalSearchConfig(
        local_search="best_first",
        best_first_prior="uniform",
        exec_mode="greedy_policy",
    )
    assert cfg.as_planner_kwargs()["full_namo_exec_mode"] == "greedy_policy"
    assert cfg.uses_greedy_policy
```

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
PYTHONPATH=src /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_local_search_config.py::test_greedy_policy_forwards_whole_problem_mode -q
```

Expected: FAIL because the mode is not in `EXEC_MODE_CHOICES`.

- [ ] **Step 2: Write the failing camera-closed-loop test**

Configure the existing fake planner with `LocalSearchConfig(..., exec_mode="greedy_policy")`, return one action from each of two bridge plan calls, notify physical success after the first, and assert the next `plan(obs)` calls the bridge again without `verify_chain`:

```python
assert planner.plan(first_obs) == first_push
planner.notify_subgoal_done(second_obs, failed=False)
assert planner.plan(second_obs) == second_push
assert len(bridge.plan_calls) == 2
assert bridge.verify_calls == []
```

Run the single test and expect FAIL because current MPC handling stores the one-action chain for reuse verification.

- [ ] **Step 3: Implement mode constants, validation, and forwarding**

Add `GREEDY_POLICY_EXEC_MODE = "greedy_policy"`, include it in `EXEC_MODE_CHOICES`, expose `uses_greedy_policy`, and use a shared `uses_unheld_greedy` property for best-first validation, `as_planner_kwargs()`, and `check_search_reaches_planner()`. Error text must name the selected mode dynamically and continue to reject held-target and non-`full_namo` combinations.

Export the constant from `planner/__init__.py` and update `run_namo.py --exec-mode` help so logs and `config.json` clearly distinguish whole-simulation `greedy_dfs` from one-real-step `greedy_policy`.

- [ ] **Step 4: Force fresh planning after each successful policy push**

In the MPC success path of `NAMOPlanner.notify_subgoal_done`, before pending suffix state is created, add a policy-only branch:

```python
if self._local_search.uses_greedy_policy:
    self._subgoals = []
    self._current_idx = 0
    self._plan_generated = False
    self._pending_reuse_chain = None
    self._pending_reuse_origin = None
    self._committed_chain = []
    self._committed_chain_origin = None
    print("[NAMOPlanner] Greedy policy step complete; replanning from fresh camera observation")
    return
```

Do not change search/reactive/`greedy_dfs` suffix behavior.

- [ ] **Step 5: Stop deterministic empty-result seed sweeps**

Import and call `retry_can_change_an_empty_result`. After a normal empty planner result is recorded, break the attempt loop when the helper returns false. Do not apply this to exceptions. Emit one contextual line explaining that the selected search is deterministic for the observed state.

Add one minimal assertion to an existing planning test that a best-first empty result with `max_planning_retries=5` calls the bridge once. Do not add seed permutations.

- [ ] **Step 6: Run focused robot tests and verify GREEN**

Run:

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
PYTHONPATH=src /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_local_search_config.py tests/test_namo_planner_chain_reuse.py -q
```

Expected: all selected tests pass, including routing, fresh planning after a policy push, and a single deterministic empty attempt.

- [ ] **Step 7: Commit the robot policy stage**

```bash
git add src/robot_control/planner/search_config.py \
        src/robot_control/planner/__init__.py \
        src/robot_control/planner/namo_planner.py \
        scripts/run_namo.py \
        tests/test_local_search_config.py \
        tests/test_namo_planner_chain_reuse.py
git commit -m "feat: execute greedy policy one real push at a time"
```

### Task 5: Document the two formal arms and metrics

**Files:**
- Modify: `real_exp/README.md`
- Modify: `real_exp/METRICS.md`

- [ ] **Step 1: Update the experiment command documentation**

Change the `greedy_dfs` section to state that `--best-first-hmax 2` controls candidate primitive depth, not total committed rollout length. Add a separate `model_greedy_policy` command using:

```bash
--exec-mode greedy_policy \
--best-first-hmax 2 \
--max-planning-retries 1 \
--shuffle-seed "$seed"
```

Document the exact closed loop: one simulated argmax moving action, one real push, fresh camera observation, fresh graph, and no suffix verification or held target.

- [ ] **Step 2: Update metrics semantics**

State that each `greedy_policy` decision is one `fresh_search` record; run totals sum all decisions. Warmup remains one-time and excluded. Physical push counts remain separate from simulation attempts, and final success still requires camera-confirmed navigation.

- [ ] **Step 3: Check documentation and commit**

Run `git diff --check`, inspect the two documented commands against `run_namo.py --help`, then commit only these files:

```bash
git add real_exp/README.md real_exp/METRICS.md
git commit -m "docs: define greedy DFS and real policy trial arms"
```

### Task 6: Final verification, integration, and push

**Files:**
- Verify all files committed above; do not stage unrelated robot worktree changes.

- [ ] **Step 1: Run backend verification**

```bash
cd /home/dhruv/projects_dhruv/namo/namo_cpp/python
PYTHONPATH=.:../build_python /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest \
  tests/test_full_namo_greedy_dfs.py \
  tests/test_best_first_protocol_defaults.py \
  tests/test_full_namo_budget_and_config.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run robot-control verification**

```bash
cd /home/dhruv/projects_dhruv/namo/robot_control
PYTHONPATH=src /home/dhruv/miniconda3/envs/namo312/bin/python -m pytest -q
```

Expected: zero failures. If unrelated baseline failures exist, report their exact names and rerun every changed-area test separately; do not claim the full suite passes.

- [ ] **Step 3: Run one plan-only closed-loop backend smoke fixture**

Use a deterministic fixture that needs more than two greedy decisions. Confirm uncapped `greedy_dfs` returns the complete action sequence and repeated `greedy_policy` invocations return the same sequence one step at a time from successive saved states. This supplements rather than replaces the minimal automated tests.

- [ ] **Step 4: Inspect commits and preserve unrelated changes**

Run `git status -sb`, `git diff --check`, and `git log --oneline` in both repositories. Confirm robot GUI changes, RVG submodule state, experiment artifacts, and unrelated tests remain unstaged.

- [ ] **Step 5: Push both feature branches**

Push `feat/horizon-q-redesign` and `real-robot` only after all required verification passes. Report commit hashes, test counts, the uncapped `hard_004` run outcome/metrics, and the exact command prepared for the first `model_greedy_policy` real trial.
