# Uniform Rollout Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `UniformRolloutSampler` that produces fresh 1-push F-characterization data via the existing modular parallel collection pipeline, with a schema designed for future chain-data extension.

**Architecture:** A new `BasePlanner`-subclass that performs no search — at each env, it exhaustively executes every reachable push primitive (per object × per edge × per depth) from the initial scene, labels each push with per-neighbor opening outcomes, and emits one region_opening-shaped `AttemptResult` per (object, neighbor) pair so downstream consumers (`batch_collection_classifier.py`, sage_learning loaders) work unchanged.

**Tech Stack:** Python 3, pytest, NumPy, `namo_rl` C++ bindings, existing `BasePlanner`/`PlannerFactory` machinery in `python/namo/core/base_planner.py`, existing `snapshot_region_connectivity` helper in `python/namo/planners/connectivity_snapshot.py`.

**Target env pool:** `/common/users/dm1487/corl2026/namo/envs_100k/` (29,849 diff-drive-car envs across `set1/` and `set2/`). Cluster invocation uses `--config-file config/namo_config_car.yaml` and `--primitive-prefix car_`.

**Spec:** `docs/superpowers/specs/2026-05-19-uniform-rollout-sampler-design.md` — read first.

---

## File Structure

**New files:**
- `python/namo/planners/sampling/uniform_rollout_sampler.py` — sampler class + helper functions + dataclasses. Single file because the components only ever co-occur and the total LOC is small (~400).
- `python/namo/data_collection/rollout_trace_loader.py` — analysis utility for reading sampler pkls and reconstructing per-(object, neighbor) F-grids.
- `python/tests/test_uniform_rollout_sampler.py` — unit tests for the sampler's helpers.
- `python/tests/test_rollout_trace_loader.py` — unit tests for the analysis utility.
- `scripts/generate_car_envs_100k_manifest.sh` — one-shot script to build the manifest file from `envs_100k/`.
- `scripts/run_uniform_rollout_collection_car.sh` — one-shot sbatch helper invoking the collection.

**Modified files:**
- `python/namo/data_collection/modular_parallel_collection.py` — add one import (line ~44), add CLI args, broaden the `attempt_results`-branch trigger condition (line ~400) so the sampler's output flows through the same per-attempt episode path as `region_opening`.

**Untouched (verified):**
- `python/namo/core/base_planner.py`
- `python/namo/planners/connectivity_snapshot.py`
- `python/namo/planners/opening/region_opening.py`
- Parallel-pool / worker / pkl-writer code paths.

---

## Task 1: Skeleton sampler class + factory registration

**Files:**
- Create: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

- [ ] **Step 1: Write the failing test for factory registration**

Create `python/tests/test_uniform_rollout_sampler.py`:

```python
"""Tests for UniformRolloutSampler."""

import namo.planners.sampling.uniform_rollout_sampler  # noqa: F401 — registers on import
from namo.core import PlannerFactory


def test_uniform_rollout_sampler_is_registered():
    available = PlannerFactory.list_available_planners()
    assert "uniform_rollout_sampler" in available
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
cd /common/home/dm1487/robotics_research/ktamp/namo
python -m pytest python/tests/test_uniform_rollout_sampler.py::test_uniform_rollout_sampler_is_registered -v
```

Expected: `ModuleNotFoundError: No module named 'namo.planners.sampling.uniform_rollout_sampler'`.

- [ ] **Step 3: Implement the minimal skeleton**

Create `python/namo/planners/sampling/uniform_rollout_sampler.py`:

```python
"""Uniform Rollout Sampler — fresh 1-push F-characterization with chain-extendable schema.

v0 collects depth-0 exhaustive data only. The schema (TransitionRecord, EnvMetadata,
SamplerAttemptResult) is designed so a follow-up spec can append depth-1 / depth-2
records without breaking existing readers. See docs/superpowers/specs/
2026-05-19-uniform-rollout-sampler-design.md for the design.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerFactory, PlannerResult


class UniformRolloutSampler(BasePlanner):
    """Exhaustively executes every reachable push primitive at the initial scene.

    Does no search. Logs every outcome. Output flows through the existing
    region_opening-style worker branch by emitting one AttemptResult per
    (object, neighbor) pair.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)

    def _setup_constraints(self) -> None:
        # No constraints needed: every reachable primitive is enumerated.
        pass

    def _initialize_algorithm(self) -> None:
        # v0 has no internal algorithm state — every search() call is independent.
        pass

    def reset(self) -> None:
        pass

    @property
    def algorithm_name(self) -> str:
        return "uniform_rollout_sampler"

    @property
    def algorithm_version(self) -> str:
        return "0.1.0"

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        # Skeleton: implemented in later tasks. Returns empty result for now.
        return PlannerResult(
            success=False,
            solution_found=False,
            action_sequence=None,
            algorithm_stats={"attempt_results": []},
        )


PlannerFactory.register_planner("uniform_rollout_sampler", UniformRolloutSampler)
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py::test_uniform_rollout_sampler_is_registered -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: skeleton UniformRolloutSampler + factory registration"
```

---

## Task 2: Dataclasses (TransitionRecord, EnvMetadata, SamplerAttemptResult)

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

- [ ] **Step 1: Write the failing dataclass test**

Append to `python/tests/test_uniform_rollout_sampler.py`:

```python
from dataclasses import asdict

from namo.planners.sampling.uniform_rollout_sampler import (
    EnvMetadata,
    SamplerAttemptResult,
    TransitionRecord,
)


def test_transition_record_roundtrip():
    rec = TransitionRecord(
        transition_id=0,
        parent_id=None,
        depth=0,
        object_id="obj_1",
        edge_idx=5,
        push_depth_idx=3,
        target_pose=(0.1, 0.2, 0.3),
        r=1,
        per_neighbor_opening={"neighbor_A": True, "neighbor_B": False},
        wall_collision=False,
        movable_collisions=[],
        push_terminated_early=False,
        sim_failure=False,
        sim_time_ms=12.3,
        state_after_se2={"obj_1": (0.1, 0.2, 0.3), "robot": (0.0, 0.0, 0.0)},
    )
    d = asdict(rec)
    assert d["transition_id"] == 0
    assert d["depth"] == 0
    assert d["r"] == 1
    assert d["per_neighbor_opening"]["neighbor_A"] is True


def test_env_metadata_fields():
    md = EnvMetadata(
        xml_file="/tmp/env.xml",
        robot_goal=(0.0, 0.0, 0.0),
        initial_state_se2={"obj_1": (0.0, 0.0, 0.0)},
        per_neighbor_region_goals={"neighbor_A": [(0.1, 0.2, 0.0)]},
        neighbor_labels=["robot_region_0", "neighbor_A"],
        static_object_info={"obj_1": {"size_x": 0.05, "size_y": 0.05}},
        collection_timestamp_utc="2026-05-20T00:00:00Z",
        sampler_version="0.1.0",
    )
    d = asdict(md)
    assert d["xml_file"] == "/tmp/env.xml"
    assert d["per_neighbor_region_goals"]["neighbor_A"] == [(0.1, 0.2, 0.0)]


def test_sampler_attempt_result_mirrors_region_opening():
    """SamplerAttemptResult must expose the fields the existing worker branch reads.

    See modular_parallel_collection.py lines 432-515 for the consumed fields.
    """
    attempt = SamplerAttemptResult(
        success=True,
        neighbour_region_label="neighbor_A",
        chosen_object_id="obj_1",
        chosen_goal=(0.1, 0.2, 0.0),
        region_goals_sampled=[(0.1, 0.2, 0.0), (0.15, 0.2, 0.0)],
        region_goal_used=(0.1, 0.2, 0.0),
        primitive_trial_log=[{"edge_idx": 0, "depth": 0, "success": True,
                              "wall_collision": False, "movable_collisions": "",
                              "stuck": False, "collision": False,
                              "reachable_after": 1}],
        chain_depth=1,
        timing_ms=42.0,
        state_observations=[{"obj_1": [0.0, 0.0, 0.0]}],
        post_action_state_observations=[{"obj_1": [0.1, 0.2, 0.0]}],
        reachable_objects_before_action=[["obj_1"]],
        reachable_objects_after_action=[["obj_1"]],
    )
    d = asdict(attempt)
    # Worker reads these in modular_parallel_collection.py line 432-515:
    for required in ("success", "neighbour_region_label", "chosen_object_id",
                     "chosen_goal", "region_goals_sampled", "region_goal_used",
                     "primitive_trial_log", "chain_depth", "timing_ms",
                     "state_observations", "post_action_state_observations",
                     "reachable_objects_before_action", "reachable_objects_after_action"):
        assert required in d, f"missing required field: {required}"
```

- [ ] **Step 2: Run, confirm fails**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "roundtrip or env_metadata or mirrors"
```

Expected: ImportError on `EnvMetadata`, `SamplerAttemptResult`, `TransitionRecord`.

- [ ] **Step 3: Implement the dataclasses**

Insert at the top of `python/namo/planners/sampling/uniform_rollout_sampler.py`, after the imports and before the class:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TransitionRecord:
    """One push attempt with its outcome.

    At v0 (depth-0 only), all records have depth=0 and parent_id=None.
    Fields are kept for chain-extendability — when the follow-up spec adds
    depth-1/2, the same record shape carries the chain bookkeeping.
    """

    transition_id: int                                   # unique within env, dense from 0
    parent_id: Optional[int]                             # always None at v0
    depth: int                                           # always 0 at v0
    object_id: str
    edge_idx: int                                        # 0..59
    push_depth_idx: int                                  # 0..9
    target_pose: Tuple[float, float, float]              # (x, y, θ) the push aimed for
    r: int                                               # 0 or 1; is_robot_goal_reachable(state_after)
    per_neighbor_opening: Dict[str, bool]                # {neighbor_label: opened?}
    wall_collision: bool
    movable_collisions: List[str]                        # object_ids hit
    push_terminated_early: bool
    sim_failure: bool                                    # True iff env.step raised
    sim_time_ms: float
    state_after_se2: Dict[str, Tuple[float, float, float]]


@dataclass
class EnvMetadata:
    """Per-env context shared across all transitions in that env.

    At v0 all transitions in an env share state_before = initial scene, so
    initial_state_se2 + per_neighbor_region_goals live here (env-level).
    """

    xml_file: str
    robot_goal: Tuple[float, float, float]
    initial_state_se2: Dict[str, Tuple[float, float, float]]
    per_neighbor_region_goals: Dict[str, List[Tuple[float, float, float]]]
    neighbor_labels: List[str]                           # all neighbors of the robot's region
    static_object_info: Dict[str, Dict[str, Any]]
    collection_timestamp_utc: str
    sampler_version: str


@dataclass
class SamplerAttemptResult:
    """Per-(object, neighbor) result shaped to match region_opening's AttemptResult.

    The worker code at modular_parallel_collection.py:432-515 reads these fields
    to build one ModularEpisodeResult per attempt. By exposing the same field
    set, the sampler reuses the existing worker branch and the existing
    batch_collection_classifier.py post-processing unchanged.
    """

    success: bool
    neighbour_region_label: str
    chosen_object_id: Optional[str] = None
    chosen_goal: Optional[Tuple[float, float, float]] = None
    region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    region_goal_used: Optional[Tuple[float, float, float]] = None
    primitive_trial_log: Optional[List[Dict[str, Any]]] = None
    chain_depth: int = 1
    timing_ms: Optional[float] = None
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None
    reachable_objects_before_action: Optional[List[List[str]]] = None
    reachable_objects_after_action: Optional[List[List[str]]] = None
    error_message: Optional[str] = None
    # Mirrored from region_opening AttemptResult for cross-compat;
    # populated where applicable, None otherwise.
    validation_method: str = "connectivity"
    connectivity_before: Optional[Dict[str, Any]] = None
    connectivity_after: Optional[Dict[str, Any]] = None
    goal_chain: Optional[List[Any]] = None              # always None at v0
    any_wall_collision: bool = False
    unique_movable_collision_count: int = 0
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: TransitionRecord, EnvMetadata, SamplerAttemptResult dataclasses"
```

---

## Task 3: Reachable-primitive enumeration helper

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

The sampler needs to enumerate every `(object_id, edge_idx, push_depth_idx)` tuple whose contact point the robot can physically reach. Reuses existing C++ bindings: `env.get_reachable_objects()` and `env.get_reachable_edges(obj)`. Depths 0..9 are tried for every reachable edge (matches existing F-char's 60×10 grid convention).

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_uniform_rollout_sampler.py`:

```python
from unittest.mock import MagicMock


def test_enumerate_reachable_primitives_combines_objects_edges_depths():
    """Enumeration is the Cartesian product of (reachable objects) × (their reachable edges) × (depths 0..9)."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["obj_1", "obj_2"]
    env.get_reachable_edges.side_effect = lambda name: {
        "obj_1": [0, 1, 2],
        "obj_2": [10],
    }[name]

    NUM_DEPTHS = 10
    prims = enumerate_reachable_primitives(env, num_depths=NUM_DEPTHS)

    # obj_1: 3 edges × 10 depths = 30
    # obj_2: 1 edge × 10 depths = 10
    assert len(prims) == 40

    # Deterministic ordering: sorted by (object_id, edge_idx, depth_idx)
    assert prims[0] == ("obj_1", 0, 0)
    assert prims[1] == ("obj_1", 0, 1)
    assert prims[9] == ("obj_1", 0, 9)
    assert prims[10] == ("obj_1", 1, 0)
    assert prims[30] == ("obj_2", 10, 0)
    assert prims[-1] == ("obj_2", 10, 9)


def test_enumerate_reachable_primitives_excludes_robot():
    """Robot is in get_reachable_objects but should not be a pushable object."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["robot", "obj_1"]
    env.get_reachable_edges.side_effect = lambda name: {"obj_1": [0]}[name]

    prims = enumerate_reachable_primitives(env, num_depths=10)

    assert all(p[0] != "robot" for p in prims)
    assert len(prims) == 10


def test_enumerate_reachable_primitives_handles_no_reachable_edges():
    """Object with empty reachable_edges contributes nothing."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["obj_1", "obj_2"]
    env.get_reachable_edges.side_effect = lambda name: {"obj_1": [], "obj_2": [0]}[name]

    prims = enumerate_reachable_primitives(env, num_depths=10)

    assert len(prims) == 10
    assert all(p[0] == "obj_2" for p in prims)
```

- [ ] **Step 2: Run, confirm fail**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "enumerate"
```

Expected: ImportError on `enumerate_reachable_primitives`.

- [ ] **Step 3: Implement the helper**

Add to `python/namo/planners/sampling/uniform_rollout_sampler.py`, below the dataclasses and above the class:

```python
# The robot appears in env.get_reachable_objects() but is never a valid push target.
# Match the convention used in region_opening's object selection.
_ROBOT_NAME_PATTERNS = ("robot", "car")


def _is_pushable_object(object_id: str) -> bool:
    """True iff this object is a candidate to be pushed (not the robot itself)."""
    lower = object_id.lower()
    return not any(pattern in lower for pattern in _ROBOT_NAME_PATTERNS)


def enumerate_reachable_primitives(
    env: namo_rl.RLEnvironment,
    num_depths: int = 10,
) -> List[Tuple[str, int, int]]:
    """Return every (object_id, edge_idx, push_depth_idx) the robot can physically attempt.

    Args:
        env: NAMO RL environment positioned at the state to enumerate from.
        num_depths: Number of push depths per edge (matches motion-primitive resolution; 10 in the
            existing F-char data).

    Returns:
        Sorted list of (object_id, edge_idx, depth_idx) tuples. Sorting is deterministic
        for reproducibility.

    Notes:
        - The robot itself (object name containing "robot" or "car") is excluded.
        - Objects with empty reachable_edges contribute nothing.
    """
    reachable_objects = [o for o in env.get_reachable_objects() if _is_pushable_object(o)]

    prims: List[Tuple[str, int, int]] = []
    for obj in sorted(reachable_objects):
        edges = env.get_reachable_edges(obj)
        for edge_idx in sorted(edges):
            for depth_idx in range(num_depths):
                prims.append((obj, edge_idx, depth_idx))
    return prims
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "enumerate"
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: enumerate_reachable_primitives helper"
```

---

## Task 4: Single-primitive execution helper

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

For each primitive, we restore env to s₀, execute the primitive, capture the outcome (success / collision flags / state_after_se2), and restore again for the next iteration. The function returns a partial `TransitionRecord` (without `transition_id`, `per_neighbor_opening`, which get filled by the caller).

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_uniform_rollout_sampler.py`:

```python
def _make_action(object_id: str, target: Tuple[float, float, float],
                 edge_idx: int, depth: int):
    """Helper: build a namo_rl.Action shaped object for tests."""
    a = MagicMock()
    a.object_id = object_id
    a.x = target[0]
    a.y = target[1]
    a.theta = target[2]
    a.edge_idx = edge_idx
    a.depth = depth
    return a


def test_execute_primitive_returns_partial_record_with_outcome():
    """execute_primitive runs env.step, captures wall_collision/stuck/movable_collisions from info."""
    from namo.planners.sampling.uniform_rollout_sampler import execute_primitive

    env = MagicMock()
    initial_state = MagicMock()
    # set_full_state returns nothing; env.step returns a StepResult-like with info dict
    step_result = MagicMock()
    step_result.info = {
        "wall_collision": "true",
        "stuck": "false",
        "movable_collisions": "obj_2",
        "robot_goal_reached": "true",
    }
    env.step.return_value = step_result
    env.is_robot_goal_reachable.return_value = True
    env.get_observation.return_value = {"obj_1": [0.5, 0.5, 0.1]}

    partial = execute_primitive(
        env=env,
        initial_state=initial_state,
        object_id="obj_1",
        edge_idx=3,
        push_depth_idx=5,
        target_pose=(0.5, 0.5, 0.1),
    )

    assert partial["object_id"] == "obj_1"
    assert partial["edge_idx"] == 3
    assert partial["push_depth_idx"] == 5
    assert partial["r"] == 1
    assert partial["wall_collision"] is True
    assert partial["push_terminated_early"] is False
    assert partial["movable_collisions"] == ["obj_2"]
    assert partial["sim_failure"] is False
    assert partial["state_after_se2"] == {"obj_1": (0.5, 0.5, 0.1)}
    env.set_full_state.assert_called_with(initial_state)
    env.step.assert_called_once()


def test_execute_primitive_catches_sim_failure():
    """If env.step raises, partial record has sim_failure=True and r=0."""
    from namo.planners.sampling.uniform_rollout_sampler import execute_primitive

    env = MagicMock()
    env.step.side_effect = RuntimeError("contact resolver failed")

    partial = execute_primitive(
        env=env,
        initial_state=MagicMock(),
        object_id="obj_1",
        edge_idx=0,
        push_depth_idx=0,
        target_pose=(0.0, 0.0, 0.0),
    )
    assert partial["sim_failure"] is True
    assert partial["r"] == 0
    assert partial["wall_collision"] is False
```

- [ ] **Step 2: Run, confirm fails**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "execute_primitive"
```

Expected: ImportError on `execute_primitive`.

- [ ] **Step 3: Implement**

Add to `python/namo/planners/sampling/uniform_rollout_sampler.py`, below `enumerate_reachable_primitives`:

```python
import time as _time


def _parse_movable_collisions(raw: str) -> List[str]:
    """info['movable_collisions'] is a comma-separated string of object IDs (or empty)."""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _se2_from_observation(obs: Dict[str, List[float]]) -> Dict[str, Tuple[float, float, float]]:
    """Convert env.get_observation() output to per-object SE(2) tuples."""
    out: Dict[str, Tuple[float, float, float]] = {}
    for name, pose in obs.items():
        if pose is None or len(pose) < 3:
            continue
        out[name] = (float(pose[0]), float(pose[1]), float(pose[2]))
    return out


def execute_primitive(
    env: namo_rl.RLEnvironment,
    initial_state: "namo_rl.RLState",
    object_id: str,
    edge_idx: int,
    push_depth_idx: int,
    target_pose: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Execute one primitive from `initial_state` and return a partial transition record.

    The partial record is missing `transition_id`, `per_neighbor_opening`, and the
    canonical dataclass wrap — the caller composes the final `TransitionRecord`.

    Restores env to `initial_state` before stepping. Captures wall_collision,
    movable_collisions, stuck, and the per-object SE(2) after the push. If the
    underlying sim raises, returns a record with sim_failure=True and r=0.
    """
    env.set_full_state(initial_state)

    action = namo_rl.Action()
    action.object_id = object_id
    action.x = float(target_pose[0])
    action.y = float(target_pose[1])
    action.theta = float(target_pose[2])
    action.edge_idx = int(edge_idx)
    action.depth = int(push_depth_idx)

    t0 = _time.perf_counter()
    sim_failure = False
    info: Dict[str, str] = {}
    try:
        step_result = env.step(action)
        info = dict(step_result.info or {})
    except Exception:
        sim_failure = True

    sim_time_ms = (_time.perf_counter() - t0) * 1000.0

    if sim_failure:
        return {
            "object_id": object_id,
            "edge_idx": edge_idx,
            "push_depth_idx": push_depth_idx,
            "target_pose": tuple(target_pose),
            "r": 0,
            "wall_collision": False,
            "movable_collisions": [],
            "push_terminated_early": False,
            "sim_failure": True,
            "sim_time_ms": sim_time_ms,
            "state_after_se2": {},
        }

    r = 1 if env.is_robot_goal_reachable() else 0
    wall_collision = info.get("wall_collision", "false").lower() == "true"
    push_terminated_early = info.get("stuck", "false").lower() == "true"
    movable_collisions = _parse_movable_collisions(info.get("movable_collisions", ""))
    state_after_se2 = _se2_from_observation(env.get_observation())

    return {
        "object_id": object_id,
        "edge_idx": edge_idx,
        "push_depth_idx": push_depth_idx,
        "target_pose": tuple(target_pose),
        "r": r,
        "wall_collision": wall_collision,
        "movable_collisions": movable_collisions,
        "push_terminated_early": push_terminated_early,
        "sim_failure": False,
        "sim_time_ms": sim_time_ms,
        "state_after_se2": state_after_se2,
    }
```

Note: `namo_rl.Action` is the binding constructor — when running the test with MagicMock env, we still construct an Action but the mock env's `step` will receive it without complaint. The test does not verify the Action's contents (it would require running real sim).

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "execute_primitive"
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: execute_primitive helper with sim-failure handling"
```

---

## Task 5: Per-neighbor opening evaluator (uses existing snapshot_region_connectivity)

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

For each post-push state, we need to know which neighbor regions of the robot's region became newly reachable. The `snapshot_region_connectivity` helper returns adjacency between regions; we diff "neighbors of the robot region at state_before" vs. "neighbors of the robot region at state_after." A neighbor that's in `_before` and merged with the robot region at `_after` (i.e. is no longer a separate neighbor — meaning passage is open) counts as opened.

Concretely: a neighbor X is "opened" if the robot's region at state_after now contains X (the wavefront from the robot floods into X). Equivalently: the `region_labels` mapping shifts so that grid cells previously labeled X are now labeled with the robot's region.

We implement this as a per-neighbor-label-comparison: a neighbor X is "opened" iff label X exists in `_before`'s adjacency-of-robot-region and *not* in `_after`'s region_labels at all (it merged into the robot region).

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_uniform_rollout_sampler.py`:

```python
def test_evaluate_per_neighbor_opening_detects_merged_neighbors():
    """A neighbor present at state_before but absent at state_after is 'opened'."""
    from namo.planners.sampling.uniform_rollout_sampler import _evaluate_opening_from_snapshots

    # state_before: robot_region with two neighbors A and B
    before_labels = {0: "robot_region_0", 1: "neighbor_A", 2: "neighbor_B"}
    before_adjacency = {
        "robot_region_0": {"neighbor_A", "neighbor_B"},
        "neighbor_A": {"robot_region_0"},
        "neighbor_B": {"robot_region_0"},
    }

    # state_after: robot_region merged with A (the passage to A opened),
    # B still separate.
    after_labels = {0: "robot_region_0", 2: "neighbor_B"}
    after_adjacency = {
        "robot_region_0": {"neighbor_B"},
        "neighbor_B": {"robot_region_0"},
    }

    result = _evaluate_opening_from_snapshots(
        before_labels=before_labels,
        before_adjacency=before_adjacency,
        after_labels=after_labels,
        after_adjacency=after_adjacency,
    )
    assert result == {"neighbor_A": True, "neighbor_B": False}


def test_evaluate_per_neighbor_opening_no_change():
    """If nothing changes, every neighbor is False."""
    from namo.planners.sampling.uniform_rollout_sampler import _evaluate_opening_from_snapshots

    labels = {0: "robot_region_0", 1: "neighbor_A"}
    adj = {"robot_region_0": {"neighbor_A"}, "neighbor_A": {"robot_region_0"}}

    result = _evaluate_opening_from_snapshots(
        before_labels=labels, before_adjacency=adj,
        after_labels=labels, after_adjacency=adj,
    )
    assert result == {"neighbor_A": False}


def test_evaluate_per_neighbor_opening_handles_missing_robot_label():
    """If robot label is missing entirely (degenerate env), return empty dict."""
    from namo.planners.sampling.uniform_rollout_sampler import _evaluate_opening_from_snapshots

    result = _evaluate_opening_from_snapshots(
        before_labels={0: "neighbor_A"},
        before_adjacency={"neighbor_A": set()},
        after_labels={0: "neighbor_A"},
        after_adjacency={"neighbor_A": set()},
    )
    assert result == {}
```

- [ ] **Step 2: Run, confirm fails**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "evaluate_per_neighbor"
```

Expected: ImportError on `_evaluate_opening_from_snapshots`.

- [ ] **Step 3: Implement**

Add to `python/namo/planners/sampling/uniform_rollout_sampler.py`:

```python
from namo.planners.connectivity_snapshot import find_robot_label


def _evaluate_opening_from_snapshots(
    before_labels: Dict[int, str],
    before_adjacency: Dict[str, Any],
    after_labels: Dict[int, str],
    after_adjacency: Dict[str, Any],
) -> Dict[str, bool]:
    """Diff two connectivity snapshots to determine per-neighbor opening.

    A neighbor X of the robot's region at state_before is 'opened' iff X no longer
    appears as a distinct region label at state_after — it merged into the robot's
    region (the passage between robot and X became open).

    Returns a dict mapping each neighbor label seen at state_before to a bool.
    Returns {} if robot label is missing at state_before (degenerate env).
    """
    robot_label = find_robot_label(before_labels)
    if robot_label is None:
        return {}

    neighbors_before = set(before_adjacency.get(robot_label, set()))
    if not neighbors_before:
        return {}

    after_label_set = set(after_labels.values())
    return {n: (n not in after_label_set) for n in neighbors_before}
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "evaluate_per_neighbor"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: per-neighbor opening evaluator from connectivity snapshots"
```

---

## Task 6: Per-env algorithm — orchestrate enumeration, execution, grouping

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`
- Test: `python/tests/test_uniform_rollout_sampler.py`

This task wires the helpers together into the sampler's `search()`. We compute one connectivity snapshot at s₀ (to know neighbors + sample region goals), then for each primitive: execute → capture per_neighbor_opening (using another snapshot at state_after) → build TransitionRecord. Finally group into `SamplerAttemptResult`s per (object, neighbor).

Because computing a `snapshot_region_connectivity` per primitive is expensive, we batch: snapshot the initial state once, and snapshot each unique state_after as needed. (At v0 every primitive lands in a distinct state_after — no batching savings — but the code structure is ready for caching.)

- [ ] **Step 1: Write a focused unit test for the grouping logic**

We test the *grouping* in isolation (without running real sim) — `group_transitions_into_attempts`. Append to test file:

```python
def test_group_transitions_into_attempts_emits_one_per_object_neighbor():
    """One AttemptResult per (object_id, neighbor_label) seen in transitions."""
    from namo.planners.sampling.uniform_rollout_sampler import group_transitions_into_attempts

    # Build 4 transitions: 2 objects × pushes opening A or B
    transitions = [
        # obj_1 push opens A
        {"object_id": "obj_1", "edge_idx": 0, "push_depth_idx": 0,
         "target_pose": (0.1, 0.0, 0.0), "r": 1, "wall_collision": False,
         "movable_collisions": [], "push_terminated_early": False,
         "sim_failure": False, "sim_time_ms": 5.0,
         "state_after_se2": {}, "per_neighbor_opening": {"A": True, "B": False}},
        # obj_1 push opens nothing
        {"object_id": "obj_1", "edge_idx": 1, "push_depth_idx": 0,
         "target_pose": (0.2, 0.0, 0.0), "r": 0, "wall_collision": True,
         "movable_collisions": [], "push_terminated_early": False,
         "sim_failure": False, "sim_time_ms": 6.0,
         "state_after_se2": {}, "per_neighbor_opening": {"A": False, "B": False}},
        # obj_2 push opens B
        {"object_id": "obj_2", "edge_idx": 0, "push_depth_idx": 5,
         "target_pose": (0.3, 0.0, 0.0), "r": 1, "wall_collision": False,
         "movable_collisions": ["obj_3"], "push_terminated_early": False,
         "sim_failure": False, "sim_time_ms": 7.0,
         "state_after_se2": {}, "per_neighbor_opening": {"A": False, "B": True}},
        # obj_2 push opens nothing
        {"object_id": "obj_2", "edge_idx": 1, "push_depth_idx": 9,
         "target_pose": (0.4, 0.0, 0.0), "r": 0, "wall_collision": False,
         "movable_collisions": [], "push_terminated_early": True,
         "sim_failure": False, "sim_time_ms": 4.0,
         "state_after_se2": {}, "per_neighbor_opening": {"A": False, "B": False}},
    ]

    initial_obs = {"obj_1": [0.0, 0.0, 0.0]}
    region_goals = {"A": [(0.1, 0.0, 0.0)], "B": [(0.3, 0.0, 0.0)]}
    neighbor_labels = ["A", "B"]
    reachable_objects = ["obj_1", "obj_2"]

    attempts = group_transitions_into_attempts(
        transitions=transitions,
        neighbor_labels=neighbor_labels,
        region_goals=region_goals,
        initial_observation=initial_obs,
        reachable_objects_before=reachable_objects,
    )

    # 2 objects × 2 neighbors = 4 attempts (some may be marked unsuccessful)
    assert len(attempts) == 4

    # obj_1 / A: trial_log has 2 entries (the 2 obj_1 trials), success=True
    obj1_A = next(a for a in attempts if a.chosen_object_id == "obj_1"
                  and a.neighbour_region_label == "A")
    assert obj1_A.success is True
    assert len(obj1_A.primitive_trial_log) == 2
    assert obj1_A.region_goals_sampled == [(0.1, 0.0, 0.0)]
    assert obj1_A.chosen_goal == (0.1, 0.0, 0.0)  # the target of the successful push

    # obj_1 / B: trial_log has 2 entries, success=False
    obj1_B = next(a for a in attempts if a.chosen_object_id == "obj_1"
                  and a.neighbour_region_label == "B")
    assert obj1_B.success is False
    assert obj1_B.chosen_goal is None

    # Verify trial-log entry shape matches existing F-char format
    entry = obj1_A.primitive_trial_log[0]
    for required_key in ("edge_idx", "depth", "success", "wall_collision",
                         "movable_collisions", "stuck", "collision", "reachable_after"):
        assert required_key in entry, f"trial_log entry missing {required_key}"
```

- [ ] **Step 2: Run, confirm fails**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "group_transitions"
```

Expected: ImportError on `group_transitions_into_attempts`.

- [ ] **Step 3: Implement the grouping function**

Add to `python/namo/planners/sampling/uniform_rollout_sampler.py`:

```python
def group_transitions_into_attempts(
    transitions: List[Dict[str, Any]],
    neighbor_labels: List[str],
    region_goals: Dict[str, List[Tuple[float, float, float]]],
    initial_observation: Dict[str, List[float]],
    reachable_objects_before: List[str],
) -> List[SamplerAttemptResult]:
    """Build one SamplerAttemptResult per (object, neighbor) pair.

    Each attempt's primitive_trial_log contains the subset of transitions involving
    that object, with success labeled per-neighbor (true iff the push opened
    THIS neighbor).

    Args:
        transitions: Flat list of dicts as produced by execute_primitive() +
            per_neighbor_opening dict added by the caller.
        neighbor_labels: All neighbor labels for the robot region at s₀.
        region_goals: Per-neighbor sampled goal points (for the region_goals_sampled
            field consumed by NAMODataVisualizer).
        initial_observation: env.get_observation() result at s₀, used to populate
            the single-entry state_observations list (matches how exhaustive-mode
            region_opening fills it).
        reachable_objects_before: env.get_reachable_objects() at s₀.

    Returns:
        List of SamplerAttemptResult, one per (object, neighbor). The order is
        sorted by (object_id, neighbor_label) for reproducibility.
    """
    objects = sorted({t["object_id"] for t in transitions})
    attempts: List[SamplerAttemptResult] = []

    for obj in objects:
        obj_transitions = [t for t in transitions if t["object_id"] == obj]
        for neighbor in sorted(neighbor_labels):
            # Build the trial_log for this (obj, neighbor) using the existing F-char schema.
            trial_log: List[Dict[str, Any]] = []
            success_target: Optional[Tuple[float, float, float]] = None
            any_wall = False
            unique_movable: set = set()

            for t in obj_transitions:
                opened = bool(t["per_neighbor_opening"].get(neighbor, False))
                entry = {
                    "edge_idx": int(t["edge_idx"]),
                    "depth": int(t["push_depth_idx"]),
                    "success": opened,
                    "wall_collision": bool(t["wall_collision"]),
                    "movable_collisions": ",".join(t["movable_collisions"]),
                    "stuck": bool(t["push_terminated_early"]),
                    "collision": bool(t["wall_collision"] or t["movable_collisions"]),
                    "reachable_after": int(t["r"]),
                }
                trial_log.append(entry)
                if opened and success_target is None:
                    success_target = tuple(t["target_pose"])
                if t["wall_collision"]:
                    any_wall = True
                for mc in t["movable_collisions"]:
                    unique_movable.add(mc)

            sampled_goals = region_goals.get(neighbor)
            attempts.append(SamplerAttemptResult(
                success=success_target is not None,
                neighbour_region_label=neighbor,
                chosen_object_id=obj,
                chosen_goal=success_target,
                region_goals_sampled=sampled_goals,
                region_goal_used=(sampled_goals[0] if sampled_goals else None),
                primitive_trial_log=trial_log,
                chain_depth=1,
                state_observations=[initial_observation],
                post_action_state_observations=None,
                reachable_objects_before_action=[reachable_objects_before],
                reachable_objects_after_action=None,
                any_wall_collision=any_wall,
                unique_movable_collision_count=len(unique_movable),
            ))

    return attempts
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v -k "group_transitions"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py \
        python/tests/test_uniform_rollout_sampler.py
git commit -m "feat: group_transitions_into_attempts produces per-(object, neighbor) AttemptResults"
```

---

## Task 7: Wire `search()` end-to-end

**Files:**
- Modify: `python/namo/planners/sampling/uniform_rollout_sampler.py`

Now we connect: snapshot at s₀ → enumerate primitives → execute each → snapshot per state_after for per-neighbor diffing → group → assemble `PlannerResult`.

This task does **not** add a new unit test — `search()` requires a real `namo_rl.RLEnvironment` and is integration-tested in Task 10. The implementation here must satisfy the per-env algorithm described in spec §3.3.

- [ ] **Step 1: Implement `search()`**

Replace the skeleton `search()` in `python/namo/planners/sampling/uniform_rollout_sampler.py` with:

```python
import datetime as _datetime

from namo.planners.connectivity_snapshot import snapshot_region_connectivity


class UniformRolloutSampler(BasePlanner):
    """Exhaustively executes every reachable push primitive at the initial scene.

    See module docstring + docs/superpowers/specs/2026-05-19-uniform-rollout-sampler-design.md.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)

    def _setup_constraints(self) -> None:
        pass

    def _initialize_algorithm(self) -> None:
        algo_params = self.config.algorithm_params or {}
        self.max_chain_depth: int = int(algo_params.get("max_chain_depth", 1))
        if self.max_chain_depth != 1:
            raise ValueError(
                f"v0 of UniformRolloutSampler only supports max_chain_depth=1; "
                f"got {self.max_chain_depth}. Chain expansion is a follow-up spec."
            )
        self.region_goal_samples_per_neighbor: int = int(
            algo_params.get("region_goal_samples_per_neighbor", 5)
        )
        self.num_depths: int = int(algo_params.get("num_depths", 10))
        self.xml_file: Optional[str] = algo_params.get("xml_file")
        self.config_file: Optional[str] = algo_params.get("config_file_path")
        self.seed: int = int(algo_params.get("seed", 42))

    def reset(self) -> None:
        pass

    @property
    def algorithm_name(self) -> str:
        return "uniform_rollout_sampler"

    @property
    def algorithm_version(self) -> str:
        return "0.1.0"

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        import numpy as np
        t_start = _time.perf_counter()

        # 1. Set the robot goal and snapshot the initial state.
        self.env.set_robot_goal(float(robot_goal[0]), float(robot_goal[1]), float(robot_goal[2]))
        initial_state = self.env.get_full_state()
        initial_observation = self.env.get_observation()
        initial_se2 = _se2_from_observation(initial_observation)
        reachable_objects_initial = list(self.env.get_reachable_objects())

        # 2. Compute connectivity at s₀, sampling K region goals per neighbor.
        rng = np.random.default_rng(self.seed)
        try:
            before_adj, _edge_objs, before_labels, before_region_goals, _snap = snapshot_region_connectivity(
                self.env,
                xml_path=self.xml_file or "",
                config_path=self.config_file or "",
                goals_per_region=self.region_goal_samples_per_neighbor,
                generate_training_data=True,
                local_info_only=False,
                use_current_state=True,
                rng=rng,
            )
        except Exception as exc:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message=f"connectivity snapshot at s₀ failed: {exc}",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        robot_label = find_robot_label(before_labels)
        if robot_label is None:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message="no robot region label at s₀",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )
        neighbor_labels = sorted(before_adj.get(robot_label, set()))

        # Convert region_goals to plain tuples (per-neighbor, only non-robot labels).
        region_goals: Dict[str, List[Tuple[float, float, float]]] = {}
        for nbr in neighbor_labels:
            bundle = before_region_goals.get(nbr)
            if bundle is None:
                region_goals[nbr] = []
                continue
            region_goals[nbr] = [(g.x, g.y, g.theta) for g in bundle.goals]

        # 3. Enumerate every reachable primitive at s₀.
        primitives = enumerate_reachable_primitives(self.env, num_depths=self.num_depths)
        if not primitives:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message="no reachable primitives at s₀",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        # 4. For each primitive: execute, snapshot state_after, diff for per-neighbor opening.
        transitions: List[Dict[str, Any]] = []
        # next_transition_id reserved for future chain-extension; unused at v0
        for tid, (obj, edge_idx, depth_idx) in enumerate(primitives):
            # Target pose is the primitive's calibrated landing pose; we don't have
            # it without running the goal strategy. The C++ skill resolves the target
            # from (object, edge_idx, depth) internally — we pass placeholder zeros and
            # set edge_idx/depth on the Action so the skill uses the database.
            target_pose = (0.0, 0.0, 0.0)
            partial = execute_primitive(
                env=self.env,
                initial_state=initial_state,
                object_id=obj,
                edge_idx=edge_idx,
                push_depth_idx=depth_idx,
                target_pose=target_pose,
            )

            # Snapshot state_after for per-neighbor opening.
            try:
                after_adj, _eo, after_labels, _rg, _snap = snapshot_region_connectivity(
                    self.env,
                    xml_path=self.xml_file or "",
                    config_path=self.config_file or "",
                    goals_per_region=0,
                    generate_training_data=False,
                    local_info_only=False,
                    use_current_state=True,
                )
                per_neighbor = _evaluate_opening_from_snapshots(
                    before_labels=before_labels, before_adjacency=before_adj,
                    after_labels=after_labels, after_adjacency=after_adj,
                )
            except Exception:
                per_neighbor = {n: False for n in neighbor_labels}

            partial["per_neighbor_opening"] = per_neighbor
            partial["transition_id"] = tid
            transitions.append(partial)

        # Restore env to s₀ once at the end so downstream code sees a stable state.
        self.env.set_full_state(initial_state)

        # 5. Group into per-(object, neighbor) AttemptResults.
        attempts = group_transitions_into_attempts(
            transitions=transitions,
            neighbor_labels=neighbor_labels,
            region_goals=region_goals,
            initial_observation=initial_observation,
            reachable_objects_before=reachable_objects_initial,
        )

        # 6. Assemble env metadata.
        env_meta = EnvMetadata(
            xml_file=self.xml_file or "",
            robot_goal=tuple(robot_goal),
            initial_state_se2=initial_se2,
            per_neighbor_region_goals=region_goals,
            neighbor_labels=neighbor_labels,
            static_object_info=dict(self.env.get_object_info()),
            collection_timestamp_utc=_datetime.datetime.utcnow().isoformat() + "Z",
            sampler_version=self.algorithm_version,
        )

        # 7. Wrap into a PlannerResult.
        from dataclasses import asdict as _asdict
        return PlannerResult(
            success=any(a.success for a in attempts),
            solution_found=False,                         # sampler doesn't 'solve' anything
            action_sequence=None,
            algorithm_stats={
                "attempt_results": attempts,             # consumed by the worker branch
                "env_metadata": _asdict(env_meta),
                "sampler_summary_stats": {
                    "n_transitions": len(transitions),
                    "n_attempts": len(attempts),
                    "n_r1": sum(1 for t in transitions if t["r"] == 1),
                    "n_sim_failures": sum(1 for t in transitions if t["sim_failure"]),
                },
            },
            search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
        )
```

(Replace the old class block; the registration line at the bottom of the file stays.)

- [ ] **Step 2: Run existing tests to confirm no regression in helpers**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler.py -v
```

Expected: all previously-passing tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add python/namo/planners/sampling/uniform_rollout_sampler.py
git commit -m "feat: wire UniformRolloutSampler.search() end-to-end (depth 0)"
```

---

## Task 8: Broaden the worker branch trigger + add CLI args

**Files:**
- Modify: `python/namo/data_collection/modular_parallel_collection.py`

The worker has a `region_opening`-special branch (line ~398–520) that fans out one `ModularEpisodeResult` per `attempt_results` entry. The sampler emits the same `attempt_results` shape but with a different algorithm name. We broaden the trigger so it fires for either planner.

- [ ] **Step 1: Read the existing trigger line**

```bash
sed -n '396,402p' /common/home/dm1487/robotics_research/ktamp/namo/python/namo/data_collection/modular_parallel_collection.py
```

Expected output (approximately):

```python
                # Special handling for region_opening planner: convert AttemptResults to episodes
                is_region_opening = task.algorithm == "region_opening"

                if is_region_opening and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:
                    # Process each AttemptResult as a separate episode
                    for attempt_idx, attempt in enumerate(planner_result.algorithm_stats['attempt_results']):
```

- [ ] **Step 2: Change the algorithm-name check to include `uniform_rollout_sampler`**

Apply the following edit to `python/namo/data_collection/modular_parallel_collection.py`:

```python
# OLD (line ~397):
is_region_opening = task.algorithm == "region_opening"

# NEW:
emits_attempt_results = task.algorithm in ("region_opening", "uniform_rollout_sampler")
is_region_opening = task.algorithm == "region_opening"  # kept for any other branch that still uses it
```

Then change the trigger on the next non-blank line:

```python
# OLD:
if is_region_opening and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:

# NEW:
if emits_attempt_results and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:
```

Use Edit tool calls for these substitutions to keep surrounding context intact.

- [ ] **Step 3: Add the import for the sampler near the existing planner imports**

In `python/namo/data_collection/modular_parallel_collection.py`, near the existing planner imports (around line 42–43):

```python
# Existing:
from namo.planners.sampling.random_sampling import RandomSamplingPlanner
from namo.planners.opening.region_opening import RegionOpeningPlanner

# Add:
from namo.planners.sampling.uniform_rollout_sampler import UniformRolloutSampler  # noqa: F401 — registers on import
```

- [ ] **Step 4: Add the CLI args**

In the same file, find the planner-specific args block (search for `--region-allow-collisions` around line 1031). Below the last `--region-*` arg, add:

```python
    # ----------------- Uniform rollout sampler arguments -----------------
    parser.add_argument("--sampler-max-chain-depth", type=int, default=1, choices=[1],
                        help="v0 supports depth 0 only (max_chain_depth=1). "
                             "Deeper depths are a follow-up spec.")
    parser.add_argument("--sampler-region-goal-samples", type=int, default=5,
                        help="K points to sample per neighbor region for goal_sample_region mask "
                             "(stored in env_metadata.per_neighbor_region_goals).")
    parser.add_argument("--sampler-num-depths", type=int, default=10,
                        help="Number of push depths per edge (matches motion-primitive resolution).")
```

Then find the algorithm_params construction block (search for `algorithm_params = {}` around line 1145). Below the `if args.algorithm == "region_opening":` block, add a parallel block:

```python
    if args.algorithm == "uniform_rollout_sampler":
        algorithm_params["max_chain_depth"] = args.sampler_max_chain_depth
        algorithm_params["region_goal_samples_per_neighbor"] = args.sampler_region_goal_samples
        algorithm_params["num_depths"] = args.sampler_num_depths
        algorithm_params["primitive_prefix"] = args.primitive_prefix
        algorithm_params["primitive_data_dir"] = args.primitive_data_dir
        algorithm_params["config_file_path"] = args.config_file
        algorithm_params["seed"] = args.seed if args.seed is not None else DEFAULT_GLOBAL_SEED
```

- [ ] **Step 5: Smoke-test the import path**

```bash
python -c "from namo.planners.sampling.uniform_rollout_sampler import UniformRolloutSampler; \
            from namo.core import PlannerFactory; \
            print('uniform_rollout_sampler' in PlannerFactory.list_available_planners())"
```

Expected output: `True`.

- [ ] **Step 6: Smoke-test argparse**

```bash
python python/namo/data_collection/modular_parallel_collection.py --help 2>&1 | grep -E "sampler-(max-chain|region-goal|num-depths)"
```

Expected: all three sampler args appear with their help text.

- [ ] **Step 7: Commit**

```bash
git add python/namo/data_collection/modular_parallel_collection.py
git commit -m "feat: wire UniformRolloutSampler into modular collection (CLI + worker branch)"
```

---

## Task 9: Manifest generation script for car envs

**Files:**
- Create: `scripts/generate_car_envs_100k_manifest.sh`

The collection pipeline needs a manifest file (one XML path per line). The existing `python/namo/data_collection/scripts/generate_xml_manifest.py` script can produce this; we wrap it for the car env pool.

- [ ] **Step 1: Find the manifest generator (sanity check it exists)**

```bash
find /common/home/dm1487/robotics_research/ktamp/namo -name "generate_xml_manifest*.py" 2>/dev/null
```

Expected: at least one match.

- [ ] **Step 2: Create the wrapper script**

Create `scripts/generate_car_envs_100k_manifest.sh`:

```bash
#!/usr/bin/env bash
# Build the manifest file for the diff-drive-car env pool at
# /common/users/dm1487/corl2026/namo/envs_100k/.
#
# Output: scripts/manifests/car_envs_100k.txt — one XML path per line.

set -euo pipefail

ENV_ROOT="/common/users/dm1487/corl2026/namo/envs_100k"
OUTPUT_DIR="$(dirname "$0")/manifests"
OUTPUT_FILE="$OUTPUT_DIR/car_envs_100k.txt"

mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.."

# Generate the manifest from the env pool.
python python/namo/data_collection/scripts/generate_xml_manifest.py \
    --input-dir "$ENV_ROOT" \
    --output "$OUTPUT_FILE"

echo ""
echo "Manifest generated: $OUTPUT_FILE"
wc -l "$OUTPUT_FILE"
```

If the generator script lives at a different path, adjust the `python` line accordingly.

- [ ] **Step 3: Make it executable**

```bash
chmod +x scripts/generate_car_envs_100k_manifest.sh
```

- [ ] **Step 4: Run it once and verify line count is ~29,849**

```bash
./scripts/generate_car_envs_100k_manifest.sh
```

Expected: `~29849 scripts/manifests/car_envs_100k.txt`.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_car_envs_100k_manifest.sh
# The manifest itself is a runtime artifact, not source — exclude.
echo "scripts/manifests/" >> .gitignore
git add .gitignore
git commit -m "chore: car-env manifest generator script"
```

---

## Task 10: Integration smoke test on 5 envs

**Files:**
- Test: `python/tests/test_uniform_rollout_sampler_integration.py`

End-to-end test: run the modular collection script on 5 envs from the car pool with `--workers 1`, verify pkl is written, schema loads, and `batch_collection_classifier.py` can produce an NPZ.

- [ ] **Step 1: Create the integration test**

Create `python/tests/test_uniform_rollout_sampler_integration.py`:

```python
"""End-to-end smoke test for UniformRolloutSampler on a tiny manifest.

Skipped if the cluster paths aren't available locally (env var SKIP_NAMO_INTEGRATION=1).
"""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "manifests" / "car_envs_100k.txt"
RUN_INTEGRATION = os.environ.get("SKIP_NAMO_INTEGRATION") != "1" and MANIFEST.exists()


@pytest.mark.skipif(not RUN_INTEGRATION, reason="requires car-env manifest")
def test_sampler_smoke_runs_5_envs_and_produces_attempt_results():
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "python", str(REPO_ROOT / "python/namo/data_collection/modular_parallel_collection.py"),
            "--algorithm", "uniform_rollout_sampler",
            "--manifest", str(MANIFEST),
            "--start-idx", "0",
            "--end-idx", "5",
            "--workers", "1",
            "--output-dir", tmp,
            "--config-file", "config/namo_config_car.yaml",
            "--primitive-prefix", "car_",
            "--sampler-max-chain-depth", "1",
            "--sampler-region-goal-samples", "5",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"collection failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        # At least one pkl was written.
        pkls = list(Path(tmp).rglob("*.pkl"))
        assert pkls, "no pkl files written"

        # First pkl contains episode_results, each with algorithm_stats and a primitive_trial_log.
        with open(pkls[0], "rb") as f:
            data = pickle.load(f)
        episodes = data.get("episode_results", [])
        assert episodes, "no episodes in first pkl"

        ep = episodes[0]
        stats = ep.get("algorithm_stats") or {}
        assert "primitive_trial_log" in stats, "primitive_trial_log missing — batch_collection_classifier needs it"
        assert isinstance(stats["primitive_trial_log"], list)
        # Validate one trial entry's shape matches existing F-char schema.
        if stats["primitive_trial_log"]:
            entry = stats["primitive_trial_log"][0]
            for key in ("edge_idx", "depth", "success", "wall_collision",
                        "movable_collisions", "stuck", "collision", "reachable_after"):
                assert key in entry, f"trial_log entry missing {key}"
```

- [ ] **Step 2: Run the integration test (gated by manifest existence)**

```bash
cd /common/home/dm1487/robotics_research/ktamp/namo
python -m pytest python/tests/test_uniform_rollout_sampler_integration.py -v
```

Expected: PASS if manifest is present (or SKIPPED if not).

- [ ] **Step 3: If the test fails, inspect logs**

The subprocess output is captured in the failure message. Common causes:
- Missing `config/namo_config_car.yaml` → confirm file exists.
- `primitive_prefix=car_` doesn't match a primitive db → verify `car_motion_primitives_15_*.dat` exists under `data/`.
- C++ skill failures during push → unrelated to sampler logic; investigate as a separate bug.

Fix the underlying issue (not the test) and re-run.

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_uniform_rollout_sampler_integration.py
git commit -m "test: end-to-end smoke test for UniformRolloutSampler on car envs"
```

---

## Task 11: F-char regression check (point-robot envs)

**Files:**
- Test: `python/tests/test_uniform_rollout_sampler_fchar_regression.py`

Existing F-char pkls at `/common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/modular_data_rlab7/` were produced by the old exhaustive `region_opening` pipeline. Running our new sampler on the same envs (point robot) should produce primitive_trial_log entries with identical (edge_idx, depth, success) tuples for the same (object, neighbor). Differences flag a behavior change worth investigating.

- [ ] **Step 1: Pick an existing pkl and one env to compare**

```bash
ls /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_full/modular_data_rlab7/*.pkl 2>/dev/null | head -1
```

If output is non-empty, note the path — we use it as the comparison source.

- [ ] **Step 2: Create the regression test**

Create `python/tests/test_uniform_rollout_sampler_fchar_regression.py`:

```python
"""Regression test: new sampler reproduces existing F-char per-primitive labels.

Runs the new sampler on one env that was also collected by the old pipeline,
verifies the depth-0 (edge_idx, depth, success) tuples match.

Skipped if the reference F-char pkls aren't accessible.
"""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path("/common/users/dm1487/namo_data/f_characterization/"
                     "1_push_exhaustive_full/modular_data_rlab7")
RUN_REGRESSION = REFERENCE_DIR.exists()


def _extract_grid(trial_log):
    """{(obj, edge, depth) -> success} from a trial_log."""
    return {(entry["edge_idx"], entry["depth"]): bool(entry["success"])
            for entry in trial_log}


@pytest.mark.skipif(not RUN_REGRESSION, reason="reference F-char pkls not available")
def test_sampler_reproduces_existing_fchar_labels_on_one_env():
    ref_pkls = sorted(REFERENCE_DIR.glob("*.pkl"))
    assert ref_pkls, "no reference pkls found"

    # Find the first reference episode that has a primitive_trial_log.
    ref_ep = None
    ref_pkl_path = None
    for pkl in ref_pkls[:5]:                           # check first few pkls
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for ep in data.get("episode_results", []):
            stats = ep.get("algorithm_stats") or {}
            if stats.get("primitive_trial_log"):
                ref_ep = ep
                ref_pkl_path = pkl
                break
        if ref_ep:
            break
    assert ref_ep, "no reference episode with primitive_trial_log found"

    xml_file = ref_ep["xml_file"]
    ref_object = ref_ep["algorithm_stats"]["chosen_object_id"]
    ref_neighbor = ref_ep["algorithm_stats"]["neighbour_region_label"]
    ref_grid = _extract_grid(ref_ep["algorithm_stats"]["primitive_trial_log"])

    # Build a one-line manifest for that env.
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "single_env.txt"
        manifest.write_text(xml_file + "\n")

        cmd = [
            "python", str(REPO_ROOT / "python/namo/data_collection/modular_parallel_collection.py"),
            "--algorithm", "uniform_rollout_sampler",
            "--manifest", str(manifest),
            "--start-idx", "0",
            "--end-idx", "1",
            "--workers", "1",
            "--output-dir", tmp,
            "--config-file", "config/namo_config.yaml",   # point robot
            "--primitive-prefix", "",                       # legacy primitives
            "--sampler-max-chain-depth", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"sampler failed:\n{result.stdout}\n{result.stderr}"

        new_pkls = list(Path(tmp).rglob("*.pkl"))
        assert new_pkls
        with open(new_pkls[0], "rb") as f:
            new_data = pickle.load(f)

        # Find the matching (object, neighbor) attempt in the new pkl.
        match = None
        for ep in new_data["episode_results"]:
            stats = ep.get("algorithm_stats") or {}
            if (stats.get("chosen_object_id") == ref_object
                    and stats.get("neighbour_region_label") == ref_neighbor):
                match = ep
                break
        assert match, (
            f"no matching attempt for (object={ref_object}, neighbor={ref_neighbor}) "
            f"in new pkl"
        )

        new_grid = _extract_grid(match["algorithm_stats"]["primitive_trial_log"])

        # Compare overlap.
        common = set(ref_grid) & set(new_grid)
        assert common, "no overlapping (edge, depth) pairs"
        mismatches = [(e, d) for (e, d) in common if ref_grid[(e, d)] != new_grid[(e, d)]]
        mismatch_rate = len(mismatches) / len(common)
        # Allow some tolerance — small numerical sim differences are acceptable.
        assert mismatch_rate < 0.05, (
            f"{len(mismatches)}/{len(common)} ({mismatch_rate*100:.1f}%) primitives "
            f"disagree between new sampler and reference F-char. Sample mismatches: "
            f"{mismatches[:5]}"
        )
```

- [ ] **Step 3: Run the regression test**

```bash
python -m pytest python/tests/test_uniform_rollout_sampler_fchar_regression.py -v
```

Expected: PASS or SKIPPED (if reference dir is unavailable).

- [ ] **Step 4: If mismatches exceed tolerance, investigate**

A high mismatch rate signals either:
1. The sampler's enumeration order produces different (object, neighbor) pairings than `region_opening` chose. In this case adjust the test to compare the union of all (object, neighbor) attempts.
2. Sim non-determinism — re-run with fixed seed to check.
3. A real behavior bug — debug the sampler.

Do NOT relax the tolerance to mask a real bug.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_uniform_rollout_sampler_fchar_regression.py
git commit -m "test: F-char regression against existing exhaustive pkls"
```

---

## Task 12: Rollout-trace loader (analysis utility)

**Files:**
- Create: `python/namo/data_collection/rollout_trace_loader.py`
- Test: `python/tests/test_rollout_trace_loader.py`

A small utility for downstream analysis: given a sampler pkl, reconstruct per-(env, object, neighbor) F-grids and per-neighbor opening dicts. Mirrors `batch_collection_classifier.extract_instances_from_pkl` but works on the new schema and exposes the richer fields (per_neighbor_opening etc.).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_rollout_trace_loader.py`:

```python
"""Tests for rollout_trace_loader."""

import pickle
import tempfile
from pathlib import Path

import numpy as np


def test_load_attempts_from_pkl_returns_one_record_per_object_neighbor():
    """Loader extracts one record per episode_results entry, surfacing the F-grid."""
    from namo.data_collection.rollout_trace_loader import load_attempts_from_pkl

    # Build a fake pkl matching the worker's output format.
    fake_pkl = {
        "task_id": "rlab7_env_000000",
        "success": True,
        "episodes_collected": 2,
        "episode_results": [
            {
                "episode_id": "ep_0",
                "algorithm": "uniform_rollout_sampler",
                "algorithm_version": "0.1.0",
                "success": True,
                "solution_found": True,
                "solution_depth": 1,
                "xml_file": "/tmp/env.xml",
                "robot_goal": (0.0, 0.0, 0.0),
                "algorithm_stats": {
                    "chosen_object_id": "obj_1",
                    "neighbour_region_label": "neighbor_A",
                    "primitive_trial_log": [
                        {"edge_idx": 0, "depth": 0, "success": True,
                         "wall_collision": False, "movable_collisions": "",
                         "stuck": False, "collision": False, "reachable_after": 1},
                        {"edge_idx": 0, "depth": 1, "success": False,
                         "wall_collision": False, "movable_collisions": "",
                         "stuck": False, "collision": False, "reachable_after": 0},
                    ],
                    "region_goals_sampled": [(0.1, 0.0, 0.0)],
                },
            },
            {
                "episode_id": "ep_1",
                "algorithm": "uniform_rollout_sampler",
                "algorithm_version": "0.1.0",
                "success": False,
                "solution_found": False,
                "solution_depth": 0,
                "xml_file": "/tmp/env.xml",
                "robot_goal": (0.0, 0.0, 0.0),
                "algorithm_stats": {
                    "chosen_object_id": "obj_1",
                    "neighbour_region_label": "neighbor_B",
                    "primitive_trial_log": [
                        {"edge_idx": 0, "depth": 0, "success": False,
                         "wall_collision": True, "movable_collisions": "",
                         "stuck": False, "collision": True, "reachable_after": 0},
                    ],
                    "region_goals_sampled": [(0.0, 0.1, 0.0)],
                },
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "fake.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(fake_pkl, f)

        records = load_attempts_from_pkl(str(pkl_path))

    assert len(records) == 2
    rec0 = records[0]
    assert rec0["xml_file"] == "/tmp/env.xml"
    assert rec0["object_id"] == "obj_1"
    assert rec0["neighbor"] == "neighbor_A"
    assert rec0["F"] == 1
    assert rec0["R"] == 2
    assert rec0["f_ratio"] == 0.5
    # f_grid is (60, 10) with NaN for unevaluated cells, 1.0 for success, 0.0 for fail
    assert rec0["f_grid"].shape == (60, 10)
    assert rec0["f_grid"][0, 0] == 1.0
    assert rec0["f_grid"][0, 1] == 0.0
    assert np.isnan(rec0["f_grid"][1, 0])      # unevaluated

    rec1 = records[1]
    assert rec1["F"] == 0
    assert rec1["R"] == 1
```

- [ ] **Step 2: Run, confirm fails**

```bash
python -m pytest python/tests/test_rollout_trace_loader.py -v
```

Expected: ImportError on `load_attempts_from_pkl`.

- [ ] **Step 3: Implement the loader**

Create `python/namo/data_collection/rollout_trace_loader.py`:

```python
"""Utility for reading UniformRolloutSampler pkls into analysis-friendly records.

Each pkl from the modular collection pipeline contains episode_results — one entry
per (object, neighbor) AttemptResult. This module flattens those into a list of
dicts with explicit F/R/f_ratio fields and a reconstructed (60, 10) f_grid.

Compatible with batch_collection_classifier.py's extract_instances_from_pkl, but
exposes the richer fields (per_neighbor_region_goals, etc.) that the new sampler
stores in env_metadata.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np


def load_attempts_from_pkl(pkl_path: str) -> List[Dict[str, Any]]:
    """Load a worker pkl and return one record per (xml, object, neighbor).

    Each record contains the f_grid, F, R, f_ratio, region_goals_sampled, and
    references back to the original episode for downstream mask rendering.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    records: List[Dict[str, Any]] = []
    for ep in data.get("episode_results", []):
        stats = ep.get("algorithm_stats") or {}
        trial_log = stats.get("primitive_trial_log")
        if not trial_log:
            continue

        f_grid = np.full((60, 10), np.nan, dtype=np.float32)
        for trial in trial_log:
            ei = int(trial["edge_idx"])
            d = int(trial["depth"])
            if 0 <= ei < 60 and 0 <= d < 10:
                f_grid[ei, d] = 1.0 if trial["success"] else 0.0

        f_count = int(np.nansum(f_grid))
        r_count = int((~np.isnan(f_grid)).sum())

        records.append({
            "pkl_path": pkl_path,
            "xml_file": ep.get("xml_file", ""),
            "object_id": stats.get("chosen_object_id"),
            "neighbor": stats.get("neighbour_region_label"),
            "robot_goal": ep.get("robot_goal"),
            "f_grid": f_grid,
            "F": f_count,
            "R": r_count,
            "f_ratio": (f_count / r_count) if r_count > 0 else 0.0,
            "region_goals_sampled": stats.get("region_goals_sampled"),
            "episode": ep,                          # full episode for mask rendering
        })

    return records
```

- [ ] **Step 4: Run, confirm pass**

```bash
python -m pytest python/tests/test_rollout_trace_loader.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/namo/data_collection/rollout_trace_loader.py \
        python/tests/test_rollout_trace_loader.py
git commit -m "feat: rollout_trace_loader analysis utility"
```

---

## Task 13: Cluster invocation script

**Files:**
- Create: `scripts/run_uniform_rollout_collection_car.sh`

A reference invocation that documents the right env/config/primitive combination. Not required for the spec to be correct, but saves you from re-deriving it each time you launch a shard.

- [ ] **Step 1: Create the script**

Create `scripts/run_uniform_rollout_collection_car.sh`:

```bash
#!/usr/bin/env bash
# Run a UniformRolloutSampler shard on the diff-drive-car env pool.
#
# Usage: ./scripts/run_uniform_rollout_collection_car.sh <START_IDX> <END_IDX> <OUTPUT_DIR>
# Example:
#   ./scripts/run_uniform_rollout_collection_car.sh 0 9950 \
#     /common/users/dm1487/namo_data/uniform_rollout_car_v0

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <START_IDX> <END_IDX> <OUTPUT_DIR>"
    exit 1
fi

START_IDX="$1"
END_IDX="$2"
OUTPUT_DIR="$3"

MANIFEST="$(dirname "$0")/manifests/car_envs_100k.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST"
    echo "Run: ./scripts/generate_car_envs_100k_manifest.sh first."
    exit 1
fi

cd "$(dirname "$0")/.."

python python/namo/data_collection/modular_parallel_collection.py \
    --algorithm uniform_rollout_sampler \
    --manifest "$MANIFEST" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --workers 100 \
    --output-dir "$OUTPUT_DIR" \
    --config-file config/namo_config_car.yaml \
    --primitive-prefix car_ \
    --sampler-max-chain-depth 1 \
    --sampler-region-goal-samples 5 \
    --sampler-num-depths 10
```

- [ ] **Step 2: Make it executable + smoke-test argparse**

```bash
chmod +x scripts/run_uniform_rollout_collection_car.sh
# Dry-run: replace the python invocation with --help to validate argparse parses everything.
sed 's|python python/namo|python python/namo|; s|--algorithm|--help \&\& exit 0 \&\& --algorithm|' \
    scripts/run_uniform_rollout_collection_car.sh | bash 2>&1 | head -5 || true
```

Skip the smoke if it's awkward — the actual test happens in Task 10's integration run.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_uniform_rollout_collection_car.sh
git commit -m "chore: car-env collection launcher script"
```

---

## Final verification

Run all sampler-related tests:

```bash
cd /common/home/dm1487/robotics_research/ktamp/namo
python -m pytest python/tests/test_uniform_rollout_sampler.py \
                 python/tests/test_uniform_rollout_sampler_integration.py \
                 python/tests/test_uniform_rollout_sampler_fchar_regression.py \
                 python/tests/test_rollout_trace_loader.py -v
```

Expected:
- All unit tests PASS (Tasks 1–7, 12).
- Integration test PASS or SKIPPED (Task 10) depending on env availability.
- F-char regression test PASS or SKIPPED (Task 11).

After all tests pass, kick off a 10-env real-cluster smoke before the full collection:

```bash
./scripts/run_uniform_rollout_collection_car.sh 0 10 /tmp/sampler_smoke
ls /tmp/sampler_smoke/modular_data_$(hostname -s)/*.pkl | head -3
```

Spot-check one pkl with the loader:

```bash
python -c "
from namo.data_collection.rollout_trace_loader import load_attempts_from_pkl
import glob
pkls = sorted(glob.glob('/tmp/sampler_smoke/modular_data_*/*.pkl'))
records = load_attempts_from_pkl(pkls[0])
for r in records[:3]:
    print(f'{r[\"object_id\"]:>10} -> {r[\"neighbor\"]:<20} F={r[\"F\"]:>3} R={r[\"R\"]:>3} ratio={r[\"f_ratio\"]:.2f}')"
```

Expected: ≥1 record printed with sensible F/R counts.

If the smoke output looks right, dispatch the full sharded run across hosts (one sbatch invocation per node, with non-overlapping `--start-idx`/`--end-idx` ranges covering 0..29849).

---

*End of implementation plan.*
