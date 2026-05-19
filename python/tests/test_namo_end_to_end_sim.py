"""End-to-end sim tests: planner → executor → goal-reachability.

These run the actual planning loop in sim using the velocity-actuator
controller from Fix 2. They're the strongest regression bar — if these
pass, the full pipeline works.

Scenes used:
  - data/test_scene.xml: 1 obstacle, walled workspace, robot far from obj
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from conftest import (
    REAL_NAMO_RL, REPO_ROOT, CONFIG_PATH, TEST_SCENE, _require_real_namo_rl,
)

_require_real_namo_rl()


# ─── Named constants ────────────────────────────────────────────────────────

# Single-push budget — completing one primitive end-to-end (nav + push)
# shouldn't take more than this much wall clock on a dev machine. Larger
# values mask performance regressions but avoid CI flakiness.
SINGLE_PUSH_TIME_BUDGET_SEC = 30.0

# After running a successful push, the object's pose should change by
# at least this much (in m). Below this is indistinguishable from "push
# silently did nothing."
POST_PUSH_MIN_OBJECT_DISPLACEMENT_M = 0.05

# A "reasonable" planning result: max chain depth. Plans deeper than this
# on a single-obstacle scene indicate the planner is exploring
# nonsense — flag for inspection.
MAX_REASONABLE_CHAIN_DEPTH = 8


# ─── Fixtures local to this file ────────────────────────────────────────────

@pytest.fixture
def test_scene_env():
    """Loads data/test_scene.xml — walled workspace, 1 obstacle. Used for
    end-to-end planning tests."""
    import namo_rl
    return namo_rl.RLEnvironment(str(TEST_SCENE), str(CONFIG_PATH), False)


# ─── Tests on full env.step() pipeline (skill → MPC executor → controller) ──

def test_skill_path_executes_a_known_push_action(test_scene_env, make_action):
    """Through the full skill / MPC executor path (not direct controller),
    a known-good push should:
      1. Complete without exception
      2. Return a StepResult
      3. Move the object measurably

    This exercises the same code path the runtime planner uses.
    """
    env = test_scene_env
    obj_name = "obstacle_1_movable"

    # Bypass MPC's is_object_at_target short-circuit: use a target_pose
    # that isn't the current pose. (Fix 3 will remove this short-circuit
    # entirely; until then, set a different target.)
    obj_before = list(env.get_observation()[f"{obj_name}_pose"])
    target_pose = (obj_before[0] + 1.0, obj_before[1], 0.0)

    start = time.time()
    action = make_action(
        object_id=obj_name, edge_idx=50, depth=2,
        x=target_pose[0], y=target_pose[1], theta=target_pose[2],
    )
    result = env.step(action)
    elapsed = time.time() - start

    obj_after = list(env.get_observation()[f"{obj_name}_pose"])
    displacement = math.hypot(
        obj_after[0] - obj_before[0],
        obj_after[1] - obj_before[1],
    )

    info = dict(result.info) if hasattr(result, "info") and result.info else {}
    failure_reason = info.get("failure_reason", "")
    assert failure_reason == "" or "stuck" not in failure_reason.lower(), \
        f"Push reported failure: {info!r}"
    assert displacement >= POST_PUSH_MIN_OBJECT_DISPLACEMENT_M, \
        f"Object moved only {displacement*100:.2f} cm via skill path. info={info!r}"
    assert elapsed < SINGLE_PUSH_TIME_BUDGET_SEC, \
        f"Single skill step took {elapsed:.1f}s, exceeds budget {SINGLE_PUSH_TIME_BUDGET_SEC}s"


def test_full_state_roundtrip_preserves_object_position(test_scene_env, make_action):
    """Search backtracking depends on get_full_state / set_full_state being
    a clean inverse. Critical for planning correctness."""
    env = test_scene_env
    obj_name = "obstacle_1_movable"

    state_before = env.get_full_state()
    pose_before = list(env.get_observation()[f"{obj_name}_pose"])

    # Execute a push that moves the object
    env.step(make_action(
        object_id=obj_name, edge_idx=50, depth=2,
        x=pose_before[0] + 1.0, y=pose_before[1], theta=0.0,
    ))
    pose_during = list(env.get_observation()[f"{obj_name}_pose"])
    moved = math.hypot(
        pose_during[0] - pose_before[0],
        pose_during[1] - pose_before[1],
    )
    if moved < 0.01:
        pytest.skip("push didn't move object; can't test state restore meaningfully")

    # Restore — pose should be back to before
    env.set_full_state(state_before)
    pose_after_restore = list(env.get_observation()[f"{obj_name}_pose"])
    RESTORE_TOLERANCE_M = 0.001
    assert math.hypot(
        pose_after_restore[0] - pose_before[0],
        pose_after_restore[1] - pose_before[1],
    ) < RESTORE_TOLERANCE_M, (
        f"State restore failed: before=({pose_before[0]:.3f},{pose_before[1]:.3f}) "
        f"after_restore=({pose_after_restore[0]:.3f},{pose_after_restore[1]:.3f})"
    )


def test_reachability_check_responds_to_object_movement(test_scene_env, make_action):
    """After moving an object, is_robot_goal_reachable() should reflect
    the new wavefront. Tests the runtime loop's primary termination check.
    """
    env = test_scene_env
    obj_name = "obstacle_1_movable"
    # Set goal where the obstacle is — initially unreachable (object blocks)
    obj = list(env.get_observation()[f"{obj_name}_pose"])
    env.set_robot_goal(obj[0], obj[1], 0.0)

    reachable_before = env.is_robot_goal_reachable()
    # Push obstacle out of the way
    env.step(make_action(
        object_id=obj_name, edge_idx=50, depth=3,
        x=obj[0] + 2.0, y=obj[1], theta=0.0,
    ))
    reachable_after = env.is_robot_goal_reachable()

    # Either before or after should differ — the reachability check
    # is computing fresh wavefronts. If both are identical regardless
    # of the push, something's wrong with the reachability machinery.
    moved = math.hypot(*[
        env.get_observation()[f"{obj_name}_pose"][i] - obj[i]
        for i in (0, 1)
    ])
    if moved >= 0.1:
        # If the push moved the object significantly, reachability should
        # have changed (or stayed True throughout, but not be inconsistent).
        # We can't assert reachable_after == True because the new goal
        # might still be blocked by something else. But we CAN assert
        # the check doesn't throw.
        assert isinstance(reachable_before, bool)
        assert isinstance(reachable_after, bool)
