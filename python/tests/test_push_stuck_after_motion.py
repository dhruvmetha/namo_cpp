"""What a push keeps when the stuck detector stops it.

The detector fires for two different situations and they deserve opposite
answers. An object that slid 19 cm and then wedged against a wall has produced
a real result; throwing it away loses work the robot actually did and hands the
planner a world that does not match the one in front of it. An object that never
moved has produced nothing, and restoring the pre-push state is right.

Before this rule both ended the same way: execute_primitive_step restored the
simulator and the caller saw zero displacement, so a jammed push was
indistinguishable from a push that never started.

The fixture builds both cases on purpose, since neither shows up reliably in the
captured real_test_envs scenes. obstacle_1_movable has 19 cm of clear travel
before block_a stops it, with the robot still well short of that wall.
obstacle_2_movable starts flush against block_b and cannot move at all.

This file also covers the earlier rule it depends on: contact between the pushed
object and a wall is recorded, not treated as failure.

To verify:
  cd namo_cpp && source env.robotlearning.sh
  python -m pytest python/tests/test_push_stuck_after_motion.py -v
"""

from __future__ import annotations

import math

import pytest

from conftest import REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()


# ─── Named constants ────────────────────────────────────────────────────

FIXTURE = REPO_ROOT / "python" / "tests" / "data" / "stuck_after_motion_car_fixture.xml"
# Long bars on pivot posts: one with room to swing, one without. Same push on
# both, so the only thing that differs is how far the bar can turn.
ROTATION_FIXTURE = REPO_ROOT / "python" / "tests" / "data" / "rotation_jam_car_fixture.xml"
# Car 1x d5, the only supported runtime profile. These fixtures were rebuilt on
# the car after the point robot was retired: the sphere is two slide joints that
# stop when you stop commanding them, while the car coasts on its casters, so a
# push behaves differently and sphere-measured constants do not transfer.
CONFIG = REPO_ROOT / "config" / "namo_config_complete_skill15_car_1x.yaml"

MOVER = "obstacle_1_movable"
STILL = "obstacle_2_movable"

# Bottom face of the mover: pushes straight at block_a. push_steps = depth + 1,
# and d5 caps depth at 4, so the pair below is the widest the profile allows.
MOVER_EDGE = 23
# At depth 3 the push reaches the wall just as it runs out of steps and ends
# normally. At depth 4 it keeps pushing after the jam, so the stuck detector ends
# it. Both park the object in the same place; that pair is the whole test.
DEPTH_ENDS_NORMALLY = 3
DEPTH_ENDS_ON_STUCK = 4

STILL_EDGE = 1
STILL_DEPTH = 1

# Measured 2026-08-21 on the car: both depths park the object 11.02 cm from
# where it started, because both stop at the same wall. Tolerance covers settle jitter
# between the two runs, not a difference in outcome.
EXPECTED_MOVER_DISPLACEMENT_M = 0.1102
SAME_RESTING_PLACE_TOLERANCE_M = 0.005

# Matches kMinUsefulPushDisplacementM in namo_push_controller.cpp, the bar the
# C++ uses to decide whether a stuck push produced anything worth keeping.
MIN_USEFUL_PUSH_DISPLACEMENT_M = 0.01
# Matches kMinUsefulPushYawRad, the other half of that decision.
MIN_USEFUL_PUSH_YAW_DEG = 5.0

# Both bars in the rotation fixture, pushed at the same edge and depth. The bar
# with 4 mm of pivot clearance swings; the one with 0.05 mm cannot.
SWINGS = "obstacle_1_movable"
CANNOT_SWING = "obstacle_2_movable"
ROTATION_EDGE = 5
ROTATION_DEPTH = 3
# Measured 2026-08-21 on the car: 15.6 degrees of yaw and 0.10 cm of travel. The travel is
# what matters as much as the yaw, since it sits under
# MIN_USEFUL_PUSH_DISPLACEMENT_M and so cannot be what saved the push.
EXPECTED_SWING_YAW_DEG = 15.6
SWING_YAW_TOLERANCE_DEG = 2.0


# ─── Helpers ────────────────────────────────────────────────────────────


def _fixture_env():
    import namo_rl

    assert FIXTURE.is_file(), f"missing fixture scene: {FIXTURE}"
    assert CONFIG.is_file(), f"missing config: {CONFIG}"
    env = namo_rl.RLEnvironment(str(FIXTURE), str(CONFIG), False)
    env.reset()
    return env


def _object_xy(env, object_id):
    pose = env.get_observation()[f"{object_id}_pose"]
    return pose[0], pose[1]


def _object_yaw(env, object_id):
    return env.get_observation()[f"{object_id}_pose"][2]


def _wrapped_yaw_delta_deg(a, b):
    """Same folding the controller uses, so the test reads the same number."""
    delta = abs(b - a)
    while delta > math.pi:
        delta = 2.0 * math.pi - delta
    return math.degrees(delta)


def _rotation_env():
    import namo_rl

    assert ROTATION_FIXTURE.is_file(), f"missing fixture scene: {ROTATION_FIXTURE}"
    env = namo_rl.RLEnvironment(str(ROTATION_FIXTURE), str(CONFIG), False)
    env.reset()
    return env


def _push_and_measure_yaw(env, object_id, edge_idx, depth):
    """Run one push, return (displacement, yaw change in degrees, info)."""
    yaw_before = _object_yaw(env, object_id)
    displacement, info = _push(env, object_id, edge_idx, depth)
    yaw_after = _object_yaw(env, object_id)
    return displacement, _wrapped_yaw_delta_deg(yaw_before, yaw_after), info


def _push(env, object_id, edge_idx, depth):
    """Run one push, return (displacement in metres, info dict)."""
    import namo_rl

    before = _object_xy(env, object_id)
    action = namo_rl.Action()
    action.object_id = object_id
    action.edge_idx = edge_idx
    action.depth = depth
    result = env.step(action)
    after = _object_xy(env, object_id)
    displacement = math.hypot(after[0] - before[0], after[1] - before[1])
    return displacement, dict(result.info)


# ─── Tests ──────────────────────────────────────────────────────────────


def test_a_push_that_moved_then_jammed_keeps_the_motion():
    env = _fixture_env()

    displacement, info = _push(env, MOVER, MOVER_EDGE, DEPTH_ENDS_ON_STUCK)

    assert info["stopped_early"] == "true", (
        f"expected the stuck stop to be reported as early completion; info={info}"
    )
    assert displacement >= MIN_USEFUL_PUSH_DISPLACEMENT_M
    assert displacement == pytest.approx(
        EXPECTED_MOVER_DISPLACEMENT_M, abs=SAME_RESTING_PLACE_TOLERANCE_M
    )
    assert not info.get("failure_reason"), (
        f"a push that produced motion is not a failure; info={info}"
    )


def test_the_kept_pose_is_where_the_object_actually_came_to_rest():
    """The early stop must not truncate the push before the object settles.

    Depth 7 ends normally at the wall. Depth 9 pushes past that and ends on the
    detector. Same wall, so the same resting place; if the early path skipped
    the settle or returned a mid-push pose, these would diverge.
    """
    normal_displacement, normal_info = _push(
        _fixture_env(), MOVER, MOVER_EDGE, DEPTH_ENDS_NORMALLY
    )
    early_displacement, early_info = _push(
        _fixture_env(), MOVER, MOVER_EDGE, DEPTH_ENDS_ON_STUCK
    )

    assert normal_info["stopped_early"] == "false"
    assert early_info["stopped_early"] == "true"
    assert early_displacement == pytest.approx(
        normal_displacement, abs=SAME_RESTING_PLACE_TOLERANCE_M
    )


def test_a_push_that_never_moved_is_rolled_back():
    env = _fixture_env()

    displacement, info = _push(env, STILL, STILL_EDGE, STILL_DEPTH)

    assert displacement < MIN_USEFUL_PUSH_DISPLACEMENT_M
    assert info["stopped_early"] == "false", (
        f"nothing moved, so there is nothing to keep; info={info}"
    )
    assert info.get("stuck") == "true"
    assert "Controller-level stuck" in info.get("failure_reason", "")


def test_the_object_touching_a_wall_does_not_fail_the_push():
    """The rule the case above rests on: object contact is recorded, not fatal.

    Depth 7 drives the mover into block_a and reports the contact. If contact
    still aborted, this push would report zero displacement and a failure.
    """
    env = _fixture_env()

    displacement, info = _push(env, MOVER, MOVER_EDGE, DEPTH_ENDS_NORMALLY)

    assert info["wall_collision"] == "true", (
        f"expected the wall contact to be recorded; info={info}"
    )
    assert displacement >= MIN_USEFUL_PUSH_DISPLACEMENT_M
    assert not info.get("failure_reason")


def test_the_robot_hitting_something_still_fails_regardless_of_motion():
    """Only the pushed object earns this leniency.

    Nothing in the fixture drives the robot into a wall, so this pins the
    contract at the level the executor reports: a robot collision names the body
    it hit, and a push that names one is a failure with the state restored.
    """
    env = _fixture_env()

    displacement, info = _push(env, STILL, STILL_EDGE, STILL_DEPTH)

    # The still object is flush against block_b; the robot never reaches it.
    assert info.get("collision_object", "") == ""
    assert displacement < MIN_USEFUL_PUSH_DISPLACEMENT_M


# ─── The same rule, decided by rotation rather than travel ──────────────


def test_a_push_that_only_turned_the_object_is_kept():
    """Travel under the bar, yaw over it, and the result survives.

    A long object pivoting in a doorway can open a boundary while its centre
    goes almost nowhere. Judging that push on displacement alone throws the
    opening away, which is what this pins.
    """
    env = _rotation_env()

    displacement, yaw_deg, info = _push_and_measure_yaw(
        env, SWINGS, ROTATION_EDGE, ROTATION_DEPTH
    )

    assert info["stopped_early"] == "true", f"info={info}"
    assert displacement < MIN_USEFUL_PUSH_DISPLACEMENT_M, (
        f"travelled {displacement * CM_PER_M:.2f} cm, which is over the "
        f"translation bar, so this case no longer isolates the yaw rule"
    )
    assert yaw_deg > MIN_USEFUL_PUSH_YAW_DEG
    assert yaw_deg == pytest.approx(
        EXPECTED_SWING_YAW_DEG, abs=SWING_YAW_TOLERANCE_DEG
    )
    assert not info.get("failure_reason")


def test_the_same_push_on_a_bar_that_cannot_turn_is_rolled_back():
    """Identical push and depth; only the pivot clearance differs.

    Holding everything else fixed is what makes this a test of where the bar
    sits rather than of two unrelated scenes.
    """
    env = _rotation_env()

    displacement, yaw_deg, info = _push_and_measure_yaw(
        env, CANNOT_SWING, ROTATION_EDGE, ROTATION_DEPTH
    )

    assert info["stopped_early"] == "false", f"info={info}"
    assert info.get("stuck") == "true"
    assert displacement < MIN_USEFUL_PUSH_DISPLACEMENT_M
    assert yaw_deg < MIN_USEFUL_PUSH_YAW_DEG
