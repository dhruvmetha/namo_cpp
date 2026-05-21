"""Regression tests for the Fix 3 MPC `is_object_at_target` short-circuit.

The bug (pre-31e886b): `execute_primitive_step` returned SUCCESS without
running the push when target_pose was within tolerance of the object's
current pose. For direct-edge skill calls (caller passes edge_idx +
depth, target_pose defaults to (0, 0, 0)), this silently no-op'd any
push on scenes where the object started at the origin.

These tests fail on pre-fix code and pass on post-fix code.
"""
from __future__ import annotations

import math

import pytest

from conftest import (
    REAL_NAMO_RL, REPO_ROOT, CONFIG_PATH, _require_real_namo_rl,
)

_require_real_namo_rl()


# Object on the primitive-gen square scene is at (0, 0, 0). Any
# target_pose within distance_threshold of that triggered the pre-fix
# short-circuit. We use (0, 0, 0) explicitly to exercise the exact
# bug condition.
PLACEHOLDER_TARGET_X = 0.0
PLACEHOLDER_TARGET_Y = 0.0
PLACEHOLDER_TARGET_THETA = 0.0

# Minimum object displacement that proves the controller actually ran.
# Pre-fix this was always 0 (controller never ran). Post-fix the canonical
# good push moves the object several meters; 5 cm is a generous lower
# bound that any working push exceeds.
MIN_DISPLACEMENT_PROVING_PUSH_RAN_M = 0.05


def test_direct_edge_with_target_at_object_pose_executes_push(
    primgen_square_env, make_action
):
    """The exact pre-fix bug condition: object at origin, target also at
    origin. Pre-fix: returns SUCCESS with displacement=0. Post-fix:
    actually runs the controller and the object moves."""
    env = primgen_square_env

    # Confirm the bug precondition: object is at origin.
    obj_pose = list(env.get_observation()["obstacle_1_movable_pose"])
    OBJ_AT_ORIGIN_TOLERANCE_M = 0.01
    assert abs(obj_pose[0]) < OBJ_AT_ORIGIN_TOLERANCE_M, \
        f"prim-gen object isn't at origin ({obj_pose[0]:.3f}); test premise invalid"
    assert abs(obj_pose[1]) < OBJ_AT_ORIGIN_TOLERANCE_M, \
        f"prim-gen object isn't at origin ({obj_pose[1]:.3f}); test premise invalid"

    # Send push with target_pose = (0, 0, 0) — same as object pose.
    # Pre-fix this hits the short-circuit. Post-fix it runs the push.
    action = make_action(
        object_id="obstacle_1_movable",
        edge_idx=50, depth=2,
        x=PLACEHOLDER_TARGET_X,
        y=PLACEHOLDER_TARGET_Y,
        theta=PLACEHOLDER_TARGET_THETA,
    )
    result = env.step(action)
    obj_pose_after = list(env.get_observation()["obstacle_1_movable_pose"])
    displacement = math.hypot(
        obj_pose_after[0] - obj_pose[0],
        obj_pose_after[1] - obj_pose[1],
    )

    info = dict(result.info) if hasattr(result, "info") and result.info else {}
    assert displacement >= MIN_DISPLACEMENT_PROVING_PUSH_RAN_M, (
        f"With target=(0,0,0) on an object-at-origin scene, the controller "
        f"didn't run the push (displacement {displacement*100:.2f} cm, "
        f"expected ≥ {MIN_DISPLACEMENT_PROVING_PUSH_RAN_M*100:.0f} cm). "
        f"This is the pre-Fix-3 short-circuit bug. info={info!r}"
    )


def test_direct_edge_with_target_far_from_object_still_works(
    primgen_square_env, make_action
):
    """Sanity: removing the short-circuit shouldn't break the case where
    target IS meaningfully different from object pose. Target (1, 1, 0)
    is well outside any distance_threshold; pre-fix this already worked,
    post-fix it should continue to work."""
    env = primgen_square_env
    obj_pose_before = list(env.get_observation()["obstacle_1_movable_pose"])

    action = make_action(
        object_id="obstacle_1_movable",
        edge_idx=50, depth=2,
        x=1.0, y=1.0, theta=0.0,
    )
    env.step(action)
    obj_pose_after = list(env.get_observation()["obstacle_1_movable_pose"])
    displacement = math.hypot(
        obj_pose_after[0] - obj_pose_before[0],
        obj_pose_after[1] - obj_pose_before[1],
    )
    assert displacement >= MIN_DISPLACEMENT_PROVING_PUSH_RAN_M, \
        f"Push with non-origin target failed (was working pre-fix); " \
        f"displacement={displacement*100:.2f} cm"


@pytest.mark.parametrize("edge_idx", [10, 31, 45, 55])
def test_short_circuit_removed_across_diverse_edges(
    primgen_square_env, make_action, edge_idx
):
    """The pre-fix short-circuit was edge-agnostic — it would fire on any
    edge if the target pose matched the object pose. Parametrize over a
    few representative edges to confirm fix coverage."""
    env = primgen_square_env
    obj_pose_before = list(env.get_observation()["obstacle_1_movable_pose"])

    action = make_action(
        object_id="obstacle_1_movable",
        edge_idx=edge_idx, depth=2,
        x=0.0, y=0.0, theta=0.0,  # placeholder target = object pose
    )
    env.step(action)
    obj_pose_after = list(env.get_observation()["obstacle_1_movable_pose"])
    displacement = math.hypot(
        obj_pose_after[0] - obj_pose_before[0],
        obj_pose_after[1] - obj_pose_before[1],
    )
    assert displacement >= MIN_DISPLACEMENT_PROVING_PUSH_RAN_M, (
        f"edge={edge_idx} with placeholder target=(0,0,0) didn't push "
        f"(displacement {displacement*100:.2f} cm). Short-circuit may "
        f"have re-introduced."
    )
