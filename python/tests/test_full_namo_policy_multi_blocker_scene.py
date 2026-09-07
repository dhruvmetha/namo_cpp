"""Greedy policy on a real two-movable doorway ranks both blockers in one call.

Integration test against the delivered rb_00034 room (v2/zig_solo0) that the
2026-09-07 real-robot session ran. Both movables sit on the robot-to-goal
boundary. With a single-blocker candidate set the policy pinned itself to
obstacle_0_movable and spent 7 to 13 stuck pushes on it against a wall before
the blacklist let obstacle_1_movable through. This test asserts the planner
now hands the opener both blockers on the first call and returns a push on one
of them, using the same checkpoint and config the table runs use.

Skips when the scene, the checkpoint or the compiled binding is missing so the
unit suite still runs on a box without the real-experiment tree.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

NAMO_CPP = Path(__file__).resolve().parents[2]
CONFIG = NAMO_CPP / "config" / "namo_config_complete_skill15_car_1x.yaml"
SCENE = Path(
    os.environ.get(
        "NAMO_RB00034_XML",
        "/home/dhruv/projects_dhruv/namo/robot_control/real_exp/environments/"
        "real_2mov/v2/zig_solo0/rb_00034/env.xml",
    )
)
CKPT = Path(
    os.environ.get(
        "NAMO_HY5U_CKPT",
        "/home/dhruv/projects_dhruv/namo/ranking/models/HY5U_s2.ckpt",
    )
)
# Goal from the room's build sheet, in simulator metres.
GOAL_M = (0.384845, 0.654918, 0.0)
# The two movables on the sheet, both blocking the doorway to the goal region.
BLOCKERS = ["obstacle_0_movable", "obstacle_1_movable"]


def _missing() -> str | None:
    if not SCENE.exists():
        return f"scene not on this box: {SCENE}"
    if not CKPT.exists():
        return f"checkpoint not on this box: {CKPT}"
    try:
        import namo_rl  # noqa: F401
    except ImportError:
        return "compiled namo_rl binding not importable"
    return None


@pytest.mark.skipif(_missing() is not None, reason=_missing() or "")
def test_greedy_policy_ranks_both_doorway_blockers_in_one_call():
    from namo.services import NAMOPlanningService

    service = NAMOPlanningService(
        config_path=str(CONFIG), primitive_data_dir=str(NAMO_CPP / "data")
    )
    result = service.plan_from_xml(
        xml_path=str(SCENE),
        robot_goal=GOAL_M,
        algorithm="full_namo",
        max_chain_depth=2,
        full_namo_local_search="best_first",
        full_namo_exec_mode="greedy_policy",
        best_first_prior="model",
        scorer_ckpt=str(CKPT),
        ml_device=os.environ.get("NAMO_TEST_DEVICE", "cuda"),
    )

    assert result.success, result.error_message
    assert len(result.actions) == 1
    pushed = str(result.actions[0].object_id)
    assert pushed in BLOCKERS

    stats = result.algorithm_stats or {}
    assert stats.get("simulations_used", 0) == 0, "greedy_policy must not roll out"
    committed = [
        entry
        for entry in stats.get("iteration_trace", [])
        if entry.get("outcome") == "policy_step_ready"
    ]
    assert len(committed) == 1
    assert committed[0]["candidate_blockers"] == BLOCKERS
    assert committed[0]["greedy_action"]["object_id"] == pushed
