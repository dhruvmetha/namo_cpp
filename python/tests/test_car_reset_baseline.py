from __future__ import annotations

import math

import pytest

from conftest import REAL_NAMO_RL, REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()


CAR_SCENE = REPO_ROOT / "data" / "nominal_primitive_scene_square_1x_car.xml"
CAR_CONFIG = REPO_ROOT / "config" / "namo_config_complete_skill15_car_1x.yaml"
OBJECT_ID = "obstacle_1_movable"
EXPECTED_EDGE_COUNT = 60
CAR_START_POSE = (-0.3333, 0.0, 0.0)
CAR_PUSH_EDGE = 50
CAR_PUSH_DEPTH = 3
MIN_PUSH_DISPLACEMENT_M = 0.01
STATE_TOLERANCE = 1e-9


def _assert_scene_files_exist() -> None:
    assert CAR_SCENE.exists(), f"missing car scene: {CAR_SCENE}"
    assert CAR_CONFIG.exists(), f"missing car config: {CAR_CONFIG}"


def _object_xy(env) -> tuple[float, float]:
    pose = env.get_observation()[f"{OBJECT_ID}_pose"]
    return pose[0], pose[1]


def _push_and_measure(env, make_action, edge_idx: int, depth: int) -> tuple[float, dict[str, str]]:
    before = _object_xy(env)
    result = env.step(make_action(object_id=OBJECT_ID, edge_idx=edge_idx, depth=depth))
    after = _object_xy(env)
    displacement = math.hypot(after[0] - before[0], after[1] - before[1])
    info = dict(result.info) if hasattr(result, "info") and result.info else {}
    return displacement, info


def _assert_same_qpos(lhs, rhs) -> None:
    assert len(lhs.qpos) == len(rhs.qpos)
    assert lhs.qpos == pytest.approx(rhs.qpos, abs=STATE_TOLERANCE)


def test_car_deferred_warmup_reset_preserves_initialized_baseline(make_action):
    _assert_scene_files_exist()

    import namo_rl

    env = namo_rl.RLEnvironment(str(CAR_SCENE), str(CAR_CONFIG), False, True)
    env.set_robot_pose(*CAR_START_POSE)
    env.warm_up()

    baseline_state = env.get_full_state()
    reachable_before = env.get_reachable_edges(OBJECT_ID)
    assert len(reachable_before) == EXPECTED_EDGE_COUNT

    displacement_before_reset, info_before = _push_and_measure(
        env, make_action, edge_idx=CAR_PUSH_EDGE, depth=CAR_PUSH_DEPTH
    )
    assert displacement_before_reset >= MIN_PUSH_DISPLACEMENT_M, (
        f"pre-reset push moved object {displacement_before_reset:.6f} m; "
        f"expected >= {MIN_PUSH_DISPLACEMENT_M:.3f} m. info={info_before!r}"
    )

    env.reset()

    restored_state = env.get_full_state()
    _assert_same_qpos(restored_state, baseline_state)

    reachable_after = env.get_reachable_edges(OBJECT_ID)
    assert len(reachable_after) == EXPECTED_EDGE_COUNT

    displacement_after_reset, info_after = _push_and_measure(
        env, make_action, edge_idx=CAR_PUSH_EDGE, depth=CAR_PUSH_DEPTH
    )
    assert displacement_after_reset >= MIN_PUSH_DISPLACEMENT_M, (
        f"post-reset push moved object {displacement_after_reset:.6f} m; "
        f"expected >= {MIN_PUSH_DISPLACEMENT_M:.3f} m. info={info_after!r}"
    )
