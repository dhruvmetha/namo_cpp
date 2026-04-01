import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def namo_env():
    """Create a headless NAMO environment for priority evaluation tests."""
    try:
        import namo_rl  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"namo_rl not importable: {e}")

    python_root = Path(__file__).resolve().parents[1]  # namo_cpp/python
    repo_root = python_root.parents[0]  # namo_cpp

    xml_path = python_root / "tests" / "data" / "geometric_transport_priority_fixture.xml"
    config_path = repo_root / "config" / "namo_config_complete_skill15.yaml"

    if not xml_path.exists():  # pragma: no cover
        pytest.skip(f"Missing XML fixture: {xml_path}")
    if not config_path.exists():  # pragma: no cover
        pytest.skip(f"Missing config: {config_path}")

    # Ensure mujoco can run headless in common setups.
    os.environ.setdefault("MUJOCO_GL", "egl")

    # The underlying C++ skill loads primitive databases from relative paths like
    # `data/motion_primitives_15_square.dat`, so run from the `namo_cpp/` root.
    prev_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        env = namo_rl.RLEnvironment(str(xml_path), str(config_path), visualize=False)
        env.reset()
        env.set_robot_goal(4.0, 0.0, 0.0)
        return env
    finally:
        os.chdir(prev_cwd)


def _priority_for_pose(env, object_name: str, pose_xyztheta):
    # env expects robot_goal as [x, y]
    robot_goal_xy = (4.0, 0.0)
    priorities = env.evaluate_primitive_priorities(object_name, [pose_xyztheta], robot_goal_xy)
    assert isinstance(priorities, (list, tuple))
    assert len(priorities) == 1
    return priorities[0]


def test_priority_clean_opening(namo_env):
    # Off the robot->goal path, no collisions => Priority 1
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [0.0, 3.0, 0.0])
    assert p == 1


def test_priority_movable_collision_opening(namo_env):
    # Overlaps obstacle_2_movable (movable collision), off path => Priority 2
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [2.0, 3.0, 0.0])
    assert p == 2


def test_priority_clean_no_opening(namo_env):
    # On the robot->goal path, no collisions => Priority 4
    # Use y=-0.25 to avoid overlapping obstacle_3_movable (used for Priority 5 test).
    # y=-0.25 is too close to the corridor wall after discretization/inflation in some builds.
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [2.0, -0.2, 0.0])
    assert p == 4


def test_priority_static_collision_opening(namo_env):
    # Collides with corridor wall (static collision), but doesn't block the path => Priority 3
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [2.0, 1.2, 0.0])
    assert p == 3


def test_priority_static_collision_no_opening(namo_env):
    # Collides with corridor wall AND blocks the path => Priority 6
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [2.0, 0.4, 0.0])
    assert p == 6


def test_priority_movable_collision_no_opening(namo_env):
    # Blocks the path AND overlaps obstacle_3_movable (movable collision) => Priority 5
    p = _priority_for_pose(namo_env, "obstacle_1_movable", [2.0, 0.0, 0.0])
    assert p == 5
