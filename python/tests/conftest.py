"""Shared pytest fixtures + namo_rl stub for tests that don't need real bindings.

Two modes:
  - When the compiled namo_rl module is on PYTHONPATH, the real binding is
    used and fixtures below provide loaded environments / planners.
  - When it isn't, a permissive stub is installed and tests that import
    from `namo.*` keep working (they exercise pure-Python logic).

Tests that *require* the real binding should declare
    pytestmark = pytest.mark.skipif(not REAL_NAMO_RL, reason="...")
or call `_require_real_namo_rl()` at module load.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Repo root = namo_cpp/. Tests are at python/tests/.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "namo_config_complete_skill15_car_1x.yaml"

# Canonical scenes used by multiple integration / e2e tests.
PRIMGEN_SCENE = REPO_ROOT / "data" / "nominal_primitive_scene_square_1x_car.xml"
TEST_SCENE = PRIMGEN_SCENE
CAR_START_POSE = (-0.3333, 0.0, 0.0)

# Detect whether the REAL namo_rl is importable. Used by tests that need
# physics, primitive generation, or actual planning.
def _detect_real_namo_rl() -> bool:
    if "namo_rl" in sys.modules:
        # Already loaded — could be stub or real
        return hasattr(sys.modules["namo_rl"], "RLEnvironment") and \
               sys.modules["namo_rl"].RLEnvironment is not object
    try:
        import namo_rl  # noqa: F401
        return True
    except ImportError:
        return False


REAL_NAMO_RL = _detect_real_namo_rl()


def _require_real_namo_rl() -> None:
    """Module-level skip helper. Use as:
        from conftest import _require_real_namo_rl
        _require_real_namo_rl()
    """
    if not REAL_NAMO_RL:
        pytest.skip(
            "real namo_rl binding not available; "
            "set PYTHONPATH to include namo_cpp/build_python "
            "to run physics-dependent tests",
            allow_module_level=True,
        )


# Install stub only if we don't have the real one (preserves legacy tests
# that rely on the stub for pure-Python imports).
if not REAL_NAMO_RL and "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    namo_rl_stub.StepResult = type("StepResult", (), {})
    sys.modules["namo_rl"] = namo_rl_stub
elif not REAL_NAMO_RL and not hasattr(sys.modules["namo_rl"], "StepResult"):
    sys.modules["namo_rl"].StepResult = type("StepResult", (), {})


# ─── Shared fixtures (require real namo_rl) ─────────────────────────────────

@pytest.fixture
def primgen_square_env():
    """Loaded RLEnvironment for the square primitive-generation scene."""
    if not REAL_NAMO_RL:
        pytest.skip("requires real namo_rl")
    import namo_rl
    env = namo_rl.RLEnvironment(str(PRIMGEN_SCENE), str(CONFIG_PATH), False, True)
    env.set_robot_pose(*CAR_START_POSE)
    env.warm_up()
    return env


@pytest.fixture(params=["square", "wide", "tall"])
def primgen_env_by_shape(request):
    """Parametrized: yields a loaded env for each of the 3 prim-gen shapes.

    Tests using this fixture run once per shape — handy for asserting the
    same property holds across all shapes.
    """
    if not REAL_NAMO_RL:
        pytest.skip("requires real namo_rl")
    import namo_rl
    scene = REPO_ROOT / "data" / f"nominal_primitive_scene_{request.param}_1x_car.xml"
    env = namo_rl.RLEnvironment(str(scene), str(CONFIG_PATH), False, True)
    env.set_robot_pose(*CAR_START_POSE)
    env.warm_up()
    return request.param, env


@pytest.fixture
def push_good_params():
    """Canonical 'good push' fixture: an (object_id, edge_idx, depth) tuple
    known to move the object on the car 1x d5 square scene
    under the current controller. Used by integration / e2e tests."""
    return dict(object_id="obstacle_1_movable", edge_idx=50, depth=2)


@pytest.fixture
def make_action():
    """Factory for namo_rl.Action with sensible defaults. Reduces test
    boilerplate from 5 lines to 1.

    Default target_pose is (0, 0, 0) — a placeholder commonly used by
    direct-edge callers that care about edge_idx + depth and don't have
    a meaningful target. Pre-Fix-3 this would silently no-op on scenes
    where the object started at the origin; post-Fix-3 the push runs
    regardless.
    """
    if not REAL_NAMO_RL:
        pytest.skip("requires real namo_rl")
    import namo_rl

    def _make(object_id: str, edge_idx: int, depth: int,
              x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        a = namo_rl.Action()
        a.object_id = object_id
        a.edge_idx = edge_idx
        a.depth = depth
        a.x = x
        a.y = y
        a.theta = theta
        return a
    return _make
