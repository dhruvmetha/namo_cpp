"""The C++ ports of robot_control's FollowPathController and NavigationController match the Python.

Car v2 (skill.car_push_follower "follow_path", skill.car_nav_mode "drive") runs C++ copies of the
real robot's controllers: PushPathFollower with align_mode on, and CarNavigationController. These
tests feed one pose and timestamp sequence to the Python controller and to the C++ port through
the namo_rl bindings, and require equal wheel commands to 1e-12 on every tick, plus the same
NavigationController state on every tick.

Units follow the Python: cm and degrees. The ports take lengths in any unit (CarNavigationParams.cm
says how long a centimetre is), so the C++ runs in cm here and no conversion sits between the two.

The pose sequences are closed loop. A unicycle model integrates the Python command each tick, then
noise and occasional heading jumps of 70-150 degrees are added, so ALIGN (60/45 degree hysteresis)
and pre/post rotation all fire. Needs robot_control checked out next to this repo.
"""
from __future__ import annotations

import math
import random
import sys
import zlib
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()

ROBOT_CONTROL = REPO_ROOT.parent / "robot_control"
if not (ROBOT_CONTROL / "src" / "robot_control").is_dir():
    pytest.skip(f"robot_control not found at {ROBOT_CONTROL}", allow_module_level=True)
sys.path.insert(0, str(ROBOT_CONTROL / "src"))

import namo_rl  # noqa: E402
from robot_control.controller import navigation as nav_module  # noqa: E402
from robot_control.controller.config import load_controller_configs  # noqa: E402
from robot_control.controller.follow_path import FollowPathController  # noqa: E402
from robot_control.core.types import Observation, WorkspaceConfig  # noqa: E402

TOL = 1e-12
CAR_CM = 7.0                      # real.yaml robot width_cm / height_cm
WHEEL_BASE_CM = 7.0               # run_namo.py:517 wheel_base=car_width_cm
CHASSIS_CM_S_PER_UNIT = 13.39     # 0.05357 m/s / 0.4: v1's calibrated push speed at the hw cap
CONTROLLER_YAML = ROBOT_CONTROL / "config" / "controller.yaml"
V2_CONFIG = REPO_ROOT / "config" / "namo_config_complete_skill15_car_v2_1x.yaml"

WORKSPACE = WorkspaceConfig(
    width=49.0, height=77.5, car_width=CAR_CM, car_height=CAR_CM,
    offset_w=0.0, offset_h=0.0, wheel_base=WHEEL_BASE_CM,
)
CONFIGS = load_controller_configs(CONTROLLER_YAML)


def _obs(x, y, theta_deg, t):
    return Observation(robot_x=x, robot_y=y, robot_theta=theta_deg, objects={}, timestamp=t)


def _integrate(pose, left, right, dt):
    x, y, th = pose
    v = 0.5 * (left + right) * CHASSIS_CM_S_PER_UNIT
    w = (right - left) / WHEEL_BASE_CM * CHASSIS_CM_S_PER_UNIT
    th_rad = math.radians(th)
    return (x + v * math.cos(th_rad) * dt, y + v * math.sin(th_rad) * dt,
            th + math.degrees(w * dt))


def _perturb(rng, pose, tick, jump_every):
    x, y, th = pose
    x += rng.gauss(0.0, 0.05)
    y += rng.gauss(0.0, 0.05)
    th += rng.gauss(0.0, 1.0)
    if jump_every and tick > 0 and tick % jump_every == 0:
        th += rng.choice([-1.0, 1.0]) * rng.uniform(70.0, 150.0)
    th = (th + 180.0) % 360.0 - 180.0
    return x, y, th


def _staircase(points, cell=0.5):
    """8-connected grid walk through `points`, like the wavefront BFS path NAMO hands the nav."""
    out = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        gx, gy = round(x0 / cell), round(y0 / cell)
        tx, ty = round(x1 / cell), round(y1 / cell)
        while (gx, gy) != (tx, ty):
            out.append((gx * cell, gy * cell))
            gx += (tx > gx) - (tx < gx)
            gy += (ty > gy) - (ty < gy)
    out.append((round(points[-1][0] / cell) * cell, round(points[-1][1] / cell) * cell))
    return out


def _cpp_follower(lookahead, goal_tolerance, max_speed):
    p = namo_rl.PushPathFollowerParams()
    p.robot_width_m = CAR_CM
    p.robot_height_m = CAR_CM
    p.wheel_base_m = WHEEL_BASE_CM
    p.lookahead_distance_m = lookahead
    p.goal_tolerance_m = goal_tolerance
    p.max_speed = max_speed
    p.max_point_gap_ratio = 0.01
    p.no_skip_ratio = 0.5
    p.wheel_deadband = 0.05
    p.align_mode = True
    return namo_rl.PushPathFollower(p)


# push.py:571-590 push path: actual_pos -> shifted mid -> 50 cm beyond. The first leg is the
# edge-to-mid length of obstacle_0 in the drive smoke-test scene: half-depth 4.6246 cm + robot
# radius 3.5 cm + edge offset 1 cm.
#
# Not 7 cm, on purpose. 7 cm resamples to exactly 100 gaps of 0.07 cm, so the no-skip gate
# (0.5 x car_size = 3.5 cm = 50 gaps) lands exactly on a path point, and whether that point is in
# the window comes down to rounding. The C++ is built with -ffast-math (fused multiply-adds in the
# resampler), the Python is not, so on such knife-edge lengths the two can pick targets one point
# apart and differ by ~1e-4. That is a float tie, not a logic difference.
LEG = 4.624618400891002 + 3.5 + 1.0


def _push_path(x, y, heading_deg):
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return [(x, y), (x + LEG * c, y + LEG * s), (x + (LEG + 50.0) * c, y + (LEG + 50.0) * s)]


# (name, path, start pose (x, y, theta_deg), heading-jump period in ticks or 0)
FOLLOW_CASES = [
    ("push_aligned", _push_path(10.0, 10.0, 0.0), (10.0, 10.0, 0.0), 0),
    ("push_70deg_off", _push_path(10.0, 10.0, 23.0), (10.0, 10.0, 93.0), 0),
    ("push_backwards", _push_path(10.0, 10.0, 90.0), (10.0, 10.0, -95.0), 0),
    ("push_jumps", _push_path(12.3, 8.1, -17.0), (12.3, 8.1, -12.0), 37),
    ("corner_120deg", [(0.0, 0.0), (20.0, 0.0), (10.0, 17.3), (0.0, 34.6)], (0.0, 0.0, 0.0), 0),
    ("zigzag_off_path", [(0.0, 0.0), (15.0, 15.0), (30.0, 0.0), (45.0, 15.0)], (-20.0, 8.0, 160.0), 53),
]


@pytest.mark.parametrize("name,path,start,jump_every", FOLLOW_CASES, ids=[c[0] for c in FOLLOW_CASES])
@pytest.mark.parametrize("role", ["push", "navigation"])
def test_follow_path_controller_wheel_commands_match(name, path, start, jump_every, role):
    if role == "push":
        # push.py:182-187 with controller.yaml push: block.
        lookahead = CONFIGS.push.lookahead_ratio * CAR_CM
        goal_tolerance = CAR_CM * 0.3
        max_speed = CONFIGS.push.max_speed
    else:
        # navigation.py:284-294 with controller.yaml navigation: block.
        lookahead = 1.0 * CAR_CM
        goal_tolerance = CONFIGS.navigation.goal_tolerance_ratio * CAR_CM
        max_speed = CONFIGS.navigation.max_speed
    py = FollowPathController(WORKSPACE, lookahead_distance=lookahead, max_speed=max_speed,
                              goal_tolerance=goal_tolerance)
    cpp = _cpp_follower(lookahead, goal_tolerance, max_speed)
    py.set_path(path)
    cpp.set_path(path)

    rng = random.Random(zlib.crc32(name.encode()))
    pose, t = start, 0.0
    modes = set()
    for tick in range(900):
        a = py.step(_obs(pose[0], pose[1], pose[2], t), None)
        left, right, cpp_mode = cpp.step(pose[0], pose[1], math.radians(pose[2]), t)
        assert abs(a.left_speed - left) <= TOL and abs(a.right_speed - right) <= TOL, (
            f"tick {tick}: python ({a.left_speed!r}, {a.right_speed!r}) "
            f"c++ ({left!r}, {right!r}) mode {py.metadata.get('mode')}/{cpp_mode}")
        if py.get_status() == "FINISHED":
            assert cpp_mode == "FINISHED"
            break
        assert (py.metadata.get("mode") == "ALIGN") == (cpp_mode == "ALIGN"), f"tick {tick}"
        modes.add(cpp_mode)
        dt = 1.0 / 30.0 + rng.uniform(-0.004, 0.004)
        t += dt
        pose = _perturb(rng, _integrate(pose, a.left_speed, a.right_speed, dt), tick, jump_every)
    if name != "push_aligned":
        assert "ALIGN" in modes, f"{name} never entered ALIGN; the case does not test it"


# (name, waypoints, start pose, goal theta (deg) or None, heading-jump period, freeze window)
NAV_CASES = [
    ("facing_away", [(5.0, 5.0), (25.0, 5.0), (25.0, 30.0)], (5.0, 5.0, 175.0), 90.0, 0, None),
    ("aligned_no_goal_theta", [(5.0, 5.0), (30.0, 12.0)], (5.0, 5.0, 10.0), None, 0, None),
    ("post_rotation_far", [(5.0, 40.0), (30.0, 40.0), (30.0, 15.0)], (5.0, 40.0, -30.0), -170.0, 0, None),
    ("heading_jumps", [(5.0, 5.0), (20.0, 20.0), (40.0, 20.0)], (5.0, 5.0, 45.0), 0.0, 41, None),
    ("already_at_goal", [(5.0, 5.0), (6.5, 5.5)], (5.0, 5.0, 0.0), 120.0, 0, None),
    # Robot stops moving mid-drive: the no-progress watchdog fires after 3 s.
    ("watchdog_blocked", [(5.0, 5.0), (40.0, 5.0)], (5.0, 5.0, 0.0), 0.0, 0, (40, 400)),
]


def _cpp_nav():
    nav = CONFIGS.navigation
    p = namo_rl.CarNavigationParams()
    p.car_width = CAR_CM
    p.car_height = CAR_CM
    p.wheel_base = WHEEL_BASE_CM
    p.cm = 1.0
    p.max_speed = nav.max_speed
    p.pre_rotation_skip_angle = nav.pre_rotation_skip_angle
    p.goal_tolerance_ratio = nav.goal_tolerance_ratio
    p.rotation_tolerance_deg = nav.rotation_tolerance_deg
    p.rotation_stable_time = nav.rotation_stable_time
    p.rotation_speed_max = nav.rotation_speed_max
    p.rotation_speed_min = nav.rotation_speed_min
    p.wheel_deadband = nav.wheel_deadband
    return namo_rl.CarNavigationController(p)


class _Clock:
    """Stands in for the time module inside navigation.py: its stable hold reads time.time()."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now


@pytest.mark.parametrize("name,waypoints,start,goal_theta,jump_every,freeze", NAV_CASES,
                         ids=[c[0] for c in NAV_CASES])
def test_navigation_controller_states_and_commands_match(monkeypatch, name, waypoints, start,
                                                         goal_theta, jump_every, freeze):
    clock = _Clock()
    monkeypatch.setattr(nav_module, "time", clock)
    py = nav_module.NavigationController(
        WORKSPACE, planner=None, nav_config=CONFIGS.navigation,
        retreat_config=CONFIGS.push, now_fn=clock.time)
    # RETREATING is not ported (the C++ fails at once). Stub the retreat set-up so the test does
    # not need the Python wavefront; the watchdog verdict is what is compared.
    monkeypatch.setattr(py, "_begin_retreat",
                        lambda obs: setattr(py, "_state", nav_module.NavigationState.RETREATING))
    cpp = _cpp_nav()

    path = _staircase(waypoints)
    current = (start[0], start[1])
    assert py.navigate_to(path[-1][0], path[-1][1], goal_theta, current, [], path=path) == \
        cpp.navigate_to(goal_theta, current, path)
    assert py.state.value == cpp.state

    rng = random.Random(zlib.crc32(name.encode()))
    pose, t = start, 0.0
    states = []
    for tick in range(3000):
        clock.now = t
        a = py.step(_obs(pose[0], pose[1], pose[2], t), None)
        left, right = cpp.step(pose[0], pose[1], pose[2], t)
        py_state = py.state.value
        if py_state == "RETREATING":
            py_state = "FAILED"
        assert py_state == cpp.state, f"tick {tick}: python {py.state.value} c++ {cpp.state}"
        assert abs(a.left_speed - left) <= TOL and abs(a.right_speed - right) <= TOL, (
            f"tick {tick} {cpp.state}: python ({a.left_speed!r}, {a.right_speed!r}) "
            f"c++ ({left!r}, {right!r})")
        if not states or states[-1] != cpp.state:
            states.append(cpp.state)
        if cpp.state in ("FINISHED", "FAILED"):
            break
        dt = 1.0 / 30.0 + rng.uniform(-0.004, 0.004)
        t += dt
        if freeze and freeze[0] <= tick < freeze[1]:
            continue
        pose = _perturb(rng, _integrate(pose, a.left_speed, a.right_speed, dt), tick, jump_every)

    expected_end = "FAILED" if name == "watchdog_blocked" else "FINISHED"
    assert states[-1] == expected_end, f"{name}: states {states}"
    if name == "watchdog_blocked":
        assert cpp.failure_cause == py._failure_cause
    if name == "facing_away":
        assert states[:2] == ["PRE_ROTATING", "FOLLOWING"] and "POST_ROTATING" in states


def test_v2_config_carries_controller_yaml_values():
    skill = yaml.safe_load(V2_CONFIG.read_text())["skill"]
    nav = CONFIGS.navigation
    for field in ("max_speed", "pre_rotation_skip_angle", "goal_tolerance_ratio",
                  "rotation_tolerance_deg", "rotation_stable_time", "rotation_speed_max",
                  "rotation_speed_min", "wheel_deadband"):
        assert skill["navigation"][field] == getattr(nav, field), field
    assert skill["push_tracker_max_speed"] == CONFIGS.push.max_speed
    assert skill["push"]["lookahead_ratio"] == CONFIGS.push.lookahead_ratio
    assert skill["push"]["approach_skip_distance_m"] * 100.0 == CONFIGS.push.approach_skip_distance
    assert skill["push"]["approach_skip_angle"] == CONFIGS.push.approach_skip_angle
    assert skill["dynamic_direction"] == CONFIGS.push.dynamic_direction
