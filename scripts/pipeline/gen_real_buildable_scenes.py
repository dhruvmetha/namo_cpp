#!/usr/bin/env python3
"""Generate NAMO scenes a human can physically build on the real 49.0 x 77.5 cm table.

Every existing car pool (`datasets/car_envs/v3/.../real_template_*`) copies the real table's SIZE
but nothing else: its inner walls have arbitrary lengths (13.8 cm, 24.4 cm, 3.0 cm) and arbitrary
yaw, and its blocks have arbitrary half-extents. None of it can be built. This generator samples
from the ACTUAL inventory instead, so each surviving scene is a build sheet.

Inventory is transcribed from the hardware repo `robot_control/config/{real,objects}.yaml`
(reported verbatim 2026-08-22, see the `reference-real-robot-bridge` memory):
  arena interior 49.0 x 77.5 cm, origin bottom-left, +x right, +y up
  bricks   19.5 x 5.5 x 10.0 cm, 3 confirmed on the table, a 4th declared (wall_9, orientation suspect)
  movables obj_1 15.0x7.0x4.0 and obj_4 12.0x7.5x5.0 confirmed; obj_2/3/5 declared but not sighted
The yaml states blocks as `width` x `depth` where DEPTH IS THE X EXTENT and WIDTH IS THE Y EXTENT.
MuJoCo half-extents are therefore (depth/2, width/2, height/2). Getting this backwards is silent;
the hardware side did it twice in one day. `real_test_envs/1push/1hop/env1` pins it: obj_1 appears
there as size="0.035 0.075 0.02", which is exactly (7.0/2, 15.0/2, 4.0/2) cm.

Geometry gate, applied before any physics runs. The wavefront inflates every obstacle by
`compute_wavefront_inflation_radius_m` (`include/wavefront/goal_tolerance_utils.hpp:34`) = 4.0 cm,
so a corridor needs 8.0 cm of clear width. READ THAT FUNCTION BEFORE CHANGING THIS NUMBER: it calls
`compute_rotation_safe_robot_radius_m`, whose name says diagonal but whose body returns
max(hx, hy) = 3.5 cm, the diagonal having been dropped because 4.95 cm "crowded out push
approaches". Believing the name gives 5.45 cm inflation and a 10.9 cm corridor rule, which rejects
buildable scenes: on the first pilot it disagreed with the C++ probe on 29 of 60, and the measured
disconnect threshold across that pilot was 4.04-4.25 cm, not 5.45. A scene is kept only if
  (a) with the blocker present, the goal is UNREACHABLE from the robot start
  (b) with the blocker deleted, the goal IS reachable
  (c) (b) still holds at `--margin-cm` of clearance instead of 8.0
(a)+(b) are the counterfactual certificate `probe_static_topology.py` documents. (c) exists because
8.0 cm is a hard threshold and ArUco pose noise is a few mm, so a scene built at the threshold is a
coin flip on hardware rather than an experiment.

This gate is NECESSARY, NOT SUFFICIENT: it proves the blocker is what separates the two regions, not
that any push can move it clear. Only the simulator decides that. SINGLE HOP is what this generator
enforces (one blocker between robot and goal); the push count is a property of the search, so label
at CHAIN DEPTH 2 and read both answers off one pass. `build_2push_validset.py` splits the trial log
by `chain_depth`, giving F (`valid_1push`) and F1' (`valid_first_push`) together, so a depth-2 run
says which scenes are 1-push solvable AND which of the rest are 2-push solvable.

Use `region_opening_exhaustive_2push_multihop_car.yaml` UNCHANGED. It already carries the four
settings that make a run exhaustive, and three of them fail silently when wrong:
`region_exhaustive_mode: true` is the only thing that populates `primitive_trial_log` at all
(region_opening.py:1703 writes None without it, and the answer key then has nothing to read);
`region_sample_k: 0` stops the planner drawing a random k-subset of the reachable candidates
(region_opening.py:2702, and the rung1 config DOES subsample, at k=25); and the two
`*_solutions_per_neighbor: 9999` caps stop it halting at the first hit, which would leave a
denominator that means nothing.

  PYTHONPATH=build_python:python python -m namo.data_collection.modular_parallel_collection \
      --config-yaml python/namo/data_collection/region_opening_exhaustive_2push_multihop_car.yaml \
      --manifest <out>/manifest.txt --output-dir <pkls> --workers 48

then tier with `probe_static_topology.py --expect-hop 1` + `label_keyhole1_difficulty.py`, exactly
as for any other pool. Leave `--target-goal-region` OFF: these scenes are single-hop by
construction, so letting the collection enumerate every neighbour of the robot region costs little
and lets the probe's hop count CONTRADICT this generator's numpy gate rather than hide a
disagreement between it and the C++ wavefront.

  python scripts/pipeline/gen_real_buildable_scenes.py --out-dir /path/scenes --num 2000 --seed 0
"""
import argparse
import json
import math
import os
import random
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Real inventory. All lengths in METRES. See module docstring for provenance.
# ---------------------------------------------------------------------------
ARENA_W = 0.490          # x extent of the interior
ARENA_H = 0.775          # y extent of the interior
BORDER_HALF = 0.010      # border wall half-thickness, inner faces at 0 and ARENA_*

BRICK_HALF = (0.0975, 0.0275, 0.050)     # 19.5 x 5.5 x 10.0 cm

# name -> (half_x, half_y, half_z), from objects.yaml (depth/2, width/2, height/2)
MOVABLES = {
    "obj_1": (0.0350, 0.0750, 0.020),    # 15.0 x 7.0 x 4.0, marker 5, ON TABLE
    "obj_4": (0.0375, 0.0600, 0.025),    # 12.0 x 7.5 x 5.0, marker 8, ON TABLE
    "obj_2": (0.0250, 0.0625, 0.025),    # 12.5 x 5.0 x 5.0, marker 6, declared only
    "obj_3": (0.0250, 0.0625, 0.025),    # 12.5 x 5.0 x 5.0, marker 7, declared only
    "obj_5": (0.0200, 0.0825, 0.025),    # 16.5 x 4.0 x 5.0, marker 2, declared only
}
ON_TABLE = ("obj_1", "obj_4")

ROBOT_HALF_X = 0.035                      # 7 x 7 cm chassis, so hx == hy
ROBOT_HALF_Y = 0.035
TIER1_MARGIN = 0.005                      # config/wavefront_inflation.yaml tier1
#: max(hx, hy), NOT hypot. See the docstring and goal_tolerance_utils.hpp:12.
WAVEFRONT_ROBOT_R = max(ROBOT_HALF_X, ROBOT_HALF_Y)
INFLATE_R = WAVEFRONT_ROBOT_R + TIER1_MARGIN          # 0.040 m -> 8.0 cm corridors

GRID_RES = 0.005                          # 5 mm rasterisation for the reachability gate


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class Rect:
    """Axis-aligned-in-its-own-frame box, placed at (cx, cy) with yaw."""

    __slots__ = ("cx", "cy", "hx", "hy", "yaw", "name", "kind")

    def __init__(self, cx, cy, hx, hy, yaw, name, kind):
        self.cx, self.cy, self.hx, self.hy, self.yaw = cx, cy, hx, hy, yaw
        self.name, self.kind = name, kind

    def corners(self, pad=0.0):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.hx + pad, self.hy + pad
        return [(self.cx + c * sx * hx - s * sy * hy, self.cy + s * sx * hx + c * sy * hy)
                for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    def contains(self, px, py, pad=0.0):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = px - self.cx, py - self.cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return abs(lx) <= self.hx + pad and abs(ly) <= self.hy + pad

    def aabb(self, pad=0.0):
        xs = [p[0] for p in self.corners(pad)]
        ys = [p[1] for p in self.corners(pad)]
        return min(xs), min(ys), max(xs), max(ys)


def _axes(r):
    c, s = math.cos(r.yaw), math.sin(r.yaw)
    return ((c, s), (-s, c))


def overlaps(a, b, pad=0.0):
    """Separating-axis test between two rotated rectangles, `pad` inflates both."""
    ca, cb = a.corners(pad / 2.0), b.corners(pad / 2.0)
    for ax, ay in _axes(a) + _axes(b):
        pa = [x * ax + y * ay for x, y in ca]
        pb = [x * ax + y * ay for x, y in cb]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def inside_arena(r, pad=0.0):
    x0, y0, x1, y1 = r.aabb(pad)
    return x0 >= 0.0 and y0 >= 0.0 and x1 <= ARENA_W and y1 <= ARENA_H


# ---------------------------------------------------------------------------
# Reachability gate
# ---------------------------------------------------------------------------
def _blocked_mask(rects, inflate):
    """Rasterise the arena, marking a cell blocked if its centre lies inside any inflated rect.

    The border is handled by inflating the arena edge inward by the same radius, which is what the
    wavefront does with the four boundary walls.
    """
    nx = int(round(ARENA_W / GRID_RES))
    ny = int(round(ARENA_H / GRID_RES))
    xs = (np.arange(nx) + 0.5) * GRID_RES
    ys = (np.arange(ny) + 0.5) * GRID_RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    blocked = (gx < inflate) | (gx > ARENA_W - inflate) | (gy < inflate) | (gy > ARENA_H - inflate)
    for r in rects:
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        dx, dy = gx - r.cx, gy - r.cy
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        blocked |= (np.abs(lx) <= r.hx + inflate) & (np.abs(ly) <= r.hy + inflate)
    return blocked


#: 8-connected, MATCHING `WavefrontPlanner::DIRECTIONS` (wavefront_planner.hpp:247). This must not be
#: 4-connected "because diagonal leaks through a corner pinch are a rasterisation artefact". The
#: planner takes those diagonal steps, so a 4-connected gate calls scenes blocked that the planner
#: walks straight through: on the first 60-scene pilot it disagreed with the C++ probe on 29 of them.
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _connected(blocked, start, goal):
    """8-connected flood fill from `start` cell to `goal` cell over free space."""
    nx, ny = blocked.shape
    si, sj = start
    gi, gj = goal
    if blocked[si, sj] or blocked[gi, gj]:
        return False
    seen = np.zeros_like(blocked)
    seen[si, sj] = True
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        if (i, j) == (gi, gj):
            return True
        for di, dj in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and not seen[a, b] and not blocked[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return False


def _cell(p):
    return (int(p[0] / GRID_RES), int(p[1] / GRID_RES))


def gate(statics, blocker, start, goal, margin_r):
    """Return (passed, reason). See module docstring for the three conditions."""
    with_block = _blocked_mask(statics + [blocker], INFLATE_R)
    si, sj = _cell(start)
    gi, gj = _cell(goal)
    nx, ny = with_block.shape
    if not (0 <= si < nx and 0 <= sj < ny and 0 <= gi < nx and 0 <= gj < ny):
        return False, "out_of_grid"
    if with_block[si, sj]:
        return False, "start_in_collision"
    if with_block[gi, gj]:
        return False, "goal_in_collision"
    if _connected(with_block, (si, sj), (gi, gj)):
        return False, "already_open"

    without = _blocked_mask(statics, INFLATE_R)
    if without[si, sj] or without[gi, gj] or not _connected(without, (si, sj), (gi, gj)):
        return False, "walls_alone_block_it"

    wide = _blocked_mask(statics, margin_r)
    if wide[si, sj] or wide[gi, gj] or not _connected(wide, (si, sj), (gi, gj)):
        return False, "no_margin"
    return True, "ok"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_scene(rng, movable_names, max_bricks, margin_r, tries=400):
    """One divider of bricks across the arena, one blocker narrowing the leftover gap.

    The arena is 49 cm wide and a brick is 19.5 cm, so two collinear bricks leave 9.8 cm and the
    scene is dead before any block is placed. The generator therefore samples the bricks with
    independent y offsets and yaw, which is how the hardware side's one working scene is built:
    the robot weaves around the wall rather than passing through a slot in it.
    """
    for _ in range(tries):
        n_bricks = rng.randint(1, max_bricks)
        y_div = rng.uniform(0.28, 0.50)
        statics = []
        ok = True
        for k in range(n_bricks):
            for _try in range(40):
                cx = rng.uniform(BRICK_HALF[0], ARENA_W - BRICK_HALF[0])
                cy = y_div + rng.uniform(-0.05, 0.05)
                yaw = math.radians(rng.uniform(-30.0, 30.0) if rng.random() < 0.75
                                   else rng.uniform(60.0, 120.0))
                r = Rect(cx, cy, BRICK_HALF[0], BRICK_HALF[1], yaw, f"wall_inner_{k+1}", "brick")
                if inside_arena(r) and not any(overlaps(r, o, 0.004) for o in statics):
                    statics.append(r)
                    break
            else:
                ok = False
                break
        if not ok:
            continue

        name = movable_names[rng.randrange(len(movable_names))]
        hx, hy, hz = MOVABLES[name]
        blocker = None
        for _try in range(60):
            bx = rng.uniform(max(hx, hy), ARENA_W - max(hx, hy))
            by = y_div + rng.uniform(-0.06, 0.06)
            byaw = math.radians(rng.uniform(0.0, 180.0))
            cand = Rect(bx, by, hx, hy, byaw, "obstacle_0_movable", name)
            if inside_arena(cand) and not any(overlaps(cand, o, 0.004) for o in statics):
                blocker = cand
                break
        if blocker is None:
            continue

        start = (rng.uniform(0.08, ARENA_W - 0.08), rng.uniform(0.08, y_div - 0.12))
        goal = (rng.uniform(0.08, ARENA_W - 0.08), rng.uniform(y_div + 0.14, ARENA_H - 0.08))
        if goal[1] <= start[1]:
            continue

        passed, reason = gate(statics, blocker, start, goal, margin_r)
        if passed:
            return {"statics": statics, "blocker": blocker, "blocker_name": name,
                    "start": start, "goal": goal, "y_div": y_div}, reason
    return None, "exhausted"


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
CAR_BODY = """    <body name="car" pos="{sx} {sy} 0.01">
      <freejoint name="car_freejoint" />
      <inertial pos="0 0 0.0375" mass="0.35" diaginertia="0.000266 0.000266 0.000286" />
      <geom name="front_chassis_collision" type="box" pos="0.0175 0 0.0375" size="0.0175 0.035 0.0325" rgba="0.3 0.3 0.7 1" />
      <geom name="rear_chassis_collision" type="box" pos="-0.0175 0 0.0375" size="0.0175 0.035 0.0325" rgba="0.25 0.25 0.6 1" />
      <geom name="front_marker" type="box" pos="0.034 0 0.0505" size="0.002 0.015 0.01" rgba="1 0.2 0.2 1" contype="0" conaffinity="0" />
      <body name="rear_support_body" pos="-0.03 0 0.0025">
        <joint name="rear_caster_joint" type="ball" damping="0.0001" />
        <inertial pos="0 0 0" mass="0.025" diaginertia="0.000001 0.000001 0.000001" />
        <geom name="rear_support" type="sphere" pos="0 0 0" size="0.0025" friction="0 0 0" rgba="0.7 0.1 0.1 1" />
      </body>
      <body name="front_support_body" pos="0.03 0 0.0025">
        <joint name="front_caster_joint" type="ball" damping="0.0001" />
        <inertial pos="0 0 0" mass="0.025" diaginertia="0.000001 0.000001 0.000001" />
        <geom name="front_support" type="sphere" pos="0 0 0" size="0.0025" friction="0 0 0" rgba="0.1 0.7 0.1 1" />
      </body>
      <body name="left_wheel" pos="0 0.0375 0.015">
        <inertial pos="0 0 0" mass="0.05" diaginertia="0.000003 0.000006 0.000003" />
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" armature="0.0001" />
        <geom name="left_wheel_collision" type="cylinder" size="0.015 0.0005" euler="90 0 0" rgba="0.1 0.1 0.1 1" />
      </body>
      <body name="right_wheel" pos="0 -0.0375 0.015">
        <inertial pos="0 0 0" mass="0.05" diaginertia="0.000003 0.000006 0.000003" />
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.01" armature="0.0001" />
        <geom name="right_wheel_collision" type="cylinder" size="0.015 0.0005" euler="90 0 0" rgba="0.1 0.1 0.1 1" />
      </body>
    </body>"""

HEADER = """<?xml version='1.0' encoding='utf-8'?>
<mujoco model="real_buildable">
  <compiler angle="degree" />
  <option timestep="0.002" integrator="implicitfast" iterations="100" cone="elliptic" />
  <default>
    <geom density="1" />
  </default>
  <asset>
    <texture builtin="gradient" height="3072" rgb1="0.3 0.5 0.7" rgb2="0 0 0" type="skybox" width="512" />
    <texture builtin="checker" height="300" mark="edge" markrgb="0.8 0.8 0.8" name="groundplane" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" type="2d" width="300" />
    <material name="groundplane" reflectance="0.2" texrepeat="5 5" texture="groundplane" texuniform="true" />
  </asset>
  <worldbody>
    <light dir="0 0 -1" directional="true" pos="0 0 1.5" />
    <geom name="floor" type="plane" condim="4" friction="0.5 0.005 0.001" material="groundplane" size="0 0 0.05" />
    <body name="walls">"""

FOOTER = """  </worldbody>
  <actuator>
    <velocity name="left_wheel_drive" joint="left_wheel_joint" ctrlrange="-25 25" kv="0.75" forcerange="-0.5 0.5" />
    <velocity name="right_wheel_drive" joint="right_wheel_joint" ctrlrange="-25 25" kv="0.75" forcerange="-0.5 0.5" />
  </actuator>
</mujoco>
"""

WALL_GEOM = ('      <geom name="{name}" type="box" condim="4" friction="1.0 0.005 0.0001" '
             'rgba="0.8 0.8 0.8 1" pos="{x:.6f} {y:.6f} {z:.6f}" euler="0 0 {yaw:.6f}" '
             'size="{hx:.6f} {hy:.6f} {hz:.6f}" />')


def to_xml(scene):
    hx, hy, hz = MOVABLES[scene["blocker_name"]]
    b = scene["blocker"]
    lines = [HEADER]
    lines.append(WALL_GEOM.format(name="wall_boundary_left", x=-BORDER_HALF, y=ARENA_H / 2,
                                  z=0.05, yaw=0.0, hx=BORDER_HALF, hy=ARENA_H / 2 + BORDER_HALF,
                                  hz=0.05))
    lines.append(WALL_GEOM.format(name="wall_boundary_right", x=ARENA_W + BORDER_HALF, y=ARENA_H / 2,
                                  z=0.05, yaw=0.0, hx=BORDER_HALF, hy=ARENA_H / 2 + BORDER_HALF,
                                  hz=0.05))
    lines.append(WALL_GEOM.format(name="wall_boundary_bottom", x=ARENA_W / 2, y=-BORDER_HALF,
                                  z=0.05, yaw=0.0, hx=ARENA_W / 2, hy=BORDER_HALF, hz=0.05))
    lines.append(WALL_GEOM.format(name="wall_boundary_top", x=ARENA_W / 2, y=ARENA_H + BORDER_HALF,
                                  z=0.05, yaw=0.0, hx=ARENA_W / 2, hy=BORDER_HALF, hz=0.05))
    for r in scene["statics"]:
        lines.append(WALL_GEOM.format(name=r.name, x=r.cx, y=r.cy, z=BRICK_HALF[2],
                                      yaw=math.degrees(r.yaw), hx=r.hx, hy=r.hy, hz=BRICK_HALF[2]))
    lines.append("    </body>")
    lines.append(CAR_BODY.format(sx=scene["start"][0], sy=scene["start"][1]))
    lines.append('    <body name="obstacle_0_movable">')
    lines.append(f'      <geom name="obstacle_0_movable" condim="4" '
                 f'pos="{b.cx:.6f} {b.cy:.6f} {hz:.6f}" euler="0 0 {math.degrees(b.yaw):.6f}" '
                 f'friction="1 0.005 0.0001" rgba="1 1 0 1" '
                 f'size="{hx:.6f} {hy:.6f} {hz:.6f}" type="box" mass="0.1" />')
    lines.append('      <joint type="free" />')
    lines.append("    </body>")
    lines.append(f'    <site name="goal" type="sphere" pos="{scene["goal"][0]:.6f} '
                 f'{scene["goal"][1]:.6f} 0.0" size="0.02" rgba="1 0 0 0.5" />')
    lines.append(FOOTER)
    return "\n".join(lines)


def to_build_sheet(scene, scene_id):
    """Human-readable placement, centres in cm from the bottom-left interior corner."""
    cm = lambda v: round(v * 100.0, 1)
    deg = lambda v: round(math.degrees(v) % 180.0, 1)
    return {
        "scene_id": scene_id,
        "arena_cm": [ARENA_W * 100, ARENA_H * 100],
        "bricks": [{"marker_hint": f"wall_{9 + i}", "center_cm": [cm(r.cx), cm(r.cy)],
                    "yaw_deg": deg(r.yaw), "size_cm": [19.5, 5.5]}
                   for i, r in enumerate(scene["statics"])],
        "blocker": {"object": scene["blocker_name"],
                    "center_cm": [cm(scene["blocker"].cx), cm(scene["blocker"].cy)],
                    "yaw_deg": deg(scene["blocker"].yaw),
                    "size_cm": [round(MOVABLES[scene["blocker_name"]][0] * 200, 1),
                                round(MOVABLES[scene["blocker_name"]][1] * 200, 1)]},
        "robot_start_cm": [cm(scene["start"][0]), cm(scene["start"][1])],
        "goal_cm": [cm(scene["goal"][0]), cm(scene["goal"][1])],
        "run_namo_goal_flag": f"--goal {cm(scene['goal'][0]):.0f} {cm(scene['goal'][1]):.0f}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num", type=int, default=500, help="scenes to emit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bricks", type=int, default=3,
                    help="3 bricks confirmed on the table; a 4th is declared but unsighted")
    ap.add_argument("--movables", default=",".join(ON_TABLE),
                    help=f"comma-separated; all known: {','.join(MOVABLES)}")
    ap.add_argument("--margin-cm", type=float, default=10.0,
                    help="post-push corridor width the scene must still admit. 8.0 is the hard "
                         "threshold; ArUco noise is a few mm, so build with headroom")
    args = ap.parse_args()

    movable_names = [m.strip() for m in args.movables.split(",") if m.strip()]
    unknown = [m for m in movable_names if m not in MOVABLES]
    if unknown:
        ap.error(f"unknown movables: {unknown}")
    margin_r = args.margin_cm / 200.0 + TIER1_MARGIN

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    manifest, sheets, rejects = [], [], {}
    attempts = 0
    while len(manifest) < args.num and attempts < args.num * 60:
        attempts += 1
        scene, reason = sample_scene(rng, movable_names, args.max_bricks, margin_r)
        if scene is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        sid = f"rb_{len(manifest):05d}"
        d = os.path.join(args.out_dir, sid)
        os.makedirs(d, exist_ok=True)
        xml_path = os.path.join(d, "env.xml")
        with open(xml_path, "w") as f:
            f.write(to_xml(scene))
        manifest.append(os.path.realpath(xml_path))
        sheets.append(to_build_sheet(scene, sid))

    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(args.out_dir, "build_sheets.json"), "w") as f:
        json.dump(sheets, f, indent=2)

    print(f"emitted {len(manifest)} scenes in {attempts} attempts -> {args.out_dir}")
    print(f"inventory: <={args.max_bricks} bricks, movables={movable_names}, "
          f"margin>={args.margin_cm} cm")
    if rejects:
        print("rejects:", dict(sorted(rejects.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
