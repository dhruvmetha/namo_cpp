#!/usr/bin/env python3
"""Baseline microbenchmark for the region-opening HOT ops (per-boundary timing).

cProfile can't see into the C++ extension, so we time each Python->C++ boundary call
that the region-opening search loop hammers, and cross-multiply by the known per-candidate
call counts (from the hot-path trace) to expose redundant-wavefront-rebuild overhead.

Ops timed (each = a full-grid rebuild + read, except step which also runs physics):
  get_reachable_objects()        1 WavefrontPlanner rebuild
  is_robot_goal_reachable()      1 WavefrontPlanner rebuild + check
  get_reachable_edges(obj)       1 WavefrontPlanner rebuild
  get_region_snapshot()          WavefrontGrid: 2 rebuilds now (double-rebuild bug) -> 1 after fix
  step(push)                     physics (settle+push+settle) + 6 WavefrontPlanner rebuilds

C++ stdout (grid/connectivity prints) is redirected to /dev/null during timing so I/O
doesn't pollute the numbers. Deterministic, single-thread:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 <python> scripts/sandbox/profile_push.py
"""
import contextlib
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts/sandbox"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl  # noqa: E402
from namo.strategies import PrimitiveGoalStrategy  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.paths import resolve  # noqa: E402
import json  # noqa: E402

CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"
DATA_DIR = f"{REPO}/data"
PRIM_PREFIX = "1x_car_d5_"
FALLBACK_GOAL = (-0.5, 1.3, 0.0)
LABELS = "/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/pure2push.json"

N_SCENES = 6
REPS = 40  # per-op repetitions for a stable median


@contextlib.contextmanager
def silence_stdout():
    """Redirect C-level fd 1 to /dev/null (silences the C++ std::cout during timing)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return st.median(ts), min(ts), max(ts)


def main():
    data = json.load(open(LABELS))
    scenes = sorted(data.keys())[:N_SCENES]
    prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)

    agg = {k: [] for k in ["reach_objs", "goal_reach", "reach_edges", "region_snap", "step_push"]}
    grid_dims = None

    for xml_key in scenes:
        with silence_stdout():
            env = namo_rl.RLEnvironment(str(resolve(xml_key)), CFG, False)
            env.reset()
            env.set_collision_checking(False)
            goal = extract_goal_with_fallback(str(resolve(xml_key)), FALLBACK_GOAL)
            env.set_robot_goal(float(goal[0]), float(goal[1]), float(goal[2]))
            reach = sorted(str(o) for o in env.get_reachable_objects())
            s0 = env.get_full_state()
            if not reach:
                continue
            obj = reach[0]
            per_edge = prim.generate_goals(obj, s0, env, 0)
            goals = [g for e in per_edge for g in e]
            goals.sort(key=lambda g: (int(g.depth), int(g.edge_idx)))
            push = goals[0] if goals else None

            def mk_step():
                env.set_full_state(s0)
                a = namo_rl.Action()
                a.object_id = obj
                a.x, a.y, a.theta = float(push.x), float(push.y), float(push.theta)
                a.edge_idx, a.depth = int(push.edge_idx), int(push.depth)
                env.step(a)

            m_reach = timed(lambda: (env.set_full_state(s0), env.get_reachable_objects()), REPS)
            m_goal = timed(lambda: (env.set_full_state(s0), env.is_robot_goal_reachable()), REPS)
            m_edge = timed(lambda: (env.set_full_state(s0), env.get_reachable_edges(obj)), REPS)
            m_snap = timed(lambda: (env.set_full_state(s0),
                                    get_region_snapshot(env, goals_per_region=0, seed=42)), REPS)
            m_step = timed(mk_step, max(6, REPS // 4)) if push else (float("nan"),) * 3

        agg["reach_objs"].append(m_reach[0])
        agg["goal_reach"].append(m_goal[0])
        agg["reach_edges"].append(m_edge[0])
        agg["region_snap"].append(m_snap[0])
        if push:
            agg["step_push"].append(m_step[0])

    def med(x):
        return st.median(x) if x else float("nan")

    wf = med(agg["reach_objs"])  # one WavefrontPlanner rebuild+read
    step = med(agg["step_push"])
    snap = med(agg["region_snap"])
    print(f"\n=== BASELINE per-op median (ms), {len(scenes)} scenes x {REPS} reps ===")
    print(f"  get_reachable_objects   {wf:8.3f}   (1 wavefront rebuild)")
    print(f"  is_robot_goal_reachable {med(agg['goal_reach']):8.3f}   (1 rebuild + check)")
    print(f"  get_reachable_edges     {med(agg['reach_edges']):8.3f}   (1 rebuild)")
    print(f"  get_region_snapshot     {snap:8.3f}   (WavefrontGrid: 2 rebuilds NOW -> 1 after fix)")
    print(f"  step(push)              {step:8.3f}   (physics + 6 wavefront rebuilds)")
    print(f"\n=== redundancy estimate (from hot-path trace) ===")
    print(f"  ~6 wavefront rebuilds inside one step @ {wf:.3f} ms = {6*wf:8.3f} ms of the {step:.3f} ms step")
    print(f"  collapsing to ~2 would save ~{4*wf:8.3f} ms/step ({100*4*wf/step:.0f}% of step) if rebuild-bound")
    print(f"  get_region_snapshot double-rebuild: halving saves ~{snap/2:.3f} ms/snapshot")


if __name__ == "__main__":
    main()
