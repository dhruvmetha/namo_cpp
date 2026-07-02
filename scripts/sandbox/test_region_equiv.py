#!/usr/bin/env python3
"""Behavior-equivalence gate for the region-opening / wavefront / push code path.

Analog of test_render_equiv.py ("158/158 diff=0"), but for the SIM/reachability path
that the wavefront cleanup touches. Drives a FIXED, deterministic sequence of pushes
through RLEnvironment directly (NO planner, NO model, NO RNG) on a stratified sample of
real test scenes, and captures the observable outputs of the wavefront/region/push code:

  - get_reachable_objects()            (sorted list[str])
  - get_reachable_edges(obj)           (sorted list[int]) per reachable object
  - is_robot_goal_reachable()          (bool)
  - get_region_snapshot(...) C++ path  (adjacency graph, region_labels, robot/goal labels,
                                         goal_reachable, goal_in_free_space)  <- the rl_env.cpp
                                         double-rebuild fix changes THIS call, so we gate it.
  - full-state qpos                    (post-push physics fingerprint; tolerance-checked)

Determinism (verified by the eval scout): set_full_state zeros qvel; step()/physics and the
wavefront BFS carry no RNG; PrimitiveGoalStrategy(shuffle_edges=False) is deterministic;
get_region_snapshot uses a fixed seed. So a behavior-preserving refactor must reproduce every
DISCRETE observable exactly (N/N), and qpos to within FP-recompilation noise.

Workflow:
  # ON THE FROZEN (pre-refactor) build, once:
  MODE=capture python scripts/sandbox/test_region_equiv.py --n-per-tier 8
  # AFTER each wavefront edit (rebuild the .so first):
  MODE=compare python scripts/sandbox/test_region_equiv.py --n-per-tier 8
  # ship only on DISCRETE N/N and qpos max|diff| ~ 0.

Run single-threaded / CPU-only for determinism (the slurm wrapper pins this):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 <python> scripts/sandbox/test_region_equiv.py ...
"""
import argparse
import json
import os
import sys
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

# Match the deployable car pipeline (scorer_beam.py): car config, car d5 primitives,
# collisions OFF = the v3 training/eval distribution.
CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"
DATA_DIR = f"{REPO}/data"
PRIM_PREFIX = "1x_car_d5_"
FALLBACK_GOAL = (-0.5, 1.3, 0.0)

LABELS = "/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/pure2push.json"
REF_DEFAULT = "/common/users/dm1487/scratch_namo/eval/region_equiv/region_equiv_ref.json"

# How many reachable objects per scene and pushes per object to exercise (deterministic caps).
MAX_OBJECTS = 2
MAX_PUSHES_PER_OBJECT = 3
QPOS_TOL = 1e-6   # continuous physics fingerprint tolerance (recompilation FP noise)


def tier(sr: float) -> str:
    # Same cut points as eval_common.bin_of: hard < 0.05, med < 0.30, else easy.
    if sr < 0.05:
        return "hard"
    if sr < 0.30:
        return "med"
    return "easy"


def select_scenes(n_per_tier: int):
    """Deterministically pick distinct xmls spanning easy/med/hard episode difficulty.

    Per-episode invariant honored: difficulty is taken PER EPISODE (solve_rate_first_push),
    never per room. A scene enters a tier bucket if it has >=1 episode in that tier.
    Stable-sorted by resolved path for reproducibility.
    """
    data = json.load(open(LABELS))
    buckets = {"easy": [], "med": [], "hard": []}
    seen = {"easy": set(), "med": set(), "hard": set()}
    for xml_key in sorted(data.keys()):
        eps = data[xml_key]
        scene_tiers = {tier(float(e.get("solve_rate_first_push", 0.0))) for e in eps}
        for t in scene_tiers:
            if xml_key not in seen[t]:
                buckets[t].append(xml_key)
                seen[t].add(xml_key)
    picked = []
    picked_set = set()
    for t in ("easy", "med", "hard"):
        for xml_key in buckets[t][:n_per_tier]:
            if xml_key not in picked_set:
                picked.append((t, xml_key))
                picked_set.add(xml_key)
    return picked


def make_env(xml_key):
    env = namo_rl.RLEnvironment(str(resolve(xml_key)), CFG, False)
    env.reset()
    env.set_collision_checking(False)
    return env


def observe(env):
    """Capture all wavefront/region-dependent observables at the CURRENT state."""
    reach = sorted(str(o) for o in env.get_reachable_objects())
    redges = {o: sorted(int(e) for e in env.get_reachable_edges(o)) for o in reach}
    goal_reach = bool(env.is_robot_goal_reachable())
    snap = get_region_snapshot(env, goals_per_region=0, seed=42, use_cpp_unified=True)
    adjacency = {str(k): sorted(str(x) for x in v) for k, v in snap["adjacency"].items()}
    edge_objects = {
        str(k): {str(nb): sorted(str(x) for x in objs) for nb, objs in nbrs.items()}
        for k, nbrs in snap["edge_objects"].items()
    }
    region_labels = {str(k): str(v) for k, v in snap["region_labels"].items()}
    qpos = [float(x) for x in env.get_full_state().qpos]
    return {
        "reach": reach,
        "redges": redges,
        "goal_reach": goal_reach,
        "adjacency": adjacency,
        "edge_objects": edge_objects,
        "region_labels": region_labels,
        "robot_label": str(snap["robot_label"]),
        "goal_label": str(snap["goal_label"]),
        "snap_goal_reachable": bool(snap["goal_reachable"]),
        "goal_in_free_space": bool(snap["goal_in_free_space"]),
        "qpos": qpos,
    }


def flat_goals(prim, obj, state, env):
    """PrimitiveGoalStrategy.generate_goals returns list-of-lists (per edge -> per depth);
    flatten to a deterministic sorted list of Goal (matches scorer_beam's iteration)."""
    per_edge = prim.generate_goals(obj, state, env, 0)  # max_goals=0 -> all; restores state
    goals = [g for edge_goals in per_edge for g in edge_goals]
    return sorted(goals, key=lambda g: (int(g.depth), int(g.edge_idx)))


def make_action(obj, g):
    a = namo_rl.Action()
    a.object_id = obj
    a.x, a.y, a.theta = float(g.x), float(g.y), float(g.theta)
    a.edge_idx, a.depth = int(g.edge_idx), int(g.depth)
    return a


def run_scene(xml_key, prim):
    """Deterministic push script for one scene -> {stage_key: observables}."""
    out = {}
    env = make_env(xml_key)
    goal = extract_goal_with_fallback(str(resolve(xml_key)), FALLBACK_GOAL)
    env.set_robot_goal(float(goal[0]), float(goal[1]), float(goal[2]))
    reach0 = sorted(str(o) for o in env.get_reachable_objects())  # warm wavefront
    s0 = env.get_full_state()
    out["s0"] = observe(env)

    chain = []  # remember first push per object for a 2-push chain
    for obj in reach0[:MAX_OBJECTS]:
        env.set_full_state(s0)
        goals = flat_goals(prim, obj, s0, env)[:MAX_PUSHES_PER_OBJECT]
        for gi, g in enumerate(goals):
            env.set_full_state(s0)
            env.step(make_action(obj, g))
            key = f"obj={obj}|edge={int(g.edge_idx)}|depth={int(g.depth)}"
            out[key] = observe(env)
            if gi == 0:
                chain.append((obj, g))

    # one deterministic 2-push chain (first push of obj0 then first push of obj1)
    if len(chain) >= 2:
        env.set_full_state(s0)
        (o0, g0), (o1, g1) = chain[0], chain[1]
        env.step(make_action(o0, g0))
        s1 = env.get_full_state()
        goals1 = flat_goals(prim, o1, s1, env)
        if goals1:
            env.set_full_state(s1)
            env.step(make_action(o1, goals1[0]))
            out[f"chain|{o0}->{o1}"] = observe(env)
    return out


def capture(scenes, prim, ref_path):
    golden = {}
    for i, (t, xml_key) in enumerate(scenes):
        base = Path(xml_key).stem
        try:
            golden[f"{i:03d}|{t}|{base}"] = run_scene(xml_key, prim)
        except Exception as e:
            golden[f"{i:03d}|{t}|{base}"] = {"__error__": repr(e)}
            print(f"  [scene {i} {base}] ERROR: {e!r}")
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(golden, open(ref_path, "w"), sort_keys=True)
    n_states = sum(len(v) for v in golden.values() if "__error__" not in v)
    print(f"CAPTURED scenes={len(golden)} states={n_states} -> {ref_path}")


def compare(scenes, prim, ref_path):
    ref = json.load(open(ref_path))
    discrete_fields = ["reach", "redges", "goal_reach", "adjacency", "edge_objects",
                       "region_labels", "robot_label", "goal_label",
                       "snap_goal_reachable", "goal_in_free_space"]
    n_states = 0
    n_match = 0
    qmax = 0.0
    mism = []
    for i, (t, xml_key) in enumerate(scenes):
        base = Path(xml_key).stem
        skey = f"{i:03d}|{t}|{base}"
        if skey not in ref:
            mism.append(f"{skey}: scene missing from REF"); continue
        try:
            now = run_scene(xml_key, prim)
        except Exception as e:
            mism.append(f"{skey}: recompute ERROR {e!r}"); continue
        refscene = ref[skey]
        if "__error__" in refscene or "__error__" in now:
            continue
        # key set must match
        if set(now.keys()) != set(refscene.keys()):
            mism.append(f"{skey}: stage-key set differs")
        for stage in refscene:
            if stage not in now:
                continue
            n_states += 1
            r, c = refscene[stage], now[stage]
            ok = all(r.get(f) == c.get(f) for f in discrete_fields)
            # qpos tolerance
            rq, cq = r.get("qpos", []), c.get("qpos", [])
            if len(rq) == len(cq) and rq:
                d = max(abs(a - b) for a, b in zip(rq, cq))
                qmax = max(qmax, d)
            if ok:
                n_match += 1
            elif len(mism) < 12:
                bad = [f for f in discrete_fields if r.get(f) != c.get(f)]
                mism.append(f"{skey} :: {stage} :: DISCRETE DIFF in {bad}")
    print(f"DISCRETE-IDENTICAL={n_match}/{n_states}   qpos max|diff|={qmax:.3e}")
    if mism:
        print("MISMATCHES:")
        for m in mism:
            print("  " + m)
    ok = (n_match == n_states) and (qmax <= QPOS_TOL)
    print("RESULT:", "PASS (behavior-identical)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=os.environ.get("MODE", "compare"), choices=["capture", "compare"])
    ap.add_argument("--n-per-tier", type=int, default=int(os.environ.get("N_PER_TIER", "8")))
    ap.add_argument("--ref", default=os.environ.get("REF", REF_DEFAULT))
    args = ap.parse_args()

    scenes = select_scenes(args.n_per_tier)
    print(f"mode={args.mode} scenes={len(scenes)} (n_per_tier={args.n_per_tier}) ref={args.ref}")
    prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)
    if args.mode == "capture":
        capture(scenes, prim, args.ref)
    else:
        compare(scenes, prim, args.ref)


if __name__ == "__main__":
    main()
