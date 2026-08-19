#!/usr/bin/env python3
"""Depth-<=2 BEAM SEARCH planner — the deployable "2-push (also does 1-push)" solver.

Value function = the VALIDATED live 1-push scorer (scripts/sandbox/live_scorer.py).
The scorer P[edge, depth] = P(this single push opens a path from the robot to its goal),
rendered from the LIVE env so it works at arbitrary mid-search states (depth-2 too).

Pipeline per scene (solve()):
  0. set_robot_goal + warm wavefront. If goal already reachable -> depth 0.
  1. DEPTH-1 candidates: for each reachable object, score_state -> (60,5); pool every
     (object, primitive Goal, P[edge,depth]) whose edge is reachable. Sort desc by P.
  2. DEPTH-1 verify (beam K1): simulate the top-K1 pushes; the FIRST that makes the goal
     reachable (verified by is_robot_goal_reachable) is the 1-push solution (highest P,
     since we go in P order). SAVE each resulting state s1 (reused at depth 2).
  3. DEPTH-2: at each saved s1, pool second-push candidates; V(s1)=max P2. Rank first
     pushes by V(s1). For first pushes in that order, simulate their top-K2 second pushes;
     first verified -> 2-push solution.

Verified-by-sim only: the scorer ranks, but EVERY returned solution is confirmed by
env.is_robot_goal_reachable() after the actual push(es).

Usage:
  # validate (3-5 scenes from each solvable manifest) + benchmark:
  python scorer_beam.py --validate
  # eval over manifest subsets:
  python scorer_beam.py --eval --n 40
"""
import argparse
import os
import sys
import time

import numpy as np

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl  # noqa: E402
from live_scorer import LiveScorer  # noqa: E402
from namo.strategies import PrimitiveGoalStrategy  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, MANIFESTS, resolve  # noqa: E402

# CHAMPION scorer (task-specified).
CHAMPION_CKPT = str(SCRATCH / "sage_outputs/scorer/sharp_s1/namo-classifier/"
                    "9yizg6i8/checkpoints/epoch017-val_loss0.2713.ckpt")
CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"
DATA_DIR = f"{REPO}/data"
PRIM_PREFIX = "1x_car_d5_"   # car d5 primitives: 60 edges x 5 depths == scorer (60,5)
FALLBACK_GOAL = (-0.5, 1.3, 0.0)

MANIFEST_1PUSH = str(MANIFESTS / "test_1push_solvable_combined.txt")
MANIFEST_2PUSH = str(MANIFESTS / "test_2push_solvable_combined.txt")


# Default OFF matches the v3 TRAINING distribution: modular_parallel_collection's
# --region-allow-collisions defaults True (object collisions allowed during a push; robot-traj
# collisions always abort). The scorer learned solvability under THAT rule, and the
# 1push/2push_solvable manifests were defined under it. `--collisions on` = STRICT mode (any
# object collision aborts the push) = real-robot-faithful: the honest deployable number, since a
# real robot can't push an object THROUGH another. The off->on gap = the "push-through tax".
COLLISIONS_OFF = True


def make_env(xml):
    env = namo_rl.RLEnvironment(str(resolve(xml)), CFG, False)  # resolve(): remap legacy data paths onto this box
    env.reset()
    if COLLISIONS_OFF:
        env.set_collision_checking(False)
    return env


from namo.planners.opening.best_first_search import make_action  # noqa: F401 - moved; re-exported for sandbox callers


def _spec(obj, g, p):
    return {"object_id": obj, "x": float(g.x), "y": float(g.y), "theta": float(g.theta),
            "edge_idx": int(g.edge_idx), "depth": int(g.depth), "P": float(p)}


def _fmt_push(s):
    return (f"obj={s['object_id']} edge={s['edge_idx']} depth={s['depth']} "
            f"P={s['P']:.3f} -> ({s['x']:.3f},{s['y']:.3f},{s['theta']:.3f})")


class BeamPlanner:
    def __init__(self, ckpt=CHAMPION_CKPT, k1=10, k2=10, n1=6, max_first=60, first_depths=(4, 3, 2)):
        self.scorer = LiveScorer(ckpt=ckpt)
        self.prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)
        self.k1 = k1
        self.k2 = k2
        self.n1 = n1                    # how many top-V(s1) first pushes to expand at depth-2
        self.max_first = max_first      # cap on the (un-P-ranked) depth-2 first-push budget
        self.first_depths = tuple(first_depths)  # which push depths to try as a FIRST move

    def _candidates(self, env, robot_goal, xml, state):
        """Pool (obj, Goal, P) over reachable objs x reachable edges x depths, sorted desc by P.

        Assumes env is currently AT `state`. All sub-calls (score_state render,
        generate_goals) read/restore that state, so env stays at `state`.
        """
        pool = []
        reach_objs = list(env.get_reachable_objects())   # warms wavefront at this state
        # snapshot reachable edges per object off the SAME warmed wavefront
        redges = {obj: set(env.get_reachable_edges(obj)) for obj in reach_objs}
        for obj in reach_objs:
            if not redges[obj]:
                continue
            try:
                P = self.scorer.score_state(env, obj, robot_goal, xml)   # (60,5), reads current state
            except Exception:
                continue
            goals_per_edge = self.prim.generate_goals(obj, state, env, max_goals=0)  # restores state
            for edge_goals in goals_per_edge:
                for g in edge_goals:
                    if g is None:
                        continue
                    e, d = int(g.edge_idx), int(g.depth)
                    if e not in redges[obj] or e >= P.shape[0] or d >= P.shape[1]:
                        continue
                    pool.append((obj, g, float(P[e, d])))
        pool.sort(key=lambda t: -t[2])
        return pool

    def _first_budget(self, env, state):
        """ALL reachable (obj, Goal) at self.first_depths — deliberately NOT P-ranked.
        The s0 scorer P predicts P(opens goal in ONE push); a 2-push chain's first push opens
        nothing yet, so P is uninformative for it. We therefore sweep a broad first-push set and
        rank by V(s1) (one-ply lookahead) downstream. env assumed AT `state`."""
        out = []
        reach_objs = list(env.get_reachable_objects())
        redges = {obj: set(env.get_reachable_edges(obj)) for obj in reach_objs}
        for obj in reach_objs:
            if not redges[obj]:
                continue
            goals_per_edge = self.prim.generate_goals(obj, state, env, max_goals=0)
            for edge_goals in goals_per_edge:
                for g in edge_goals:
                    if g is None:
                        continue
                    e, d = int(g.edge_idx), int(g.depth)
                    if e in redges[obj] and d in self.first_depths:
                        out.append((obj, g))
        if len(out) > self.max_first:   # bound: subsample uniformly across the list (keeps edge spread)
            idx = np.linspace(0, len(out) - 1, self.max_first).astype(int)
            out = [out[i] for i in sorted(set(idx.tolist()))]
        return out

    def solve(self, env, robot_goal, xml, K1=None, K2=None):
        K1 = self.k1 if K1 is None else K1
        K2 = self.k2 if K2 is None else K2
        t0 = time.time()
        n_sims = 0

        env.set_robot_goal(*robot_goal)
        env.get_reachable_objects()   # warm wavefront so is_robot_goal_reachable is valid at s0
        if env.is_robot_goal_reachable():
            return {"solved": True, "depth": 0, "plan": [], "time_s": time.time() - t0,
                    "n_sims": 0}

        s0 = env.get_full_state()

        # ---- DEPTH-1: top-K1 by P (the scorer IS predictive for 1-push solutions) ----
        pool1 = self._candidates(env, robot_goal, xml, s0)
        first_states = []   # (spec1, s1) for EVERY first push we simulate (depth-1 + broad budget)
        seen = set()        # (obj, edge, depth) already simulated
        for (obj, g, p) in pool1[:K1]:
            env.set_full_state(s0)
            env.step(make_action(obj, g))
            n_sims += 1
            key = (obj, int(g.edge_idx), int(g.depth)); seen.add(key)
            spec1 = _spec(obj, g, p)
            if env.is_robot_goal_reachable():
                return {"solved": True, "depth": 1, "plan": [spec1],
                        "time_s": time.time() - t0, "n_sims": n_sims}
            first_states.append((spec1, env.get_full_state()))

        # ---- DEPTH-2: sweep a BROAD first-push budget (NOT P-ranked — P is blind to 2-push
        #      first moves), simulate each, then rank ALL first pushes by V(s1)=max second-push P. ----
        env.set_full_state(s0)
        for (obj, g) in self._first_budget(env, s0):
            key = (obj, int(g.edge_idx), int(g.depth))
            if key in seen:
                continue
            seen.add(key)
            env.set_full_state(s0)
            env.step(make_action(obj, g))
            n_sims += 1
            spec1 = _spec(obj, g, 0.0)
            if env.is_robot_goal_reachable():        # late 1-push catch (P missed it)
                return {"solved": True, "depth": 1, "plan": [spec1],
                        "time_s": time.time() - t0, "n_sims": n_sims}
            first_states.append((spec1, env.get_full_state()))

        # one-ply lookahead value of every simulated first push
        ranked = []   # (V, spec1, s1, pool2)
        for (spec1, s1) in first_states:
            env.set_full_state(s1)
            pool2 = self._candidates(env, robot_goal, xml, s1)
            V = pool2[0][2] if pool2 else -1.0
            ranked.append((V, spec1, s1, pool2))
        ranked.sort(key=lambda t: -t[0])

        # verify the top-N1 first pushes (by V) x their top-K2 second pushes
        for (V, spec1, s1, pool2) in ranked[:self.n1]:
            for (obj2, g2, p2) in pool2[:K2]:
                env.set_full_state(s1)
                env.step(make_action(obj2, g2))
                n_sims += 1
                if env.is_robot_goal_reachable():
                    spec2 = _spec(obj2, g2, p2)
                    return {"solved": True, "depth": 2, "plan": [spec1, spec2],
                            "V1": float(V), "time_s": time.time() - t0, "n_sims": n_sims}

        return {"solved": False, "depth": None, "plan": [], "time_s": time.time() - t0,
                "n_sims": n_sims, "best_V": float(ranked[0][0]) if ranked else None,
                "n_first": len(first_states)}


# --------------------------------------------------------------------------------------------------
def read_manifest(path, n=None):
    out = []
    with open(path) as f:
        for line in f:
            x = line.strip()
            if x and not x.startswith("#"):
                out.append(x)
    return out if n is None else out[:n]


def run_scene(planner, xml, K1, K2, verbose=False):
    env = make_env(xml)
    goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
    res = planner.solve(env, goal, xml, K1=K1, K2=K2)
    res["xml"] = xml
    if verbose:
        tag = f"depth={res['depth']}" if res["solved"] else "UNSOLVED"
        print(f"  [{tag}] {os.path.basename(xml)}  t={res['time_s']:.2f}s sims={res['n_sims']}")
        for i, s in enumerate(res["plan"]):
            print(f"      push{i+1}: {_fmt_push(s)}")
    return res


def validate(planner, K1, K2, n=5):
    print("\n" + "=" * 70)
    print("VALIDATION A: 1-push-solvable scenes (expect verified depth-1 solutions)")
    print("=" * 70)
    for xml in read_manifest(MANIFEST_1PUSH, n):
        run_scene(planner, xml, K1, K2, verbose=True)

    print("\n" + "=" * 70)
    print("VALIDATION B: 2-push-solvable scenes (expect verified depth-2 where depth-1 failed)")
    print("=" * 70)
    found2 = 0
    # 2push manifest has many scenes solvable in 1 push too; scan until we show a few depth-2.
    for xml in read_manifest(MANIFEST_2PUSH, 30):
        res = run_scene(planner, xml, K1, K2, verbose=True)
        if res["solved"] and res["depth"] == 2:
            found2 += 1
            if found2 >= 4:
                break
    print(f"\n  -> showed {found2} verified depth-2 solutions (depth-1 beam failed, two pushes opened the goal)")


def evaluate(planner, K1, K2, n, only="both", manifest=None):
    print("\n" + "=" * 70)
    print(f"EVAL  (K1={K1} K2={K2}, n<={n} per manifest, only={only})")
    print("=" * 70)
    rows = []
    if manifest:
        manifests = [(os.path.basename(manifest).replace(".txt", ""), manifest)]
    else:
        manifests = [("test_1push_solvable", MANIFEST_1PUSH), ("test_2push_solvable", MANIFEST_2PUSH)]
        if only == "1push":
            manifests = manifests[:1]
        elif only == "2push":
            manifests = manifests[1:]
    for name, path in manifests:
        xmls = read_manifest(path, n)
        d_le1 = d_le2 = total = 0
        depth_hist = {0: 0, 1: 0, 2: 0, None: 0}
        times, sims = [], []
        t_start = time.time()
        for i, xml in enumerate(xmls):
            try:
                res = run_scene(planner, xml, K1, K2, verbose=False)
            except Exception as e:
                print(f"  ERR {os.path.basename(xml)}: {e}")
                continue
            total += 1
            depth_hist[res["depth"]] = depth_hist.get(res["depth"], 0) + 1
            if res["solved"] and res["depth"] <= 1:
                d_le1 += 1
            if res["solved"] and res["depth"] <= 2:
                d_le2 += 1
            times.append(res["time_s"])
            sims.append(res["n_sims"])
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(xmls)}  d<=1={d_le1} d<=2={d_le2}  "
                      f"elapsed={time.time()-t_start:.0f}s", flush=True)
        rows.append({
            "name": name, "total": total,
            "d_le1": d_le1, "d_le2": d_le2,
            "pct_le1": 100.0 * d_le1 / max(1, total),
            "pct_le2": 100.0 * d_le2 / max(1, total),
            "depth_hist": depth_hist,
            "mean_t": float(np.mean(times)) if times else 0.0,
            "mean_sims": float(np.mean(sims)) if sims else 0.0,
        })

    print("\n" + "=" * 70)
    print("RESULT TABLE")
    print("=" * 70)
    hdr = f"{'manifest':<22}{'n':>5}{'%<=1':>8}{'%<=2':>8}{'gain':>8}{'t/scene':>10}{'sims/scene':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        gain = r["pct_le2"] - r["pct_le1"]
        print(f"{r['name']:<22}{r['total']:>5}{r['pct_le1']:>8.1f}{r['pct_le2']:>8.1f}"
              f"{gain:>8.1f}{r['mean_t']:>9.2f}s{r['mean_sims']:>12.1f}")
    print()
    for r in rows:
        print(f"  {r['name']}: depth histogram {r['depth_hist']} (None=unsolved)")
    two = next((r for r in rows if r["name"] == "test_2push_solvable"), None)
    if two:
        print(f"\n  HEADLINE: depth-2 lifts the 2-push-solvable set "
              f"{two['pct_le1']:.1f}% -> {two['pct_le2']:.1f}%  "
              f"(+{two['pct_le2']-two['pct_le1']:.1f} pp from composing two pushes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CHAMPION_CKPT)
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=10)
    ap.add_argument("--n1", type=int, default=6, help="top-V(s1) first pushes expanded at depth-2")
    ap.add_argument("--max-first", type=int, default=60, help="cap on depth-2 first-push budget")
    ap.add_argument("--first-depths", default="4,3,2", help="push depths tried as a FIRST move")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="scenes per manifest in --eval")
    ap.add_argument("--val-n", type=int, default=5)
    ap.add_argument("--collisions", choices=["off", "on"], default="off",
                    help="off=match training (object collisions allowed); on=strict/real-robot-faithful")
    ap.add_argument("--only", choices=["both", "1push", "2push"], default="both",
                    help="which manifest(s) to eval")
    ap.add_argument("--manifest", default=None, help="override: eval ONLY this manifest path")
    a = ap.parse_args()

    global COLLISIONS_OFF
    COLLISIONS_OFF = (a.collisions == "off")
    print(f"collisions={'OFF (training-match)' if COLLISIONS_OFF else 'ON (strict/real-robot)'}")
    fdepths = tuple(int(x) for x in a.first_depths.split(","))
    planner = BeamPlanner(ckpt=a.ckpt, k1=a.k1, k2=a.k2, n1=a.n1, max_first=a.max_first,
                          first_depths=fdepths)
    print(f"device={planner.scorer.device}  ckpt={os.path.basename(a.ckpt)}  "
          f"K1={a.k1} K2={a.k2} N1={a.n1} max_first={a.max_first} first_depths={fdepths}")

    if a.validate:
        validate(planner, a.k1, a.k2, n=a.val_n)
    if a.eval:
        evaluate(planner, a.k1, a.k2, a.n, only=a.only, manifest=a.manifest)
    if not (a.validate or a.eval):
        print("nothing to do; pass --validate and/or --eval")


if __name__ == "__main__":
    main()
