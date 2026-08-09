#!/usr/bin/env python3
"""Physics fingerprint: execute a FIXED list of pushes and dump the resulting object poses.

Run it on two boxes (or two builds) and diff the JSON. Identical output = the same physics; any
nonzero deviation = collected labels are not interchangeable between them.

Why this exists: the recorded cross-box verification (0.000 mm, 48 pushes) is dated 2026-07-14 and
PREDATES d6088d0, the sticky-collision fix to include/planning/namo_push_controller.hpp. Mixing
Amarel-collected data with CS evaluation is a standing assumption of the whole project, so it has to
be re-established against the current controller rather than inherited.

Deterministic by construction: pushes are enumerated from the reset state in sorted (object, edge,
depth) order, each executed from a fresh reset, so nothing depends on scene order or timing.

  python scripts/replay_physics_fingerprint.py --manifest <scenes.txt> --out fp.json [--per-scene 8]
  # then, off-box:  python scripts/replay_physics_fingerprint.py --compare a.json b.json
"""
import argparse, json, os, sys

def compare(a_path, b_path):
    A = json.load(open(a_path)); B = json.load(open(b_path))
    ka, kb = set(A["pushes"]), set(B["pushes"])
    common = sorted(ka & kb)
    print(f"A: {a_path}  build={A.get('build', {}).get('git_sha', '?')[:7]} host={A.get('host')}")
    print(f"B: {b_path}  build={B.get('build', {}).get('git_sha', '?')[:7]} host={B.get('host')}")
    print(f"pushes: A={len(ka)} B={len(kb)} common={len(common)}"
          f"{'  ⚠ NON-OVERLAPPING KEYS' if ka != kb else ''}")
    if not common:
        print("NO COMMON PUSHES — cannot compare"); return 2
    worst_mm = 0.0; worst_key = None; worst_deg = 0.0
    for k in common:
        for pa, pb in zip(A["pushes"][k], B["pushes"][k]):
            dx, dy = abs(pa[0] - pb[0]), abs(pa[1] - pb[1])
            dth = abs(pa[2] - pb[2])
            dth = min(dth, 2 * 3.141592653589793 - dth)
            mm = max(dx, dy) * 1000.0
            if mm > worst_mm: worst_mm, worst_key = mm, k
            worst_deg = max(worst_deg, dth * 180.0 / 3.141592653589793)
    print(f"max position deviation: {worst_mm:.6f} mm   (worst: {worst_key})")
    print(f"max yaw deviation:      {worst_deg:.6f} deg")
    oa, ob = A.get("opened", {}), B.get("opened", {})
    both = [k for k in common if k in oa and k in ob]
    if both:
        flips = [k for k in both if oa[k] != ob[k]]
        print(f"LABEL FLIPS (goal opened?): {len(flips)}/{len(both)} = {100.0*len(flips)/len(both):.3f}%")
        for k in flips[:5]:
            print(f"   flip: {k}  A={oa[k]} B={ob[k]}")
    ok = worst_mm < 1e-3 and worst_deg < 1e-3
    print("IDENTICAL — data is interchangeable" if ok else
          "DIFFERENT — do NOT mix data collected on these two builds")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--out")
    ap.add_argument("--config", default="config/namo_config_complete_skill15_car_1x.yaml")
    ap.add_argument("--per-scene", type=int, default=8)
    ap.add_argument("--reverse", action="store_true",
                    help="try candidates in reverse order; same pushes, different history. Any diff vs "
                         "forward means outcomes depend on what ran BEFORE them (warmstart carry-over)")
    ap.add_argument("--no-restore", action="store_true",
                    help="fresh env.reset() per push instead of set_full_state (isolates warmstart)")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))

    for d in ("build_python", "python", "scripts", "scripts/sandbox", "scripts/pipeline"):
        if d not in sys.path:
            sys.path.insert(0, d)
    import namo_rl
    from scorer_beam import make_action                      # canonical Action builder
    from namo.strategies import PrimitiveGoalStrategy
    from scorer_beam import DATA_DIR, PRIM_PREFIX

    build = {}
    if os.path.exists("build_python/BUILD_INFO"):
        for line in open("build_python/BUILD_INFO"):
            if "=" in line:
                k, v = line.strip().split("=", 1); build[k] = v

    prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)
    scenes = [l.split("\t")[0].strip() for l in open(a.manifest) if l.strip()]
    out = {"host": os.uname().nodename, "build": build, "config": a.config, "pushes": {}}

    for xml in scenes:
        if not os.path.exists(xml):
            continue
        env = namo_rl.RLEnvironment(xml, a.config, False)
        env.reset()
        s0 = env.get_full_state()
        objs = sorted(env.get_reachable_objects())
        # Deterministic enumeration: sorted objects, sorted edges, ascending depth, each push executed
        # from a fresh reset so the fingerprint cannot depend on ordering or on prior pushes.
        taken = 0
        for obj in objs:
            env.set_full_state(s0)
            redges = set(env.get_reachable_edges(obj))
            goals_per_edge = prim.generate_goals(obj, s0, env, max_goals=0)
            flat = []
            for eg in goals_per_edge:
                for g in eg:
                    if g is None:
                        continue
                    e, d = int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1))
                    if e in redges and d >= 0:
                        flat.append((e, d, g))
            # Select the candidate SET canonically, THEN choose execution order. Reversing the sort
            # itself would change which candidates the per-scene cap admits, making the two runs
            # incomparable (84 pushes each, only 4 keys in common) rather than testing order.
            sel = sorted(flat, key=lambda t: (t[0], t[1]))[:max(0, a.per_scene - taken)]
            if a.reverse:
                sel = list(reversed(sel))
            for e, d, g in sel:
                if taken >= a.per_scene:
                    break
                if a.no_restore:
                    # A fresh reset re-derives MuJoCo's warmstart from scratch. set_full_state does NOT
                    # restore that cache, so it can carry box-local state into the push -- this flag is
                    # what separates "the physics differs" from "the RESTORE PATH differs".
                    env = namo_rl.RLEnvironment(xml, a.config, False); env.reset()
                else:
                    env.set_full_state(s0)
                try:
                    env.step(make_action(obj, g))
                except Exception:
                    continue
                obs = env.get_observation()
                key = f"{os.path.basename(xml)}|{obj}|{e}|{d}"
                out["pushes"][key] = [[round(float(v), 12) for v in obs[f"{o}_pose"]]
                                      for o in objs if f"{o}_pose" in obs]
                # The LABEL, not just the pose. Millimetres only matter where they flip this bit --
                # this is what turns "1.2 mm of order-dependence" into "N cells got a different label".
                out.setdefault("opened", {})[key] = bool(env.is_robot_goal_reachable())
                taken += 1
            if taken >= a.per_scene:
                break

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out['pushes'])} pushes over {len(scenes)} scenes")
    print(f"build git_sha={build.get('git_sha', '?')[:7]} cpp_tree={build.get('cpp_tree', '?')[:7]} host={out['host']}")


if __name__ == "__main__":
    main()
