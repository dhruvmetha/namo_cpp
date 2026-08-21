"""Replay a region_opening solution from its results pkl, dumping qpos only for
the successful pushes (no failed trials)."""
import argparse, os, pickle, sys
from pathlib import Path

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
sys.path.insert(0, str(REPO / f"build_python_mjxrl_{os.uname().nodename.split('.')[0]}"))
sys.path.insert(0, str(REPO / "python"))

import namo_rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-pkl", required=True)
    ap.add_argument("--xml", required=True, help="Original env XML")
    ap.add_argument("--namo-config", default=str(REPO / "config/namo_config_car.yaml"))
    ap.add_argument("--qpos-out", required=True)
    ap.add_argument("--chain-length", type=int, default=None,
                    help="Only replay episodes whose action_sequence has this length (default: replay all)")
    ap.add_argument("--success-only", action="store_true",
                    help="Skip episodes where success=False")
    ap.add_argument("--episode-idx", type=int, default=None,
                    help="After filtering, only replay the Nth (0-indexed) matching episode")
    args = ap.parse_args()

    os.environ["NAMO_QPOS_DUMP"] = args.qpos_out
    # Teleport is now the default for the controller; no env var needed.
    # If you want a real-nav replay (for kinematic-honest videos), run with
    # NAMO_REAL_NAV=1 in the environment.
    if Path(args.qpos_out).exists():
        Path(args.qpos_out).unlink()

    with open(args.results_pkl, "rb") as f:
        r = pickle.load(f)
    episodes = r.get("episode_results") or []

    env = namo_rl.RLEnvironment(args.xml, args.namo_config, visualize=False)
    env.reset()

    matched_so_far = -1
    for ep_i, ep in enumerate(episodes):
        actions = ep.get("action_sequence") or []
        if not actions:
            continue
        if args.chain_length is not None and len(actions) != args.chain_length:
            continue
        if args.success_only and not ep.get("success"):
            continue
        matched_so_far += 1
        if args.episode_idx is not None and matched_so_far != args.episode_idx:
            continue
        print(f"--- episode {ep_i}: {len(actions)} action(s) ---")
        for a in actions:
            obj = a["object_id"]
            x, y, theta = a["target"]
            action = namo_rl.Action()
            action.object_id = obj
            action.x = float(x); action.y = float(y); action.theta = float(theta)
            # If the recorded action carries explicit edge_idx/depth, use them so the
            # C++ skill bypasses MPC and reproduces the exact primitive the search ran.
            # Older pkls without these keys fall back to -1/-1 (MPC mode).
            action.edge_idx = int(a.get("edge_idx", -1))
            action.depth = int(a.get("depth", -1))
            print(f"  push {obj} edge={action.edge_idx} depth={action.depth} -> ({x:.3f}, {y:.3f}, {theta:.3f})")
            res = env.step(action)
            print(f"    done={res.done}  reward={res.reward}")
        env.reset()  # reset between episodes so each replay is clean

    print(f"qpos written to {args.qpos_out}")


if __name__ == "__main__":
    main()
