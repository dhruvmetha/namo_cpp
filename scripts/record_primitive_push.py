"""Record a single push primitive in sim and write it to an MP4 at 1× sim-real.

Reuses the production render_chain_to_mp4 path (sim_replay_subprocess), which
runs ``namo_rl.RLEnvironment.step()`` on a tiny chain of pushes, captures
per-tick qpos via ``NAMO_QPOS_DUMP``, then renders offscreen with
``mujoco.Renderer``. Same C++ push controller, same wheel-tracker, same
calibrated ``push_tracker_max_speed`` as production primitive generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# robot_control hosts the render_chain_to_mp4 helper.
ROBOT_CTRL_SRC = Path("/home/dhruv/projects_dhruv/namo/robot_control/src")
sys.path.insert(0, str(ROBOT_CTRL_SRC))

from robot_control.diagnostics.sim_replay import render_chain_to_mp4


SCENES = {
    "square": "data/nominal_primitive_scene_square_1x_car.xml",
    "wide": "data/nominal_primitive_scene_wide_1x_car.xml",
    "tall": "data/nominal_primitive_scene_tall_1x_car.xml",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene",
        choices=list(SCENES.keys()),
        default="wide",
        help="Which 1×-car primitive scene to record (default: wide).",
    )
    p.add_argument(
        "--edge", type=int, default=0,
        help="Edge index 0..59 (default: 0, a corner of the top face).",
    )
    p.add_argument(
        "--push-steps", type=int, default=10,
        help="Number of push steps in the primitive (default: 10 = max).",
    )
    p.add_argument(
        "--config",
        default="config/namo_config_complete_skill15_car_1x.yaml",
        help="namo_rl YAML config (defaults to the car 1x config).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output MP4 path. Default: /tmp/primitive_<scene>_edge<edge>_steps<n>.mp4",
    )
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(
        f"/tmp/primitive_{args.scene}_edge{args.edge}_steps{args.push_steps}.mp4"
    )

    chain = [{
        "object_id": "obstacle_1_movable",
        "edge_idx": int(args.edge),
        "push_steps": int(args.push_steps),
        "depth": int(args.push_steps - 1),  # convention used by sim_replay
    }]

    print(f"[record] scene={args.scene}  edge={args.edge}  push_steps={args.push_steps}")
    print(f"[record] xml    = {SCENES[args.scene]}")
    print(f"[record] config = {args.config}")
    print(f"[record] out    = {out}")

    result = render_chain_to_mp4(
        start_xml=SCENES[args.scene],
        namo_config=args.config,
        chain=chain,
        output_mp4=str(out),
    )
    if result is None:
        print("[record] FAILED — see [sim_replay] / [sim_replay_subprocess] lines above.")
        return 1
    print(f"[record] ok → {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
