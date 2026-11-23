#!/usr/bin/env python3
"""Side-by-side video of one push action run by two cars (for example car v1 and car v2).

Each arm runs env.step((object, edge, depth)) in its own child process with NAMO_QPOS_DUMP set,
as python/scripts/car_v2_drive_video.py does for drives. The parent replays the push part of
each dump (phase 3: settle, push, settle) in pip mujoco, both arms aligned on the first push
tick. Top row: the block from straight above, with its starting outline drawn on the floor.
Bottom row (dropped by --top-only): an angled view with the car body see-through. Text per arm:
block travel so far (centre of mass, not qpos), car speed, and wheel rim speed; a rim speed well
above the car speed means the tyres are slipping. Frames every 17 ticks are real time at 30 fps.

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python python/scripts/car_push_compare_video.py \\
      --build-python build_python --obj obstacle_1_movable --edge 30 --depth 4 \\
      --arm "car v1=<v1 scene>=config/namo_config_complete_skill15_car_1x.yaml" \\
      --arm "car v2=<v2 scene>=config/namo_config_complete_skill15_car_v2_1x.yaml" --out <dir>/push.mp4
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from car_v2_drive_video import add_polyline, model  # noqa: E402

DT = 0.002
FPS = 30
STRIDE = 17
SETTLE = 100                      # kSettleSteps in namo_push_controller.cpp: ticks before the push
PW, PH = 640, 480                 # one panel


def child(args):
    import namo_rl
    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    a = namo_rl.Action()
    a.object_id, a.edge_idx, a.depth = args.obj, args.edge, args.depth
    a.x = a.y = a.theta = 0.0
    r = env.step(a)
    Path(args.result).write_text(json.dumps({k: str(v) for k, v in dict(r.info).items()}))


def run_arm(args, name, xml, config, tmp):
    stem = f"{args.out.stem}__{''.join(c if c.isalnum() else '_' for c in name)}"   # one per video and arm
    dump, result = tmp / f"{stem}.qpos", tmp / f"{stem}.json"
    dump.unlink(missing_ok=True)
    env = dict(os.environ, NAMO_QPOS_DUMP=str(dump),
               PYTHONPATH=f"{Path(args.build_python).resolve()}:{Path.cwd() / 'python'}")
    subprocess.run([sys.executable, __file__, "--child", "--xml", xml, "--config", config,
                    "--obj", args.obj, "--edge", str(args.edge), "--depth", str(args.depth),
                    "--result", str(result)], env=env, check=True, stdout=subprocess.DEVNULL)
    rows = [ln.split() for ln in open(dump) if ln.strip()]
    Q = np.array([[float(x) for x in r[2:]] for r in rows if r[0] == "3"])
    return Q, json.loads(result.read_text())


class Arm:
    """One car's replay: models for both views, per-tick block travel, car and rim speeds."""

    def __init__(self, name, xml, Q, info, obj):
        import mujoco
        self.name, self.Q, self.info = name, Q, info
        self.m_top, self.m_side = model(xml, PW, PH), model(xml, PW, PH)
        self.m_top.vis.global_.orthographic = 1
        for m in (self.m_top, self.m_side):
            car = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "car")
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == car and m.geom_size[g][0] > 0.01:     # chassis boxes, not the marker
                    m.geom_rgba[g][3] = 0.35
        m = self.m_top
        self.d = mujoco.MjData(m)
        self.r_top = mujoco.Renderer(self.m_top, height=PH, width=PW, max_geom=2000)
        self.r_side = mujoco.Renderer(self.m_side, height=PH, width=PW, max_geom=2000)
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
        wl = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")]
        wr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")]
        rad = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_collision")][0]
        com, corners = [], None
        bg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bid][0]
        for i, q in enumerate(Q):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(m, self.d)
            mujoco.mj_comPos(m, self.d)
            com.append(self.d.xipos[self.bid][:2].copy())
            if i == SETTLE:
                R, c, h = self.d.geom_xmat[bg].reshape(3, 3), self.d.geom_xpos[bg], m.geom_size[bg]
                corners = [c + R @ np.array([sx * h[0], sy * h[1], 0]) for sx, sy in
                           ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
        self.com = np.array(com)
        self.outline = [p[:2] for p in corners]
        self.travel = 1000 * np.hypot(*(self.com - self.com[SETTLE]).T)
        xy = Q[:, qa:qa + 2]
        w, x, y, z = Q[:, qa + 3:qa + 7].T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self.car0, self.yaw0 = xy[SETTLE], yaw[SETTLE]
        k = 5
        self.v = np.zeros(len(Q)); self.rim = np.zeros(len(Q))
        self.v[k:-k] = np.sum((xy[2 * k:] - xy[:-2 * k]) * np.c_[np.cos(yaw), np.sin(yaw)][k:-k], 1) / (2 * k * DT)
        wheel = (Q[:, wl] + Q[:, wr]) / 2          # both wheels' mean angle; forward is +qpos on both
        self.rim[k:-k] = rad * (wheel[2 * k:] - wheel[:-2 * k]) / (2 * k * DT)
        self.n_push = len(Q) - 2 * SETTLE if not info.get("failure_reason") else len(Q) - SETTLE

    def frame(self, i, top, side=None):
        import cv2
        import mujoco
        i = min(i, len(self.Q) - 1)
        self.d.qpos[:] = self.Q[i]
        mujoco.mj_forward(self.m_top, self.d)
        out = []
        for r, cam in ((self.r_top, top), (self.r_side, side))[: 1 if side is None else 2]:
            r.update_scene(self.d, camera=cam)
            add_polyline(r.scene, self.outline, (1.0, 1.0, 1.0, 0.9), 0.0012, 0.001)
            add_polyline(r.scene, self.com[SETTLE: i + 1: 8], (1.0, 0.3, 0.3, 1.0), 0.0015, 0.0015)
            out.append(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        t = (i - SETTLE) * DT
        phase = "settling" if i < SETTLE else ("pushing" if i < SETTLE + self.n_push else "settling, push done")
        lines = [f"{self.name}   t = {t:5.2f} s   {phase}",
                 f"block moved {self.travel[i]:5.1f} mm",
                 f"car {100 * self.v[i]:5.2f} cm/s   wheel rim {100 * self.rim[i]:5.2f} cm/s"]
        band = out[0][: 30 * len(lines) + 14]
        band[:] = (0.35 * band).astype(band.dtype)
        for row, txt in enumerate(lines):
            cv2.putText(out[0], txt, (14, 30 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--xml"); ap.add_argument("--config"); ap.add_argument("--result")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edge", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--arm", action="append", help="name=scene=config")
    ap.add_argument("--build-python")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--top-only", action="store_true", help="drop the angled row")
    args = ap.parse_args()
    if args.child:
        return child(args)

    import cv2
    import mujoco
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in args.arm:
        name, xml, config = spec.split("=")
        Q, info = run_arm(args, name, xml, config, args.out.parent)
        arms.append(Arm(name, xml, Q, info, args.obj))
        a = arms[-1]
        p = slice(SETTLE, SETTLE + a.n_push)
        print(f"{name}: failure_reason={info.get('failure_reason', '')!r} push {a.n_push * DT:.2f} s, "
              f"block moved {a.travel[SETTLE + a.n_push - 1]:.1f} mm, end {a.travel[-1]:.1f} mm; "
              f"push means: car {100 * a.v[p].mean():.2f} cm/s, wheel rim {100 * a.rim[p].mean():.2f} cm/s")

    # Top view: frame every arm's car start and the block's whole path, with 9 cm to spare.
    pts = np.vstack([np.vstack([a.com[SETTLE:], a.car0[None]]) for a in arms])
    lo, hi = pts.min(0) - 0.09, pts.max(0) + 0.09
    top = mujoco.MjvCamera(); top.type = mujoco.mjtCamera.mjCAMERA_FREE
    top.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0]
    top.elevation = -89.9; top.azimuth = 90; top.distance = 1.0
    for a in arms:
        a.m_top.vis.global_.fovy = max(hi[1] - lo[1], (hi[0] - lo[0]) * PH / PW)   # orthographic: metres
    # Angled view: from the side of the push (90 deg off the car's heading at the first push tick),
    # looking down at 55 deg over the 10 cm walls at the point between the car and the block.
    mid = (arms[0].car0 + arms[0].com[SETTLE]) / 2
    side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    side.lookat[:] = [mid[0], mid[1], 0.03]; side.distance = 0.36; side.elevation = -55
    side.azimuth = math.degrees(arms[0].yaw0) + 90

    raw = args.out.with_suffix(".raw.mp4")
    if args.top_only:
        side = None
    rows = 1 if side is None else 2
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (PW * len(arms), rows * PH))
    n = max(len(a.Q) for a in arms)
    for i in list(range(0, n, STRIDE)) + [n - 1] * FPS:
        img = np.hstack([a.frame(i, top, side) for a in arms])
        cv2.putText(img, f"push {args.obj} edge {args.edge} depth {args.depth}: white = block's start, "
                    f"red = its centre's path", (14, rows * PH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(args.out)], check=True)
    raw.unlink()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
