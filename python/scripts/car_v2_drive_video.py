#!/usr/bin/env python3
"""Video of the car driving to one pose after another under the ported 30 Hz controller.

A child process runs env.navigate_to_pose for each leg on the C++ bindings with
NAMO_QPOS_DUMP set (the dump opens once per process, so one child per video). The parent
replays the dumped qpos in pip mujoco and renders two panels side by side: the whole room
from above, and a camera that follows the car. Both draw the planned path on the floor
(green) and the path driven so far (orange). Frames are every 17 ticks, which is real time
at 30 fps.

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python python/scripts/car_v2_drive_video.py \\
      --xml <scene> --config config/namo_config_complete_skill15_car_v2_1x.yaml \\
      --build-python build_python --leg 0.09,0.2225,0 --leg 0.38,0.10,-90 --out <dir>/drive.mp4
Each --leg is x,y in metres and the final heading in degrees.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DT = 0.002
FPS = 30
STRIDE = 17                       # ticks per frame: 500 Hz / 17 = 29.4 frames of sim per second
W = 5                             # half window for speeds, ticks (10 ms centred differences)
ROOM_W, CHASE_W, H = 480, 800, 720


def child(args):
    import namo_rl
    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    legs = []
    for leg in args.leg:
        x, y, th = (float(v) for v in leg.split(","))
        r = dict(env.navigate_to_pose(x, y, math.radians(th)))
        legs.append(dict(goal=[x, y, th], ok=bool(r["ok"]), failure_reason=r["failure_reason"],
                         nav_time_s=float(r["nav_time_s"]), path=[list(p) for p in r["path"]],
                         final_pose=list(r["final_pose"])))
        if not r["ok"]:
            break
    Path(args.result).write_text(json.dumps(legs))


def yaw_of(q):
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def centred(x):
    out = np.zeros_like(x)
    out[W:-W] = (x[2 * W:] - x[:-2 * W]) / (2 * W * DT)
    return out


def model(xml, width, height):
    import mujoco
    root = ET.parse(xml).getroot()
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.set("offwidth", str(width)); g.set("offheight", str(height))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def add_polyline(scn, pts, rgba, width, z):
    import mujoco
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom or np.hypot(b[0] - a[0], b[1] - a[1]) < 1e-6:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.array([a[0], a[1], z]), np.array([b[0], b[1], z]))
        scn.ngeom += 1


def render(args, legs, Q):
    import cv2
    import mujoco
    m_room = model(args.xml, ROOM_W, H)
    m_chase = model(args.xml, CHASE_W, H)
    d = mujoco.MjData(m_room)
    r_room = mujoco.Renderer(m_room, height=H, width=ROOM_W, max_geom=4000)
    r_chase = mujoco.Renderer(m_chase, height=H, width=CHASE_W, max_geom=4000)
    jid = mujoco.mj_name2id(m_room, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    a = m_room.jnt_qposadr[jid]
    xy, yaw = Q[:, a:a + 2], yaw_of(Q[:, a + 3:a + 7])
    hd = np.c_[np.cos(yaw), np.sin(yaw)]
    v = np.sum(np.c_[centred(xy[:, 0]), centred(xy[:, 1])] * hd, axis=1)
    yr = np.degrees(centred(yaw))

    # Which leg each tick belongs to: leg k's drive is nav_time_s / DT ticks long.
    ends = np.cumsum([int(round(l["nav_time_s"] / DT)) for l in legs])
    leg_of = np.searchsorted(ends, np.arange(len(Q)), side="right").clip(0, len(legs) - 1)

    xs = [float(g.get("pos").split()[0]) for g in ET.parse(args.xml).getroot().iter("geom")
          if g.get("name", "").startswith("wall_boundary")]
    ys = [float(g.get("pos").split()[1]) for g in ET.parse(args.xml).getroot().iter("geom")
          if g.get("name", "").startswith("wall_boundary")]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max((max(ys) - min(ys)) / 2, (max(xs) - min(xs)) / 2 * H / ROOM_W) + 0.03

    room = mujoco.MjvCamera(); room.type = mujoco.mjtCamera.mjCAMERA_FREE
    room.lookat[:] = [cx, cy, 0.0]; room.elevation = -89.9; room.azimuth = 90
    room.orthographic = 1; room.distance = 1.0
    m_room.vis.global_.orthographic = 1           # the free camera needs both flags
    m_room.vis.global_.fovy = 2 * span            # and then reads fovy as metres
    chase = mujoco.MjvCamera(); chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.distance = 0.38; chase.elevation = -35; chase.azimuth = 120
    opt = mujoco.MjvOption()

    raw = args.out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ROOM_W + CHASE_W, H))
    look = xy[0].copy()
    frames = list(range(0, len(Q), STRIDE)) + [len(Q) - 1] * FPS   # hold the last frame 1 s
    for i in frames:
        d.qpos[:] = Q[i]
        mujoco.mj_forward(m_room, d)
        k = leg_of[i]
        trail = xy[: i + 1 : 4]
        look += 0.15 * (xy[i] - look) if i else 0.0
        chase.lookat[:] = [look[0], look[1], 0.02]
        imgs = []
        for r, cam, w_path in ((r_room, room, 0.0025), (r_chase, chase, 0.0012)):
            r.update_scene(d, camera=cam, scene_option=opt)
            add_polyline(r.scene, legs[k]["path"], (0.1, 0.85, 0.2, 0.9), w_path, 0.0015)
            add_polyline(r.scene, trail, (1.0, 0.55, 0.0, 1.0), w_path * 0.8, 0.002)
            imgs.append(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        img = np.hstack(imgs)
        g = legs[k]["goal"]
        lines = [f"car v2, ported 30 Hz controller, real time   t = {i * DT:5.2f} s",
                 f"leg {k + 1}/{len(legs)} to ({100 * g[0]:.1f}, {100 * g[1]:.1f}) cm, heading {g[2]:.0f} deg",
                 f"speed {100 * v[i]:5.2f} cm/s   turn rate {yr[i]:6.1f} deg/s"]
        for row, txt in enumerate(lines):
            cv2.putText(img, txt, (ROOM_W + 20, 40 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "green: planned path   orange: driven", (ROOM_W + 20, H - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(args.out)], check=True)
    raw.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--leg", action="append", required=True, help="x,y,heading_deg")
    ap.add_argument("--build-python")
    ap.add_argument("--result")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.child:
        return child(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump, result = args.out.with_suffix(".qpos"), args.out.with_suffix(".legs.json")
    dump.unlink(missing_ok=True)
    env = dict(os.environ, NAMO_QPOS_DUMP=str(dump),
               PYTHONPATH=f"{Path(args.build_python).resolve()}:{Path.cwd() / 'python'}")
    cmd = [sys.executable, __file__, "--child", "--xml", args.xml, "--config", args.config,
           "--result", str(result)] + [f"--leg={leg}" for leg in args.leg]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
    legs = json.loads(result.read_text())
    for k, l in enumerate(legs):
        print(f"leg {k + 1}: ok={l['ok']} {l['failure_reason']} {l['nav_time_s']:.2f} s, "
              f"end {np.round(l['final_pose'][:2], 4).tolist()} heading {math.degrees(l['final_pose'][2]):.1f} deg")
    Q = np.array([[float(x) for x in ln.split()[2:]] for ln in open(dump) if ln.strip()])
    render(args, legs, Q)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
