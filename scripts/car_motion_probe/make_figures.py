#!/usr/bin/env python3
"""Plots and clips from the probe traces. Rendering only replays saved qpos, so it changes nothing.

  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 python make_figures.py --xml <pool scene> --out <dir>
"""
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import car_probe_lib as L

W, H, FPS = 1280, 720, 30
REAL = int(round(1 / (FPS * L.DT)))          # ticks per frame at real time (17)
V1_WHEEL_R = 0.015                            # these figures are of the v1 baseline
V1_CRUISE_OMEGA = L.CRUISE_MS / V1_WHEEL_R


def plots(out):
    s = np.load(out / "traces" / "d_stop.npz")
    a = np.load(out / "traces" / "a_straight.npz")
    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    for col, (tr, title) in enumerate([(a, "start from rest, wheels stepped to cruise"),
                                       (s, "stop from cruise, wheels stepped to 0")]):
        k = slice(0, 60)
        tt = 1000 * L.DT * np.arange(60)
        off = sum(tr[f"n_{g}"] for g in ("wL", "wR", "cF", "cB"))[k] == 0
        for axx in ax[:, col]:
            axx.fill_between(tt, 0, 1, where=off, transform=axx.get_xaxis_transform(), color="0.85",
                             label="no support touching floor")
        ax[0, col].plot(tt, 100 * tr["v_fwd"][k], label="chassis speed")
        ax[0, col].plot(tt, 100 * V1_WHEEL_R * tr["omega_L"][k], "--", label="wheel surface speed r*omega")
        ax[0, col].axhline(100 * L.CRUISE_MS, color="k", lw=0.6, ls=":", label="commanded 5.36 cm/s")
        ax[0, col].set_ylabel("cm/s"); ax[0, col].set_title(title); ax[0, col].legend(fontsize=7)
        for g, lab in [("wL", "left wheel"), ("cF", "front caster"), ("cB", "rear caster")]:
            ax[1, col].plot(tt, tr[f"fn_{g}"][k], label=lab)
        ax[1, col].set_ylabel("floor normal force, N"); ax[1, col].legend(fontsize=7)
        ax[2, col].plot(tt, tr["tau_L"][k], label="left wheel torque")
        ax[2, col].axhline(0.5, color="r", lw=0.6, ls=":"); ax[2, col].axhline(-0.5, color="r", lw=0.6, ls=":",
                                                                               label="forcerange +-0.5 N m")
        ax[2, col].set_ylabel("N m"); ax[2, col].set_xlabel("ms after the command step"); ax[2, col].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "plots" / "start_stop_transients.png", dpi=130); plt.close(fig)

    e = np.load(out / "traces" / "e_push.npz")
    n = len(e["ctrl"]); t = L.DT * np.arange(n)
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(t, e["ctrl"][:, 0], label="left wheel command"); ax[0].plot(t, e["ctrl"][:, 1], label="right wheel command")
    ax[0].axhline(V1_CRUISE_OMEGA, color="k", lw=0.6, ls=":"); ax[0].set_ylabel("rad/s"); ax[0].legend(fontsize=8)
    ax[0].set_title("one v1 push (obstacle_1, edge 30, depth 4) through the C++ controller")
    ax[1].plot(t[5:-5], 100 * e["v_cpp"], label="chassis speed (C++ qpos, 10 ms difference)")
    ax[1].axhline(100 * L.CRUISE_MS, color="k", lw=0.6, ls=":", label="commanded"); ax[1].set_ylabel("cm/s"); ax[1].legend(fontsize=8)
    ax[2].plot(t, 1000 * e["cte"], label="cross-track error (C++ qpos)")
    for y in (7, -7):
        ax[2].axhline(y, color="r", lw=0.6, ls=":")
    ax[2].set_ylabel("mm"); ax[2].legend(fontsize=8, title="dotted: follower's 7 mm CTE deadband", title_fontsize=8)
    ax[3].plot(t, e["slip_L"], lw=0.7, label="left wheel slip (replay)"); ax[3].plot(t, e["slip_R"], lw=0.7, label="right wheel slip (replay)")
    ax[3].fill_between(t, -1, 1.2, where=e["n_block"] > 0, color="0.9", label="car touching block (replay)")
    ax[3].set_ylabel("slip ratio"); ax[3].set_xlabel("s into the push"); ax[3].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "push_timeseries.png", dpi=130); plt.close(fig)


def _model(xml):
    root = ET.parse(xml).getroot()
    vis = root.find("visual") if root.find("visual") is not None else ET.SubElement(root, "visual")
    g = vis.find("global") if vis.find("global") is not None else ET.SubElement(vis, "global")
    g.set("offwidth", str(W)); g.set("offheight", str(H))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def clip(xml, qpos, frames, path, cam_fn, text_fn):
    m = _model(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=H, width=W)
    car = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "car")
    raw = path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in frames:
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        cam = cam_fn(d.xpos[car])
        r.update_scene(d, camera=cam)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        for row, txt in enumerate(text_fn(i)):
            cv2.putText(img, txt, (20, 40 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(path)], check=True)
    raw.unlink()


def clips(xml, out):
    def side(target):
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = [target[0], target[1], 0.02]; c.distance = 0.22; c.elevation = -4; c.azimuth = 90
        return c

    s = np.load(out / "traces" / "d_stop.npz")          # 1000 ticks cruise from rest, then 500 ticks stopping
    q = s["qpos"]; v = np.r_[np.zeros(1000), s["v_fwd"]]
    a = np.load(out / "traces" / "a_straight.npz")
    v[:1000] = a["v_fwd"][:1000]                          # same command and start state as the first 1000 ticks
    slow = list(range(0, 60))                             # first 120 ms, one tick per frame (17x slow motion)
    frames = slow + list(range(60, 1000, REAL)) + list(range(1000, 1060)) + list(range(1060, 1500, REAL))
    speed = lambda i: ("SLOW MOTION 17x" if (i < 60 or 1000 <= i < 1060) else "real time")
    clip(out / "xml" / "v1_open_floor.xml", q, frames, out / "clips" / "v1_straight_start_stop.mp4", side,
         lambda i: [f"v1 car, open floor   t = {1000 * L.DT * i:6.0f} ms   {speed(i)}",
                    f"chassis speed {100 * v[i]:5.2f} cm/s (command {100 * L.CRUISE_MS:.2f} until 2000 ms, then 0)"])

    def top(target):
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = [target[0], target[1], 0.0]; c.distance = 0.55; c.elevation = -70; c.azimuth = 90
        return c

    e = np.load(out / "traces" / "e_push.npz")
    S, cq, ctrl = int(e["settle"]), e["cpp_qpos"], e["ctrl"]
    n = len(ctrl)
    vv = np.r_[np.zeros(5), e["v_cpp"], np.zeros(5)]
    def txt(i):
        k = i - S
        if 0 <= k < n:
            return [f"v1 push, C++ controller   t = {k * L.DT:4.2f} s   real time",
                    f"wheel command L {ctrl[k, 0]:5.2f}  R {ctrl[k, 1]:5.2f} rad/s   chassis {100 * vv[k]:5.2f} cm/s"]
        return ["v1 push, C++ controller   settle (wheels 0)", ""]
    clip(xml, cq, list(range(0, len(cq), REAL)), out / "clips" / "v1_push_obstacle1_edge30_d4.mp4", top, txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    plots(out)
    clips(args.xml, out)


if __name__ == "__main__":
    main()
