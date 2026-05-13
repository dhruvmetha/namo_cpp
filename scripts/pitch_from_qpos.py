"""Extract chassis pitch from a NAMO_QPOS_DUMP file and plot it over time.

The dump format is one line per sim tick:
  <phase> <nq> <q0> <q1> ... <q(nq-1)>

The car body is added by the runtime such that its freejoint qpos sits
somewhere in the qpos vector.  The freejoint layout is
  [x, y, z, qw, qx, qy, qz]
so for the car we want indices [car_q .. car_q+6].

We don't know `car_q` from the dump alone, so the user passes the env XML
and we ask MuJoCo for `model.jnt_qposadr[car_freejoint]`.

Outputs a PNG with chassis pitch (deg) over sim time, with phase boundaries
shaded.  Also prints a per-push-phase peak pitch table.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


def quat_pitch(w, x, y, z):
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("qpos_dump")
    ap.add_argument("--output", default=None)
    ap.add_argument("--timestep", type=float, default=0.002)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    car_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    if car_jnt < 0:
        raise SystemExit("car_freejoint not found in model")
    car_q0 = int(model.jnt_qposadr[car_jnt])
    print(f"car_freejoint qpos start index = {car_q0}")
    print(f"model.nq = {model.nq}")

    phases, ticks, pitches, zs, xs, ys = [], [], [], [], [], []
    with open(args.qpos_dump) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ":" in ln:
                phase_str, qstr = ln.split(":", 1)
                phase = int(phase_str)
                qs = np.fromstring(qstr, sep=" ", dtype=float)
            else:
                vals = np.fromstring(ln, sep=" ", dtype=float)
                if vals.size < 2:
                    continue
                phase = int(vals[0])
                nq = int(vals[1])
                qs = vals[2:2 + nq]
            if qs.size < car_q0 + 7:
                continue
            x, y, z = qs[car_q0], qs[car_q0 + 1], qs[car_q0 + 2]
            qw, qx, qy, qz = qs[car_q0 + 3], qs[car_q0 + 4], qs[car_q0 + 5], qs[car_q0 + 6]
            phases.append(phase)
            ticks.append(len(ticks))
            pitches.append(math.degrees(quat_pitch(qw, qx, qy, qz)))
            zs.append(z * 1000)
            xs.append(x); ys.append(y)

    phases = np.array(phases)
    ticks = np.array(ticks)
    pitches = np.array(pitches)
    zs = np.array(zs)
    t = ticks * args.timestep
    print(f"loaded {len(ticks)} ticks ({t[-1]:.1f}s) phases seen: {sorted(set(phases.tolist()))}")
    print(f"overall: pitch min/max = {pitches.min():+.2f}/{pitches.max():+.2f} deg, "
          f"|peak| = {np.max(np.abs(pitches)):.2f} deg")
    print(f"         car_z min/max = {zs.min():.2f}/{zs.max():.2f} mm")

    # Find contiguous phase=3 push segments (push controller dumps with phase=3)
    # and per-segment peak pitch
    print()
    print("Per push-phase segment peak |pitch|:")
    in_seg = False
    seg_start = 0
    seg_idx = 0
    for i in range(len(phases)):
        if phases[i] == 3 and not in_seg:
            in_seg = True
            seg_start = i
        elif phases[i] != 3 and in_seg:
            in_seg = False
            seg = pitches[seg_start:i]
            seg_idx += 1
            print(f"  push#{seg_idx:>2}  ticks [{seg_start:>6} .. {i:>6}]  "
                  f"len={i - seg_start:>5}  peak |pitch| = {np.max(np.abs(seg)):.2f} deg")
    if in_seg:
        seg = pitches[seg_start:]
        seg_idx += 1
        print(f"  push#{seg_idx:>2}  ticks [{seg_start:>6} .. end]      "
              f"len={len(seg):>5}  peak |pitch| = {np.max(np.abs(seg)):.2f} deg")

    out = Path(args.output) if args.output else Path(args.qpos_dump).with_suffix(".pitch.png")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Phase shading
    phase_colors = {0: "#eaeaea", 1: "#ffe1b3", 2: "#cfe5ff", 3: "#fdc1c1", 4: "#d4f0c1"}
    boundaries = [0]
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            boundaries.append(i)
    boundaries.append(len(phases))
    for ax in axes:
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            ax.axvspan(t[s], t[e - 1], color=phase_colors.get(int(phases[s]), "#ffffff"), alpha=0.35)

    axes[0].plot(t, pitches, lw=0.7, color="#222")
    axes[0].axhline(0, color="#888", lw=0.5)
    axes[0].set_ylabel("chassis pitch (deg)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, zs, lw=0.7, color="#226")
    axes[1].set_ylabel("car body z (mm)")
    axes[1].set_xlabel("sim time (s)")
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Runtime chassis pose — {Path(args.qpos_dump).name}")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
