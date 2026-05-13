"""Plot per-tick log written by log_one_push.py.

Renders a 4-panel figure: object pose, car pose, wheel ctrl, wheel velocity.
Phase backgrounds are color-shaded (pre-settle / push / post-settle).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"empty log: {path}")
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    out["tick"] = out["tick"].astype(int)
    out["phase"] = out["phase"].astype(int)
    return out


def shade_phases(ax, t, phase):
    colors = {0: "#dddddd", 1: "#ffe1b3", 2: "#cfe5ff", 3: "#fdc1c1", 4: "#d4f0c1"}
    labels = {0: "pre-settle", 1: "push", 2: "post-settle", 3: "exit-ramp", 4: "entry-ramp"}
    drawn = set()
    boundaries = [0]
    for i in range(1, len(phase)):
        if phase[i] != phase[i - 1]:
            boundaries.append(i)
    boundaries.append(len(phase))
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        p = phase[s]
        lbl = labels[p] if p not in drawn else None
        drawn.add(p)
        ax.axvspan(t[s], t[e - 1], color=colors[p], alpha=0.5, label=lbl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    d = load_csv(args.csv)
    t = d["tick"] * 0.002  # sim seconds (timestep=0.002)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"Push primitive log: {args.csv.name}", fontsize=12)

    # 1) Object pose
    ax = axes[0]
    shade_phases(ax, t, d["phase"])
    ax.plot(t, d["obj_x"] * 1000, label="obj_x (mm)")
    ax.plot(t, d["obj_y"] * 1000, label="obj_y (mm)")
    ax.plot(t, np.degrees(d["obj_theta"]), label="obj_θ (deg)", linestyle="--")
    ax.set_ylabel("object")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # 2) Car pose
    ax = axes[1]
    shade_phases(ax, t, d["phase"])
    ax.plot(t, d["car_x"] * 1000, label="car_x (mm)")
    ax.plot(t, d["car_y"] * 1000, label="car_y (mm)")
    ax.plot(t, np.degrees(d["car_theta"]), label="car_θ (deg)", linestyle="--")
    ax.set_ylabel("car")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # 3) Wheel ctrl
    ax = axes[2]
    shade_phases(ax, t, d["phase"])
    ax.plot(t, d["wheel_left_ctrl"], label="left ctrl (rad/s)")
    ax.plot(t, d["wheel_right_ctrl"], label="right ctrl (rad/s)", linestyle="--")
    ax.set_ylabel("wheel ctrl")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # 4) Wheel velocity
    ax = axes[3]
    shade_phases(ax, t, d["phase"])
    ax.plot(t, d["wheel_left_vel"], label="left vel (rad/s)")
    ax.plot(t, d["wheel_right_vel"], label="right vel (rad/s)", linestyle="--")
    ax.set_ylabel("wheel vel")
    ax.set_xlabel("sim time (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(args.output) if args.output else args.csv.with_suffix(".png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
