#!/usr/bin/env python3
"""Visualize GT-successful pushes vs model-predicted pushes for NAMO 1-push test episodes.

For a handful of one-push test episodes spanning difficulty (chosen by solve_rate), produce ONE
composite PNG per episode containing:

  Viz B (parameter grid): three 60x5 heatmaps side by side --
      GT f_grid | Hz q-grid | NoHz q-grid.   rows = edge/contact 0-59, cols = push depth 0-4.
      GT: green=opens(1) / grey=tried-but-failed(0) / white=unreachable(NaN).
      q : a continuous colormap over the model's P(this push opens the goal).

  Viz A (scene overlay, WORLD/meter coords): a top-down matplotlib render of the scene (equal aspect,
      xlim/ylim = world bounds). The labeled object's candidate push TARGETS are drawn as the object's
      RESULTING footprint (oriented box at the primitive's world target), colored by the model's q,
      with GT-opener footprints (f_grid==1) outlined bold green. One panel for Hz, one for NoHz.
      Also: the object START footprint (bold black), static walls (grey), other movables (light), and
      the robot goal (red star).

The (edge 0-59, depth 0-4) indexing is shared by the f_grid, the model q-grid, and the primitive
target table -- they line up cell-for-cell.

Run from the repo root after `source env.amarel.sh`:
    python scripts/sandbox/viz_pushes.py
    python scripts/sandbox/viz_pushes.py --targets 0.9,0.33,0.05 --labels easy,mid,hard
    python scripts/sandbox/viz_pushes.py --episodes <xml>:<object_id> ...   # explicit picks
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

import namo_rl  # noqa: E402
from scorer_beam import BeamPlanner, FALLBACK_GOAL  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import resolve  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Config (task-specified)
# ------------------------------------------------------------------------------------------------
CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"
LABELS_JSON = "/scratch/dm1487/datasets/namo_testset_v1/labels/onepush_episodes.json"
PRIM_DAT = f"{REPO}/data/motion_primitives_1x_car_d5_square.dat"
OUT_DIR = "/scratch/dm1487/eval/viz"

HZ_CKPT = ("/scratch/dm1487/sage_outputs/scorer/qfull_v3_v4hq_s1/namo-classifier/"
           "qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt")
NOHZ_CKPT = ("/scratch/dm1487/sage_outputs/scorer/qfull_nohz_v3_v4hq_s1/namo-classifier/"
             "wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt")

N_EDGES, N_DEPTHS = 60, 5


# ------------------------------------------------------------------------------------------------
# Primitive table  ->  prim[edge][depth] = (dx, dy, dtheta)  (object-local SE2 deltas)
# ------------------------------------------------------------------------------------------------
def load_prim_table(path=PRIM_DAT):
    """Return (60,5,3) float32 table of object-local (dx,dy,dtheta); depth = push_steps-1."""
    table = np.full((N_EDGES, N_DEPTHS, 3), np.nan, dtype=np.float32)
    with open(path, "rb") as f:
        count = struct.unpack("I", f.read(4))[0]
        for _ in range(count):
            dx, dy, dth, e, ps = struct.unpack("fffBB", f.read(14))
            d = ps - 1
            if 0 <= e < N_EDGES and 0 <= d < N_DEPTHS:
                table[e, d] = (dx, dy, dth)
    return table


def world_targets(prim, ox, oy, oth):
    """(60,5,3) world (wx,wy,wth) for every primitive cell, from object pose (ox,oy,oth)."""
    c, s = math.cos(oth), math.sin(oth)
    out = np.full_like(prim, np.nan)
    dx, dy, dth = prim[..., 0], prim[..., 1], prim[..., 2]
    out[..., 0] = ox + dx * c - dy * s
    out[..., 1] = oy + dx * s + dy * c
    out[..., 2] = oth + dth
    return out


# ------------------------------------------------------------------------------------------------
# GT f_grid
# ------------------------------------------------------------------------------------------------
def build_f_grid(rec):
    """(60,5): NaN=unreachable/untried, 0=tried-failed, 1=opens (valid)."""
    g = np.full((N_EDGES, N_DEPTHS), np.nan, dtype=np.float32)
    for e, d in rec["tried"]:
        if 0 <= e < N_EDGES and 0 <= d < N_DEPTHS:
            g[e, d] = 0.0
    for e, d in rec["valid"]:
        if 0 <= e < N_EDGES and 0 <= d < N_DEPTHS:
            g[e, d] = 1.0
    return g


# ------------------------------------------------------------------------------------------------
# Episode selection
# ------------------------------------------------------------------------------------------------
def flatten_records(labels):
    """[(xml, rec, solve_rate, resolved_xml_exists)] over every (room, record)."""
    out = []
    for xml, recs in labels.items():
        rx = str(resolve(xml))
        ok = os.path.exists(rx)
        for rec in recs:
            out.append((xml, rec, float(rec.get("solve_rate", float("nan"))), ok))
    return out


def select_by_targets(labels, targets, labelnames):
    """Pick one (xml, rec) per target solve_rate, nearest match, distinct rooms, xml must exist."""
    flat = [r for r in flatten_records(labels) if r[3] and np.isfinite(r[2])]
    picked, used_xml = [], set()
    for tgt, name in zip(targets, labelnames):
        cands = [r for r in flat if r[0] not in used_xml]
        if not cands:
            cands = flat
        # nearest solve_rate; tiebreak prefers more 'tried' cells (richer panel)
        best = min(cands, key=lambda r: (abs(r[2] - tgt), -len(r[1]["tried"])))
        used_xml.add(best[0])
        picked.append((name, tgt, best[0], best[1]))
    return picked


def select_explicit(labels, specs):
    """specs: ['<xml>:<object_id>', ...]  ->  list of (name, solve_rate, xml, rec)."""
    out = []
    for i, spec in enumerate(specs):
        xml, _, obj = spec.partition(":")
        recs = labels.get(xml)
        if recs is None:
            raise SystemExit(f"xml not in labels: {xml}")
        rec = next((r for r in recs if (not obj or r["object_id"] == obj)), recs[0])
        out.append((f"ep{i}", float(rec.get("solve_rate", float("nan"))), xml, rec))
    return out


# ------------------------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------------------------
def box_corners(cx, cy, theta, hx, hy):
    """4x2 world corners of an oriented box (half-extents hx,hy) centered at (cx,cy), yaw theta."""
    c, s = math.cos(theta), math.sin(theta)
    loc = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.array([cx, cy])


def quat_yaw(info):
    """Yaw (z-rotation) from a wall's quaternion fields; identity -> 0."""
    qw = info.get("quat_w", 1.0)
    qz = info.get("quat_z", 0.0)
    return 2.0 * math.atan2(qz, qw)


# ------------------------------------------------------------------------------------------------
# Scene render (matplotlib, world coords)
# ------------------------------------------------------------------------------------------------
def draw_scene(ax, env, bounds, obj_id, obj_center, obj_theta, obj_hx, obj_hy, goal, view=None):
    """Draw walls (grey), other movables (light), labeled-object START (bold black), goal (star).

    `bounds` = world bounds (drives nothing but is the floor for the view). `view` = the actual
    axis limits (padded to include off-bounds commanded-target footprints); defaults to `bounds`.
    """
    oi = env.get_object_info()
    obs = env.get_observation()
    xmin, xmax, ymin, ymax = view if view is not None else bounds

    # static walls
    for name, info in oi.items():
        if not name.startswith("wall"):
            continue
        cx, cy = info.get("pos_x"), info.get("pos_y")
        if cx is None or cy is None:
            continue
        corners = box_corners(cx, cy, quat_yaw(info), info["size_x"], info["size_y"])
        ax.add_patch(Polygon(corners, closed=True, facecolor="0.55", edgecolor="0.3",
                             linewidth=0.5, zorder=1))

    # other movables (light)
    for name, info in oi.items():
        if not name.endswith("_movable") or name == obj_id:
            continue
        pose = obs.get(f"{name}_pose")
        if pose is None:
            continue
        corners = box_corners(pose[0], pose[1], pose[2], info["size_x"], info["size_y"])
        ax.add_patch(Polygon(corners, closed=True, facecolor="#cfe8ff", edgecolor="#5a8fbf",
                             linewidth=0.8, alpha=0.85, zorder=2))

    # labeled object START footprint (bold black)
    start = box_corners(obj_center[0], obj_center[1], obj_theta, obj_hx, obj_hy)
    ax.add_patch(Polygon(start, closed=True, fill=False, edgecolor="black",
                         linewidth=2.6, zorder=6))

    # robot goal (red star)
    ax.plot([goal[0]], [goal[1]], marker="*", markersize=22, color="red",
            markeredgecolor="black", markeredgewidth=0.8, zorder=7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")


def draw_candidates(ax, wt, fgrid, q, obj_hx, obj_hy, norm, cmap):
    """Overlay candidate target footprints (reachable cells only) colored by q; GT openers bold green."""
    reach = np.argwhere(~np.isnan(fgrid))
    for e, d in reach:
        cx, cy, cth = wt[e, d]
        corners = box_corners(cx, cy, cth, obj_hx, obj_hy)
        opener = fgrid[e, d] == 1.0
        ax.add_patch(Polygon(corners, closed=True, facecolor=cmap(norm(q[e, d])),
                             edgecolor="none", alpha=0.45, zorder=3))
        if opener:
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="lime",
                                 linewidth=2.2, zorder=5))
    # GT-opener target CENTERS as bold markers (read on top of the cloud)
    for e, d in np.argwhere(fgrid == 1.0):
        ax.plot([wt[e, d, 0]], [wt[e, d, 1]], marker="o", markersize=4,
                color="lime", markeredgecolor="black", markeredgewidth=0.5, zorder=5)


# ------------------------------------------------------------------------------------------------
# Heatmaps (Viz B)
# ------------------------------------------------------------------------------------------------
def draw_gt_heatmap(ax, fgrid):
    """white=NaN, grey=tried-failed(0), green=opens(1)."""
    rgb = np.ones((N_EDGES, N_DEPTHS, 3), dtype=float)            # white background (NaN)
    rgb[fgrid == 0.0] = (0.6, 0.6, 0.6)                            # grey
    rgb[fgrid == 1.0] = (0.15, 0.75, 0.2)                         # green
    ax.imshow(rgb, aspect="auto", origin="upper", interpolation="nearest")
    ax.set_title(f"GT f_grid  ({int((fgrid == 1.0).sum())} openers / "
                 f"{int((~np.isnan(fgrid)).sum())} tried)", fontsize=10)
    _heat_axes(ax)


def draw_q_heatmap(ax, q, title, norm, cmap):
    im = ax.imshow(q, aspect="auto", origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"{title}  (q {q.min():.2f}-{q.max():.2f})", fontsize=10)
    _heat_axes(ax)
    return im


def _heat_axes(ax):
    ax.set_xlabel("push depth")
    ax.set_ylabel("edge / contact (0-59)")
    ax.set_xticks(range(N_DEPTHS))


# ------------------------------------------------------------------------------------------------
# Per-episode composite
# ------------------------------------------------------------------------------------------------
def run_episode(name, xml, rec, prim, scorers, out_dir):
    obj_id = rec["object_id"]
    ox, oy = rec["object_center"]
    oth = float(rec["object_theta"])
    sr = float(rec.get("solve_rate", float("nan")))
    rx = str(resolve(xml))

    env = namo_rl.RLEnvironment(rx, CFG, False)
    env.reset()
    goal = extract_goal_with_fallback(rx, FALLBACK_GOAL)
    env.set_robot_goal(*goal)
    env.get_reachable_objects()                       # warm the wavefront

    oi = env.get_object_info()[obj_id]
    hx, hy = oi["size_x"], oi["size_y"]               # half-extents the scorer uses
    obs_pose = env.get_observation()[f"{obj_id}_pose"]

    fgrid = build_f_grid(rec)
    wt = world_targets(prim, ox, oy, oth)

    # model q for both checkpoints (h=1; region_samples defaults None -> final-goal conditioning)
    q = {}
    for tag, pl in scorers.items():
        q[tag] = pl.scorer.score_state(env, obj_id, goal, rx, h=1).astype(np.float32)

    # shared q color scale across Hz & NoHz so the two panels are comparable
    qcat = np.concatenate([q["Hz"].ravel(), q["NoHz"].ravel()])
    norm = Normalize(vmin=float(qcat.min()), vmax=float(qcat.max()))
    cmap = matplotlib.colormaps["viridis"]

    bounds = env.get_world_bounds()
    # pad the view so commanded-target footprints that overrun a wall stay visible
    reach = np.argwhere(~np.isnan(fgrid))
    view_pts = [box_corners(*wt[e, d], hx, hy) for e, d in reach]
    view_pts.append(box_corners(ox, oy, oth, hx, hy))
    allc = np.vstack(view_pts)
    m = 0.03
    view = (min(bounds[0], allc[:, 0].min()) - m, max(bounds[1], allc[:, 0].max()) + m,
            min(bounds[2], allc[:, 1].min()) - m, max(bounds[3], allc[:, 1].max()) + m)

    # ---- precision checks ------------------------------------------------------------------------
    n_open = int((fgrid == 1.0).sum())
    n_tried = int((~np.isnan(fgrid)).sum())
    start_err = math.hypot(obs_pose[0] - ox, obs_pose[1] - oy)
    valid = world_targets(prim, ox, oy, oth)
    cand_centers = valid[..., :2].reshape(-1, 2)
    cand_centers = cand_centers[~np.isnan(cand_centers).any(axis=1)]
    cand_dist = np.linalg.norm(cand_centers - np.array([ox, oy]), axis=1)
    xmin, xmax, ymin, ymax = bounds
    off = int(((cand_centers[:, 0] < xmin) | (cand_centers[:, 0] > xmax) |
               (cand_centers[:, 1] < ymin) | (cand_centers[:, 1] > ymax)).sum())
    opener_centers = wt[fgrid == 1.0][:, :2] if n_open else np.zeros((0, 2))
    opener_off = int(((opener_centers[:, 0] < xmin) | (opener_centers[:, 0] > xmax) |
                      (opener_centers[:, 1] < ymin) | (opener_centers[:, 1] > ymax)).sum()) if n_open else 0

    print(f"\n=== episode [{name}] solve_rate={sr:.3f}  obj={obj_id} ===")
    print(f"  room: {os.path.basename(xml)}")
    print(f"  GT openers (f_grid==1): {n_open}   tried cells: {n_tried}")
    print(f"  Hz   q min/max: {q['Hz'].min():.4f} / {q['Hz'].max():.4f}")
    print(f"  NoHz q min/max: {q['NoHz'].min():.4f} / {q['NoHz'].max():.4f}")
    print(f"  START footprint vs drawn movable pose: {start_err*1000:.3f} mm (should be ~0)")
    print(f"  candidate cloud: {len(cand_centers)}/300 targets, dist-from-object "
          f"min/mean/max = {cand_dist.min()*1000:.1f}/{cand_dist.mean()*1000:.1f}/{cand_dist.max()*1000:.1f} mm, "
          f"{off} off-screen")
    if n_open:
        # peak-on-opener diagnostic: rank of best opener under each model (over tried cells)
        tried_mask = ~np.isnan(fgrid)
        for tag in ("Hz", "NoHz"):
            order = np.argsort(-q[tag][tried_mask])
            ranks = {tuple(c) for c in np.argwhere(fgrid == 1.0)}
            tried_idx = np.argwhere(tried_mask)
            sorted_cells = [tuple(tried_idx[i]) for i in order]
            best_rank = min((sorted_cells.index(c) for c in ranks), default=-1)
            print(f"  {tag}: best GT-opener ranks #{best_rank+1}/{n_tried} among tried "
                  f"(q@opener max={max(q[tag][e, d] for e, d in ranks):.3f})")
        print(f"  GT-opener target footprints off-screen: {opener_off}/{n_open}")

    # ---- figure ----------------------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.9)

    axA_hz = fig.add_subplot(gs[0, 0:6])
    axA_nohz = fig.add_subplot(gs[0, 6:12])
    for ax, tag in ((axA_hz, "Hz"), (axA_nohz, "NoHz")):
        draw_scene(ax, env, bounds, obj_id, (ox, oy), oth, hx, hy, goal, view=view)
        draw_candidates(ax, wt, fgrid, q[tag], hx, hy, norm, cmap)
        ax.set_title(f"Viz A: scene + {tag} q over candidate footprints "
                     f"(GT openers = green)", fontsize=11)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=[axA_hz, axA_nohz], fraction=0.025, pad=0.01, label="model q")

    axB_gt = fig.add_subplot(gs[1, 0:4])
    axB_hz = fig.add_subplot(gs[1, 4:8])
    axB_nohz = fig.add_subplot(gs[1, 8:12])
    draw_gt_heatmap(axB_gt, fgrid)
    draw_q_heatmap(axB_hz, q["Hz"], "Hz q-grid", norm, cmap)
    im = draw_q_heatmap(axB_nohz, q["NoHz"], "NoHz q-grid", norm, cmap)
    fig.colorbar(im, ax=[axB_hz, axB_nohz], fraction=0.04, pad=0.02, label="model q")

    fig.suptitle(f"[{name}]  solve_rate={sr:.3f}   obj={obj_id}   "
                 f"openers={n_open}/{n_tried} tried   room={os.path.basename(xml)}",
                 fontsize=13)

    os.makedirs(out_dir, exist_ok=True)
    room = os.path.splitext(os.path.basename(xml))[0]
    out_path = os.path.join(out_dir, f"viz_{name}_{room}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")
    return out_path


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS_JSON)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--targets", default="0.9,0.33,0.05",
                    help="comma solve_rate targets (one episode picked nearest each)")
    ap.add_argument("--labels-names", default="easy,mid,hard")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="explicit '<xml>:<object_id>' picks (overrides --targets)")
    ap.add_argument("--hz-ckpt", default=HZ_CKPT)
    ap.add_argument("--nohz-ckpt", default=NOHZ_CKPT)
    a = ap.parse_args()

    labels = json.load(open(a.labels))
    prim = load_prim_table()
    print(f"loaded {len(labels)} rooms; prim table cells filled = "
          f"{int((~np.isnan(prim[..., 0])).sum())}/300")

    if a.episodes:
        picks = [(n, x, r) for (n, _sr, x, r) in select_explicit(labels, a.episodes)]
    else:
        tgts = [float(x) for x in a.targets.split(",")]
        names = a.labels_names.split(",")
        picks = [(n, x, r) for (n, _t, x, r) in select_by_targets(labels, tgts, names)]

    print("loading scorers (CPU ok)...")
    scorers = {"Hz": BeamPlanner(ckpt=a.hz_ckpt), "NoHz": BeamPlanner(ckpt=a.nohz_ckpt)}

    out_paths = []
    for name, xml, rec in picks:
        out_paths.append(run_episode(name, xml, rec, prim, scorers, a.out_dir))

    print("\n==== PNGs ====")
    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
