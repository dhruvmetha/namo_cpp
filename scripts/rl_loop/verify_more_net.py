#!/usr/bin/env python3
"""Does MORE's trained network actually predict anything, on states it did not train on?

Write this down as a habit: a falling loss is not evidence a network learned the task.
The first full run here reported 1.82 for an entire epoch while every output pixel had
already walked to -8000 against labels in [-1, 1.2]. The loss looked fine because it was
a cumulative mean and because only 9 of 250,880 output pixels per sample carry any
weight at all. Predictions against held-out labels are what actually caught it.

Three checks, in the order they catch things:

  range      predictions must live near the label range. A net predicting thousands has
             diverged, whatever the loss says.
  ordering   openings (label > 1) must score ABOVE non-openings. This is the only thing
             the search needs, since it uses the net to order pushes, not to calibrate.
  geometry   unreachable contacts must score below reachable ones, when the arm was
             trained with the reachability negatives.

Split is by room, never by state: many states share an xml and a state-level split would
leak the same scene into both halves.

  python verify_more_net.py --h5 more_pilot_chunked.h5 --ckpt model_s1/more_epoch001.pt
"""
import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo.rl_loop.more_net import MorePushNet, read_contact_values  # noqa: E402

NUM_EDGES, NUM_DEPTHS = 60, 5


def _rotate(x, angle):
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=True)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=True)


def _rotate_pixel(px, angle, size):
    centre = (size - 1) / 2.0
    y, x = px[:, 0] - centre, px[:, 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([y * cos - x * sin + centre, y * sin + x * cos + centre], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--states", type=int, default=200, help="held-out states to score")
    ap.add_argument("--holdout-frac", type=float, default=0.1, help="rooms held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    xml = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["xml"][:]])
    rooms = np.unique(xml)
    rng = np.random.default_rng(args.seed)
    held = set(rng.choice(rooms, max(1, int(len(rooms) * args.holdout_frac)), replace=False))
    pool = np.flatnonzero(np.array([x in held for x in xml]))
    if len(pool) == 0:
        raise SystemExit("no held-out states; is the corpus one room?")
    pick = np.sort(rng.choice(pool, min(args.states, len(pool)), replace=False))
    print(f"rooms {len(rooms)}  held out {len(held)}  states scored {len(pick)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu")
    weights = blob.get("model", blob)
    net = MorePushNet(in_channels=weights["body.conv1.weight"].shape[1], num_depths=NUM_DEPTHS)
    net.load_state_dict(weights)
    net.to(dev).eval()
    ang = torch.tensor(2 * math.pi * np.arange(NUM_EDGES) / NUM_EDGES,
                       dtype=torch.float32, device=dev)

    P, L, E, R = [], [], [], []
    with torch.no_grad():
        for i in pick:
            ctx = torch.tensor(f["ctx"][int(i)].astype(np.float32), device=dev)
            cpx = torch.tensor(f["contact_px"][int(i)], device=dev)
            pred = net(_rotate(ctx.unsqueeze(0).expand(NUM_EDGES, -1, -1, -1), ang))
            rpx = _rotate_pixel(cpx, ang, ctx.shape[-1]).unsqueeze(1)
            P.append(read_contact_values(pred, rpx, patch=3)[:, 0, :].cpu().numpy())
            L.append(f["label"][int(i)]); E.append(f["evidence"][int(i)]); R.append(f["reach_mask"][int(i)])
    P, L, E, R = (np.stack(v) for v in (P, L, E, R))

    ok = True
    lo, hi = P.min(), P.max()
    print(f"\nrange     predictions {lo:8.3f} .. {hi:8.3f}   labels {L.min():.3f} .. {L.max():.3f}")
    if not (-10 < lo and hi < 10):
        print("          FAIL: predictions are far outside the label range; the run diverged")
        ok = False

    opened, other = (E == 1) & (L > 1), (E == 1) & (L <= 1)
    if opened.sum() and other.sum():
        a, b = P[opened].mean(), P[other].mean()
        print(f"ordering  openings {a:7.3f}  vs non-openings {b:7.3f}  gap {a-b:+.3f}"
              f"   (n={opened.sum()} / {other.sum()})")
        if a <= b:
            print("          FAIL: openings do not score above non-openings, so the search "
                  "would gain nothing from this net")
            ok = False
    else:
        print("ordering  skipped: no openings among the held-out sampled cells")

    if (R == 0).sum():
        a, b = P[R == 0].mean(), P[R > 0].mean()
        print(f"geometry  unreachable {a:7.3f}  vs reachable {b:7.3f}  gap {a-b:+.3f}")
        if a >= b:
            print("          WARN: unreachable contacts do not score below reachable ones")

    m = E == 1
    if L[m].std() > 1e-9:
        print(f"\ncorrelation with sampled labels: {np.corrcoef(P[m], L[m])[0, 1]:.3f}")
    print("\nVERDICT:", "usable" if ok else "NOT usable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
