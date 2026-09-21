#!/usr/bin/env python3
"""Turn MORE's collected trees into a training H5.

The collector (more_collect.py) stores, per episode, one row per child of every
expanded node plus the simulator state each push was taken from. This renders those
states and writes what lifelong_trainer needs.

ONE PICTURE PER STATE, not per action. MORE's dataset rotates the image so the push
points in a canonical direction and paints the label at the rotated start pixel
(dataset.py:376), which makes a training SAMPLE per action. Storing pre-rotated images
would multiply the corpus by the action count. Instead the state's picture is stored
once and the rotation happens at load time, which is cheap on a GPU and leaves the
samples identical.

Rendered at 224, MORE's IMAGE_SIZE. `live_scorer` already crops 1024 -> 0.5 m @224
before downsampling to 64, so this asks the same renderer for the stage it already
computes.

Schema:
  ctx          (n, C, 224, 224) f16   the state's scene crop
  contact_px   (n, 60, 2)       f32   pixel of each contact edge, at 224
  label        (n, 60, 5)       f32   max(child.q) per (edge, depth)
  weight       (n, 60, 5)       f32   visit count capped at 50, 0 where no evidence
  reach_mask   (n, 60, 5)       f32   1 where the push is reachable
  evidence     (n, 60, 5)       i8    1 sampled, 0 nobody looked
  xml          (n,)             str   room id, for grouping the train/val split
  object_id    (n,)             str
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live_scorer  # noqa: E402

RENDER_SIZE = 224          # MORE constants.py IMAGE_SIZE
VISIT_CAP = 50             # dataset.py:191, `weight = num_visits if < 50 else 50`
NUM_EDGES, NUM_DEPTHS = 60, 5


def _board(rows, key, dtype=np.float32):
    out = np.zeros((NUM_EDGES, NUM_DEPTHS), dtype=dtype)
    for r in rows:
        e, d = int(r["edge"]), int(r["push_depth"])
        if 0 <= e < NUM_EDGES and 0 <= d < NUM_DEPTHS:
            out[e, d] = r[key]
    return out


def build(shards, out_path, render_config, reach_negative=-1.0, limit=0):
    """Render every stored state and write one row per state.

    reach_negative: the label for a cell whose contact the geometry says the robot
    cannot reach. HY5U gets roughly 231 of these per board for free, since reachability
    is geometry and costs no simulator call, so matching only the simulator budget
    would still leave MORE hundreds of millions of labels short. Withholding them
    handicaps the baseline on something that is not the method's fault. MORE has no
    equivalent signal, so this is a named deviation. Pass --no-reach-negative to drop
    them and measure what they were worth.
    """
    live_scorer.OUT = RENDER_SIZE                       # before _Renderer reads it
    from namo.rl_loop.build_train_h5 import _Renderer, _rlstate
    from scorer_beam import make_env, FALLBACK_GOAL
    from namo.core.xml_goal_parser import extract_goal_with_fallback

    renderer = _Renderer(render_config)
    acc = {k: [] for k in ("ctx", "contact_px", "label", "weight", "reach_mask",
                           "evidence", "xml", "object_id")}
    env_cache = {}
    skipped = 0

    for shard in shards:
        for line in open(shard):
            ep = json.loads(line)
            xml, obj = ep["xml"], ep["object_id"]
            by_uid = {}
            for r in ep["rows"]:
                by_uid.setdefault(r["uid"], []).append(r)
            for uid, rows in by_uid.items():
                qpos = ep["states"].get(uid)
                if qpos is None:
                    skipped += 1
                    continue
                try:
                    if xml not in env_cache:
                        env = make_env(xml)
                        goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
                        env.set_robot_goal(*goal)
                        env_cache[xml] = (env, goal)
                    env, goal = env_cache[xml]
                    env.set_full_state(_rlstate(qpos, [0.0] * len(qpos)))
                    env.get_reachable_objects()
                    ctx, _meta = renderer.render_ctx(env, obj, goal, xml)
                    cpx = renderer.contact_px_live(env, obj)
                except Exception as exc:                # noqa: BLE001
                    print(f"SKIP {xml} {uid}: {type(exc).__name__} {exc}", flush=True)
                    skipped += 1
                    continue

                label = _board(rows, "label")
                visits = _board(rows, "num_visits")
                evidence = np.zeros((NUM_EDGES, NUM_DEPTHS), np.int8)
                reach = np.zeros((NUM_EDGES, NUM_DEPTHS), np.float32)
                for r in rows:
                    e, d = int(r["edge"]), int(r["push_depth"])
                    if 0 <= e < NUM_EDGES and 0 <= d < NUM_DEPTHS:
                        reach[e, d] = 1.0
                        if r["evidence"] == "sampled":
                            evidence[e, d] = 1
                weight = np.minimum(visits, VISIT_CAP).astype(np.float32)
                # MORE writes a row for every child of an expanded node, so an action
                # nobody tried carries label 0 with weight 1. Keeping that is faithful;
                # the evidence board is what lets the trainer drop them instead.
                weight[(evidence == 0) & (reach > 0)] = 1.0
                if reach_negative:
                    label[reach == 0] = reach_negative
                    weight[reach == 0] = 1.0

                acc["ctx"].append(ctx.astype(np.float16))
                acc["contact_px"].append(cpx.astype(np.float32))
                acc["label"].append(label)
                acc["weight"].append(weight)
                acc["reach_mask"].append(reach)
                acc["evidence"].append(evidence)
                acc["xml"].append(xml)
                acc["object_id"].append(obj)
                if limit and len(acc["ctx"]) >= limit:
                    break
            if limit and len(acc["ctx"]) >= limit:
                break
        if limit and len(acc["ctx"]) >= limit:
            break

    n = len(acc["ctx"])
    print(f"rows {n}  skipped {skipped}", flush=True)
    if not n:
        raise SystemExit("nothing rendered")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vlen = h5py.string_dtype("utf-8")
    with h5py.File(out_path, "w") as f:
        for k in ("ctx", "contact_px", "label", "weight", "reach_mask", "evidence"):
            f.create_dataset(k, data=np.stack(acc[k]), compression="lzf")
        for k in ("xml", "object_id"):
            f.create_dataset(k, data=np.array(acc[k], dtype=object), dtype=vlen)
        f.attrs["render_size"] = RENDER_SIZE
        f.attrs["visit_cap"] = VISIT_CAP
        f.attrs["reach_negative"] = reach_negative
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="collector JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-config", default="", help="defaults to the deploy scorer config")
    ap.add_argument("--no-reach-negative", action="store_true",
                    help="drop the free geometric negatives, to measure what they were worth")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke runs)")
    args = ap.parse_args()

    cfg = args.render_config
    if not cfg:
        from scorer_beam import CFG
        cfg = CFG
    build(args.shards, args.out, cfg,
          reach_negative=(0.0 if args.no_reach_negative else -1.0), limit=args.limit)


if __name__ == "__main__":
    main()
