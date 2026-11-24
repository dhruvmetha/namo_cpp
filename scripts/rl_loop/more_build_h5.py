#!/usr/bin/env python3
"""Turn MORE collection trees (more_collect.py JSONL) into a training H5.

MORE's save_mcts_data (mcts_main.py:180-212) writes one row per child of every node with
children: the parent's picture, the action, label max(child.q) and the child's visit count. Every
child of MORE's collection tree was simulated (mcts/nodes.py:47-63), so every row is physics
evidence; an action the tree never simulated has no row and gets no loss.

ONE PICTURE PER STATE, not per action: MORE rotates each sample by its push angle at load time
(dataset.py:376), so the state's picture is stored once and train_more.py rotates it per row.

The H5 stores what the trees measured and nothing the trainer decides. Unreachable cells are
marked by reach_mask = 0 and carry label 0, weight 0 here; what they become (MORE's label 0 at
weight 0.001, or our label -1 at weight 1) is the trainer's --invalid choice.

Schema (n states):
  ctx          (n, C, S, S) f16   the state's scene crop, S = --render-size (224 PushNet, 64 EdgeCrossAttn)
  contact_px   (n, 60, 2)   f32   contact pixel of each edge in (column, row), at S
  label        (n, 60, 5)   f32   max(child.q) where a row exists, else 0
  weight       (n, 60, 5)   f32   min(visits, 50) where a row exists, else 0
  evidence     (n, 60, 5)   i8    1 where a row exists (a simulated action)
  reach_mask   (n, 60, 5)   f32   1 where the action is legal at that state (the candidates() pool)
  xml, object_id (n,)       str   room id (for the room split) and pushed object
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

from namo.rl_loop.more_training import VISIT_CAP  # noqa: E402

NUM_EDGES, NUM_DEPTHS = 60, 5
SCHEMA = "more-trees-v2"


def boards(rows, legal):
    """label, weight, evidence, reach for one saved state."""
    label = np.zeros((NUM_EDGES, NUM_DEPTHS), np.float32)
    weight = np.zeros((NUM_EDGES, NUM_DEPTHS), np.float32)
    evidence = np.zeros((NUM_EDGES, NUM_DEPTHS), np.int8)
    reach = np.zeros((NUM_EDGES, NUM_DEPTHS), np.float32)
    for e, d in legal:
        if 0 <= e < NUM_EDGES and 0 <= d < NUM_DEPTHS:
            reach[e, d] = 1.0
    for r in rows:
        e, d = int(r["edge"]), int(r["push_depth"])
        if 0 <= e < NUM_EDGES and 0 <= d < NUM_DEPTHS:
            label[e, d] = r["label"]
            weight[e, d] = min(int(r["num_visits"]), VISIT_CAP)    # dataset.py:191
            evidence[e, d] = 1
            reach[e, d] = 1.0
    return label, weight, evidence, reach


def build(shards, out_path, render_config, render_size=224, limit=0):
    live_scorer.OUT = render_size                       # before _Renderer reads it
    from namo.rl_loop.build_train_h5 import _Renderer, _rlstate
    from scorer_beam import make_env, FALLBACK_GOAL
    from namo.core.xml_goal_parser import extract_goal_with_fallback

    renderer = _Renderer(render_config)
    acc = {k: [] for k in ("ctx", "contact_px", "label", "weight", "reach_mask", "evidence", "xml", "object_id")}
    env_cache, skipped = {}, 0
    for shard in shards:
        for line in open(shard):
            ep = json.loads(line)
            if ep.get("schema") != SCHEMA:
                raise SystemExit(f"{shard}: not a {SCHEMA} shard (re-collect with more_collect.py)")
            xml, obj = ep["xml"], ep["object_id"]
            by_key = {}
            for r in ep["rows"]:
                by_key.setdefault(r["key"], []).append(r)
            for key, rows in by_key.items():
                try:
                    if xml not in env_cache:
                        env = make_env(xml)
                        goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
                        env.set_robot_goal(*goal)
                        env_cache[xml] = (env, goal)
                    env, goal = env_cache[xml]
                    qpos = ep["states"][key]
                    env.set_full_state(_rlstate(qpos, [0.0] * len(qpos)))
                    env.get_reachable_objects()
                    ctx, _meta = renderer.render_ctx(env, obj, goal, xml)
                    cpx = renderer.contact_px_live(env, obj)
                except Exception as exc:                # noqa: BLE001  one bad state must not kill a build
                    print(f"SKIP {xml} {key}: {type(exc).__name__} {exc}", flush=True)
                    skipped += 1
                    continue
                label, weight, evidence, reach = boards(rows, ep["legal"].get(key, []))
                for k, v in (("ctx", ctx.astype(np.float16)), ("contact_px", cpx.astype(np.float32)),
                             ("label", label), ("weight", weight), ("reach_mask", reach),
                             ("evidence", evidence), ("xml", xml), ("object_id", obj)):
                    acc[k].append(v)
                if limit and len(acc["ctx"]) >= limit:
                    break
            if limit and len(acc["ctx"]) >= limit:
                break
        if limit and len(acc["ctx"]) >= limit:
            break

    n = len(acc["ctx"])
    print(f"states {n}  skipped {skipped}", flush=True)
    if not n:
        raise SystemExit("nothing rendered")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vlen = h5py.string_dtype("utf-8")
    with h5py.File(out_path, "w") as f:
        for k in ("ctx", "contact_px", "label", "weight", "reach_mask", "evidence"):
            arr = np.stack(acc[k])
            # One state per chunk: training reads one state at a time, and a chunk spanning
            # thousands of states once starved a GPU for eight hours.
            f.create_dataset(k, data=arr, compression="lzf", chunks=(1,) + arr.shape[1:])
        for k in ("xml", "object_id"):
            f.create_dataset(k, data=np.array(acc[k], dtype=object), dtype=vlen)
        f.attrs["render_size"] = render_size
        f.attrs["visit_cap"] = VISIT_CAP
        f.attrs["schema"] = SCHEMA
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="more_collect.py JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-size", type=int, default=224,
                    help="224 for MORE's PushNet (constants.py IMAGE_SIZE), 64 for EdgeCrossAttn")
    ap.add_argument("--render-config", default="", help="defaults to the deploy scorer config")
    ap.add_argument("--limit", type=int, default=0, help="stop after N states (smoke runs)")
    args = ap.parse_args()
    cfg = args.render_config
    if not cfg:
        from scorer_beam import CFG
        cfg = CFG
    build(args.shards, args.out, cfg, render_size=args.render_size, limit=args.limit)


if __name__ == "__main__":
    main()
