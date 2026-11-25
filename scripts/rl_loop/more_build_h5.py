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
  episode      (n,)         i64   the state's episode ordinal over the input shards in order, so a
                                  budget cut is the prefix of states with episode < its episode count

Big builds run as parallel parts over consecutive chunks of the ordered episodes (--episode-offset), joined by
more_merge_h5.py; states are appended as rendered, so memory stays flat.
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


class _Writer:
    """Appends one state at a time, so a build's memory does not grow with its size."""

    def __init__(self, out_path, render_size):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(out_path, "w")
        self.n = 0
        self.ds = {}
        self.render_size = render_size

    def _make(self, k, v):
        if k in ("xml", "object_id"):
            self.ds[k] = self.f.create_dataset(k, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"),
                                               chunks=(4096,))
        elif k == "episode":
            self.ds[k] = self.f.create_dataset(k, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(4096,))
        else:
            # One state per chunk: training reads one state at a time, and a chunk spanning
            # thousands of states once starved a GPU for eight hours.
            self.ds[k] = self.f.create_dataset(k, shape=(0,) + v.shape, maxshape=(None,) + v.shape, dtype=v.dtype,
                                               compression="lzf", chunks=(1,) + v.shape)

    def add(self, rec):
        for k, v in rec.items():
            if k not in self.ds:
                self._make(k, v)
            d = self.ds[k]
            d.resize(self.n + 1, axis=0)
            d[self.n] = v
        self.n += 1

    def close(self):
        self.f.attrs["render_size"] = self.render_size
        self.f.attrs["visit_cap"] = VISIT_CAP
        self.f.attrs["schema"] = SCHEMA
        self.f.close()


def episodes(shards, offset):
    """(episode ordinal, record) over the shards read in order; ordinals start at offset."""
    n = offset
    for shard in shards:
        with open(shard) as fh:
            for line in fh:
                yield n, json.loads(line)
                n += 1


def build(shards, out_path, render_config, render_size=224, limit=0, offset=0):
    """Render every saved state of the shards' episodes into out_path; episode ordinals start at offset."""
    from namo.rl_loop.build_train_h5 import _Renderer, _rlstate
    from scorer_beam import make_env, FALLBACK_GOAL
    from namo.core.xml_goal_parser import extract_goal_with_fallback

    renderer = _Renderer(render_config, out=render_size)
    writer = _Writer(out_path, render_size)
    env_cache, skipped, n_eps = {}, 0, 0
    for ordinal, ep in episodes(shards, offset):
        if ep.get("schema") != SCHEMA:
            raise SystemExit(f"episode {ordinal}: not a {SCHEMA} record (re-collect with more_collect.py)")
        n_eps += 1
        xml, obj = ep["xml"], ep["object_id"]
        by_key = {}
        for r in ep["rows"]:
            by_key.setdefault(r["key"], []).append(r)
        for key, rows in by_key.items():
            try:
                if xml not in env_cache:
                    env_cache.clear()                   # episodes arrive grouped by room
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
            except Exception as exc:                    # noqa: BLE001  one bad state must not kill a build
                print(f"SKIP {xml} {key}: {type(exc).__name__} {exc}", flush=True)
                skipped += 1
                continue
            label, weight, evidence, reach = boards(rows, ep["legal"].get(key, []))
            writer.add({"ctx": ctx.astype(np.float16), "contact_px": cpx.astype(np.float32), "label": label,
                        "weight": weight, "reach_mask": reach, "evidence": evidence, "xml": xml,
                        "object_id": obj, "episode": ordinal})
            if limit and writer.n >= limit:
                break
        if limit and writer.n >= limit:
            break
        if n_eps % 200 == 0:
            print(f"episodes {n_eps} states {writer.n} skipped {skipped}", flush=True)

    n = writer.n
    writer.f.attrs["episodes"] = n_eps
    writer.f.attrs["episode_offset"] = offset
    writer.f.attrs["skipped_states"] = skipped
    writer.close()
    print(f"episodes {n_eps} states {n}  skipped {skipped}", flush=True)
    if not n:
        Path(out_path).unlink()
        raise SystemExit("nothing rendered")
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="more_collect.py JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render-size", type=int, default=224,
                    help="224 for MORE's PushNet (constants.py IMAGE_SIZE), 64 for EdgeCrossAttn")
    ap.add_argument("--render-config", default="", help="defaults to the deploy scorer config")
    ap.add_argument("--limit", type=int, default=0, help="stop after N states (smoke runs)")
    ap.add_argument("--episode-offset", type=int, default=0,
                    help="ordinal of the first episode in --shards within the whole ordered collection")
    args = ap.parse_args()
    cfg = args.render_config
    if not cfg:
        from scorer_beam import CFG
        cfg = CFG
    build(args.shards, args.out, cfg, render_size=args.render_size, limit=args.limit,
          offset=args.episode_offset)


if __name__ == "__main__":
    main()
