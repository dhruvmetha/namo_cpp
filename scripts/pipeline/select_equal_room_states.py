#!/usr/bin/env python3
"""Pick SPAR's and MORE's training states on the same rooms, with equal states per room (card
EXP-2026-11-27-more-spar-equal-rooms). Rooms are the H5 `xml` strings, the same Amarel paths in both files.

  rooms      MORE collection JSONL (more_cut.py parts) + SPAR H5 -> rooms where MORE has at least one state,
             all of which must be SPAR rooms; writes rooms.json and spar_rows.npy (SPAR's rows on them).
  spar-split the filtered SPAR H5 -> the rooms SPAR's trainer holds out (Q2DataModule.setup from the trainer's
             worktree, its defaults) and SPAR's training states per room; writes spar_split.json.
  more-rows  MORE H5 sources (read as one table, as subset_h5_rows.py does) + spar_split.json -> more_rows.npy:
             in each SPAR training room, as many MORE states as SPAR has there, drawn at random (--seed); a room
             with fewer MORE states gives all of them and the shortfall is drawn from the spare MORE states of
             the other training rooms, so the totals match. Held-out rooms give nothing.

Usage:
  python select_equal_room_states.py rooms --parts cut/part_0*.jsonl --spar-h5 hybrid_train_v1.h5 --out DIR
  python select_equal_room_states.py spar-split --spar-h5 DIR/spar.h5 --trainer-repo <worktree> --out DIR
  python select_equal_room_states.py more-rows --more-h5 a.h5 b.h5 --split DIR/spar_split.json --out DIR
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def rooms(args):
    states = Counter()
    for p in args.parts:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                states[r["xml"]] += len(r["states"])
    more_rooms = {x for x, n in states.items() if n > 0}
    with h5py.File(args.spar_h5, "r") as f:
        spar_xml = decode(f["xml"][:])
    spar_rooms = set(spar_xml)
    missing = more_rooms - spar_rooms
    if missing:
        raise SystemExit(f"{len(missing)} MORE rooms are not SPAR rooms, e.g. {sorted(missing)[:3]}")
    rows = np.flatnonzero(np.isin(spar_xml, sorted(more_rooms))).astype(np.int64)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "spar_rows.npy", rows)
    (out / "rooms.json").write_text(json.dumps(dict(
        rooms=sorted(more_rooms), more_states_from_jsonl=int(sum(states[x] for x in more_rooms)),
        more_episodes_rooms=len(states), spar_rows=int(len(rows)), spar_rows_total=int(len(spar_xml)),
        parts=[str(Path(p).resolve()) for p in args.parts], spar_h5=str(Path(args.spar_h5).resolve()))))
    print(f"rooms {len(more_rooms)} (MORE states {sum(states[x] for x in more_rooms)}), "
          f"SPAR rows on them {len(rows)} of {len(spar_xml)}")


def spar_split(args):
    sys.path[:0] = [str(Path(args.trainer_repo) / "python"), str(Path(args.trainer_repo) / "scripts")]
    from namo.rl_loop.sage_ext.q2_dataset import Q2DataModule
    import namo.rl_loop.sage_ext.q2_dataset as q2
    if not Path(q2.__file__).resolve().is_relative_to(Path(args.trainer_repo).resolve()):
        raise SystemExit(f"imported {q2.__file__}, not the trainer's worktree")
    dm = Q2DataModule(args.spar_h5)
    dm.setup()
    with h5py.File(args.spar_h5, "r") as f:
        xml = decode(f["xml"][:])
    train = Counter(xml[dm.train_idx]); val = set(xml[dm.val_idx])
    if set(train) & val:
        raise SystemExit("a room is on both sides of SPAR's split")
    out = Path(args.out)
    (out / "spar_split.json").write_text(json.dumps(dict(
        train_rooms={k: int(v) for k, v in sorted(train.items())}, heldout_rooms=sorted(val),
        train_states=int(len(dm.train_idx)), heldout_states=int(len(dm.val_idx)),
        trainer_module=str(Path(q2.__file__).resolve()), spar_h5=str(Path(args.spar_h5).resolve()))))
    print(f"SPAR train {len(dm.train_idx)} states in {len(train)} rooms, held out {len(dm.val_idx)} in {len(val)}")


def more_rows(args):
    split = json.loads(Path(args.split).read_text())
    target = split["train_rooms"]
    xml = []
    for p in args.more_h5:
        with h5py.File(p, "r") as f:
            xml.append(decode(f["xml"][:]))
    xml = np.concatenate(xml)
    by_room = {}
    for i in np.flatnonzero(np.isin(xml, list(target))):
        by_room.setdefault(xml[i], []).append(i)
    absent = set(target) - set(by_room)
    if absent:
        raise SystemExit(f"{len(absent)} SPAR training rooms have no MORE state, e.g. {sorted(absent)[:3]}")
    rng = np.random.default_rng(args.seed)
    keep, spare, short = [], [], 0
    for room in sorted(target):
        idx = rng.permutation(np.asarray(by_room[room], dtype=np.int64))
        n = target[room]
        keep.append(idx[:n]); spare.append(idx[n:])
        short += max(0, n - len(idx))
    spare = np.concatenate(spare)
    keep.append(rng.choice(spare, short, replace=False) if short else np.empty(0, np.int64))
    rows = np.sort(np.concatenate(keep))
    assert len(rows) == split["train_states"] and len(np.unique(rows)) == len(rows)
    out = Path(args.out)
    np.save(out / "more_rows.npy", rows)
    got = Counter(xml[rows])
    (out / "more_rows.json").write_text(json.dumps(dict(
        states=int(len(rows)), rooms=len(got), shortfall_redrawn=int(short),
        rooms_short=int(sum(1 for r in target if len(by_room[r]) < target[r])),
        more_states_in_training_rooms=int(sum(len(v) for v in by_room.values())),
        more_table_rows=int(len(xml)), seed=args.seed, sources=[str(Path(p).resolve()) for p in args.more_h5])))
    print(f"MORE {len(rows)} states in {len(got)} rooms, shortfall {short} redrawn")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rooms"); a.add_argument("--parts", nargs="+", required=True)
    a.add_argument("--spar-h5", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("spar-split"); b.add_argument("--spar-h5", required=True)
    b.add_argument("--trainer-repo", required=True); b.add_argument("--out", required=True)
    c = sub.add_parser("more-rows"); c.add_argument("--more-h5", nargs="+", required=True)
    c.add_argument("--split", required=True); c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"rooms": rooms, "spar-split": spar_split, "more-rows": more_rows}[args.cmd](args)


if __name__ == "__main__":
    main()
