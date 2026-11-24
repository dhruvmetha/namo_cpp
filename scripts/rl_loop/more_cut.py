#!/usr/bin/env python3
"""Merge a sharded MORE collection (more_collect.py JSONL) and cut it at simulator-call budgets.

The collection ran one worker per slice of a shuffled room list (manifest.txt). A worker that
SLURM preempted or timed out is finished by a rerun from the last room it wrote (w_NNNNN_TAG.jsonl,
TAG = r1, r2, ... in rerun order); for a room in both, the rerun's records win, because a killed
worker may have written only some of that room's episodes.

Every worker slice must be complete: its own log, or its latest rerun's log, ends with a CENSUS
line. Rooms that yield no episode (already open, no edge objects, env errors) leave no record, so
the CENSUS lines are the only proof that a slice was walked to the end. The script refuses an
incomplete collection unless --allow-incomplete.

Episodes are ordered by (manifest index, object id), which is the order a single worker walking the
whole list would have produced them. Cuts fall at room boundaries: a room is kept when the calls
before it are below the budget, so a cut overshoots its budget by at most one room. The ordered
stream is written as consecutive part files, one per budget interval, so cut k is parts 0..k and
the learning-curve points share files with the full cut. cuts.json records each cut's totals.

  python more_cut.py --raw RAW --manifest manifest.txt --per 54 --workers 1600 \
      --budgets 1.5e6 3e6 6e6 12e6 24.6e6 --out OUT
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^w_(\d{5})(?:_r(\d+))?\.jsonl$")


def census_done(log):
    if not log.exists():
        return None
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if line.startswith("CENSUS "):
            return json.loads(line[len("CENSUS "):])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--per", type=int, required=True, help="rooms per worker (collect.sbatch PER)")
    ap.add_argument("--workers", type=int, required=True, help="NSHARDS * cores")
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="cumulative call budgets, ascending")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    budgets = [int(b) for b in args.budgets]
    assert budgets == sorted(budgets)

    rooms = [ln.strip() for ln in open(args.manifest) if ln.strip()]
    index = {x: i for i, x in enumerate(rooms)}

    files = defaultdict(dict)                       # worker -> {rerun tag (0 = original): path}
    for p in raw.iterdir():
        m = NAME.match(p.name)
        if m:
            files[int(m.group(1))][int(m.group(2) or 0)] = p

    # Completeness and census, per worker slice.
    census_tot = defaultdict(float)
    missing = []
    for g in range(args.workers):
        s, e = g * args.per, min((g + 1) * args.per, len(rooms))
        if s >= e:
            continue
        tags = files.get(g, {})
        done = None
        for t in sorted(tags, reverse=True):        # latest rerun first
            done = census_done(tags[t].with_suffix(".log"))
            if done is not None:
                break
        if done is None:
            missing.append(g)
            continue
        for k, v in done.items():
            if isinstance(v, (int, float)):
                census_tot[k] += v
    if missing and not args.allow_incomplete:
        raise SystemExit(f"{len(missing)} worker slices have no CENSUS line: {missing[:40]}")

    # Pass 1: one small entry per record; reruns replace earlier records of the same room.
    entries = {}                                    # (room index, object id) -> entry
    owner = {}                                      # room index -> (worker, tag) whose records stand
    bad_lines = 0
    for g in sorted(files):
        for t in sorted(files[g]):                  # original, then r1, r2, ...
            p = files[g][t]
            seen_rooms = set()
            with open(p, "rb") as f:
                off = 0
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:    # a line cut off by the kill
                        bad_lines += 1
                        off += len(line)
                        continue
                    i = index[r["xml"]]
                    if i not in seen_rooms:
                        seen_rooms.add(i)
                        if i in owner and owner[i] != (g, t):
                            for k in [k for k in entries if k[0] == i]:
                                del entries[k]
                        owner[i] = (g, t)
                    entries[(i, r["object_id"])] = {"path": str(p), "off": off, "len": len(line),
                                                    "calls": int(r["calls"]), "solved": bool(r["solved"]),
                                                    "rows": len(r["rows"])}
                    off += len(line)
    order = sorted(entries)

    # Cut points at room boundaries.
    out.mkdir(parents=True, exist_ok=True)
    cuts, part_ends = [], []
    cum = 0
    bi = 0
    room_of = [k[0] for k in order]
    for n, k in enumerate(order):
        while bi < len(budgets) and cum >= budgets[bi] and (n == 0 or room_of[n] != room_of[n - 1]):
            part_ends.append(n)
            bi += 1
        cum += entries[k]["calls"]
    part_ends += [len(order)] * (len(budgets) - len(part_ends))   # budget beyond the collection
    part_ends.append(len(order))                                  # the tail past the last budget

    handles = {}
    start = 0
    tot = {"episodes": 0, "calls": 0, "solved": 0, "rows": 0}
    rooms_seen = set()
    for pi, end in enumerate(part_ends):
        name = f"part_{pi:02d}.jsonl"
        with open(out / name, "wb") as w:
            for k in order[start:end]:
                ent = entries[k]
                h = handles.get(ent["path"]) or handles.setdefault(ent["path"], open(ent["path"], "rb"))
                h.seek(ent["off"])
                w.write(h.read(ent["len"]))
                tot["episodes"] += 1
                tot["calls"] += ent["calls"]
                tot["solved"] += int(ent["solved"])
                tot["rows"] += ent["rows"]
                rooms_seen.add(k[0])
        label = f"budget_{budgets[pi]}" if pi < len(budgets) else "all"
        cuts.append({"cut": label, "parts": [f"part_{j:02d}.jsonl" for j in range(pi + 1)],
                     "last_manifest_index": order[end - 1][0] if end else None,
                     "rooms_with_episodes": len(rooms_seen), **tot})
        start = end
    for h in handles.values():
        h.close()

    summary = {"raw": str(raw), "manifest": args.manifest, "workers": args.workers, "per": args.per,
               "incomplete_workers": missing, "undecodable_lines": bad_lines,
               "census_sum_final_logs": dict(census_tot),
               "rerun_files": sorted(str(p) for g in files for t, p in files[g].items() if t),
               "cuts": cuts}
    (out / "cuts.json").write_text(json.dumps(summary, indent=1))
    for c in cuts:
        print(f"{c['cut']:>18}  episodes {c['episodes']:>7}  calls {c['calls']:>10}  solved {c['solved']:>7}"
              f"  rows {c['rows']:>9}  rooms {c['rooms_with_episodes']:>6}  last idx {c['last_manifest_index']}")
    print(f"incomplete workers {len(missing)}, undecodable lines {bad_lines}")


if __name__ == "__main__":
    main()
