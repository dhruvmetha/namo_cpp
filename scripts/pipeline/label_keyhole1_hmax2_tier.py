#!/usr/bin/env python3
"""Re-tier keyhole-1 episodes on the exact hmax=2 axis.

`kh1_scenes.jsonl` tiers each scene by its keyhole-1 **1-push** solve rate. The search this
project actually deploys runs at `hmax=2`, so the matching difficulty axis is the fraction of
tried pushes that open the keyhole **within two pushes**:

    solve_rate_hmax2 = |valid_1push  U  valid_first_push| / |tried_1push|

`valid_first_push` is the set of setup pushes that lead to an opening on the second push, so the
union is exactly "this push starts a chain that opens keyhole 1 within the budget". Both sets and
the denominator come from `kh1_2push_key.json`.

Output is a drop-in replacement for `kh1_scenes.jsonl`: same field names, so
`report_multihop_showcase.py --candidates` renders the same tables on the new axis.

Joins are on `os.path.realpath`, and the keyhole-1 episode inside a scene is matched by
(object_id, region) — a scene has many episodes and the wrong one carries the wrong difficulty
(docs/pipeline/multi_episode_rooms.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_common import bin_of  # noqa: E402  the one canonical hard/med/easy rulebook


def _canon(path: str) -> str:
    return os.path.realpath(path.replace("/scache/scratch/", "/scratch/", 1))


def _pick_episode(episodes, objects, region):
    """The keyhole-1 episode: same pushed object, same target region."""
    wanted = set(objects or ())
    exact = [e for e in episodes if e.get("object_id") in wanted and e.get("region") == region]
    if exact:
        return exact[0], "object+region"
    by_object = [e for e in episodes if e.get("object_id") in wanted]
    if by_object:
        return by_object[0], "object_only"
    return None, "no_match"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, required=True, help="kh1_scenes.jsonl")
    parser.add_argument("--key", type=Path, required=True, help="kh1_2push_key.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key = {_canon(k): v for k, v in json.loads(args.key.read_text(encoding="utf-8")).items()}
    how = Counter()
    tiers = Counter()
    moved = Counter()
    written = 0

    with args.output.open("w", encoding="utf-8") as out:
        for line in args.scenes.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            canon = _canon(row["xml_path"])
            episodes = key.get(canon)
            if not episodes:
                how["no_key_entry"] += 1
                continue
            episode, mode = _pick_episode(
                episodes, row.get("keyhole_objects"), row.get("target_region")
            )
            how[mode] += 1
            if episode is None:
                continue
            tried = len({tuple(p) for p in episode.get("tried_1push", ())})
            if not tried:
                how["no_tried"] += 1
                continue
            opens = {tuple(p) for p in episode.get("valid_1push", ())} | {
                tuple(p) for p in episode.get("valid_first_push", ())
            }
            rate = len(opens) / tried
            tier = bin_of(rate)
            tiers[tier] += 1
            if row.get("tier"):
                moved["%s->%s" % (row["tier"], tier)] += 1
            out.write(
                json.dumps(
                    {
                        "xml_path": row["xml_path"],
                        "tier": tier,
                        "tier_1push": row.get("tier"),
                        "solve_rate_hmax2": rate,
                        "solve_rate_1push": row.get("solve_rate_1push_best"),
                        "n_open_within_2push": len(opens),
                        "n_tried_1push": tried,
                        "any_1push_solvable": bool(episode.get("is_1push_solvable")),
                        "any_2push_solvable": bool(episode.get("is_2push_solvable")),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1

    print("written %d" % written)
    print("episode match:", dict(how))
    print("hmax2 tiers:", dict(sorted(tiers.items())))
    print("1push -> hmax2 movement:", dict(sorted(moved.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
