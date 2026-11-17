#!/usr/bin/env python3
"""Derive the chained-2-hop eval population: scenes whose keyhole-2 sweep completed.

A row qualifies when its keyhole-1 opener succeeded (`"success" in kh1_failure_reasons`) and a
keyhole-2 key was actually written (`kh1_key` non-empty). The round-2 rows point at the
post-push round-1 XMLs; this maps each back to the ORIGINAL pool XML, because the eval must
solve the full 2-hop problem from the initial state, not resume mid-chain.

    set1__benchmark_4__run_0092__env_0092_pair_001.xml
      -> generated/set1/benchmark_4/run_0092/env_0092_pair_001.xml

The population grows as the chain runs, so this is re-runnable and the manifest is an input to
the eval rather than something baked into it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ORIG_ROOT = "/common/users/dm1487/scratch_namo/multihop_aug9_hy5u/scale_20260817_0000/generated"


def original_xml(round1_path: str, root: str) -> str | None:
    stem = Path(round1_path).name
    parts = stem.split("__")
    if len(parts) != 4:
        return None
    setname, bench, run, env = parts
    return os.path.join(root, setname, bench, run, env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--orig-root", default=ORIG_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-output", default=None,
                        help="jsonl recording original<-round1 pairs plus the row's kh1 fields")
    args = parser.parse_args()

    stats = Counter()
    seen: dict[str, dict] = {}
    for path in args.rows:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows"] += 1
            reasons = row.get("kh1_failure_reasons") or []
            if "success" not in reasons:
                stats["no_success"] += 1
                continue
            if not row.get("kh1_key"):
                stats["empty_kh1_key"] += 1
                continue
            orig = original_xml(row["xml_path"], args.orig_root)
            if orig is None:
                stats["unparsable_basename"] += 1
                continue
            if not os.path.exists(orig):
                stats["original_missing"] += 1
                continue
            stats["qualified"] += 1
            canon = os.path.realpath(orig)
            if canon in seen:
                stats["duplicate_original"] += 1
                continue
            seen[canon] = {
                "xml_path": canon,
                "round1_xml": row["xml_path"],
                "kh1_pushes": row.get("kh1_pushes"),
                "kh1_openers_1push": row.get("kh1_openers_1push"),
                "kh1_candidate_openers": row.get("kh1_candidate_openers"),
                "status": row.get("status"),
            }

    with open(args.output, "w", encoding="utf-8") as fh:
        for canon in sorted(seen):
            fh.write(canon + "\n")
    if args.map_output:
        with open(args.map_output, "w", encoding="utf-8") as fh:
            for canon in sorted(seen):
                fh.write(json.dumps(seen[canon], sort_keys=True) + "\n")

    print("stats:", dict(stats))
    print("unique original scenes written: %d -> %s" % (len(seen), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
