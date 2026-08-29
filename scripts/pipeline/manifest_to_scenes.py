#!/usr/bin/env python3
"""Turn a scene pool's `manifest.txt` into the scenes list `exhaustive_hmax2.py` reads.

The generator writes a manifest of XML paths and nothing else, so the object names have to come off
the XML itself. Every movable body it emits is named `obstacle_<i>_movable`, one per entry of the
scene's blocker list, so the names sort into the same order the build sheet lists them in and index
i lines up with `solo_opens[i]`.

Single-movable pools get `{"xml":..., "object_id": "obstacle_0_movable"}`, the legacy shape, so the
output is a drop-in for pools that predate two-movable scenes. Anything with more gets
`{"xml":..., "object_ids": [...]}`. `exhaustive_hmax2.py` accepts both.

  python scripts/pipeline/manifest_to_scenes.py --out scenes.json \
      $NAMO_SCRATCH/real_buildable_2mov/v1/solo{0,1,2}/manifest.txt
"""
import argparse
import json
import os
import re

BODY = re.compile(r'<body name="(obstacle_\d+_movable)"')


def objects_in(xml):
    with open(xml) as f:
        return sorted(set(BODY.findall(f.read())), key=lambda n: int(n.split("_")[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+", help="one or more manifest.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenes, counts = [], {}
    for m in args.manifests:
        for line in open(m):
            xml = line.strip()
            if not xml:
                continue
            objs = objects_in(xml)
            counts[len(objs)] = counts.get(len(objs), 0) + 1
            scenes.append({"xml": xml, "object_id": objs[0]} if len(objs) == 1
                          else {"xml": xml, "object_ids": objs})
    with open(args.out, "w") as f:
        json.dump(scenes, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(scenes)} scenes, movables per scene {dict(sorted(counts.items()))}")
    missing = [s["xml"] for s in scenes if not os.path.exists(s["xml"])]
    if missing:
        print(f"WARNING {len(missing)} manifest paths do not exist, first: {missing[0]}")


if __name__ == "__main__":
    main()
