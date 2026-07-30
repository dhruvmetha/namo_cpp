#!/usr/bin/env python3
"""Build a region-opening manifest that runs only specified episode objects."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from namo.paths import resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--alignment-key", default="manifest_missing_gt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    alignment = json.load(open(args.alignment))
    episodes = alignment[args.alignment_key]
    seen = set()
    lines = []
    for episode in episodes:
        xml = episode["xml"]
        object_id = episode["object_id"]
        region = episode["region"]
        key = (str(resolve(xml)), object_id, region)
        if key in seen:
            raise RuntimeError(f"duplicate target episode: {key}")
        seen.add(key)

        root = ET.parse(resolve(xml)).getroot()
        movable = sorted({
            body.get("name")
            for body in root.findall(".//body")
            if (body.get("name") or "").endswith("_movable")
        })
        if object_id not in movable:
            raise RuntimeError(f"target {object_id} absent from {xml}")
        skips = ",".join(f"{region}:{name}" for name in movable if name != object_id)
        lines.append(f"{xml}\t{skips}" if skips else xml)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"episodes": len(episodes), "unique_keys": len(seen), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
