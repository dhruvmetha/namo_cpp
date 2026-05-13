"""Strip a populated env XML into a template (walls only, no obstacles, no goal site).

Walks aug9/{easy,medium}/set{1,2}/benchmark_{1..5}/, takes the first env_config_*.xml
from each, removes movable obstacles and the goal site, writes to an output tree.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def strip_env(src_xml: Path, out_xml: Path) -> int:
    """Strip movable obstacles and goal site from src_xml, write to out_xml.
    Returns number of obstacles removed."""
    tree = ET.parse(src_xml)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"no <worldbody> in {src_xml}")

    removed = 0
    for child in list(worldbody):
        name = child.get("name", "")
        # Remove movable obstacle bodies
        if child.tag == "body" and re.match(r"obstacle_\d+_movable", name):
            worldbody.remove(child)
            removed += 1
            continue
        # Remove the goal site (will be re-placed by generator)
        if child.tag == "site" and name == "goal":
            worldbody.remove(child)
            continue

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root",
                    default="/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/"
                            "resources/models/custom_walled_envs/aug9")
    ap.add_argument("--out-root", default="templates/aug9")
    ap.add_argument("--source-difficulty", default="medium",
                    help="Which difficulty subdir under src_root to read templates from "
                         "(easy and medium share wall layouts; pick one).")
    ap.add_argument("--sets", nargs="+", default=["set1", "set2"])
    ap.add_argument("--benchmarks", nargs="+", default=[f"benchmark_{i}" for i in range(1, 6)])
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    summary = []
    for s in args.sets:
        for bench in args.benchmarks:
            bench_dir = src_root / args.source_difficulty / s / bench
            if not bench_dir.is_dir():
                summary.append((str(bench_dir), "MISSING", 0))
                continue
            envs = sorted(bench_dir.glob("env_config_*.xml"))
            if not envs:
                summary.append((str(bench_dir), "NO ENVS", 0))
                continue
            src = envs[0]
            out = out_root / s / f"{bench}.xml"
            n = strip_env(src, out)
            summary.append((str(out), f"OK (from {src.name})", n))

    print(f"{'output':<70} {'status':<28} {'obstacles_removed'}")
    for o, s, n in summary:
        print(f"{o:<70} {s:<28} {n}")


if __name__ == "__main__":
    main()
