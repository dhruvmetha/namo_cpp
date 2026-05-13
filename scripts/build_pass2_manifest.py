"""Parse Pass-1 region_opening results and emit a Pass-2 manifest of XMLs that
need chain-depth=2 retry.

An XML is sent to Pass 2 if NONE of its Pass-1 episodes found a solution.
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1-dir", required=True,
                    help="Directory containing Pass-1 *_results.pkl files")
    ap.add_argument("--output", required=True,
                    help="Output manifest path (one XML per line)")
    args = ap.parse_args()

    xml_status = defaultdict(lambda: {"any_success": False, "episodes": 0})

    for pkl in Path(args.pass1_dir).rglob("*_results.pkl"):
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for ep in data.get("episode_results", []):
            xml = ep.get("xml_file")
            if not xml:
                continue
            xml_status[xml]["episodes"] += 1
            if ep.get("success"):
                xml_status[xml]["any_success"] = True

    pass2 = sorted([x for x, s in xml_status.items() if not s["any_success"]])
    with open(args.output, "w") as f:
        for xml in pass2:
            f.write(xml + "\n")

    total = len(xml_status)
    pass1_ok = sum(1 for s in xml_status.values() if s["any_success"])
    print(f"Pass-1 XMLs processed: {total}")
    print(f"  succeeded:           {pass1_ok}")
    print(f"  → Pass-2 manifest:   {len(pass2)} XMLs written to {args.output}")


if __name__ == "__main__":
    main()
