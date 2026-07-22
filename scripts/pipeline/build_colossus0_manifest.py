#!/usr/bin/env python3
"""Select an exact, deterministic 60/40 Colossus-0 pair-XML manifest."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--aug9", type=int, default=600_000)
    parser.add_argument("--feb", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = [line.strip() for line in open(args.clean_manifest) if line.strip()]
    if len(paths) != len(set(paths)):
        raise RuntimeError("clean manifest contains duplicate XML paths")
    aug9 = [path for path in paths if "/aug9/" in path]
    feb = [path for path in paths if "/feb/" in path]
    if len(aug9) < args.aug9 or len(feb) < args.feb:
        raise RuntimeError(
            f"insufficient clean XMLs: aug9={len(aug9)}/{args.aug9}, feb={len(feb)}/{args.feb}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(aug9)
    rng.shuffle(feb)
    selected = aug9[: args.aug9] + feb[: args.feb]
    rng.shuffle(selected)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x") as handle:
        handle.writelines(f"{path}\n" for path in selected)
    report = {
        "seed": args.seed,
        "clean_aug9_available": len(aug9),
        "clean_feb_available": len(feb),
        "selected_aug9": args.aug9,
        "selected_feb": args.feb,
        "selected_total": len(selected),
    }
    with open(args.report, "x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
