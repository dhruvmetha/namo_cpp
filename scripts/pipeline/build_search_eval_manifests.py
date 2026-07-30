#!/usr/bin/env python3
"""Build search-eligible eval manifests by applying the registered episode exclusions."""
import argparse
import json
import os
from pathlib import Path

import yaml

from namo.paths import resolve


def _matches(xml, row, exclusion):
    return (
        os.path.realpath(resolve(xml)).endswith(exclusion["xml_suffix"])
        and row["object_id"] == exclusion["object_id"]
        and row.get("region") == exclusion.get("region")
    )


def _filter(source, exclusions):
    raw = json.load(open(source))
    out = {}
    removed = []
    for xml, rows in raw.items():
        keep = []
        for row in rows:
            match = next((item for item in exclusions if _matches(xml, row, item)), None)
            if match is None:
                keep.append(row)
            else:
                removed.append((xml, row["object_id"], row.get("region")))
        if keep:
            out[xml] = keep
    return out, removed


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(data, stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval_sets.yaml")
    parser.add_argument("--onepush-source", required=True)
    parser.add_argument("--twopush-source", required=True)
    parser.add_argument("--twopush-gt-divisions-source", required=True)
    parser.add_argument("--twopush-sampled-divisions-source", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    exclusions = cfg["search_eval_exclusions"]
    onepush_exclusions = [item for item in exclusions if item["horizon"] == "1push"]
    twopush_exclusions = [item for item in exclusions if item["horizon"] == "2push"]
    out_dir = Path(args.out_dir)
    builds = (
        (args.onepush_source, onepush_exclusions, out_dir / "onepush_search_eval.json"),
        (args.twopush_source, twopush_exclusions, out_dir / "pure2push_search_eval.json"),
        (args.twopush_gt_divisions_source, twopush_exclusions, out_dir / "pure2push_gt_divisions_search_eval.json"),
        (args.twopush_sampled_divisions_source, twopush_exclusions, out_dir / "pure2push_divisions_search_eval.json"),
    )
    for source, selected, destination in builds:
        data, removed = _filter(source, selected)
        if len(removed) != len(selected):
            raise RuntimeError(f"{source}: removed {len(removed)} episodes, expected {len(selected)}")
        _write(destination, data)
        print(f"{destination}: {sum(map(len, data.values()))} episodes; removed {len(removed)}")


if __name__ == "__main__":
    main()
