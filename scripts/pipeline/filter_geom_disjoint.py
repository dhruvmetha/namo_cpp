#!/usr/bin/env python3
"""Filter generated XMLs against the canonical test set by full room geometry."""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for path in (REPO / "build_python", REPO / "python", REPO / "scripts", REPO / "scripts/pipeline"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verify_geom_disjoint import geom_sig, load_xmls, sig_map  # noqa: E402


def _load_generated(spec: str):
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "**", "*.xml"), recursive=True))
    return load_xmls(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-xmls", required=True)
    parser.add_argument("--test-xmls", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    generated = _load_generated(args.gen_xmls)
    test_xmls = load_xmls(args.test_xmls)
    test_n, test_full_to_xml, _ = sig_map(test_xmls, workers=args.workers)
    test_signatures = set(test_full_to_xml)

    unique_generated = list(dict.fromkeys(generated))
    signatures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for xml_path, (full, _) in zip(
            unique_generated,
            executor.map(geom_sig, unique_generated, chunksize=64),
        ):
            signatures[xml_path] = full

    kept = []
    dropped = []
    unparseable = []
    for xml_path in generated:
        signature = signatures.get(xml_path)
        if signature is None:
            unparseable.append(xml_path)
        elif signature in test_signatures:
            dropped.append(xml_path)
        else:
            kept.append(xml_path)

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "x") as handle:
        handle.writelines(f"{xml_path}\n" for xml_path in kept)

    report = {
        "n_gen_xmls": len(generated),
        "n_gen_unique_paths": len(unique_generated),
        "n_test_xmls": len(test_xmls),
        "n_test_parseable": test_n,
        "n_test_unique_scenes": len(test_signatures),
        "n_kept_xmls": len(kept),
        "n_dropped_xmls": len(dropped),
        "n_unparseable": len(unparseable),
        "clean_input": not dropped,
        "dropped_examples": dropped[:10],
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "dropped_examples"}, indent=2))


if __name__ == "__main__":
    main()
