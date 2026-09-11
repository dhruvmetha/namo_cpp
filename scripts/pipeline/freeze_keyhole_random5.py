#!/usr/bin/env python3
"""Reproduce the historical set2/benchmark_5 five-seed Full-NAMO release.

The median-call thresholds (10/50), finite-tier sampling, and retained unresolved
scenes belong to this recorded protocol; they do not set general testbed policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from assemble_keyhole_random5 import (
    SEEDS, TECHNICAL_FAILURES, certified_descriptor, read_rows, write_json, write_rows,
)

# User-approved whole-problem median-call boundaries; unrelated to donor tiers.
EASY_MAX = 10
MEDIUM_MAX = 50
FINITE_TIERS = ("easy", "medium", "hard")
FULL_PROBLEM_BUDGET = 20000


def difficulty_for_runs(runs):
    assert set(runs) == set(SEEDS)
    assert all(isinstance(r["total_calls"], int) and
               0 <= r["total_calls"] <= FULL_PROBLEM_BUDGET for r in runs.values())
    successes = sorted(r["total_calls"] for r in runs.values() if r["solved"])
    if any(r["failure_kind"] in TECHNICAL_FAILURES for r in runs.values()):
        return "technical_error", None, len(successes)
    if len(successes) < 3:
        return "unresolved", None, len(successes)
    median = successes[2]  # Failures rank above all successful counts.
    return ("easy" if median <= EASY_MAX else "medium" if median <= MEDIUM_MAX else "hard",
            median, len(successes))


def choose(pool, seed, per_tier):
    selected = []
    for tier in FINITE_TIERS:
        candidates = sorted((r for r in pool if r["difficulty"] == tier),
                            key=lambda r: r["geometry_id"])
        selected.extend(random.Random(f"{seed}:{tier}").sample(candidates, per_tier))
    selected.extend(r for r in pool if r["difficulty"] == "unresolved")
    return sorted(selected, key=lambda r: (r["difficulty"], r["geometry_id"]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--cert-results", type=Path, required=True)
    parser.add_argument("--recert-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-tier", type=int, default=200)
    args = parser.parse_args()
    old, out = args.previous_root, args.output_root
    out.mkdir(parents=True, exist_ok=False)
    provenance = out / "provenance"
    provenance.mkdir()
    sources_file, results_file = old / "input/manifest.jsonl", old / "aggregate/per_scene.jsonl"
    sources = read_rows(sources_file)
    sources_by_xml = {r["xml_path"]: r for r in sources}
    assert len(sources_by_xml) == len(sources)
    certificates = [(r, args.cert_results) for r in read_rows(args.cert_results)]
    certificates.extend((json.loads(p.read_text()), p)
                        for p in sorted(args.recert_results.glob("task_*.json")))
    certified, certificate_records = {}, {}
    for result, path in certificates:
        descriptor = certified_descriptor(result, path)
        if descriptor:
            certified[result["geometry_id"]] = descriptor
            certificate_records[result["geometry_id"]] = result

    pool = []
    for baseline in read_rows(results_file):
        original = sources_by_xml[baseline["xml_path"]]
        if original["cohort"] != "mixed_push_horizons":
            descriptor = dict(original, label_evidence="completed_onepush_chain")
        else:
            descriptor = certified.get(original["geometry_id"])
            if descriptor is None:
                continue
            assert Path(descriptor["xml_path"]).read_bytes() == Path(original["xml_path"]).read_bytes()
        tier, median, successes = difficulty_for_runs(baseline["runs"])
        pool.append(dict(descriptor, source_xml=original["xml_path"],
                         source_manifest=original["source_manifest"],
                         difficulty=tier, median_total_calls_until_success=median,
                         median_is_censored=tier == "unresolved",
                         successes_out_of_5=successes, runs=baseline["runs"]))
    assert len({r["geometry_id"] for r in pool}) == len(pool)
    pool.sort(key=lambda r: r["geometry_id"])
    write_rows(provenance / "selection_pool.jsonl", pool)
    write_rows(provenance / "technical_errors_excluded.jsonl",
               [r for r in pool if r["difficulty"] == "technical_error"])
    selected = choose(pool, args.seed, args.per_tier)
    expected_counts = {tier: args.per_tier for tier in FINITE_TIERS}
    expected_counts["unresolved"] = sum(r["difficulty"] == "unresolved" for r in pool)
    assert Counter(r["difficulty"] for r in selected) == expected_counts

    active, by_source_xml, xml_hashes = [], {}, set()
    for chosen in selected:
        row = dict(chosen)
        folder = Path(row["difficulty"])
        if row["difficulty"] == "unresolved":
            folder /= f'{row["successes_out_of_5"]}_of_5'
        relative = folder / f'{row["geometry_id"]}.xml'
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source_xml"], target)
        assert digest(target) == digest(row["source_xml"])
        xml = ET.parse(target).getroot()
        assert not xml.findall(".//include") and not any(e.get("file") for e in xml.iter())
        assert xml.find(".//site[@name='goal']") is not None
        row.update(xml_relative_path=relative.as_posix(),
                   xml_path=str(args.install_root / relative), xml_sha256=digest(target))
        xml_hashes.add(row["xml_sha256"])
        active.append(row)
        by_source_xml[row["source_xml"]] = row
    assert len(xml_hashes) == len(active)

    # All references used by evaluators point to the destination. Source paths stay archival.
    write_rows(out / "manifest.jsonl", active)
    write_json(out / "population.json", {"name": out.name, "scenes": [
        {"xml_path": r["xml_path"], "cluster_id": "geometry:" + r["geometry_id"],
         "difficulty": r["difficulty"], "horizon_pattern": r["horizon_pattern"]} for r in active]})
    groups = defaultdict(list)
    for row in active:
        groups["."].append(row)
        groups[row["difficulty"]].append(row)
        if row["difficulty"] == "unresolved":
            groups[f'unresolved/{row["successes_out_of_5"]}_of_5'].append(row)
    for successes in range(3):
        groups.setdefault(f"unresolved/{successes}_of_5", [])
    for folder, rows in groups.items():
        directory = out / folder
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "xmls.txt").open("x") as handle:
            handle.writelines(r["xml_path"] + "\n" for r in rows)
    with (out / "benchmark_600.txt").open("x") as handle:
        handle.writelines(r["xml_path"] + "\n" for r in active if r["difficulty"] in FINITE_TIERS)

    # Preserve original generation/replay evidence without reopening the simulator.
    selected_by_manifest = defaultdict(set)
    for r in active:
        selected_by_manifest[r["source_manifest"]].add(r["source_xml"])
    generation = []
    for path, wanted in selected_by_manifest.items():
        found = [r for r in read_rows(path) if r["xml_path"] in wanted]
        assert {r["xml_path"] for r in found} == wanted
        generation.extend(found)
    write_rows(provenance / "generation_records.jsonl", generation)
    write_rows(provenance / "selected_certificates.jsonl",
               [certificate_records[r["geometry_id"]] for r in active
                if r["geometry_id"] in certificate_records])

    # Saved successful chains are small. Failure kinds/calls are already in the manifest;
    # the 70 GiB of detailed failed-search iteration traces remain at their source.
    saved_solutions = Counter()
    random_dir = out / "random_solutions"
    random_dir.mkdir()
    for seed in SEEDS:
        records = []
        for path in sorted((old / "raw" / f"random_s{seed}").glob("shard_*/solved.jsonl")):
            for record in read_rows(path):
                chosen = by_source_xml.get(record["xml_path"])
                if chosen is None:
                    continue
                assert chosen["runs"][seed]["solved"]
                assert record["simulation_budget_used"] == chosen["runs"][seed]["total_calls"]
                records.append(dict(record, source_xml=record["xml_path"],
                                    xml_path=chosen["xml_path"], geometry_id=chosen["geometry_id"],
                                    seed=int(seed), source_result_file=str(path)))
        assert len(records) == sum(r["runs"][seed]["solved"] for r in active)
        assert len({r["geometry_id"] for r in records}) == len(records)
        write_rows(random_dir / f"seed_{seed}.jsonl", records)
        saved_solutions[seed] = len(records)
        shutil.copyfile(old / "raw" / f"random_s{seed}" / "shard_0000/run_config.json",
                        provenance / f"random_config_seed_{seed}.json")

    for src in [old / "campaign_record.json", old / "runfiles/runtime_sha256.txt",
                Path(__file__), Path(__file__).with_name("assemble_keyhole_random5.py")]:
        shutil.copyfile(src, provenance / src.name)
    source_paths = [sources_file, results_file, args.cert_results]
    source_paths.extend(sorted(args.recert_results.glob("task_*.json")))
    write_json(provenance / "source_sha256.json",
               {str(p): digest(p) for p in source_paths})
    summary = {
        "name": out.name, "status": "frozen_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": args.seed, "python_version": sys.version,
        "selection_algorithm": "For each tier: random.Random(str(seed)+':'+tier).sample(sorted_by_geometry_id, 200). All unresolved retained.",
        "selection_code_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
        "install_root": str(args.install_root), "baseline_source": str(old),
        "pool_counts": dict(Counter(r["difficulty"] for r in pool)),
        "selected_counts": dict(Counter(r["difficulty"] for r in active)),
        "unresolved_by_successes": dict(Counter(str(r["successes_out_of_5"]) for r in active if r["difficulty"] == "unresolved")),
        "selected_by_difficulty_and_pattern": {
            tier: dict(Counter(r["horizon_pattern"] for r in active if r["difficulty"] == tier))
            for tier in (*FINITE_TIERS, "unresolved")},
        "seeds": list(map(int, SEEDS)), "scene_count": len(active),
        "random_seed_runs": len(active) * len(SEEDS), "saved_random_solutions": dict(saved_solutions),
        "thresholds_total_simulator_calls": {"easy_max": EASY_MAX, "medium_max": MEDIUM_MAX},
        "simulation_budget_total": FULL_PROBLEM_BUDGET,
        "median_rule": "Third sorted successful call count if at least 3/5 succeed; otherwise null/unresolved. All five runs count. Technical-error scenes excluded.",
        "success_condition": "Full-scene end-goal reachability",
        "wall_template": "set2/benchmark_5", "horizon_quotas": False,
        "unresolved_is_not_proof_of_unsolvability": True,
        "completed_pool_only": True,
    }
    write_json(out / "summary.json", summary)
    readme = f"""# Frozen two-keyhole Full-NAMO benchmark

200 easy (median <=10 total calls), 200 medium (11-50), 200 hard (>50), plus all {expected_counts['unresolved']} unresolved environments, grouped by 0/5, 1/5, or 2/5 random successes.

Selection seed: {args.seed}. Uniform random sampling without replacement within each finite difficulty tier; no push-horizon or donor-tier quotas. Candidates sorted by full geometry ID before sampling. The exact candidate pool, source hashes, selection code and commit are recorded in provenance/ and summary.json. Later cluster results do not change this release.

Difficulty uses uniform-random Full-NAMO ordering seeds 7000, 8000, 9000, 10000, 11000; sampler seed 42; total budget 20,000 per run. The median is the third sorted successful count when at least three seeds succeed, with failures ordered above successes. Otherwise it is unresolved, not an invented finite cost. Technical-error scenes are excluded and listed in provenance/.

Each XML is a complete robot/start/objects/end-goal problem, not an isolated local region-opening episode. All scenes use set2/benchmark_5 walls. 11/12/21/22 labels are independent of difficulty; label_evidence distinguishes original completed one-push chains from exhaustive independent gate certification.

Use benchmark_600.txt for the balanced 600, xmls.txt or population.json for the full release including unresolved, or each category's xmls.txt. Absolute evaluation paths refer to this dhruv-linux installation; xml_relative_path supports relocation. Write evaluation outputs outside this read-only release.

manifest.jsonl includes all five outcome/call/failure records for every selected scene. random_solutions/ includes every saved successful random chain. Original generator/replay records and selected gate certificates are included in provenance/. Detailed failed-search iteration traces remain on Amarel; their outcome summaries are preserved here.

Full-NAMO success means end-goal reachability. The baseline uses evaluation-default teleport navigation to push approaches. No model or planner was changed, no new search was run, and no requirement to reduce graph hops after every opening was introduced.

Verify copied files with: sha256sum -c SHA256SUMS
"""
    with (out / "README.md").open("x") as handle:
        handle.write(readme)
    with (out / "SHA256SUMS").open("x") as handle:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{digest(path)}  {path.relative_to(out)}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
