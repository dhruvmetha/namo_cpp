#!/usr/bin/env python3
"""Reproduce the historical September 2026 labeled-random5 campaign assembly.

The prepare/publish entrypoints retain that campaign's 535 certified scenes,
458 missing baselines, 177 pilot recertifications, and 31,779 one-push-chain
scenes. These counts do not define the current diverse-template testbed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

PATTERNS = ("11", "12", "21", "22")
SEEDS = ("7000", "8000", "9000", "10000", "11000")
TECHNICAL_FAILURES = {"planner_invariant_violation", "runner_exception"}


def read_rows(path):
    """Read JSONL without depending on the simulator."""
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    """Write a new JSONL artifact without overwriting an earlier result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def certified_descriptor(result, evidence_path):
    """Select definite gate labels only when the complete-scene witness succeeds."""
    if result["status"] != "ok":
        raise ValueError(f"certification task error: {result['task_id']}")
    gates = result["gates"]
    if len(gates) != 2:
        raise ValueError(f"missing gate in {result['task_id']}")
    for gate in gates:
        assert gate["tried_onepush_count"] == gate["reachable_edge_count"] * 5
        assert len(gate["onepush_trials"]) == gate["tried_onepush_count"]
        assert gate["initial_closed"] and gate["target_point_count"] == 100
        if gate["label"] == "2":
            assert gate["valid_onepush_count"] == 0
            assert gate["witness"]["done"] == [True, True]
            assert gate["witness"]["opened"] == [False, True]
        elif gate["label"] == "1":
            assert gate["valid_onepush_count"] > 0
    pattern = "".join(gate["label"] for gate in gates)
    if pattern not in PATTERNS or not result["full_witness"]["success"]:
        return None
    return {
        "geometry_id": result["geometry_id"],
        "xml_path": result["source_xml"],
        "horizon_pattern": pattern,
        "tier_pair": result["tier_pair"],
        "cohort": "independently_certified",
        "testbed": {"category": f"{pattern}_{result['tier_pair']}", "scene_type": "clean"},
        "label_evidence": "independent_exhaustive_local_opening",
        "certificate_file": str(evidence_path),
        "certificate_task_id": result["task_id"],
    }


def prepare(root, previous, cert_root, pilot):
    """Prepare only missing random runs and previously unreconciled pilot tasks."""
    previous_rows = read_rows(previous / "input/manifest.jsonl")
    previous_by_geometry = {row["geometry_id"]: row for row in previous_rows}
    cert_file = cert_root / "testbed/certification_results.jsonl"
    certified = read_rows(cert_file)
    known_geometries = {row["geometry_id"] for row in certified}
    selected, missing = [], []
    for result in certified:
        row = certified_descriptor(result, cert_file)
        if row is None:
            continue
        selected.append(row)
        old = previous_by_geometry.get(row["geometry_id"])
        if old:
            assert Path(old["xml_path"]).read_bytes() == Path(row["xml_path"]).read_bytes()
        else:
            missing.append(row)
    remaining = [row for row in read_rows(pilot) if row["geometry_identity"]["full"] not in known_geometries]
    tasks = []
    for index, row in enumerate(remaining):
        geometry = row["geometry_identity"]["full"]
        pattern = row["pilot"]["horizon_pattern"]
        actions = row["replay"]["actions"]
        assert pattern in PATTERNS and [len(hop) for hop in actions] == list(map(int, pattern))
        assert row["replay"]["status"] == "solved" and geometry in previous_by_geometry
        tasks.append({
            "task_index": index, "task_id": f"{index:04d}_{geometry[:16]}",
            "geometry_id": geometry, "source_xml": row["xml_path"],
            "tier_pair": row["pilot"]["tier_pair"], "source_cell": row["pilot"]["cell"],
            "witness_actions": actions,
        })
    assert len(selected) == 535 and len(missing) == 458 and len(tasks) == 177
    output = root / "input"
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "certified_latest.jsonl", selected)
    write_rows(output / "manifest.jsonl", missing)
    write_rows(output / "pilot_tasks.jsonl", tasks)
    with (output / "xmls.txt").open("x") as handle:
        handle.writelines(row["xml_path"] + "\n" for row in missing)
    summary = {
        "certified_latest": len(selected), "missing_random_scenes": len(missing),
        "missing_seed_runs": len(missing) * len(SEEDS), "pilot_recertification_scenes": len(tasks),
        "latest_patterns": dict(Counter(row["horizon_pattern"] for row in selected)),
        "missing_patterns": dict(Counter(row["horizon_pattern"] for row in missing)),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (previous / "input/manifest.jsonl", previous / "aggregate/per_scene.jsonl", cert_file, pilot)},
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def publish(root, previous, cert_root, analysis_script):
    """Publish labeled manifests and five-seed difficulty evidence with provenance."""
    spec = importlib.util.spec_from_file_location("random5_analysis", analysis_script)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    old_sources = read_rows(previous / "input/manifest.jsonl")
    old_results = {row["xml_path"]: row for row in read_rows(previous / "aggregate/per_scene.jsonl")}
    new_sources = read_rows(root / "input/manifest.jsonl")
    new_results = {row["xml_path"]: row for row in read_rows(root / "aggregate/per_scene.jsonl")}
    result_by_geometry = {row["geometry_id"]: (old_results[row["xml_path"]], str(previous / "aggregate/per_scene.jsonl")) for row in old_sources}
    for row in new_sources:
        assert row["geometry_id"] not in result_by_geometry
        result_by_geometry[row["geometry_id"]] = (new_results[row["xml_path"]], str(root / "aggregate/per_scene.jsonl"))

    descriptors = [dict(row, label_evidence="completed_onepush_chain") for row in old_sources if row["cohort"] != "mixed_push_horizons"]
    assert len(descriptors) == 31779 and all(row["horizon_pattern"] == "11" for row in descriptors)
    descriptors.extend(read_rows(root / "input/certified_latest.jsonl"))
    tasks = read_rows(root / "input/pilot_tasks.jsonl")
    paths = sorted((root / "recert/raw/results").glob("task_*.json"))
    recert = [json.loads(path.read_text()) for path in paths]
    assert len(recert) == len(tasks) == 177
    assert {row["task_index"] for row in recert} == set(range(len(tasks)))
    for result, path in zip(recert, paths):
        assert result["task_id"] == tasks[result["task_index"]]["task_id"]
        descriptor = certified_descriptor(result, path)
        if descriptor:
            descriptors.append(descriptor)
    assert len({row["geometry_id"] for row in descriptors}) == len(descriptors)
    per_scene, technical = [], []
    for descriptor in descriptors:
        original, source = result_by_geometry[descriptor["geometry_id"]]
        assert set(original["runs"]) == set(SEEDS)
        row = copy.deepcopy(original)
        row.update({key: descriptor[key] for key in ("geometry_id", "xml_path", "horizon_pattern", "tier_pair", "cohort", "label_evidence")})
        row.update(category=descriptor["testbed"]["category"], scene_type=descriptor["testbed"]["scene_type"], baseline_source=source)
        bad_seeds = []
        for seed, run in row["runs"].items():
            assert isinstance(run["total_calls"], int) and 0 <= run["total_calls"] <= 20000
            if run["failure_kind"] in TECHNICAL_FAILURES:
                bad_seeds.append(seed)
                technical.append({"geometry_id": row["geometry_id"], "xml_path": row["xml_path"], "seed": seed, "baseline_source": source, **run})
        row["baseline_valid"] = not bad_seeds
        if bad_seeds:
            row["median_total_calls_until_success"] = None
            row["difficulty_status"] = "technical_error"
        else:
            calls = sorted(run["total_calls"] for run in row["runs"].values() if run["solved"])
            row["successes_out_of_5"] = len(calls)
            row["median_is_censored"] = len(calls) < 3
            row["median_total_calls_until_success"] = calls[2] if len(calls) >= 3 else None
            row["difficulty_status"] = "censored" if len(calls) < 3 else "finite"
        per_scene.append(row)

    def summarize(scenes):
        runs = [run for row in scenes for run in row["runs"].values()]
        finite = [row["median_total_calls_until_success"] for row in scenes if row["difficulty_status"] == "finite"]
        return {"scenes": len(scenes), "difficulty_status": dict(Counter(row["difficulty_status"] for row in scenes)),
                "successes_out_of_5": dict(Counter(row["successes_out_of_5"] for row in scenes)),
                "scene_median_calls": analysis.distribution(finite), "runs": analysis.summarize_runs(runs)}

    output = root / "testbed"
    output.mkdir(exist_ok=False)
    write_rows(output / "manifest.jsonl", descriptors)
    write_rows(output / "per_scene.jsonl", per_scene)
    write_rows(output / "technical_errors.jsonl", technical)
    for pattern in PATTERNS:
        write_rows(output / "patterns" / pattern / "manifest.jsonl", [row for row in descriptors if row["horizon_pattern"] == pattern])
    latest = read_rows(cert_root / "testbed/certification_results.jsonl")
    write_rows(output / "unresolved_or_replay_failed.jsonl", [{"geometry_id": row["geometry_id"], "xml_path": row["source_xml"], "gate_labels": "".join(g["label"] for g in row.get("gates", [])), "full_witness": row.get("full_witness"), "status": row["status"]} for row in latest + recert if not (row["full_witness"]["success"] and all(g["label"] in ("1", "2") for g in row["gates"]))])
    write_json(output / "population.json", {"name": "two-keyhole-labeled-random5-v4", "scenes": [{"xml_path": row["xml_path"], "cluster_id": "geometry:" + row["geometry_id"], "horizon_pattern": row["horizon_pattern"]} for row in descriptors]})
    summary = {
        "scene_count": len(per_scene), "seed_run_count": len(per_scene) * 5, "seeds": list(map(int, SEEDS)),
        "total_simulation_budget": 20000, "full_namo_success": "end-goal reachability",
        "difficulty_labels_assigned": False, "technical_error_runs": len(technical),
        "new_random_scenes": len(new_sources), "pilot_recertified_scenes": len(recert),
        "label_evidence": dict(Counter(row["label_evidence"] for row in descriptors)),
        "censoring_rule": "Failures sort above successes; median is the third successful call count if at least three seeds succeed. Technical-error scenes have no difficulty estimate.",
        "all": summarize(per_scene),
        "by_pattern": {pattern: summarize([row for row in per_scene if row["horizon_pattern"] == pattern]) for pattern in PATTERNS},
        "by_pattern_and_donor_tier": {key: summarize([row for row in per_scene if row["horizon_pattern"] + "_" + row["tier_pair"] == key]) for key in sorted({row["horizon_pattern"] + "_" + row["tier_pair"] for row in per_scene})},
    }
    write_json(output / "summary.json", summary)
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}\n" for path in sorted(output.rglob("*")) if path.is_file()]
    with (output / "SHA256SUMS").open("x") as handle:
        handle.writelines(checksums)
    print(json.dumps({key: summary[key] for key in ("scene_count", "seed_run_count", "technical_error_runs", "label_evidence")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "publish"))
    args = parser.parse_args()
    root = Path(os.environ["RUN_ROOT"])
    previous = Path(os.environ["PREVIOUS_ROOT"])
    cert_root = Path(os.environ["CERT_SOURCE_ROOT"])
    if args.command == "prepare":
        prepare(root, previous, cert_root, Path(os.environ["PILOT_SOURCE"]))
    else:
        publish(root, previous, cert_root, Path(os.environ["RANDOM_ANALYSIS_SCRIPT"]))
