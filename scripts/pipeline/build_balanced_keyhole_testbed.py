#!/usr/bin/env python3
"""Reproduce the historical mixed-context two-keyhole testbed correction.

This entrypoint requires the original 428-scene primary cohort and 280 K1
discovery tasks, with their recorded host/category quotas. It preserves that
campaign's provenance; it is not the current diverse-template testbed policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("MM", "MH", "HM", "HH")
TIERS = {
    "MM": ("medium", "medium"),
    "MH": ("medium", "hard"),
    "HM": ("hard", "medium"),
    "HH": ("hard", "hard"),
}
POSITION_TOLERANCE_M = 0.002
ANGLE_TOLERANCE_RAD = math.radians(1.0)
CLEAN_PER_CATEGORY = 11
INTERACTIONS_PER_CATEGORY = {1: 32, 2: 64}
HOST_CAP = {1: 8, 2: 32}
MIN_HOSTS = 4
MIN_MOTION_MARGIN = 1.25
EXPECTED_PRIMARY = 428
EXPECTED_DISCOVERY_TASKS = 280


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing manifest: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(row: dict) -> str:
    value = row.get("geometry_identity", {}).get("full")
    if not isinstance(value, str) or not value:
        raise RuntimeError("row is missing a full geometry identity")
    return value


def category(row: dict) -> str:
    tiers = tuple(donor.get("tier") for donor in row.get("donors", []))
    for label, expected in TIERS.items():
        if tiers == expected:
            return label
    raise RuntimeError(f"unsupported donor tiers: {tiers!r}")


def scene_type(row: dict) -> str:
    mode = row.get("composition", {}).get("mode")
    if mode == "same_template":
        return "clean"
    if mode == "same_template_contact":
        return "interaction"
    raise RuntimeError(f"unsupported composition mode: {mode!r}")


def clean_sources(original_root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [
        row
        for label in CATEGORIES
        for row in read_jsonl(original_root / "clean" / label.lower() / "manifest.jsonl")
    ]
    if len(rows) != 70:
        raise RuntimeError(f"expected 70 clean hosts, found {len(rows)}")
    index: dict[str, dict] = {}
    for row in rows:
        if scene_type(row) != "clean":
            raise RuntimeError("clean source manifest contains an interaction")
        path = os.path.realpath(row["xml_path"])
        if path in index:
            raise RuntimeError(f"ambiguous clean source path: {path}")
        index[path] = row
    if len({geometry(row) for row in rows}) != 70:
        raise RuntimeError("clean source geometries are not unique")
    return rows, index


def pose_delta(before: list[float], after: list[float]) -> tuple[float, float]:
    translation = math.hypot(after[0] - before[0], after[1] - before[1])
    theta = after[2] - before[2]
    return translation, abs(math.atan2(math.sin(theta), math.cos(theta)))


def interaction_evidence(row: dict) -> dict:
    composition = row["composition"]
    hop = int(composition["interaction_effect"]["intended_hop"])
    context = list(composition.get("clutter_object_ids") or [])
    poses = row["replay"]["object_pose_trace"]
    infos = row["replay"]["action_info_trace"]
    if hop not in (1, 2) or not context or len(poses) != 3 or len(infos) != 2:
        raise RuntimeError("incomplete interaction replay evidence")
    other = 2 if hop == 1 else 1
    intended = []
    unintended = []
    for object_id in context:
        try:
            delta = pose_delta(poses[hop - 1][object_id], poses[hop][object_id])
            other_delta = pose_delta(poses[other - 1][object_id], poses[other][object_id])
        except KeyError as error:
            raise RuntimeError(f"missing context object in replay: {object_id}") from error
        intended.append((*delta, object_id))
        unintended.append((*other_delta, object_id))
    translation, angle, measured_object = max(
        intended,
        key=lambda item: max(
            item[0] / POSITION_TOLERANCE_M,
            item[1] / ANGLE_TOLERANCE_RAD,
        ),
    )
    collisions = {
        token.strip()
        for token in infos[hop - 1].get("movable_collisions", "").split(",")
        if token.strip()
    }
    return {
        "angle_rad": angle,
        "collision_object_ids": sorted(collisions.intersection(context)),
        "intended_hop": hop,
        "measured_object_id": measured_object,
        "motion_margin": max(
            translation / POSITION_TOLERANCE_M,
            angle / ANGLE_TOLERANCE_RAD,
        ),
        "translation_m": translation,
        "unintended_hop_stable": all(
            position <= POSITION_TOLERANCE_M and rotation <= ANGLE_TOLERANCE_RAD
            for position, rotation, _ in unintended
        ),
        "unintended_motion": [
            {"object_id": object_id, "translation_m": position, "angle_rad": rotation}
            for position, rotation, object_id in unintended
        ],
    }


def annotate(row: dict, source_index: dict[str, dict]) -> dict:
    tagged = copy.deepcopy(row)
    label = category(tagged)
    kind = scene_type(tagged)
    if kind == "clean":
        base = geometry(tagged)
        evidence = None
    else:
        source = os.path.realpath(tagged["composition"]["source_xml"])
        if source not in source_index:
            raise RuntimeError(f"interaction has no clean source: {source}")
        clean = source_index[source]
        if category(clean) != label:
            raise RuntimeError(f"interaction and clean source categories differ: {source}")
        if clean["geometry_identity"]["walls"] != tagged["geometry_identity"]["walls"]:
            raise RuntimeError(f"interaction and clean source walls differ: {source}")
        base = geometry(clean)
        evidence = interaction_evidence(tagged)
    tagged["evaluation"] = {
        "base_geometry_identity": base,
        "category": label,
        "cluster_id": "clean-source:"
        + hashlib.sha256(base.encode("utf-8")).hexdigest()[:20],
        "scene_type": kind,
    }
    if evidence is not None:
        tagged["evaluation"]["interaction"] = evidence
    return tagged


def replay_is_valid(row: dict) -> bool:
    replay = row.get("replay") or {}
    trace = [state.get("goal_reachable") for state in replay.get("reachability_trace", [])]
    return row.get("hops") == 2 and replay.get("status") == "solved" and trace == [False, False, True]


def robust_for_primary(row: dict, hop: int) -> bool:
    evidence = row["evaluation"].get("interaction") or {}
    return (
        replay_is_valid(row)
        and evidence.get("intended_hop") == hop
        and evidence.get("motion_margin", 0.0) >= MIN_MOTION_MARGIN
        and bool(evidence.get("collision_object_ids"))
        and evidence.get("unintended_hop_stable") is True
    )


def unique_by_geometry(rows: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: (geometry(item), os.path.realpath(item["xml_path"]))):
        unique.setdefault(geometry(row), row)
    return list(unique.values())


def select_round_robin(rows: list[dict], count: int, cap: int) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["evaluation"]["base_geometry_identity"]].append(row)
    for group in groups.values():
        group.sort(key=geometry)
    selected: list[dict] = []
    per_host: Counter[str] = Counter()
    while len(selected) < count:
        progressed = False
        for host in sorted(groups):
            if len(selected) == count:
                break
            if groups[host] and per_host[host] < cap:
                selected.append(groups[host].pop(0))
                per_host[host] += 1
                progressed = True
        if not progressed:
            break
    return selected, {
        "available": len(rows),
        "available_hosts": len(groups),
        "selected": len(selected),
        "selected_hosts": len(per_host),
        "selected_per_host": dict(sorted(per_host.items())),
    }


def load_discovery(discovery_root: Path) -> list[dict]:
    rows: list[dict] = []
    for task in range(EXPECTED_DISCOVERY_TASKS):
        result = discovery_root / f"task_{task:03d}" / "results"
        summary_path = result / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"missing discovery result: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_rows = read_jsonl(result / "manifest.jsonl")
        if summary.get("accepted") != len(task_rows) or summary.get("contact_hops") != [1]:
            raise RuntimeError(f"invalid K1 discovery result in task {task}")
        rows.extend(task_rows)
    return rows


def prepare(original_root: Path, discovery_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite: {output_dir}")
    _, source_index = clean_sources(original_root)
    original = read_jsonl(original_root / "testbed" / "manifest.jsonl")
    if len(original) != 14124 or not all(replay_is_valid(row) for row in original):
        raise RuntimeError("original replay-valid testbed is incomplete")
    discovered = load_discovery(discovery_root)
    original = [annotate(row, source_index) for row in original]
    discovered = [annotate(row, source_index) for row in discovered]
    clean = [row for row in original if scene_type(row) == "clean"]
    interactions = unique_by_geometry(
        [row for row in original if scene_type(row) == "interaction"] + discovered
    )

    selected: list[dict] = []
    report: dict[str, dict] = {}
    for label in CATEGORIES:
        clean_pool = sorted([row for row in clean if category(row) == label], key=geometry)
        if len(clean_pool) < CLEAN_PER_CATEGORY:
            raise RuntimeError(f"{label} clean shortfall: {len(clean_pool)}/{CLEAN_PER_CATEGORY}")
        selected.extend(clean_pool[:CLEAN_PER_CATEGORY])
        report[label] = {"clean": len(clean_pool[:CLEAN_PER_CATEGORY])}
        for hop in (1, 2):
            pool = [
                row
                for row in interactions
                if category(row) == label and robust_for_primary(row, hop)
            ]
            chosen, stats = select_round_robin(
                pool, INTERACTIONS_PER_CATEGORY[hop], HOST_CAP[hop]
            )
            if len(chosen) != INTERACTIONS_PER_CATEGORY[hop] or stats["selected_hosts"] < MIN_HOSTS:
                raise RuntimeError(
                    f"{label} K{hop} shortfall: selected={len(chosen)}/"
                    f"{INTERACTIONS_PER_CATEGORY[hop]}, hosts={stats['selected_hosts']}/{MIN_HOSTS}, "
                    f"available={len(pool)} from {stats['available_hosts']} hosts"
                )
            selected.extend(chosen)
            report[label][f"K{hop}"] = stats

    order = {("clean", 0): 0, ("interaction", 1): 1, ("interaction", 2): 2}
    selected.sort(
        key=lambda row: (
            CATEGORIES.index(category(row)),
            order[
                (
                    scene_type(row),
                    (row["evaluation"].get("interaction") or {}).get("intended_hop", 0),
                )
            ],
            geometry(row),
        )
    )
    if len(selected) != EXPECTED_PRIMARY or len({geometry(row) for row in selected}) != EXPECTED_PRIMARY:
        raise RuntimeError("primary selection is not exactly 428 unique geometries")
    if len({os.path.realpath(row["xml_path"]) for row in selected}) != EXPECTED_PRIMARY:
        raise RuntimeError("primary selection repeats an XML path")
    output_dir.mkdir(parents=True)
    for index, row in enumerate(selected):
        row["evaluation"]["selection_index"] = index
        write_jsonl(output_dir / "validation_inputs" / f"scene_{index:03d}.jsonl", [row])
    write_jsonl(output_dir / "primary_candidates.jsonl", selected)
    summary = {
        "discovered_k1": len(discovered),
        "original_population": len(original),
        "primary_candidates": len(selected),
        "selection": report,
    }
    write_json(output_dir / "selection_report.json", summary)
    return summary


def population(name: str, rows: list[dict]) -> dict:
    return {
        "name": name,
        "scenes": [
            {
                "base_geometry_identity": row["evaluation"]["base_geometry_identity"],
                "category": row["evaluation"]["category"],
                "cluster_id": row["evaluation"]["cluster_id"],
                "geometry_identity": geometry(row),
                "interaction_hop": (row["evaluation"].get("interaction") or {}).get("intended_hop"),
                "scene_type": row["evaluation"]["scene_type"],
                "xml_path": row["xml_path"],
            }
            for row in rows
        ],
    }


def stratum_counts(rows: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        suffix = "clean" if scene_type(row) == "clean" else f"K{row['evaluation']['interaction']['intended_hop']}"
        counts[f"{category(row)}_{suffix}"] += 1
    return dict(sorted(counts.items()))


def write_xml_hashes(path: Path, rows: list[dict]) -> None:
    xmls = sorted({os.path.realpath(row["xml_path"]) for row in rows})
    missing = [xml for xml in xmls if not Path(xml).is_file()]
    if missing:
        raise RuntimeError(f"missing referenced XML: {missing[0]}")
    path.write_text(
        "".join(f"{sha256(Path(xml))}  {xml}\n" for xml in xmls), encoding="utf-8"
    )


def finalize(original_root: Path, prepared: Path, validation: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    _, source_index = clean_sources(original_root)
    candidates = read_jsonl(prepared / "primary_candidates.jsonl")
    if len(candidates) != EXPECTED_PRIMARY:
        raise RuntimeError(f"expected {EXPECTED_PRIMARY} candidates")
    accepted: list[dict] = []
    runtimes: list[dict] = []
    for index, candidate in enumerate(candidates):
        task = validation / f"task_{index:03d}"
        rows = read_jsonl(task / "manifest.jsonl")
        rejected = read_jsonl(task / "rejected.jsonl")
        summary = json.loads((task / "summary.json").read_text(encoding="utf-8"))
        if len(rows) != 1 or rejected or summary.get("accepted") != 1:
            raise RuntimeError(f"validation task {index} did not accept exactly one scene")
        row = annotate(rows[0], source_index)
        if geometry(row) != geometry(candidate) or not replay_is_valid(row):
            raise RuntimeError(f"validation task {index} changed or failed its scene")
        if scene_type(row) == "interaction":
            hop = row["evaluation"]["interaction"]["intended_hop"]
            if not robust_for_primary(row, hop):
                raise RuntimeError(f"validation task {index} lost interaction evidence")
        for key in ("base_geometry_identity", "category", "cluster_id", "scene_type"):
            if row["evaluation"][key] != candidate["evaluation"][key]:
                raise RuntimeError(f"validation task {index} changed {key}")
        row["evaluation"]["selection_index"] = index
        runtime = summary.get("runtime") or {}
        if runtime.get("slurm_array_task_id") != str(index):
            raise RuntimeError(f"validation task {index} has wrong process provenance")
        accepted.append(row)
        runtimes.append(runtime)

    expected_counts = {
        **{f"{label}_clean": CLEAN_PER_CATEGORY for label in CATEGORIES},
        **{
            f"{label}_K{hop}": count
            for label in CATEGORIES
            for hop, count in INTERACTIONS_PER_CATEGORY.items()
        },
    }
    if stratum_counts(accepted) != dict(sorted(expected_counts.items())):
        raise RuntimeError("fresh replay changed primary stratum counts")
    signatures = {
        (
            runtime.get("binding_sha256"),
            runtime.get("code_commit"),
            runtime.get("namo_path"),
            runtime.get("python"),
        )
        for runtime in runtimes
    }
    if len(signatures) != 1:
        raise RuntimeError("validation tasks used inconsistent code or bindings")

    original = [
        annotate(row, source_index)
        for row in read_jsonl(original_root / "testbed" / "manifest.jsonl")
    ]
    original_geometry = {geometry(row) for row in original}
    new_k1 = [
        row
        for row in accepted
        if scene_type(row) == "interaction"
        and row["evaluation"]["interaction"]["intended_hop"] == 1
        and geometry(row) not in original_geometry
    ]
    stress = original + new_k1
    if len({geometry(row) for row in stress}) != len(stress):
        raise RuntimeError("stress population contains duplicate geometry")

    output.mkdir(parents=True)
    write_jsonl(output / "manifest_primary.jsonl", accepted)
    write_json(output / "population_primary.json", population("two-keyhole-primary-balanced-v2", accepted))
    write_json(output / "population_stress.json", population("two-keyhole-stress-corrected-v2", stress))
    write_json(output / "validation_runtime.json", runtimes)
    write_xml_hashes(output / "PRIMARY_XML_SHA256SUMS", accepted)
    write_xml_hashes(output / "STRESS_XML_SHA256SUMS", stress)
    summary = {
        "primary_rows": len(accepted),
        "primary_strata": stratum_counts(accepted),
        "stress_rows": len(stress),
        "stress_new_k1": len(new_k1),
        "unique_validation_array_tasks": len(
            {runtime.get("slurm_array_task_id") for runtime in runtimes}
        ),
    }
    write_json(output / "summary.json", summary)
    index_files = [
        "manifest_primary.jsonl",
        "population_primary.json",
        "population_stress.json",
        "validation_runtime.json",
        "PRIMARY_XML_SHA256SUMS",
        "STRESS_XML_SHA256SUMS",
        "summary.json",
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in index_files),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("prepare")
    select.add_argument("--original-root", type=Path, required=True)
    select.add_argument("--discovery-root", type=Path, required=True)
    select.add_argument("--out-dir", type=Path, required=True)
    publish = commands.add_parser("finalize")
    publish.add_argument("--original-root", type=Path, required=True)
    publish.add_argument("--prepared-dir", type=Path, required=True)
    publish.add_argument("--validation-root", type=Path, required=True)
    publish.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.original_root, args.discovery_root, args.out_dir)
    else:
        result = finalize(
            args.original_root, args.prepared_dir, args.validation_root, args.out_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
