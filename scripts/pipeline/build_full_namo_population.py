#!/usr/bin/env python3
"""Build a frozen, structurally valid, geometry-disjoint Full NAMO population."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from probe_static_topology import DROP_RULES
from verify_geom_disjoint import geom_sig, load_xmls


OUTPUT_FILENAMES = (
    "population.json",
    "accepted_scenes.txt",
    "dropped_scenes.jsonl",
    "population_audit.json",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(raw: object, base: Path, context: str) -> str:
    """Resolve one nonempty scene path to the canonical realpath identity."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a nonempty path string")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    return os.path.realpath(path)


def _load_manifest(path: Path) -> tuple[str, ...]:
    """Load a line manifest and reject duplicate canonical scene identities."""
    scenes: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            scene = _canonical_path(raw, path.parent, f"{path}:{line_number}")
            if scene in seen:
                raise ValueError(f"{path}:{line_number}: duplicate manifest scene {scene}")
            seen.add(scene)
            scenes.append(scene)
    if not scenes:
        raise ValueError(f"candidate manifest is empty: {path}")
    return tuple(scenes)


def _load_probe(path: Path) -> dict[str, dict[str, object]]:
    """Load zero-simulation probe rows by canonical scene identity."""
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: probe row must be a JSON object")
            scene = _canonical_path(
                raw.get("xml_path"),
                path.parent,
                f"{path}:{line_number}.xml_path",
            )
            if scene in rows:
                raise ValueError(f"{path}:{line_number}: duplicate probe scene {scene}")
            rows[scene] = raw
    if not rows:
        raise ValueError(f"probe JSONL is empty: {path}")
    return rows


def _structural_reasons(row: dict[str, object], expect_hop: int) -> list[str]:
    """Return zero-simulation structural rejection reasons for one probe row."""
    reasons: list[str] = []
    for rule in DROP_RULES:
        if rule == "hop_mismatch":
            if bool(row.get(rule)) or row.get("hop_count") != expect_hop:
                reasons.append(rule)
        elif bool(row.get(rule)):
            reasons.append(rule)
    if row.get("goal_in_free_space") is not True:
        reasons.append("goal_not_in_free_space")
    return reasons


def _training_signatures(
    train_specs: Sequence[Path],
) -> tuple[set[str], set[str], dict[str, int]]:
    """Load all registered training rooms and return full and floorplan signatures."""
    train_paths: list[str] = []
    for spec in train_specs:
        train_paths.extend(os.path.realpath(path) for path in load_xmls(str(spec)))
    unique_paths = tuple(dict.fromkeys(train_paths))
    full_signatures: set[str] = set()
    wall_signatures: set[str] = set()
    unparseable = 0
    for scene in unique_paths:
        full, walls = geom_sig(scene)
        if full is None or walls is None:
            unparseable += 1
            continue
        full_signatures.add(full)
        wall_signatures.add(walls)
    return full_signatures, wall_signatures, {
        "reference_files": len(train_specs),
        "xml_paths": len(train_paths),
        "unique_xml_paths": len(unique_paths),
        "unparseable_xml_paths": unparseable,
        "unique_room_signatures": len(full_signatures),
        "unique_floorplan_signatures": len(wall_signatures),
    }


def _assert_outputs_absent(out_dir: Path) -> None:
    """Protect an existing frozen population from accidental mutation."""
    existing = [out_dir / name for name in OUTPUT_FILENAMES if (out_dir / name).exists()]
    if existing:
        raise ValueError(f"refusing to overwrite frozen output: {existing[0]}")


def _write_json(path: Path, value: object) -> None:
    """Write stable, human-readable JSON to a new file."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write deterministic compact JSONL to a new file."""
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def build_population(
    *,
    manifest_path: Path,
    probe_jsonl: Path,
    train_specs: Sequence[Path],
    name: str,
    expect_hop: int,
    out_dir: Path,
) -> dict[str, object]:
    """Validate candidates and write one immutable Full NAMO population and audit."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    probe_jsonl = Path(probe_jsonl).expanduser().resolve()
    train_specs = tuple(Path(path).expanduser().resolve() for path in train_specs)
    out_dir = Path(out_dir).expanduser().resolve()
    if not name.strip():
        raise ValueError("population name must be nonempty")
    if expect_hop <= 0:
        raise ValueError("expected hop count must be positive")
    _assert_outputs_absent(out_dir)

    candidates = _load_manifest(manifest_path)
    probe_rows = _load_probe(probe_jsonl)
    candidate_set = set(candidates)
    probe_set = set(probe_rows)
    if candidate_set != probe_set:
        raise ValueError(
            "probe population mismatch: "
            f"{len(candidate_set - probe_set)} missing, "
            f"{len(probe_set - candidate_set)} extra"
        )

    train_full, train_walls, train_counts = _training_signatures(train_specs)
    accepted: list[dict[str, str]] = []
    dropped: list[dict[str, object]] = []
    drop_counts: Counter[str] = Counter()
    accepted_full: list[str] = []
    accepted_walls: list[str] = []
    candidate_floorplan_overlap = 0
    training_leaks = 0

    for scene in sorted(candidates):
        row = probe_rows[scene]
        reasons = _structural_reasons(row, expect_hop)
        full, walls = geom_sig(scene)
        if full is None or walls is None:
            reasons.append("unparseable_geometry")
        else:
            if walls in train_walls:
                candidate_floorplan_overlap += 1
            if full in train_full:
                reasons.append("training_geometry_leak")
                training_leaks += 1

        if reasons:
            unique_reasons = list(dict.fromkeys(reasons))
            drop_counts.update(unique_reasons)
            dropped.append(
                {
                    "xml_path": scene,
                    "reasons": unique_reasons,
                    "probe_error": row.get("error"),
                }
            )
            continue

        accepted.append(
            {
                "xml_path": scene,
                "cluster_id": f"floorplan:{walls}",
            }
        )
        accepted_full.append(full)
        accepted_walls.append(walls)

    if not accepted:
        raise ValueError("structural and leakage checks removed every candidate")

    population = {"name": name.strip(), "scenes": accepted}
    audit: dict[str, object] = {
        "name": name.strip(),
        "expected_hop_count": expect_hop,
        "sources": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "probe_jsonl": str(probe_jsonl),
            "probe_jsonl_sha256": _sha256(probe_jsonl),
            "training_references": [str(path) for path in train_specs],
        },
        "counts": {
            "input_scenes": len(candidates),
            "accepted_scenes": len(accepted),
            "dropped_scenes": len(dropped),
            "training_scene_leaks": training_leaks,
        },
        "drop_reasons": dict(sorted(drop_counts.items())),
        "training": train_counts,
        "geometry": {
            "accepted_unique_room_signatures": len(set(accepted_full)),
            "accepted_room_variants": len(accepted_full) - len(set(accepted_full)),
            "accepted_unique_floorplans": len(set(accepted_walls)),
            "candidate_floorplans_shared_with_train": candidate_floorplan_overlap,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "population.json", population)
    with (out_dir / "accepted_scenes.txt").open("x", encoding="utf-8") as stream:
        stream.write("".join(f"{row['xml_path']}\n" for row in accepted))
    _write_jsonl(out_dir / "dropped_scenes.jsonl", dropped)
    _write_json(out_dir / "population_audit.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    """Run the held-out population build from command-line inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-jsonl", type=Path, required=True)
    parser.add_argument(
        "--train-xmls",
        type=Path,
        action="append",
        required=True,
        help="registered training H5/TXT/JSON reference; repeat for multiple corpora",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--expect-hop", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    audit = build_population(
        manifest_path=args.manifest,
        probe_jsonl=args.probe_jsonl,
        train_specs=args.train_xmls,
        name=args.name,
        expect_hop=args.expect_hop,
        out_dir=args.out_dir,
    )
    counts = audit["counts"]
    print(
        f"accepted {counts['accepted_scenes']}/{counts['input_scenes']} scenes; "
        f"dropped {counts['dropped_scenes']}; "
        f"training leaks {counts['training_scene_leaks']}"
    )
    for filename in OUTPUT_FILENAMES:
        print(f"wrote {args.out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
