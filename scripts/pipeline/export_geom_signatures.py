#!/usr/bin/env python3
"""Export complete training-room geometry as a compact signature reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from verify_geom_disjoint import (
    SIGNATURE_REFERENCE_SCHEMA,
    load_xmls,
    sig_map,
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source reference file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_signatures(
    *,
    train_specs: Sequence[Path],
    out_path: Path,
    workers: int = 32,
) -> dict[str, object]:
    """Write a deterministic, fail-closed geometry reference for training rooms."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    specs = sorted({Path(path).expanduser().resolve() for path in train_specs})
    if not specs:
        raise ValueError("at least one training XML reference is required")
    out_path = Path(out_path).expanduser().resolve()
    if out_path.exists():
        raise ValueError(f"refusing to overwrite signature reference: {out_path}")

    sources: list[dict[str, str]] = []
    xml_paths: list[str] = []
    for spec in specs:
        sources.append({"path": str(spec), "sha256": _sha256(spec)})
        xml_paths.extend(os.path.realpath(path) for path in load_xmls(str(spec)))
    if not xml_paths:
        raise ValueError("training references contain no XML paths")

    unique_paths = tuple(dict.fromkeys(xml_paths))
    parsed, full_to_xmls, walls_to_full = sig_map(unique_paths, workers=workers)
    if parsed != len(unique_paths):
        raise ValueError(
            f"{len(unique_paths) - parsed} of {len(unique_paths)} unique training XMLs "
            "are unparseable"
        )
    if not full_to_xmls or not walls_to_full:
        raise ValueError("training references produced no geometry signatures")

    artifact: dict[str, object] = {
        "schema": SIGNATURE_REFERENCE_SCHEMA,
        "sources": sources,
        "counts": {
            "xml_paths": len(xml_paths),
            "unique_xml_paths": len(unique_paths),
            "unique_room_signatures": len(full_to_xmls),
            "unique_floorplan_signatures": len(walls_to_full),
        },
        "full_signatures": sorted(full_to_xmls),
        "wall_signatures": sorted(walls_to_full),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Export a compact geometry reference from registered training sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-xmls", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)

    artifact = export_signatures(
        train_specs=args.train_xmls,
        out_path=args.out,
        workers=args.workers,
    )
    counts = artifact["counts"]
    print(
        f"exported {counts['unique_room_signatures']} room signatures from "
        f"{counts['unique_xml_paths']} unique XML paths"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
