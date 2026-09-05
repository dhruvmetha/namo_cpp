#!/usr/bin/env python3
"""Select a fixed clean/contact mixture for each ordered two-keyhole difficulty pair."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


PAIR_TIERS = {
    "mm": ("medium", "medium"),
    "mh": ("medium", "hard"),
    "hm": ("hard", "medium"),
    "hh": ("hard", "hard"),
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _row_tiers(row: dict) -> tuple[str, str]:
    return tuple(donor["tier"] for donor in row["donors"])


def _interaction_hop(row: dict) -> int:
    return int(row["composition"]["interaction_effect"]["intended_hop"])


def _ordered_contacts(rows: list[dict]) -> list[dict]:
    by_hop = {
        hop: [row for row in rows if _interaction_hop(row) == hop]
        for hop in (1, 2)
    }
    ordered = []
    while by_hop[1] or by_hop[2]:
        for hop in (1, 2):
            if by_hop[hop]:
                ordered.append(by_hop[hop].pop(0))
    return ordered


def select_pair(
    clean_rows: list[dict],
    contact_rows: list[dict],
    tiers: tuple[str, str],
    *,
    per_pair: int,
    max_contacts: int,
    geometry_seen: set[str],
) -> list[dict]:
    for row in clean_rows + contact_rows:
        if _row_tiers(row) != tiers:
            raise RuntimeError(f"row tiers {_row_tiers(row)} do not match expected {tiers}")
    selected = []
    selected_clean_sources = set()
    for row in _ordered_contacts(contact_rows):
        identity = row["geometry_identity"]["full"]
        if identity in geometry_seen:
            continue
        selected.append(copy.deepcopy(row))
        geometry_seen.add(identity)
        selected_clean_sources.add(row["composition"]["source_xml"])
        if len(selected) == min(max_contacts, per_pair):
            break
    clean_order = [
        row for row in clean_rows if row["xml_path"] not in selected_clean_sources
    ] + [row for row in clean_rows if row["xml_path"] in selected_clean_sources]
    for row in clean_order:
        if len(selected) == per_pair:
            break
        identity = row["geometry_identity"]["full"]
        if identity in geometry_seen:
            continue
        selected.append(copy.deepcopy(row))
        geometry_seen.add(identity)
    if len(selected) != per_pair:
        raise RuntimeError(f"{tiers}: selected {len(selected)} of required {per_pair}")
    for index, row in enumerate(selected):
        row["cohort"] = {
            "source_type": (
                "contact" if row["composition"]["mode"] == "same_template_contact" else "clean"
            ),
            "pair_tiers": list(tiers),
            "pair_index": index,
        }
    return selected


def build_cohort(
    clean_root: Path,
    contact_root: Path,
    output_dir: Path,
    *,
    per_pair: int,
    max_contacts: int,
) -> dict:
    geometry_seen: set[str] = set()
    selected = []
    by_pair = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair, tiers in PAIR_TIERS.items():
        rows = select_pair(
            _read_jsonl(clean_root / pair / "manifest.jsonl"),
            _read_jsonl(contact_root / pair / "manifest.jsonl"),
            tiers,
            per_pair=per_pair,
            max_contacts=max_contacts,
            geometry_seen=geometry_seen,
        )
        selected.extend(rows)
        type_counts = Counter(row["cohort"]["source_type"] for row in rows)
        interaction_hops = Counter(
            str(_interaction_hop(row))
            for row in rows
            if row["cohort"]["source_type"] == "contact"
        )
        by_pair[pair] = {
            "tiers": list(tiers),
            "selected": len(rows),
            "source_types": dict(sorted(type_counts.items())),
            "interaction_hops": dict(sorted(interaction_hops.items())),
        }
        pair_dir = output_dir / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        (pair_dir / "xmls.txt").write_text(
            "".join(row["xml_path"] + "\n" for row in rows), encoding="utf-8"
        )

    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
    )
    (output_dir / "xmls.txt").write_text(
        "".join(row["xml_path"] + "\n" for row in selected), encoding="utf-8"
    )
    summary = {
        "clean_root": str(clean_root.resolve()),
        "contact_root": str(contact_root.resolve()),
        "per_pair": per_pair,
        "max_contacts_per_pair": max_contacts,
        "selected": len(selected),
        "unique_geometry": len(geometry_seen),
        "by_pair": by_pair,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-pair", type=int, default=10)
    parser.add_argument("--max-contacts-per-pair", type=int, default=3)
    args = parser.parse_args()
    summary = build_cohort(
        args.clean_root,
        args.contact_root,
        args.out_dir,
        per_pair=args.per_pair,
        max_contacts=args.max_contacts_per_pair,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
