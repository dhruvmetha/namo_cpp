#!/usr/bin/env python3
"""Aggregate sharded Full-NAMO solvability results into one solved manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _template_key(xml_path: str) -> str:
    parts = Path(xml_path).parts
    for index, part in enumerate(parts[:-1]):
        if part in {"set1", "set2"} and parts[index + 1].startswith("benchmark_"):
            return f"{part}/{parts[index + 1]}"
    return "unknown"


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def aggregate(eval_root: Path, output_dir: Path) -> Dict[str, Any]:
    shard_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(eval_root.glob("shard_*/summary.json"))
    ]
    solved_by_xml = {
        row["xml_path"]: row
        for row in _read_jsonl(eval_root.glob("shard_*/solved.jsonl"))
    }
    unsolved_by_xml = {
        row["xml_path"]: row
        for row in _read_jsonl(eval_root.glob("shard_*/unsolved.jsonl"))
        if row["xml_path"] not in solved_by_xml
    }
    solved = [solved_by_xml[key] for key in sorted(solved_by_xml)]
    unsolved = [unsolved_by_xml[key] for key in sorted(unsolved_by_xml)]
    all_rows = solved + unsolved

    selected_by_template = Counter(_template_key(row["xml_path"]) for row in all_rows)
    solved_by_template = Counter(_template_key(row["xml_path"]) for row in solved)
    template_rows = {
        key: {
            "evaluated": selected_by_template[key],
            "solved": solved_by_template[key],
            "solve_rate": solved_by_template[key] / selected_by_template[key],
        }
        for key in sorted(selected_by_template)
    }

    summary = {
        "completed_shards": len(shard_summaries),
        "input_count": sum(row["input_env_count"] for row in shard_summaries),
        "selected_exact_hop_count": sum(row["selected_env_count"] for row in shard_summaries),
        "path_length_mismatch_count": sum(
            row["input_env_count"] - row["selected_env_count"] - row["selection_error_count"]
            for row in shard_summaries
        ),
        "selection_error_count": sum(row["selection_error_count"] for row in shard_summaries),
        "evaluated_count": len(all_rows),
        "solved_count": len(solved),
        "unsolved_count": len(unsolved),
        "solve_rate": len(solved) / len(all_rows) if all_rows else 0.0,
        "failure_kinds": dict(sorted(Counter(
            str(row.get("failure_kind") or row.get("outcome") or "unknown")
            for row in unsolved
        ).items())),
        "by_template": template_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "solved.jsonl", solved)
    _write_jsonl(output_dir / "unsolved.jsonl", unsolved)
    (output_dir / "solved_xmls.txt").write_text(
        "".join(f"{row['xml_path']}\n" for row in solved),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.eval_root, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
