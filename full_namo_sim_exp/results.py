from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from full_namo_sim_exp.experiment_io import Arm, Experiment, normalize_scene_id


@dataclass(frozen=True)
class Outcome:
    solved: bool
    simulator_calls: int
    wall_time_seconds: float


def load_arm_results(experiment: Experiment, arm: Arm) -> dict[str, Outcome]:
    rows: dict[str, Outcome] = {}
    root = experiment.aggregate_root(arm)
    for filename, solved in (("solved.jsonl", True), ("unsolved.jsonl", False)):
        path = root / filename
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise ValueError(f"missing aggregate file: {path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            scene = normalize_scene_id(raw.get("xml_path"))
            if scene in rows:
                raise ValueError(f"{path}:{line_number}: duplicate scene {scene}")
            calls = raw.get("simulation_budget_used_total")
            time_ms = raw.get("search_time_ms")
            if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                raise ValueError(f"{path}:{line_number}: invalid simulator-call total")
            if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)):
                raise ValueError(f"{path}:{line_number}: invalid search_time_ms")
            if not math.isfinite(float(time_ms)) or time_ms < 0:
                raise ValueError(f"{path}:{line_number}: invalid search_time_ms")
            rows[scene] = Outcome(solved, calls, float(time_ms) / 1000.0)
    expected = set(experiment.population.scene_ids)
    if set(rows) != expected:
        raise ValueError(
            f"{arm.name}: aggregate population mismatch: "
            f"{len(expected - set(rows))} missing, {len(set(rows) - expected)} extra"
        )
    return {scene: rows[scene] for scene in experiment.population.scene_ids}


def load_all_results(experiment: Experiment) -> dict[str, dict[str, Outcome]]:
    return {arm.name: load_arm_results(experiment, arm) for arm in experiment.arms}
