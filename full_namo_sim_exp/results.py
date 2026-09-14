from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from full_namo_sim_exp.experiment_io import Arm, Experiment, normalize_scene_id


FROZEN_MODEL_ARM = "HY5U_s2_search"
FROZEN_RANDOM_SEEDS = (7000, 8000, 9000, 10000, 11000)
FROZEN_RANDOM_ARMS = tuple(f"random_s{seed}" for seed in FROZEN_RANDOM_SEEDS)
FROZEN_DIFFICULTIES = ("easy", "medium", "hard")


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


@dataclass(frozen=True)
class FrozenResults:
    """Matched Full NAMO tasks from one normalized timed campaign snapshot."""

    campaign: str
    call_cap: int
    scenes: tuple[dict, ...]
    outcomes: dict[str, dict[str, Outcome]]
    source: Path
    source_sha256: str
    excluded_geometry_ids: tuple[str, ...]
    provenance: dict


def load_frozen_results(path: Path) -> FrozenResults:
    """Select the frozen 100/100/100 tasks and validate all six required arms.

    This snapshot identifies one full robot-to-goal task per geometry ID;
    cross-arm problem IDs and XML hashes are checked before selecting tiers.
    The existing unresolved stratum is excluded by its frozen label only.
    Failures among the selected tasks are retained.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("format") != "full_namo_timed_v1":
        raise ValueError(f"{path}: expected a full_namo_timed_v1 snapshot")
    scenes = payload["scenes"]
    ids = [scene["geometry_id"] for scene in scenes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: empty or duplicate Full NAMO task identities")
    if any(scene.get("difficulty") not in (*FROZEN_DIFFICULTIES, "unresolved") for scene in scenes):
        raise ValueError(f"{path}: missing or invalid frozen difficulty labels")
    selected = tuple(scene for scene in scenes if scene["difficulty"] in FROZEN_DIFFICULTIES)
    counts = Counter(scene["difficulty"] for scene in selected)
    if counts != {tier: 100 for tier in FROZEN_DIFFICULTIES}:
        raise ValueError(f"{path}: expected 100 easy/medium/hard tasks, found {dict(counts)}")
    cap = payload["call_cap"]
    if type(cap) is not int or cap < 1:
        raise ValueError(f"{path}: simulation call cap must be a positive integer")
    selected_ids = {scene["geometry_id"] for scene in selected}
    expected_ids = set(ids)
    xml_hashes = {scene["geometry_id"]: scene["xml_sha256"] for scene in scenes}
    problem_ids = {}
    outcomes = {}
    for arm in (FROZEN_MODEL_ARM, *FROZEN_RANDOM_ARMS):
        rows = payload["arms"].get(arm, {})
        if set(rows) != expected_ids:
            raise ValueError(f"{path}/{arm}: missing or extra tasks in the source population")
        outcomes[arm] = {}
        for scene_id in ids:
            row = rows[scene_id]
            context = f"{arm}/{scene_id}"
            solved, calls, seconds = row["solved"], row["total_calls"], row["t_wall"]
            if not isinstance(solved, bool) or type(calls) is not int or not 0 <= calls <= cap:
                raise ValueError(f"{context}: invalid outcome or simulator-call count")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"{context}: invalid wall-clock seconds")
            if not row.get("problem_id") or row.get("xml_sha256") != xml_hashes[scene_id]:
                raise ValueError(f"{context}: missing problem ID or mismatched XML hash")
            if arm == FROZEN_MODEL_ARM:
                problem_ids[scene_id] = row["problem_id"]
            elif problem_ids[scene_id] != row["problem_id"]:
                raise ValueError(f"{context}: problem identity differs between methods")
            if scene_id in selected_ids:
                outcomes[arm][scene_id] = Outcome(solved, calls, float(seconds))
    return FrozenResults(
        campaign=payload["campaign"], call_cap=cap, scenes=selected, outcomes=outcomes,
        source=path, source_sha256=hashlib.sha256(raw).hexdigest(),
        excluded_geometry_ids=tuple(scene_id for scene_id in ids if scene_id not in selected_ids),
        provenance=payload["provenance"],
    )
