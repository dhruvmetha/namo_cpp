from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from full_namo_sim_exp.experiment_io import Arm, Experiment, normalize_scene_id
from full_namo_sim_exp.run_shard import shard_scenes


INFRASTRUCTURE_FAILURES = {"runner_exception", "selection_error"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return raw


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        yield line_number, row


def _validate_terminal_row(
    row: dict[str, Any],
    *,
    solved: bool,
    source: Path,
    line_number: int,
    expected_path_length: int,
) -> tuple[str, dict[str, Any]]:
    context = f"{source}:{line_number}"
    scene = normalize_scene_id(row.get("xml_path"))
    failure_kind = row.get("failure_kind")
    outcome = row.get("outcome")
    if failure_kind in INFRASTRUCTURE_FAILURES or outcome in INFRASTRUCTURE_FAILURES:
        raise ValueError(f"{context}: infrastructure failure {failure_kind or outcome}")
    calls = row.get("simulation_budget_used_total")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise ValueError(f"{context}: simulation_budget_used_total must be nonnegative")
    time_ms = row.get("search_time_ms")
    if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)):
        raise ValueError(f"{context}: search_time_ms must be numeric")
    if not math.isfinite(float(time_ms)) or time_ms < 0:
        raise ValueError(f"{context}: search_time_ms must be finite and nonnegative")
    if row.get("timing_scope") != "full_namo_planner_search":
        raise ValueError(
            f"{context}: timing_scope must be 'full_namo_planner_search'"
        )
    if row.get("path_length_n") != expected_path_length:
        raise ValueError(
            f"{context}: path_length_n expected {expected_path_length}, "
            f"received {row.get('path_length_n')!r}"
        )
    recorded_solved = row.get("solved")
    if recorded_solved is not None and recorded_solved is not solved:
        raise ValueError(f"{context}: solved field contradicts containing file")
    normalized = dict(row)
    normalized["xml_path"] = scene
    normalized["solved"] = solved
    return scene, normalized


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_run_config(experiment: Experiment, arm: Arm, path: Path) -> None:
    raw = _read_json(path)
    protocol = experiment.protocol
    expected = {
        "path_length": protocol.path_length,
        "path_length_role": "declared_population_metadata_only",
        "config_file": str(protocol.config_file),
        "primitive_data_dir": str(protocol.primitive_data_dir),
        "primitive_prefix": protocol.primitive_prefix,
        "rollout_samples_per_state": None,
        "region_frontier_beam_width": None,
        "region_max_chain_depth": protocol.region_max_chain_depth,
        "region_success_min_reachable": protocol.region_success_min_reachable,
        "goals_per_region": protocol.goals_per_region,
        "seed": protocol.evaluation_seed,
        "evaluation_seed": protocol.evaluation_seed,
        "ordering_seed": arm.seed,
        "goal_strategy": "scorer",
        "use_cpp_snapshot": True,
        "simulation_budget": protocol.simulation_budget_per_keyhole,
        "simulation_budget_scope": "keyhole",
        "local_search": "best_first",
        "best_first_prior": arm.prior,
        "region_selection_strategy": "ml_first",
        "scorer_ckpt": str(arm.checkpoint),
        "ml_device": "cpu",
        "workers": 1,
        "full_namo_max_iterations": None,
        "audit_next_keyhole_reachability": False,
        "preserve_next_keyhole_access": False,
        "max_push_steps": protocol.max_push_steps,
        "prefilter_applied": False,
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise ValueError(
                f"{path}: {field} expected {value!r}, received {raw.get(field)!r}"
            )


def aggregate_arm(experiment: Experiment, arm: Arm) -> dict[str, Any]:
    raw_root = experiment.run_root / "raw"
    expected_shards = tuple(
        experiment.raw_shard_root(index) for index in range(experiment.protocol.n_shards)
    )
    actual_shards = tuple(sorted(path for path in raw_root.glob("shard_*") if path.is_dir()))
    if actual_shards != expected_shards:
        raise ValueError(
            f"{arm.name}: raw shard directories do not match the configured shard set"
        )
    rows: dict[str, dict[str, Any]] = {}
    solved_by_scene: dict[str, bool] = {}
    for shard_index, shard_root in enumerate(expected_shards):
        arm_root = shard_root / arm.name
        _validate_run_config(experiment, arm, arm_root / "run_config.json")
        summary = _read_json(arm_root / "summary.json")
        input_count = summary.get("input_env_count")
        selected_count = summary.get("selected_env_count")
        selection_errors = summary.get("selection_error_count")
        if selection_errors != 0 or selected_count != input_count:
            raise ValueError(
                f"{shard_root}: expected every frozen scene to be selected; "
                f"input={input_count}, selected={selected_count}, "
                f"selection_errors={selection_errors}"
            )
        expected_shard_scenes = set(shard_scenes(experiment, shard_index))
        shard_rows: set[str] = set()
        for filename, solved in (("solved.jsonl", True), ("unsolved.jsonl", False)):
            source = arm_root / filename
            for line_number, raw in _read_jsonl(source):
                scene, row = _validate_terminal_row(
                    raw,
                    solved=solved,
                    source=source,
                    line_number=line_number,
                    expected_path_length=experiment.protocol.path_length,
                )
                if scene in rows:
                    raise ValueError(f"{source}:{line_number}: duplicate scene {scene}")
                rows[scene] = row
                solved_by_scene[scene] = solved
                shard_rows.add(scene)
        if shard_rows != expected_shard_scenes:
            raise ValueError(
                f"{shard_root}: scene mismatch: "
                f"{len(expected_shard_scenes - shard_rows)} missing, "
                f"{len(shard_rows - expected_shard_scenes)} extra"
            )
        if input_count != len(expected_shard_scenes):
            raise ValueError(
                f"{shard_root}: input_env_count expected {len(expected_shard_scenes)}, "
                f"received {input_count!r}"
            )

    expected = set(experiment.population.scene_ids)
    actual = set(rows)
    if actual != expected:
        raise ValueError(
            f"{arm.name}: population mismatch: "
            f"{len(expected - actual)} missing, {len(actual - expected)} extra"
        )
    solved_rows = [
        rows[scene] for scene in experiment.population.scene_ids if solved_by_scene[scene]
    ]
    unsolved_rows = [
        rows[scene] for scene in experiment.population.scene_ids if not solved_by_scene[scene]
    ]
    output = experiment.aggregate_root(arm)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "solved.jsonl", solved_rows)
    _write_jsonl(output / "unsolved.jsonl", unsolved_rows)
    failure_kinds = Counter(
        str(row.get("failure_kind") or row.get("outcome") or "unknown")
        for row in unsolved_rows
    )
    summary = {
        "experiment": experiment.name,
        "population": experiment.population.name,
        "population_size": len(experiment.population.scene_ids),
        "arm": arm.name,
        "label": arm.label,
        "prior": arm.prior,
        "seed": arm.seed,
        "completed_shards": len(expected_shards),
        "solved_count": len(solved_rows),
        "unsolved_count": len(unsolved_rows),
        "success_rate_percent": 100.0 * len(solved_rows) / len(rows),
        "failure_kinds": dict(sorted(failure_kinds.items())),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_all(experiment: Experiment) -> dict[str, dict[str, Any]]:
    return {arm.name: aggregate_arm(experiment, arm) for arm in experiment.arms}
