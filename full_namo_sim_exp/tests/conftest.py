from __future__ import annotations

import json
from pathlib import Path

import pytest

from full_namo_sim_exp import provenance
from full_namo_sim_exp.experiment_io import Arm, Experiment


SCENES = tuple(f"/tmp/full_namo_sim_exp_fixture/room_{index}.xml" for index in range(4))


@pytest.fixture(autouse=True)
def clean_repository_state(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_repository_state", lambda: ("test-commit", ""))
    monkeypatch.setattr(
        provenance,
        "_runtime_state",
        lambda: {"namo_rl_sha256": "test-runtime", "sage_commit": "test-sage"},
        raising=False,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def experiment_path(tmp_path: Path) -> Path:
    for index, scene in enumerate(SCENES):
        path = Path(scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<mujoco model='room_{index}'/>\n", encoding="utf-8")
    write_json(
        tmp_path / "population.json",
        {
            "name": "heldout-full-namo-v1",
            "scenes": [
                {"xml_path": scene, "cluster_id": f"base_{index // 2}"}
                for index, scene in enumerate(SCENES)
            ],
        },
    )
    (tmp_path / "model.ckpt").write_bytes(b"checkpoint")
    (tmp_path / "config.yaml").write_text(
        "motion_primitives:\n  max_push_steps: 5\n",
        encoding="utf-8",
    )
    (tmp_path / "primitive_data").mkdir()
    (tmp_path / "primitive_data/1x_car_d5_fixture.dat").write_bytes(b"primitive")
    config = {
        "name": "full-namo-final",
        "population": "population.json",
        "run_root": "run",
        "model": {
            "name": "hy5u",
            "label": "Sage (Hybrid)",
            "checkpoint": "model.ckpt",
            "seed": 42,
        },
        "random_seeds": [101, 102, 103, 104, 105],
        "protocol": {
            "evaluation_seed": 42,
            "path_length": 2,
            "config_file": "config.yaml",
            "primitive_data_dir": "primitive_data",
            "primitive_prefix": "1x_car_d5_",
            "max_push_steps": 5,
            "simulation_budget_per_keyhole": 900,
            "region_max_chain_depth": 2,
            "goals_per_region": 100,
            "region_success_min_reachable": 20,
            "n_shards": 2,
            "slurm_partition": "main-redhat",
            "hardware_constraint": "icelake",
        },
        "analysis": {
            "bootstrap_replicates": 200,
            "bootstrap_seed": 7,
            "simulator_call_cutoffs": [2, 5, 10, 30],
            "wall_time_cutoffs_seconds": [1, 5, 30],
        },
    }
    path = tmp_path / "experiment.json"
    write_json(path, config)
    return path


def terminal_row(
    scene: str,
    *,
    solved: bool,
    calls: int,
    seconds: float,
) -> dict[str, object]:
    return {
        "xml_path": scene,
        "solved": solved,
        "simulation_budget_used_total": calls,
        "search_time_ms": 1000.0 * seconds,
        "timing_scope": "full_namo_planner_search",
        "path_length_n": 2,
        **({} if solved else {"failure_kind": "simulation_budget_exhausted"}),
    }


def write_aggregate(
    root: Path,
    solved_indices: set[int],
    *,
    call_offset: int = 0,
) -> None:
    solved_rows: list[dict[str, object]] = []
    unsolved_rows: list[dict[str, object]] = []
    for index, scene in enumerate(SCENES):
        solved = index in solved_indices
        row = terminal_row(
            scene,
            solved=solved,
            calls=call_offset + 2 * (index + 1),
            seconds=0.5 * (call_offset + index + 1),
        )
        (solved_rows if solved else unsolved_rows).append(row)
    write_jsonl(root / "solved.jsonl", solved_rows)
    write_jsonl(root / "unsolved.jsonl", unsolved_rows)


def write_run_config(root: Path, experiment: Experiment, arm: Arm) -> None:
    protocol = experiment.protocol
    write_json(
        root / "run_config.json",
        {
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
        },
    )
