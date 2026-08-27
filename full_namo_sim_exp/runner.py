from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from namo import solvability_runner as base


_ORDERING_SEED: int | None = None


def with_ordering_seed(config: Any, ordering_seed: int) -> Any:
    """Change search ordering without changing evaluation geometry randomness."""
    params = dict(config.algorithm_params or {})
    params["shuffle_seed"] = int(ordering_seed)
    config.algorithm_params = params
    return config


def build_planner_config(task: base.SolveTask) -> Any:
    config = base.build_full_namo_planner_config(task)
    if _ORDERING_SEED is not None:
        with_ordering_seed(config, _ORDERING_SEED)
    return config


def run_manifest_without_prefilter(
    *,
    repo_root: Path,
    input_dir: str | None,
    manifest_path: str | None,
    path_length: int,
    output_dir: str,
    config_file: str,
    goal_strategy: str,
    region_max_chain_depth: int,
    primitive_data_dir: str,
    primitive_prefix: str,
    rollout_samples_per_state: int | None,
    region_frontier_beam_width: int | None,
    region_success_min_reachable: int,
    goals_per_region: int,
    seed: int,
    use_cpp_snapshot: bool,
    simulation_budget: int,
    simulation_budget_scope: str,
    local_search: str,
    best_first_prior: str,
    region_selection_strategy: str,
    scorer_ckpt: str | None,
    ml_device: str,
    workers: int,
    full_namo_max_iterations: int | None,
    audit_next_keyhole_reachability: bool,
    preserve_next_keyhole_access: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Run every manifest scene; path_length is metadata, never a filter."""
    config_path = base._resolve_config_path(repo_root, config_file)
    base.require_canonical_runtime_config(config_path)
    primitive_root = Path(primitive_data_dir)
    if not primitive_root.is_absolute():
        primitive_root = repo_root / primitive_root
    primitive_data_dir_resolved = str(primitive_root.resolve())
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    max_push_steps = base.derive_max_push_steps(config_path)
    base.require_canonical_primitive_profile(primitive_prefix, max_push_steps)
    ordering_seed = seed if _ORDERING_SEED is None else _ORDERING_SEED
    run_config = {
        "input_dir": input_dir,
        "manifest": manifest_path,
        "path_length": int(path_length),
        "path_length_role": "declared_population_metadata_only",
        "config_file": config_path,
        "goal_strategy": goal_strategy,
        "region_max_chain_depth": int(region_max_chain_depth),
        "primitive_data_dir": primitive_data_dir_resolved,
        "primitive_prefix": primitive_prefix,
        "rollout_samples_per_state": rollout_samples_per_state,
        "region_frontier_beam_width": region_frontier_beam_width,
        "region_success_min_reachable": int(region_success_min_reachable),
        "goals_per_region": int(goals_per_region),
        "seed": int(seed),
        "evaluation_seed": int(seed),
        "ordering_seed": int(ordering_seed),
        "use_cpp_snapshot": bool(use_cpp_snapshot),
        "simulation_budget": int(simulation_budget),
        "simulation_budget_scope": simulation_budget_scope,
        "local_search": local_search,
        "best_first_prior": best_first_prior,
        "region_selection_strategy": region_selection_strategy,
        "scorer_ckpt": scorer_ckpt,
        "ml_device": ml_device,
        "workers": int(workers),
        "full_namo_max_iterations": full_namo_max_iterations,
        "audit_next_keyhole_reachability": bool(audit_next_keyhole_reachability),
        "preserve_next_keyhole_access": bool(preserve_next_keyhole_access),
        "max_push_steps": max_push_steps,
        "prefilter_applied": False,
    }
    base._write_json(output_root / "run_config.json", run_config)

    xml_files = base.get_xml_files(
        input_dir=input_dir,
        manifest_path=manifest_path,
        limit=limit,
    )
    base._write_selected_envs(output_root / "selected_envs.txt", xml_files)
    tasks = [
        base.SolveTask(
            xml_path=xml_path,
            path_length_n=path_length,
            config_path=config_path,
            goal_strategy=goal_strategy,
            region_max_chain_depth=region_max_chain_depth,
            primitive_data_dir=primitive_data_dir_resolved,
            primitive_prefix=primitive_prefix,
            rollout_samples_per_state=rollout_samples_per_state,
            region_frontier_beam_width=region_frontier_beam_width,
            region_success_min_reachable=region_success_min_reachable,
            goals_per_region=goals_per_region,
            seed=seed,
            use_cpp_snapshot=use_cpp_snapshot,
            simulation_budget=simulation_budget,
            simulation_budget_scope=simulation_budget_scope,
            local_search=local_search,
            best_first_prior=best_first_prior,
            region_selection_strategy=region_selection_strategy,
            scorer_ckpt=scorer_ckpt,
            ml_device=ml_device,
            full_namo_max_iterations=full_namo_max_iterations,
            max_push_steps=max_push_steps,
            audit_next_keyhole_reachability=audit_next_keyhole_reachability,
            preserve_next_keyhole_access=preserve_next_keyhole_access,
        )
        for xml_path in xml_files
    ]

    solved_rows: list[dict[str, Any]] = []
    unsolved_rows: list[dict[str, Any]] = []
    for result in base._iter_solve_results(tasks, workers):
        (solved_rows if result["kind"] == "solved" else unsolved_rows).append(
            result["row"]
        )
    solved_rows.sort(key=lambda row: row["xml_path"])
    unsolved_rows.sort(key=lambda row: row["xml_path"])
    base._write_jsonl(output_root / "solved.jsonl", solved_rows)
    base._write_jsonl(output_root / "unsolved.jsonl", unsolved_rows)
    summary = {
        "input_env_count": len(xml_files),
        "selected_env_count": len(xml_files),
        "solved_count": len(solved_rows),
        "unsolved_count": len(unsolved_rows),
        "simulation_budget_exhausted_count": sum(
            row.get("outcome") == "simulation_budget_exhausted" for row in unsolved_rows
        ),
        "selection_error_count": 0,
        "planner_failure_count": sum(
            row.get("outcome") == "planner_failure" for row in unsolved_rows
        ),
    }
    base._write_json(output_root / "summary.json", summary)
    return summary


def timed_solve_environment_task(task: base.SolveTask) -> dict[str, Any]:
    """Time the complete FullNAMOPlanner.search call for one scene."""
    try:
        env = base.namo_rl.RLEnvironment(task.xml_path, task.config_path, False)
        planner = base.FullNAMOPlanner(env, build_planner_config(task))
        robot_goal = base.extract_goal_from_xml(task.xml_path)
        start = time.perf_counter()
        result = planner.search(robot_goal)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)

        budget_stats = dict(result.algorithm_stats or {})
        budget_fields = {
            key: budget_stats[key]
            for key in (
                "simulation_budget_limit",
                "simulation_budget_used",
                "simulation_budget_remaining",
                "simulation_budget_scope",
                "simulation_budget_limit_per_keyhole",
                "simulation_budget_used_total",
                "simulation_budget_used_by_keyhole",
                "simulation_budget_keyholes_attempted",
            )
            if key in budget_stats
        }
        trace_fields = (
            {"iteration_trace": budget_stats["iteration_trace"]}
            if "iteration_trace" in budget_stats
            else {}
        )
        timing_fields = {
            "search_time_ms": elapsed_ms,
            "timing_scope": "full_namo_planner_search",
        }
        if result.success:
            actions = list(result.action_sequence or [])
            return {
                "kind": "solved",
                "row": {
                    "xml_path": task.xml_path,
                    "path_length_n": task.path_length_n,
                    "solved": True,
                    "solution_length": len(actions),
                    "solution": [base.serialize_action(action) for action in actions],
                    **budget_fields,
                    **trace_fields,
                    **timing_fields,
                },
            }
        failure_kind = str(budget_stats.get("failure_kind") or "")
        outcome = (
            "simulation_budget_exhausted"
            if failure_kind == "simulation_budget_exhausted"
            else "planner_failure"
        )
        return {
            "kind": "unsolved",
            "row": {
                "xml_path": task.xml_path,
                "path_length_n": task.path_length_n,
                "solved": False,
                "outcome": outcome,
                "failure_kind": failure_kind or None,
                "failure_subkind": budget_stats.get("failure_subkind"),
                "error_message": result.error_message or None,
                **budget_fields,
                **trace_fields,
                **timing_fields,
            },
        }
    except Exception as exc:
        return {
            "kind": "unsolved",
            "row": {
                "xml_path": task.xml_path,
                "path_length_n": task.path_length_n,
                "solved": False,
                "outcome": "planner_failure",
                "failure_kind": "runner_exception",
                "failure_subkind": None,
                "error_message": str(exc),
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    global _ORDERING_SEED
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--ordering-seed", type=int, required=True)
    custom_args, base_argv = custom.parse_known_args(raw_argv)
    base.build_arg_parser().parse_args(base_argv)
    _ORDERING_SEED = custom_args.ordering_seed
    base.solve_environment_task = timed_solve_environment_task
    base.run_exact_n_solvability = run_manifest_without_prefilter
    return base.cli_main(base_argv)


if __name__ == "__main__":
    raise SystemExit(main())
