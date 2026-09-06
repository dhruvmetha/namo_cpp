"""Exact-n full-pipeline solvability runner for Full NAMO."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import namo_rl
import yaml

from namo.core import PlannerConfig
from namo.core.xml_goal_parser import extract_goal_from_xml
from namo.environment_selection import (
    RegionPathAnalysis,
    analyze_environment_path_length,
    get_xml_files,
)
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from namo.planners.utils import PushAttemptBudget
from namo.runtime_profile import (
    CANONICAL_CONFIG,
    CANONICAL_NUM_DEPTHS,
    CANONICAL_PRIMITIVE_PREFIX,
    require_canonical_primitive_profile,
    require_canonical_runtime_config,
)


DEFAULT_CONFIG = CANONICAL_CONFIG
DEFAULT_GOAL_STRATEGY = "random_rollout"
DEFAULT_GOALS_PER_REGION = 100
DEFAULT_REGION_MAX_CHAIN_DEPTH = 2
DEFAULT_REGION_SUCCESS_MIN_REACHABLE = 1
DEFAULT_SIMULATION_BUDGET = 100_000
DEFAULT_SEED = 42
@dataclass(frozen=True)
class SolveTask:
    xml_path: str
    path_length_n: int
    config_path: str
    goal_strategy: str
    region_max_chain_depth: int
    primitive_data_dir: str
    primitive_prefix: str
    rollout_samples_per_state: Optional[int]
    region_frontier_beam_width: Optional[int]
    region_success_min_reachable: int
    goals_per_region: int
    seed: int
    use_cpp_snapshot: bool
    simulation_budget: int
    simulation_budget_scope: str = "full_problem"
    local_search: str = "region_bfs"
    best_first_prior: str = "model"
    region_selection_strategy: str = "ml_first"
    scorer_ckpt: Optional[str] = None
    ml_device: str = "cpu"
    full_namo_max_iterations: Optional[int] = None
    max_push_steps: Optional[int] = None
    audit_next_keyhole_reachability: bool = False
    preserve_next_keyhole_access: bool = False
    shuffle_seed: Optional[int] = None


def _load_namo_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def derive_max_push_steps(config_path: str) -> Optional[int]:
    cfg = _load_namo_config(config_path)
    value = (cfg.get("motion_primitives", {}) or {}).get("max_push_steps")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def serialize_action(action: Any) -> Dict[str, Any]:
    return {
        "object_id": str(getattr(action, "object_id", "")),
        "edge_idx": int(getattr(action, "edge_idx", -1)),
        "depth": int(getattr(action, "depth", -1)),
        "target": [
            float(getattr(action, "x", 0.0)),
            float(getattr(action, "y", 0.0)),
            float(getattr(action, "theta", 0.0)),
        ],
    }


def build_full_namo_planner_config(task: SolveTask) -> PlannerConfig:
    push_budget = PushAttemptBudget(limit=task.simulation_budget)
    algorithm_params: Dict[str, Any] = {
        "goal_strategy": task.goal_strategy,
        "primitive_data_dir": task.primitive_data_dir,
        "primitive_prefix": task.primitive_prefix,
        "xml_file": task.xml_path,
        "namo_config_path": task.config_path,
        "region_max_chain_depth": task.region_max_chain_depth,
        "region_success_min_reachable": task.region_success_min_reachable,
        "region_use_cpp_unified_wavefront": task.use_cpp_snapshot,
        "full_namo_use_cpp_unified_wavefront": task.use_cpp_snapshot,
        "region_snapshot_seed": task.seed,
        "shuffle_seed": task.seed if task.shuffle_seed is None else task.shuffle_seed,
        "ml_seed": task.seed,
        "push_budget": push_budget,
        "full_namo_budget_scope": task.simulation_budget_scope,
        "full_namo_keyhole_simulation_budget": task.simulation_budget,
        "full_namo_local_search": task.local_search,
        "best_first_prior": task.best_first_prior,
        "best_first_hmax": task.region_max_chain_depth,
        "best_first_agg": "mean5",
        "best_first_combine": "q",
        "best_first_raw": True,
        "full_namo_audit_next_keyhole_reachability": task.audit_next_keyhole_reachability,
        "full_namo_preserve_next_keyhole_access": task.preserve_next_keyhole_access,
        "region_selection_strategy": task.region_selection_strategy,
        "ml_device": task.ml_device,
    }
    if task.scorer_ckpt is not None:
        algorithm_params["scorer_ckpt"] = task.scorer_ckpt
    if task.rollout_samples_per_state is not None:
        algorithm_params["rollout_samples_per_state"] = int(task.rollout_samples_per_state)
    if task.region_frontier_beam_width is not None:
        algorithm_params["region_frontier_beam_width"] = int(task.region_frontier_beam_width)
    if task.full_namo_max_iterations is not None:
        algorithm_params["full_namo_max_iterations"] = int(task.full_namo_max_iterations)
    if task.max_push_steps is not None:
        algorithm_params["max_push_steps"] = int(task.max_push_steps)

    return PlannerConfig(
        goals_per_region=task.goals_per_region,
        random_seed=task.seed,
        verbose=False,
        collect_stats=True,
        algorithm_params=algorithm_params,
    )


def solve_environment_task(task: SolveTask) -> Dict[str, Any]:
    try:
        env = namo_rl.RLEnvironment(task.xml_path, task.config_path, False)
        planner = FullNAMOPlanner(env, build_full_namo_planner_config(task))
        robot_goal = extract_goal_from_xml(task.xml_path)
        result = planner.search(robot_goal)

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
        # Per-iteration trace: the only record of whether a run committed a keyhole before
        # failing, which the terminal failure_kind alone cannot distinguish.
        trace_fields = (
            {"iteration_trace": budget_stats["iteration_trace"]}
            if "iteration_trace" in budget_stats
            else {}
        )

        if result.success:
            action_sequence = list(result.action_sequence or [])
            return {
                "kind": "solved",
                "row": {
                    "xml_path": task.xml_path,
                    "path_length_n": task.path_length_n,
                    "solution_length": len(action_sequence),
                    "solution": [serialize_action(action) for action in action_sequence],
                    **budget_fields,
                    **trace_fields,
                },
            }

        failure_kind = str((result.algorithm_stats or {}).get("failure_kind") or "")
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
                "outcome": outcome,
                "failure_kind": failure_kind or None,
                "failure_subkind": (result.algorithm_stats or {}).get("failure_subkind"),
                "error_message": result.error_message or None,
                **budget_fields,
                **trace_fields,
            },
        }
    except Exception as exc:
        return {
            "kind": "unsolved",
            "row": {
                "xml_path": task.xml_path,
                "path_length_n": task.path_length_n,
                "outcome": "planner_failure",
                "failure_kind": "runner_exception",
                "failure_subkind": None,
                "error_message": str(exc),
            },
        }


def _write_json(path: Path, payload: Dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_selected_envs(path: Path, xml_paths: Sequence[str]):
    content = "\n".join(xml_paths)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _build_task(
    analysis: RegionPathAnalysis,
    *,
    config_path: str,
    goal_strategy: str,
    region_max_chain_depth: int,
    primitive_data_dir: str,
    primitive_prefix: str,
    rollout_samples_per_state: Optional[int],
    region_frontier_beam_width: Optional[int],
    region_success_min_reachable: int,
    goals_per_region: int,
    seed: int,
    use_cpp_snapshot: bool,
    simulation_budget: int,
    simulation_budget_scope: str,
    local_search: str,
    best_first_prior: str,
    region_selection_strategy: str,
    scorer_ckpt: Optional[str],
    ml_device: str,
    full_namo_max_iterations: Optional[int],
    max_push_steps: Optional[int],
    audit_next_keyhole_reachability: bool,
    preserve_next_keyhole_access: bool,
    shuffle_seed: Optional[int],
) -> SolveTask:
    return SolveTask(
        xml_path=analysis.xml_path,
        path_length_n=analysis.path_length_n,
        config_path=config_path,
        goal_strategy=goal_strategy,
        region_max_chain_depth=region_max_chain_depth,
        primitive_data_dir=primitive_data_dir,
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
        shuffle_seed=shuffle_seed,
    )


def _resolve_config_path(repo_root: Path, config_file: str) -> str:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = repo_root / config_file
    return str(config_path.resolve())


def _iter_solve_results(tasks: Sequence[SolveTask], workers: int) -> Iterable[Dict[str, Any]]:
    if workers <= 1:
        for task in tasks:
            yield solve_environment_task(task)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(solve_environment_task, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def run_exact_n_solvability(
    *,
    repo_root: Path,
    input_dir: Optional[str],
    manifest_path: Optional[str],
    path_length: int,
    output_dir: str,
    config_file: str = DEFAULT_CONFIG,
    goal_strategy: str = DEFAULT_GOAL_STRATEGY,
    region_max_chain_depth: int = DEFAULT_REGION_MAX_CHAIN_DEPTH,
    primitive_data_dir: str = "data",
    primitive_prefix: str = CANONICAL_PRIMITIVE_PREFIX,
    rollout_samples_per_state: Optional[int] = None,
    region_frontier_beam_width: Optional[int] = None,
    region_success_min_reachable: int = DEFAULT_REGION_SUCCESS_MIN_REACHABLE,
    goals_per_region: int = DEFAULT_GOALS_PER_REGION,
    seed: int = DEFAULT_SEED,
    use_cpp_snapshot: bool = True,
    simulation_budget: int = DEFAULT_SIMULATION_BUDGET,
    simulation_budget_scope: str = "full_problem",
    local_search: str = "region_bfs",
    best_first_prior: str = "model",
    region_selection_strategy: str = "ml_first",
    scorer_ckpt: Optional[str] = None,
    ml_device: str = "cpu",
    workers: int = 1,
    full_namo_max_iterations: Optional[int] = None,
    audit_next_keyhole_reachability: bool = False,
    preserve_next_keyhole_access: bool = False,
    shuffle_seed: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    config_path = _resolve_config_path(repo_root, config_file)
    require_canonical_runtime_config(config_path)
    primitive_root = Path(primitive_data_dir)
    if not primitive_root.is_absolute():
        primitive_root = repo_root / primitive_data_dir
    primitive_data_dir_resolved = str(primitive_root.resolve())
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    effective_max_push_steps = derive_max_push_steps(config_path)
    require_canonical_primitive_profile(primitive_prefix, effective_max_push_steps)
    effective_primitive_prefix = primitive_prefix

    run_config = {
        "input_dir": input_dir,
        "manifest": manifest_path,
        "path_length": int(path_length),
        "config_file": config_path,
        "goal_strategy": goal_strategy,
        "region_max_chain_depth": int(region_max_chain_depth),
        "primitive_data_dir": primitive_data_dir_resolved,
        "primitive_prefix": effective_primitive_prefix,
        "rollout_samples_per_state": rollout_samples_per_state,
        "region_frontier_beam_width": region_frontier_beam_width,
        "region_success_min_reachable": int(region_success_min_reachable),
        "goals_per_region": int(goals_per_region),
        "seed": int(seed),
        "shuffle_seed": int(seed if shuffle_seed is None else shuffle_seed),
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
        "max_push_steps": effective_max_push_steps,
    }
    _write_json(output_root / "run_config.json", run_config)

    xml_files = get_xml_files(input_dir=input_dir, manifest_path=manifest_path, limit=limit)

    selected_analyses: List[RegionPathAnalysis] = []
    unsolved_rows: List[Dict[str, Any]] = []
    for xml_path in xml_files:
        analysis = analyze_environment_path_length(
            xml_path,
            config_path,
            use_cpp_snapshot=use_cpp_snapshot,
        )
        if analysis.selection_error:
            unsolved_rows.append(
                {
                    "xml_path": analysis.xml_path,
                    "path_length_n": analysis.path_length_n,
                    "outcome": "selection_error",
                    "failure_kind": "selection_error",
                    "failure_subkind": None,
                    "error_message": analysis.selection_error,
                }
            )
            continue
        if analysis.path_length_n == path_length:
            selected_analyses.append(analysis)

    selected_xmls = [analysis.xml_path for analysis in selected_analyses]
    _write_selected_envs(output_root / "selected_envs.txt", selected_xmls)

    tasks = [
        _build_task(
            analysis,
            config_path=config_path,
            goal_strategy=goal_strategy,
            region_max_chain_depth=region_max_chain_depth,
            primitive_data_dir=primitive_data_dir_resolved,
            primitive_prefix=effective_primitive_prefix,
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
            max_push_steps=effective_max_push_steps,
            audit_next_keyhole_reachability=audit_next_keyhole_reachability,
            preserve_next_keyhole_access=preserve_next_keyhole_access,
            shuffle_seed=shuffle_seed,
        )
        for analysis in selected_analyses
    ]

    solved_rows: List[Dict[str, Any]] = []
    for result in _iter_solve_results(tasks, workers):
        if result["kind"] == "solved":
            solved_rows.append(result["row"])
        else:
            unsolved_rows.append(result["row"])

    solved_rows.sort(key=lambda row: row["xml_path"])
    unsolved_rows.sort(key=lambda row: row["xml_path"])
    _write_jsonl(output_root / "solved.jsonl", solved_rows)
    _write_jsonl(output_root / "unsolved.jsonl", unsolved_rows)

    summary = {
        "input_env_count": len(xml_files),
        "selected_env_count": len(selected_xmls),
        "solved_count": len(solved_rows),
        "unsolved_count": len(unsolved_rows),
        "simulation_budget_exhausted_count": sum(
            1 for row in unsolved_rows if row.get("outcome") == "simulation_budget_exhausted"
        ),
        "selection_error_count": sum(
            1 for row in unsolved_rows if row.get("outcome") == "selection_error"
        ),
        "planner_failure_count": sum(
            1 for row in unsolved_rows if row.get("outcome") == "planner_failure"
        ),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-n Full NAMO solvability evaluation with a full-problem or per-keyhole push budget."
    )
    parser.add_argument("--xml-dir", type=str, help="Directory containing XML environments")
    parser.add_argument("--manifest", type=str, help="Manifest listing XML environments")
    parser.add_argument("--path-length", type=int, required=True, help="Exact initial shortest region-path length")
    parser.add_argument(
        "--config-file",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"NAMO config file path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--goal-strategy",
        type=str,
        default=DEFAULT_GOAL_STRATEGY,
        help=f"Region-opening goal strategy (default: {DEFAULT_GOAL_STRATEGY})",
    )
    parser.add_argument(
        "--region-max-chain-depth",
        type=int,
        default=DEFAULT_REGION_MAX_CHAIN_DEPTH,
        help=f"Max local push-chain depth (default: {DEFAULT_REGION_MAX_CHAIN_DEPTH})",
    )
    parser.add_argument(
        "--region-allow-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow local opening search to continue after collisions (default: true)",
    )
    parser.add_argument(
        "--primitive-data-dir",
        type=str,
        default="data",
        help="Directory containing primitive data files (default: data)",
    )
    parser.add_argument(
        "--primitive-prefix",
        type=str,
        default=CANONICAL_PRIMITIVE_PREFIX,
        choices=[CANONICAL_PRIMITIVE_PREFIX],
        help="Canonical car 1x d5 primitive prefix",
    )
    parser.add_argument(
        "--rollout-samples-per-state",
        type=int,
        default=None,
        help="Optional cap on sampled rollout goals per state",
    )
    parser.add_argument(
        "--region-frontier-beam-width",
        type=int,
        default=None,
        help="Optional local-search frontier beam width",
    )
    parser.add_argument(
        "--region-success-min-reachable",
        type=int,
        default=DEFAULT_REGION_SUCCESS_MIN_REACHABLE,
        help=f"Minimum reachable sampled goals required to count an opening (default: {DEFAULT_REGION_SUCCESS_MIN_REACHABLE})",
    )
    parser.add_argument(
        "--goals-per-region",
        type=int,
        default=DEFAULT_GOALS_PER_REGION,
        help=f"Number of sampled goals per region for opening validation (default: {DEFAULT_GOALS_PER_REGION})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional push-ordering seed; defaults to --seed",
    )
    parser.add_argument(
        "--use-cpp-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the unified C++ snapshot backend for both selection and planning (default: true)",
    )
    parser.add_argument(
        "--simulation-budget",
        type=int,
        default=DEFAULT_SIMULATION_BUDGET,
        help=f"Max env.step(...) calls for the selected budget scope (default: {DEFAULT_SIMULATION_BUDGET})",
    )
    parser.add_argument(
        "--simulation-budget-scope",
        choices=("full_problem", "keyhole"),
        default="full_problem",
        help="Reset the simulation budget for every targeted boundary when set to keyhole",
    )
    parser.add_argument(
        "--local-search",
        choices=("region_bfs", "best_first"),
        default="region_bfs",
        help="Local keyhole search loop; best_first reuses scripts/sandbox/eval_bestfirst.py",
    )
    parser.add_argument(
        "--best-first-prior",
        choices=("model", "uniform"),
        default="model",
        help="Priority source for --local-search best_first",
    )
    parser.add_argument(
        "--region-selection-strategy",
        choices=("ml_first", "cost_first"),
        default="ml_first",
        help="How the local opener orders frontier states (default: ml_first)",
    )
    parser.add_argument(
        "--scorer-ckpt",
        type=str,
        default=None,
        help="Checkpoint used by model-ranked best-first search or --goal-strategy scorer",
    )
    parser.add_argument(
        "--ml-device",
        type=str,
        default="cpu",
        help="Device for scorer inference (default: cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel environment solves (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for run manifests",
    )
    parser.add_argument(
        "--full-namo-max-iterations",
        type=int,
        default=None,
        help="Optional outer Full NAMO iteration cap; omitted by default",
    )
    parser.add_argument(
        "--audit-next-keyhole-reachability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For every committed opening with another keyhole ahead, compare that next "
            "keyhole's middle-region contact set before the opening with its reachable "
            "contact set afterward"
        ),
    )
    parser.add_argument(
        "--preserve-next-keyhole-access",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep searching after a locally open first keyhole unless every contact edge "
            "that was reachable on the next blocker remains reachable"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional input-environment limit for debugging",
    )
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.xml_dir and not args.manifest:
        parser.error("must provide either --xml-dir or --manifest")
    if args.xml_dir and args.manifest:
        parser.error("provide only one of --xml-dir or --manifest")

    repo_root = Path(__file__).resolve().parents[2]
    run_exact_n_solvability(
        repo_root=repo_root,
        input_dir=args.xml_dir,
        manifest_path=args.manifest,
        path_length=args.path_length,
        output_dir=args.output_dir,
        config_file=args.config_file,
        goal_strategy=args.goal_strategy,
        region_max_chain_depth=args.region_max_chain_depth,
        primitive_data_dir=args.primitive_data_dir,
        primitive_prefix=args.primitive_prefix,
        rollout_samples_per_state=args.rollout_samples_per_state,
        region_frontier_beam_width=args.region_frontier_beam_width,
        region_success_min_reachable=args.region_success_min_reachable,
        goals_per_region=args.goals_per_region,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        use_cpp_snapshot=args.use_cpp_snapshot,
        simulation_budget=args.simulation_budget,
        simulation_budget_scope=args.simulation_budget_scope,
        local_search=args.local_search,
        best_first_prior=args.best_first_prior,
        region_selection_strategy=args.region_selection_strategy,
        scorer_ckpt=args.scorer_ckpt,
        ml_device=args.ml_device,
        workers=args.workers,
        full_namo_max_iterations=args.full_namo_max_iterations,
        audit_next_keyhole_reachability=args.audit_next_keyhole_reachability,
        preserve_next_keyhole_access=args.preserve_next_keyhole_access,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
