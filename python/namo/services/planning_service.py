"""Stable XML-based planning facade for external NAMO consumers.

The service owns simulator and planner construction. Callers such as
``robot_control`` remain responsible for observation conversion, coordinate
mapping, and physical execution policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import namo_rl

from namo.core import PlannerConfig, PlannerResult


_ML_GOAL_STRATEGIES = frozenset(
    {
        "ml",
        "ml_primitive",
        "ml_async",
        "ml_primitive_async",
        "ml_driven_async",
    }
)
_MOTION_PRIMITIVE_FILENAME_MARKER = "motion_primitives_"


def _create_planner(name: str, env: Any, config: PlannerConfig) -> Any:
    """Create a registered planner without importing planner internals eagerly."""
    from namo.core import PlannerFactory
    import namo.planners  # noqa: F401 - registers planner implementations

    return PlannerFactory.create_planner(name, env, config)


@dataclass(frozen=True)
class NAMOAction:
    """One executable NAMO push in simulator object coordinates."""

    object_id: str
    edge_idx: int
    depth: int


@dataclass
class NAMOPlanResult:
    """External planning result with executable actions and diagnostics."""

    success: bool
    actions: List[NAMOAction] = field(default_factory=list)
    search_time_ms: float = 0.0
    error_message: str = ""
    algorithm_stats: Optional[Dict[str, Any]] = None


class NAMOPlanningService:
    """Construct NAMO environments and invoke registered planners from XML."""

    def __init__(
        self,
        config_path: str,
        primitive_data_dir: str = "data",
        verbose: bool = False,
        enable_viewer: bool = False,
        pause_after_load: bool = False,
    ) -> None:
        self._config_path = config_path
        self._primitive_data_dir = primitive_data_dir
        self._verbose = verbose
        self._enable_viewer = enable_viewer
        self._pause_after_load = pause_after_load
        self._parsed_namo_config: Optional[Dict[str, Any]] = None
        self._cached_goal_model: Optional[Any] = None
        self._cached_goal_model_signature: Optional[Tuple[Any, ...]] = None

    def _create_environment(
        self,
        xml_path: str,
        starting_robot_pose: Optional[Tuple[float, float, float]],
    ) -> Any:
        """Load an environment, placing a freejoint robot before warm-up."""
        defer_warmup = starting_robot_pose is not None
        env = namo_rl.RLEnvironment(
            xml_path,
            self._config_path,
            self._enable_viewer,
            defer_warmup,
        )
        if starting_robot_pose is not None:
            env.set_robot_pose(*starting_robot_pose)
            env.warm_up()
        return env

    def _load_namo_config(self) -> Dict[str, Any]:
        """Load and cache the NAMO YAML used by both planning layers."""
        if self._parsed_namo_config is None:
            import yaml

            try:
                with open(self._config_path, encoding="utf-8") as config_file:
                    self._parsed_namo_config = yaml.safe_load(config_file) or {}
            except (OSError, yaml.YAMLError):
                self._parsed_namo_config = {}
        return self._parsed_namo_config

    def _derive_primitive_prefix(self) -> str:
        """Derive the Python primitive prefix from the C++ config filename."""
        config = self._load_namo_config()
        primitive_file = (config.get("system", {}) or {}).get(
            "motion_primitives_file"
        )
        if not primitive_file:
            return ""

        stem = Path(str(primitive_file)).stem
        if not stem.startswith(_MOTION_PRIMITIVE_FILENAME_MARKER):
            return ""
        variant = stem[len(_MOTION_PRIMITIVE_FILENAME_MARKER) :]
        if not variant:
            return ""

        prefix = f"{variant}_"
        sentinel = Path(self._primitive_data_dir) / (
            f"{prefix}motion_primitives_15_square.dat"
        )
        return prefix if sentinel.exists() else ""

    def _max_push_steps_from_config(self) -> Optional[int]:
        """Return the configured primitive push-step cap when present."""
        value = (self._load_namo_config().get("motion_primitives", {}) or {}).get(
            "max_push_steps"
        )
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_or_load_goal_model(
        self,
        goal_strategy: str,
        algorithm_params: Dict[str, Any],
    ) -> Optional[Any]:
        """Load one goal model per unique path/device/sampler configuration."""
        if goal_strategy.lower() not in _ML_GOAL_STRATEGIES:
            return None

        model_path = algorithm_params.get("ml_goal_model_path")
        if not model_path:
            return None

        device = algorithm_params.get("ml_device", "cuda")
        sampler_method = algorithm_params.get("ml_sampler_method")
        num_steps = algorithm_params.get("ml_num_steps")
        signature = (
            str(Path(model_path).expanduser().resolve(strict=False)),
            str(device),
            sampler_method,
            None if num_steps is None else int(num_steps),
        )
        if (
            self._cached_goal_model is not None
            and signature == self._cached_goal_model_signature
        ):
            return self._cached_goal_model

        from sage_learning.goal_inference_model import GoalInferenceModel

        load_start = time.perf_counter()
        model = GoalInferenceModel(
            model_path=model_path,
            device=device,
            sampler_method=sampler_method,
            num_steps=num_steps,
            namo_config_path=self._config_path,
        )
        self._cached_goal_model = model
        self._cached_goal_model_signature = signature
        if self._verbose:
            elapsed_ms = (time.perf_counter() - load_start) * 1000.0
            print(
                f"[NAMOPlanningService] loaded goal model in {elapsed_ms:.0f}ms",
                flush=True,
            )
        self._warmup_goal_model(model, algorithm_params)
        return model

    def _warmup_goal_model(
        self,
        model: Any,
        algorithm_params: Dict[str, Any],
    ) -> None:
        """Warm supported goal models so replanning excludes compile latency."""
        if not hasattr(model, "warmup"):
            return
        model.warmup(
            samples=int(algorithm_params.get("ml_samples", 32)),
            num_steps=int(algorithm_params.get("ml_num_steps") or 20),
            repeats=3,
        )

    def preload_goal_model(self, goal_strategy: str, **kwargs: Any) -> None:
        """Eagerly load the model needed by an ML goal strategy."""
        self._get_or_load_goal_model(goal_strategy, kwargs)

    def analyze_reachability_from_xml(
        self,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
        analysis_mode: bool = False,
        starting_robot_pose: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Return the C++ wavefront reachability summary for one XML state."""
        start_time = time.perf_counter()
        try:
            env = self._create_environment(xml_path, starting_robot_pose)
            env.set_robot_goal(*robot_goal)
            summary = dict(env.get_reachability_summary(analysis_mode))
            summary["compute_time_ms"] = (
                time.perf_counter() - start_time
            ) * 1000.0
            return summary
        except Exception as exc:
            return {
                "goal_reachable": False,
                "analysis_mode": analysis_mode,
                "objects": {},
                "compute_time_ms": (time.perf_counter() - start_time) * 1000.0,
                "error_message": f"Reachability failed for {xml_path}: {exc}",
            }

    def plan_from_xml(
        self,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
        algorithm: str = "full_namo",
        goal_strategy: str = "primitive",
        max_chain_depth: int = 1,
        max_solutions_per_neighbor: int = 1,
        timeout_per_neighbour_sec: Optional[float] = None,
        allow_collisions: bool = True,
        frontier_beam_width: int = 10000,
        chain_link_cost: int = 11,
        selection_strategy: str = "cost_first",
        goals_per_region: int = 10,
        starting_robot_pose: Optional[Tuple[float, float, float]] = None,
        **kwargs: Any,
    ) -> NAMOPlanResult:
        """Run a registered NAMO planner against an XML environment."""
        start_time = time.perf_counter()
        try:
            env = self._create_environment(xml_path, starting_robot_pose)
            if self._pause_after_load and self._enable_viewer:
                self._pause_with_viewer(env, xml_path, robot_goal)

            algorithm_params: Dict[str, Any] = {
                "primitive_data_dir": self._primitive_data_dir,
                "goal_strategy": goal_strategy,
                "xml_file": xml_path,
                "namo_config_path": self._config_path,
                "region_max_chain_depth": max_chain_depth,
                "region_max_solutions_per_neighbor": max_solutions_per_neighbor,
                "region_max_recorded_solutions_per_neighbor": (
                    max_solutions_per_neighbor
                ),
                "region_allow_collisions": allow_collisions,
                "region_frontier_beam_width": frontier_beam_width,
                "region_chain_link_cost": chain_link_cost,
                "region_selection_strategy": selection_strategy,
                "region_ml_ignore_blacklist": False,
            }
            if timeout_per_neighbour_sec is not None:
                algorithm_params["region_timeout_per_neighbour_sec"] = (
                    timeout_per_neighbour_sec
                )
            algorithm_params.update(kwargs)

            if "primitive_prefix" not in algorithm_params:
                prefix = self._derive_primitive_prefix()
                if prefix:
                    algorithm_params["primitive_prefix"] = prefix
            if "max_push_steps" not in algorithm_params:
                max_push_steps = self._max_push_steps_from_config()
                if max_push_steps is not None:
                    algorithm_params["max_push_steps"] = max_push_steps

            goal_model = self._get_or_load_goal_model(
                goal_strategy,
                algorithm_params,
            )
            if goal_model is not None:
                algorithm_params["preloaded_goal_model"] = goal_model

            config = PlannerConfig(
                verbose=self._verbose,
                goals_per_region=goals_per_region,
                algorithm_params=algorithm_params,
            )
            planner = _create_planner(algorithm, env, config)
            env.set_robot_goal(*robot_goal)
            result = planner.search(robot_goal)
            actions = self._extract_actions(result)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if self._verbose:
                print(
                    f"[NAMOPlanningService] total={elapsed_ms:.0f}ms "
                    f"actions={len(actions)} success={bool(result.success)}",
                    flush=True,
                )
            return NAMOPlanResult(
                success=bool(result.success) and bool(actions),
                actions=actions,
                search_time_ms=elapsed_ms,
                error_message=str(getattr(result, "error_message", "") or ""),
                algorithm_stats=getattr(result, "algorithm_stats", None),
            )
        except Exception as exc:
            return NAMOPlanResult(
                success=False,
                search_time_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=f"Planning failed for {xml_path}: {exc}",
            )

    @staticmethod
    def _extract_actions(result: PlannerResult) -> List[NAMOAction]:
        """Return only planner actions carrying an executable edge and depth."""
        actions: List[NAMOAction] = []
        for action in result.action_sequence or []:
            edge_idx = int(getattr(action, "edge_idx", -1))
            depth = int(getattr(action, "depth", -1))
            if edge_idx < 0 or depth < 0:
                continue
            actions.append(
                NAMOAction(
                    object_id=str(action.object_id),
                    edge_idx=edge_idx,
                    depth=depth,
                )
            )
        return actions

    @staticmethod
    def _pause_with_viewer(
        env: Any,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
    ) -> None:
        """Keep the viewer responsive until the operator presses Enter."""
        import select
        import sys

        if sys.platform == "win32":
            input(f"Environment {xml_path} loaded for goal {robot_goal}; press Enter")
            return

        print(
            f"Environment {xml_path} loaded for goal {robot_goal}; press Enter",
            flush=True,
        )
        while True:
            env.render()
            if select.select([sys.stdin], [], [], 0.03)[0]:
                sys.stdin.readline()
                return
