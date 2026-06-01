"""NAMO planning service for external integration.

Wraps RegionOpeningPlanner with a clean interface for use by robot_control.
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import namo_rl

from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
import namo.planners  # noqa: F401  — triggers PlannerFactory registration for all algorithms


@dataclass
class NAMOAction:
    """A single NAMO push action."""

    object_id: str  # e.g., "obstacle_1_movable"
    edge_idx: int  # 0-59 (points_per_face * 4 faces)
    depth: int  # 0-9 (push_steps = depth + 1)


@dataclass
class NAMOPlanResult:
    """Result from NAMO planning."""

    success: bool
    actions: List[NAMOAction] = field(default_factory=list)
    search_time_ms: float = 0.0
    error_message: str = ""
    algorithm_stats: Optional[Dict[str, Any]] = None


class NAMOPlanningService:
    """Service for NAMO planning from XML environment files.

    Wraps RegionOpeningPlanner with a simple interface suitable for
    integration with robot_control.

    Usage:
        service = NAMOPlanningService("config/namo_config_complete.yaml")
        result = service.plan_from_xml(
            xml_path="/tmp/env.xml",
            robot_goal=(3.0, 2.4, 0.0),  # simulation coordinates (meters)
            algorithm="full_namo",
        )
        if result.success:
            for action in result.actions:
                print(f"Push {action.object_id} edge={action.edge_idx} depth={action.depth}")
    """

    def __init__(
        self,
        config_path: str,
        primitive_data_dir: str = "data",
        verbose: bool = False,
        enable_viewer: bool = False,
        pause_after_load: bool = False,
    ):
        """Initialize the planning service.

        Args:
            config_path: Path to NAMO config YAML (e.g., namo_config_complete.yaml)
            primitive_data_dir: Directory containing motion primitive data
            verbose: Enable verbose logging
            enable_viewer: Enable MuJoCo visualization window
            pause_after_load: Pause for user input after loading XML (for inspection)
        """
        self._config_path = config_path
        self._primitive_data_dir = primitive_data_dir
        self._verbose = verbose
        self._enable_viewer = enable_viewer
        self._pause_after_load = pause_after_load
        self._cached_goal_model = None
        self._cached_goal_model_signature = None

    # Strategy names that use a GoalInferenceModel
    _ML_GOAL_STRATEGIES = frozenset({
        "ml", "ml_primitive",
        "ml_fallback", "ml_primitive_fallback",
        "ml_async", "ml_primitive_async",
        "ml_driven_async",
    })

    def _get_or_load_goal_model(self, goal_strategy: str, algo_params: Dict[str, Any]):
        """Return a cached GoalInferenceModel, loading it on first call.

        Returns None when *goal_strategy* does not need an ML model.
        """
        if goal_strategy.lower() not in self._ML_GOAL_STRATEGIES:
            return None

        model_path = algo_params.get("ml_goal_model_path")
        if not model_path:
            return None

        device = algo_params.get("ml_device", "cuda")
        sampler_method = algo_params.get("ml_sampler_method")
        num_steps = algo_params.get("ml_num_steps")
        requested_signature = (
            str(Path(model_path).expanduser().resolve(strict=False)),
            str(device),
            sampler_method,
            None if num_steps is None else int(num_steps),
        )

        if (
            self._cached_goal_model is not None
            and self._cached_goal_model_signature == requested_signature
        ):
            return self._cached_goal_model

        self._cached_goal_model = None
        self._cached_goal_model_signature = None
        try:
            from sage_learning.goal_inference_model import GoalInferenceModel

            print(f"Loading GoalInferenceModel from {model_path} (device={device})...")
            load_start = time.time()
            self._cached_goal_model = GoalInferenceModel(
                model_path=model_path,
                device=device,
                sampler_method=sampler_method,
                num_steps=num_steps,
                namo_config_path=self._config_path,
            )
            self._cached_goal_model_signature = requested_signature
            load_ms = (time.time() - load_start) * 1000
            print(f"GoalInferenceModel cached in NAMOPlanningService ({load_ms:.0f}ms)")

            # Warmup: run dummy inferences to compile CUDA kernels
            self._warmup_goal_model(algo_params, device)
        except Exception as e:
            print(f"Failed to load GoalInferenceModel: {e}")
            self._cached_goal_model = None
            self._cached_goal_model_signature = None
            return None

        return self._cached_goal_model

    def _warmup_goal_model(self, algo_params: Dict[str, Any], device: str) -> None:
        """Run dummy inferences to compile CUDA kernels."""
        try:
            model = self._cached_goal_model
            num_samples = algo_params.get("ml_samples", 32)
            num_steps = algo_params.get("ml_num_steps") or 20

            print(f"Warming up goal model ({num_samples} samples x 3 runs)...")
            warmup_start = time.time()

            if model is not None and hasattr(model, "warmup"):
                model.warmup(
                    samples=num_samples,
                    num_steps=num_steps,
                    repeats=3,
                )

            warmup_ms = (time.time() - warmup_start) * 1000
            print(f"Goal model warmed up ({warmup_ms:.0f}ms)")
        except Exception as e:
            print(f"Warmup failed (will warmup on first real inference): {e}")

    def preload_goal_model(self, goal_strategy: str, **kwargs) -> None:
        """Eagerly load and warm up the ML goal model.

        Call before plan_from_xml() so load+warmup cost is not counted
        in planning time.
        """
        self._get_or_load_goal_model(goal_strategy, kwargs)

    # Python primitive files are named '{prefix}motion_primitives_15_{shape}.dat'.
    # The C++ executor instead reads system.motion_primitives_file from the namo
    # config. These helpers bridge the two so both halves use the same robot's
    # primitives, sourced from one place (the config). Parsing is best-effort and
    # cached; any failure degrades to the prior no-prefix / no-cap behavior.
    _MP_FILENAME_MARKER = "motion_primitives_"

    def _load_namo_config(self) -> Dict[str, Any]:
        """Parse the namo config YAML once and cache it. {} on any failure."""
        cached = getattr(self, "_parsed_namo_config", None)
        if cached is None:
            try:
                import yaml
                with open(self._config_path) as f:
                    cached = yaml.safe_load(f) or {}
            except Exception:
                cached = {}
            self._parsed_namo_config = cached
        return cached

    def _derive_primitive_prefix(self) -> str:
        """Map the config's motion_primitives_file to a Python primitive prefix.

        'data/motion_primitives_1x_car.dat' -> variant '1x_car' -> prefix
        '1x_car_'. Returns '' unless '{prefix}motion_primitives_15_square.dat'
        actually exists under primitive_data_dir, so a config whose naming we
        don't recognize (or a missing set) safely falls back to legacy behavior.
        """
        import os
        cfg = self._load_namo_config()
        mp_file = (cfg.get("system", {}) or {}).get("motion_primitives_file")
        if not mp_file:
            return ""
        stem = os.path.splitext(os.path.basename(str(mp_file)))[0]
        if not stem.startswith(self._MP_FILENAME_MARKER):
            return ""
        variant = stem[len(self._MP_FILENAME_MARKER):]
        if not variant:
            return ""
        prefix = f"{variant}_"
        sentinel = os.path.join(
            self._primitive_data_dir,
            f"{prefix}motion_primitives_15_square.dat",
        )
        return prefix if os.path.exists(sentinel) else ""

    def _max_push_steps_from_config(self) -> Optional[int]:
        """Read motion_primitives.max_push_steps from the config, or None."""
        cfg = self._load_namo_config()
        val = (cfg.get("motion_primitives", {}) or {}).get("max_push_steps")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def analyze_reachability_from_xml(
        self,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
        analysis_mode: bool = False,
        starting_robot_pose: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Compute unified C++ wavefront reachability for one XML state.

        ``starting_robot_pose`` (x_m, y_m, theta_rad) is required for the car
        robot — its freejoint spawn pos in the included little_car.xml can't
        be overridden through a top-level <include>, so without teleport the
        wavefront BFS starts from the XML spawn (typically the origin) and
        reachability answers describe the wrong world. Mirrors plan_from_xml's
        defer_warmup branch. Sphere XMLs bake pose into the geom; pass None.
        """
        start_time = time.time()
        try:
            defer_warmup = starting_robot_pose is not None
            env = namo_rl.RLEnvironment(
                xml_path, self._config_path, self._enable_viewer, defer_warmup
            )
            if defer_warmup:
                env.set_robot_pose(*starting_robot_pose)
                env.warm_up()
            env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])
            summary = env.get_reachability_summary(analysis_mode)
            summary["compute_time_ms"] = (time.time() - start_time) * 1000
            return summary
        except Exception as e:
            return {
                "goal_reachable": False,
                "analysis_mode": analysis_mode,
                "objects": {},
                "compute_time_ms": (time.time() - start_time) * 1000,
                "error_message": str(e),
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
        """Plan from an XML environment file.

        Args:
            xml_path: Path to MuJoCo XML environment file
            robot_goal: Target robot position (x, y, theta) in simulation meters
            algorithm: Planning algorithm ("full_namo" or "region_opening")
            goal_strategy: Goal sampling strategy ("primitive", "ml", etc.)
            max_chain_depth: Maximum chain depth for multi-push solutions (1 or 2)
            max_solutions_per_neighbor: Maximum solutions to find per neighbor
            timeout_per_neighbour_sec: Timeout per neighbor in seconds
            allow_collisions: Allow collisions during push (don't terminate on collision)
            frontier_beam_width: Beam width for frontier search (0 = unbounded)
            chain_link_cost: Additional cost per chain link beyond first push
            selection_strategy: Frontier priority ("cost_first" or "ml_first")
            goals_per_region: Number of robot goal samples per region for validation
            **kwargs: Additional algorithm parameters

        Returns:
            NAMOPlanResult with success status and list of actions
        """
        start_time = time.time()

        try:
            # When the caller provides starting_robot_pose, we know the XML's
            # XML-default robot pose may overlap obstacles (the car case —
            # little_car.xml fixes the freejoint spawn at the origin and we
            # can't override it through a top-level <include>). Skip the
            # ctor's built-in warm_up, teleport the robot to the actual
            # observation pose, THEN run warm_up so the 3 physics ticks
            # integrate with the correct starting state.
            #
            # Sphere XMLs bake the robot pose into the geom and don't need
            # this — pass starting_robot_pose=None for them and the env
            # behaves exactly as before.
            defer_warmup = starting_robot_pose is not None
            env = namo_rl.RLEnvironment(
                xml_path,
                self._config_path,
                self._enable_viewer,
                defer_warmup,
            )
            if defer_warmup:
                env.set_robot_pose(*starting_robot_pose)
                env.warm_up()

            # Pause for inspection if requested
            if self._pause_after_load and self._enable_viewer:
                import sys
                import select

                print("\n" + "=" * 60)
                print("PAUSED: Environment loaded from XML")
                print(f"  XML: {xml_path}")
                print(f"  Goal: {robot_goal}")
                print("  Viewer window is open - interact with it (rotate, zoom)")
                print("=" * 60)
                print("Press ENTER in terminal to continue with planning...")

                # Render loop - keep viewer responsive while waiting for input
                while True:
                    env.render()
                    # Check for input without blocking (Unix only)
                    if sys.platform != 'win32':
                        if select.select([sys.stdin], [], [], 0.03)[0]:
                            sys.stdin.readline()
                            break
                    else:
                        # Windows fallback - just render and use shorter timeout
                        import msvcrt
                        if msvcrt.kbhit():
                            msvcrt.getch()
                            break
                        time.sleep(0.03)
                print()

            # Build algorithm parameters matching region_opening_collection.yaml
            algo_params = {
                "primitive_data_dir": self._primitive_data_dir,
                "goal_strategy": goal_strategy,
                "xml_file": xml_path,
                # Path to the NAMO config, so ML strategies that lazily build
                # their own GoalInferenceModel generate region masks with the
                # correct robot footprint (the preloaded model already carries
                # this; this covers the non-preloaded path).
                "namo_config_path": self._config_path,
                # Region opening specific parameters
                "region_max_chain_depth": max_chain_depth,
                "region_max_solutions_per_neighbor": max_solutions_per_neighbor,
                "region_max_recorded_solutions_per_neighbor": max_solutions_per_neighbor,
                "region_allow_collisions": allow_collisions,
                "region_frontier_beam_width": frontier_beam_width,
                "region_chain_link_cost": chain_link_cost,
                "region_selection_strategy": selection_strategy,
                "region_ml_ignore_blacklist": False,
            }
            if timeout_per_neighbour_sec is not None:
                algo_params["region_timeout_per_neighbour_sec"] = timeout_per_neighbour_sec
            algo_params.update(kwargs)

            # Keep Python goal-generation primitives in sync with the C++ executor.
            # Only C++ reads the namo config YAML; the Python strategies pick a
            # primitive file by filename prefix. Derive that prefix (and the
            # push-step cap) from the SAME config so goal generation/alignment
            # never silently uses a different robot's primitives than the
            # executor (e.g. legacy 30 cm point-robot pushes of 0.5-6 m vs the
            # 7 cm car's 0.02-0.43 m). Caller-supplied values win; we only fill
            # defaults, and _derive_primitive_prefix() returns "" (legacy) unless
            # the resolved file actually exists — so existing setups never break.
            if "primitive_prefix" not in algo_params:
                derived_prefix = self._derive_primitive_prefix()
                if derived_prefix:
                    algo_params["primitive_prefix"] = derived_prefix
            if "max_push_steps" not in algo_params:
                cap = self._max_push_steps_from_config()
                if cap is not None:
                    algo_params["max_push_steps"] = cap

            # Inject cached ML goal model (avoids reloading on every replan)
            cached_model = self._get_or_load_goal_model(goal_strategy, algo_params)
            if cached_model is not None:
                algo_params["preloaded_goal_model"] = cached_model

            # Create planner config
            config = PlannerConfig(
                verbose=self._verbose,
                goals_per_region=goals_per_region,
                algorithm_params=algo_params,
            )

            # Create planner via factory (supports region_opening, full_namo, etc.)
            planner = PlannerFactory.create_planner(algorithm, env, config)

            # Set robot goal
            env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

            # Run search
            search_start = time.time()
            result = planner.search(robot_goal)
            search_only_ms = (time.time() - search_start) * 1000

            # Extract actions from result
            actions = self._extract_actions(result)

            total_ms = (time.time() - start_time) * 1000
            print(
                f"[NAMOPlanningService] search={search_only_ms:.0f}ms, "
                f"total={total_ms:.0f}ms, actions={len(actions)}"
            )

            return NAMOPlanResult(
                success=result.success and len(actions) > 0,
                actions=actions,
                search_time_ms=total_ms,
                algorithm_stats=result.algorithm_stats,
            )

        except Exception as e:
            search_time_ms = (time.time() - start_time) * 1000
            return NAMOPlanResult(
                success=False,
                actions=[],
                search_time_ms=search_time_ms,
                error_message=str(e),
            )

    def _extract_actions(self, result: PlannerResult) -> List[NAMOAction]:
        """Extract NAMOAction list from PlannerResult.

        Filters to only actions with valid edge_idx and depth.
        """
        actions = []

        if not result.action_sequence:
            return actions

        for action in result.action_sequence:
            # Only include actions with valid edge_idx and depth
            edge_idx = getattr(action, "edge_idx", -1)
            depth = getattr(action, "depth", -1)

            if edge_idx >= 0 and depth >= 0:
                actions.append(
                    NAMOAction(
                        object_id=action.object_id,
                        edge_idx=edge_idx,
                        depth=depth,
                    )
                )

        return actions
