"""Stable XML-based planning facade for external NAMO consumers.

The service owns simulator and planner construction. Callers such as
``robot_control`` remain responsible for observation conversion, coordinate
mapping, and physical execution policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import namo_rl

from namo.core import PlannerConfig, PlannerResult


_ML_GOAL_STRATEGIES = frozenset(
    {
        "ml",
        "ml_primitive",
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


@dataclass
class BoundaryOpeningResult:
    """Outcome of solving ONE pinned region boundary.

    Distinct from NAMOPlanResult because an external executor needs things that
    result cannot carry: the state to continue from, why a boundary failed, and
    the fact that a boundary was *already* open -- which NAMOPlanResult reports
    as failure, since it derives success from a non-empty action list.
    """

    success: bool = False
    # True when the boundary already cleared the bar with zero pushes. Success,
    # but with nothing to execute.
    already_open: bool = False
    actions: List[NAMOAction] = field(default_factory=list)
    # The labels resolved for THIS call. Not durable -- labels renumber whenever
    # free space changes. Recorded for diagnostics, and so a caller that wants
    # to exclude this boundary from its next selection can build the pair out of
    # two labels from the same snapshot rather than mixing one of these with a
    # label it persisted before the last push.
    resolved_source: str = ""
    resolved_target: str = ""
    blocking_objects: List[str] = field(default_factory=list)
    # Echo of the points the opening was graded against, so a run log records
    # the criterion and not just the verdict.
    graded_points: List[Tuple[float, float]] = field(default_factory=list)
    failure_reason: str = ""
    boundary_exhausted: bool = False
    # RLState is not picklable; the repo's convention is plain qpos/qvel lists.
    resulting_state: Optional[Dict[str, List[float]]] = None
    simulations_used: int = 0
    simulation_budget_limit: Optional[int] = None
    search_time_ms: float = 0.0
    target_summary: Optional[Dict[str, Any]] = None
    error_message: str = ""


def _boundary_object_set(
    edge_objects: Dict[str, Dict[str, Sequence[str]]], source: str, target: str
) -> Set[str]:
    """Objects blocking the source-target boundary, either direction."""
    forward = edge_objects.get(source, {}).get(target) or []
    reverse = edge_objects.get(target, {}).get(source) or []
    return {str(o) for o in forward} | {str(o) for o in reverse}


def _resolve_boundary_target(
    snapshot: Dict[str, Any],
    blocking_objects: Optional[Sequence[str]],
    target_hint: Optional[str],
) -> Tuple[Optional[str], str]:
    """Find which immediate neighbour is the caller's boundary, in this snapshot.

    A caller cannot persist a region label: labels are ordinal, so they renumber
    whenever a push re-partitions free space. The durable handle is the set of
    objects blocking the boundary, so that is tried first; a label hint is only
    a fallback for the first call, before anything has moved.

    Reports ``ambiguous_boundary`` when two neighbours match the objects equally
    well, because then the objects do not name one boundary and the caller has
    to re-choose at the outer level.
    """
    robot_label = snapshot.get("robot_label")
    neighbours = sorted(snapshot.get("adjacency", {}).get(robot_label, ()))
    if not neighbours:
        return None, "no_immediate_neighbors"

    if blocking_objects:
        wanted = {str(o) for o in blocking_objects}
        edge_objects = snapshot.get("edge_objects", {})
        scored = []
        for neighbour in neighbours:
            boundary = _boundary_object_set(edge_objects, robot_label, neighbour)
            overlap = len(wanted & boundary)
            if overlap:
                # Rank on overlap first, then on how much else the boundary
                # carries. A neighbour blocked by exactly `wanted` beats one
                # blocked by `wanted` plus two objects the caller never saw.
                scored.append(((overlap, -len(boundary ^ wanted)), neighbour))
        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            # Two neighbours match the caller's objects equally well, so the
            # object set does not identify one boundary. Taking the first by
            # label order would be a coin flip that reads as a decision.
            return None, "ambiguous_boundary"
        if scored:
            return scored[0][1], ""

    if target_hint and target_hint in neighbours:
        return target_hint, ""

    # The boundary merged away, or the robot is no longer beside it. The caller
    # must re-choose at the outer level; this is a normal outcome, not an error.
    return None, "target_not_immediate_neighbor"


def _region_search_params(
    *,
    goal_strategy: str,
    max_chain_depth: int,
    max_solutions_per_neighbor: int,
    allow_collisions: bool,
    frontier_beam_width: int,
    chain_link_cost: int,
    selection_strategy: str,
    timeout_per_neighbour_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Translate caller-facing option names into the region_* keys openers read.

    One copy, because two drifted. plan_from_xml built this inline and
    solve_boundary_from_xml built none of it, so a caller holding a boundary
    across pushes got the openers' own defaults. The one that mattered was
    region_max_chain_depth, which defaults to 1: a held boundary could never be
    opened by a setup push followed by a finish push, which is the entire reason
    to hold it.
    """
    params: Dict[str, Any] = {
        "goal_strategy": goal_strategy,
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
        params["region_timeout_per_neighbour_sec"] = timeout_per_neighbour_sec
    return params


def _stale_boundaries(
    adjacency: Dict[str, Any], blocked_boundaries: Optional[Sequence[Tuple[str, str]]]
) -> List[Tuple[str, str]]:
    """Blocked pairs that name no edge in this snapshot.

    Region labels are ordinal, so a blocklist built before a push can name
    boundaries that no longer exist. Excluding those is harmless, the routing
    just proceeds as if they were absent, but a caller that never hears about it
    cannot tell a working blocklist from one that has gone stale.
    """
    from namo.planners.full_namo.full_namo_planner import boundary_key

    stale = set()
    for a, b in blocked_boundaries or ():
        if b not in (adjacency.get(a) or ()) and a not in (adjacency.get(b) or ()):
            stale.add(boundary_key(a, b))
    return sorted(stale)


def _reporting_attempt(attempts: Sequence[Any]) -> Optional[Any]:
    """The attempt that produced the returned plan.

    An opener sweeps every candidate object and records one AttemptResult per
    object, failures included, then takes its action sequence from the first
    attempt that succeeded. A boundary solved by the second candidate therefore
    has a failed attempt at index 0. Reading index 0 hands the caller a
    ``success=True`` result stamped with the reason the first candidate failed
    and with no resulting state, so read the successful attempt instead.
    """
    for attempt in attempts:
        if getattr(attempt, "success", False):
            return attempt
    return attempts[0] if attempts else None


def _durable_state(state: Any) -> Optional[Dict[str, List[float]]]:
    """RLState -> plain lists, the repo's convention for a storable state."""
    if state is None:
        return None
    return {"qpos": [float(v) for v in state.qpos], "qvel": [float(v) for v in state.qvel]}


@dataclass
class BoundarySelection:
    """Which boundary to open next, and the points that define it.

    Returned without solving anything. The points are sampled once here so the
    caller can freeze them: re-sampling on a later call would grade against a
    different target, because a push re-partitions free space.
    """

    found: bool = False
    # Valid only for the snapshot this was computed from -- labels are ordinal.
    # Pass blocking_objects, not this, to identify the boundary later.
    target_label: str = ""
    target_points: List[Tuple[float, float]] = field(default_factory=list)
    blocking_objects: List[str] = field(default_factory=list)
    region_path: List[str] = field(default_factory=list)
    goal_already_reachable: bool = False
    # Pairs from `blocked_boundaries` that name no edge in this snapshot. A
    # caller that carried a blocklist across a push sees here which entries
    # stopped meaning anything, instead of silently routing around nothing.
    stale_blocked_boundaries: List[Tuple[str, str]] = field(default_factory=list)
    failure_reason: str = ""
    error_message: str = ""


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

    def select_boundary_from_xml(
        self,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
        *,
        blocked_boundaries: Optional[Sequence[Tuple[str, str]]] = None,
        starting_robot_pose: Optional[Tuple[float, float, float]] = None,
        goals_per_region: Optional[int] = None,
        region_snapshot_seed: int = 42,
    ) -> BoundarySelection:
        """Choose the next region boundary to open, without solving it.

        Applies exactly the rule FullNAMOPlanner uses -- shortest region path to
        the goal region, then its first hop -- via the shared `find_region_path`,
        so an external executor driving the loop one push at a time makes the
        same choice the in-process planner would.

        The caller freezes the returned points and passes them to
        solve_boundary_from_xml on every subsequent call, so the success bar
        stops moving when the scene does.

        `blocked_boundaries` lets a caller exclude boundaries it has already
        exhausted, as region-label pairs. Labels are ordinal and renumber
        whenever a push re-partitions free space, so a blocklist is only valid
        while the scene has not changed. Build it inside one selection episode
        and drop it after a push. Entries that name no edge in the current
        snapshot come back in `stale_blocked_boundaries` instead of quietly
        excluding a boundary that no longer exists.
        """
        try:
            from namo.planners import get_region_snapshot
            from namo.planners.full_namo.full_namo_planner import (
                boundary_key,
                find_region_path,
            )

            env = self._create_environment(xml_path, starting_robot_pose)
            env.set_robot_goal(*robot_goal)
            if env.is_robot_goal_reachable():
                return BoundarySelection(goal_already_reachable=True)

            config_kwargs: Dict[str, Any] = {}
            if goals_per_region is not None:
                config_kwargs["goals_per_region"] = goals_per_region
            resolved_goals_per_region = PlannerConfig(**config_kwargs).goals_per_region

            # local_info_only=False: the path to the goal region needs the whole
            # graph, not just the robot's immediate neighbours.
            snapshot = get_region_snapshot(
                env,
                goals_per_region=resolved_goals_per_region,
                local_info_only=False,
                seed=int(region_snapshot_seed),
                use_xml_goal=True,
            )
            adjacency = snapshot.get("adjacency", {})
            stale = _stale_boundaries(adjacency, blocked_boundaries)

            robot_label = snapshot.get("robot_label")
            goal_label = snapshot.get("goal_label")
            if not robot_label or not goal_label:
                return BoundarySelection(
                    failure_reason="missing_region_labels",
                    stale_blocked_boundaries=stale,
                )

            blocked = {boundary_key(a, b) for a, b in (blocked_boundaries or ())}
            path = find_region_path(adjacency, robot_label, goal_label, blocked)
            if path is None:
                return BoundarySelection(
                    failure_reason="region_path_exhausted",
                    stale_blocked_boundaries=stale,
                )
            if len(path) < 2:
                # Robot and goal share a region yet the goal is unreachable --
                # a graph/reachability disagreement, not something to open.
                return BoundarySelection(
                    failure_reason="same_region_but_goal_unreachable",
                    region_path=list(path),
                    stale_blocked_boundaries=stale,
                )

            target = path[1]
            bundle = (snapshot.get("region_goals") or {}).get(target)
            points = [
                (float(g.x), float(g.y)) for g in (bundle.goals if bundle else [])
            ]
            if not points:
                return BoundarySelection(
                    failure_reason="target_region_has_no_sampled_points",
                    target_label=target,
                    region_path=list(path),
                    stale_blocked_boundaries=stale,
                )

            return BoundarySelection(
                found=True,
                target_label=target,
                target_points=points,
                blocking_objects=sorted(
                    _boundary_object_set(
                        snapshot.get("edge_objects", {}), robot_label, target
                    )
                ),
                region_path=list(path),
                stale_blocked_boundaries=stale,
            )
        except Exception as exc:  # noqa: BLE001 - facade boundary
            return BoundarySelection(
                failure_reason="exception",
                error_message=f"Boundary selection failed for {xml_path}: {exc}",
            )

    def solve_boundary_from_xml(
        self,
        xml_path: str,
        robot_goal: Tuple[float, float, float],
        target_points: Sequence[Tuple[float, float]],
        *,
        blocking_objects: Optional[Sequence[str]] = None,
        target_neighbor: Optional[str] = None,
        local_search: str = "region_bfs",
        starting_robot_pose: Optional[Tuple[float, float, float]] = None,
        goals_per_region: Optional[int] = None,
        # Same names and defaults as plan_from_xml. A held boundary is solved by
        # the same openers under the same protocol, so a caller must not have to
        # know the region_* keys to reach them.
        goal_strategy: str = "primitive",
        max_chain_depth: int = 1,
        max_solutions_per_neighbor: int = 1,
        allow_collisions: bool = True,
        frontier_beam_width: int = 10000,
        chain_link_cost: int = 11,
        selection_strategy: str = "cost_first",
        timeout_per_neighbour_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> BoundaryOpeningResult:
        """Open ONE specific region boundary, graded against caller-supplied points.

        ``plan_from_xml`` solves the whole problem: it rebuilds the region graph
        and chooses its own next boundary every call. An executor that runs one
        physical push at a time cannot use that, because the choice is remade
        after every push and a setup push can strand itself against a boundary
        the next call no longer targets. This method takes the choice as input.

        ``target_points`` are in simulator metres and are the whole success
        criterion: sample them once, pass the same list every call, and the bar
        stays fixed no matter how the scene re-partitions. ``blocking_objects``
        is the durable identity of the boundary; the label is re-resolved per
        call because labels renumber.

        The search options carry the same names and defaults as
        ``plan_from_xml`` and route through the same mapping, so holding a
        boundary cannot silently search differently from planning the whole
        problem. ``max_chain_depth`` is the one that bites: at the openers' own
        default of 1 no setup-then-finish chain exists, which is the only reason
        to hold a boundary in the first place.

        Returns a typed result rather than raising for the ordinary failures --
        a boundary that merged away or ran out of pushes is an outcome the
        caller must handle, not an exception.
        """
        start_time = time.perf_counter()
        points = [(float(px), float(py)) for px, py in target_points]
        if not points:
            raise ValueError("target_points must not be empty")
        if local_search not in ("region_bfs", "best_first"):
            raise ValueError(
                f"Unknown local_search {local_search!r}. Valid: region_bfs, best_first"
            )

        def _elapsed_ms() -> float:
            return (time.perf_counter() - start_time) * 1000.0

        try:
            from namo.planners import get_region_snapshot

            env = self._create_environment(xml_path, starting_robot_pose)
            env.set_robot_goal(*robot_goal)

            config_kwargs: Dict[str, Any] = {"verbose": self._verbose}
            if goals_per_region is not None:
                config_kwargs["goals_per_region"] = goals_per_region
            resolved_goals_per_region = PlannerConfig(**config_kwargs).goals_per_region

            snapshot = get_region_snapshot(
                env,
                goals_per_region=resolved_goals_per_region,
                local_info_only=True,
                seed=int(kwargs.get("region_snapshot_seed", 42)),
                use_xml_goal=True,
            )
            resolved_target, resolve_error = _resolve_boundary_target(
                snapshot, blocking_objects, target_neighbor
            )
            if resolved_target is None:
                return BoundaryOpeningResult(
                    failure_reason=resolve_error,
                    graded_points=points,
                    search_time_ms=_elapsed_ms(),
                )

            robot_label = str(snapshot.get("robot_label") or "")
            boundary = sorted(
                _boundary_object_set(
                    snapshot.get("edge_objects", {}), robot_label, resolved_target
                )
            )

            algorithm_params: Dict[str, Any] = {
                "primitive_data_dir": self._primitive_data_dir,
                "xml_file": xml_path,
                "namo_config_path": self._config_path,
                "region_target_points": points,
                **_region_search_params(
                    goal_strategy=goal_strategy,
                    max_chain_depth=max_chain_depth,
                    max_solutions_per_neighbor=max_solutions_per_neighbor,
                    allow_collisions=allow_collisions,
                    frontier_beam_width=frontier_beam_width,
                    chain_link_cost=chain_link_cost,
                    selection_strategy=selection_strategy,
                    timeout_per_neighbour_sec=timeout_per_neighbour_sec,
                ),
            }
            algorithm_params.update(kwargs)
            if "primitive_prefix" not in algorithm_params:
                prefix = self._derive_primitive_prefix()
                if prefix:
                    algorithm_params["primitive_prefix"] = prefix
            if "max_push_steps" not in algorithm_params:
                max_push_steps = self._max_push_steps_from_config()
                if max_push_steps is not None:
                    algorithm_params["max_push_steps"] = max_push_steps

            goal_model = self._get_or_load_goal_model(goal_strategy, algorithm_params)
            if goal_model is not None:
                algorithm_params["preloaded_goal_model"] = goal_model

            config = PlannerConfig(
                algorithm_params=algorithm_params, **config_kwargs
            )
            # BestFirstRegionOpeningPlanner is not registered with PlannerFactory,
            # and neither opener is reachable through it with a pinned target, so
            # construct directly -- as solvability_runner already does.
            if local_search == "best_first":
                from namo.planners.opening.best_first_region_opening import (
                    BestFirstRegionOpeningPlanner,
                )

                opener = BestFirstRegionOpeningPlanner(env, config)
            else:
                from namo.planners.opening.region_opening import RegionOpeningPlanner

                opener = RegionOpeningPlanner(env, config)

            result = opener.search(robot_goal, target_neighbor=resolved_target)
            return self._boundary_result(
                result, robot_label, resolved_target, boundary, points, _elapsed_ms()
            )
        except Exception as exc:  # noqa: BLE001 - facade boundary
            return BoundaryOpeningResult(
                failure_reason="exception",
                graded_points=points,
                error_message=f"Boundary opening failed for {xml_path}: {exc}",
                search_time_ms=_elapsed_ms(),
            )

    @staticmethod
    def _boundary_result(
        result: PlannerResult,
        resolved_source: str,
        resolved_target: str,
        blocking_objects: List[str],
        points: List[Tuple[float, float]],
        elapsed_ms: float,
    ) -> BoundaryOpeningResult:
        """Flatten an opener's PlannerResult into the external boundary result."""
        stats = result.algorithm_stats or {}
        attempts = stats.get("attempt_results") or []
        summary = stats.get("target_summary") or {}
        # The opener aggregates across attempts, so its verdict already accounts
        # for a sweep that failed on one object and solved on another. The
        # per-attempt reason is the fallback for openers that build no summary.
        attempt = _reporting_attempt(attempts)
        failure_reason = str(
            summary.get("failure_reason")
            or getattr(attempt, "failure_reason", "")
            or ""
        )

        actions = [
            NAMOAction(str(a.object_id), int(a.edge_idx), int(a.depth))
            for a in (result.action_sequence or [])
            if int(a.edge_idx) >= 0 and int(a.depth) >= 0
        ]
        return BoundaryOpeningResult(
            # Deliberately NOT `and bool(actions)`: a boundary that was already
            # open is a success with nothing to execute.
            success=bool(result.success),
            already_open=failure_reason == "already_accessible",
            actions=actions,
            resolved_source=resolved_source,
            resolved_target=resolved_target,
            blocking_objects=blocking_objects,
            graded_points=points,
            failure_reason=failure_reason,
            boundary_exhausted=bool(summary.get("boundary_exhausted", False)),
            resulting_state=_durable_state(getattr(attempt, "resulting_state", None)),
            simulations_used=int(stats.get("simulation_budget_used") or 0),
            simulation_budget_limit=stats.get("simulation_budget_limit"),
            search_time_ms=elapsed_ms,
            target_summary=summary or None,
            error_message=result.error_message or "",
        )

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
        goals_per_region: Optional[int] = None,
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
                "xml_file": xml_path,
                "namo_config_path": self._config_path,
                **_region_search_params(
                    goal_strategy=goal_strategy,
                    max_chain_depth=max_chain_depth,
                    max_solutions_per_neighbor=max_solutions_per_neighbor,
                    allow_collisions=allow_collisions,
                    frontier_beam_width=frontier_beam_width,
                    chain_link_cost=chain_link_cost,
                    selection_strategy=selection_strategy,
                    timeout_per_neighbour_sec=timeout_per_neighbour_sec,
                ),
            }
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

            # Omitting the key defers to PlannerConfig's canonical 100. This
            # facade used to default it to 10, so a caller that did not set it
            # was graded at ">=2 of 10 sampled points" rather than the canonical
            # ">=20 of 100" -- the same fraction over a far noisier sample.
            config_kwargs: Dict[str, Any] = {
                "verbose": self._verbose,
                "algorithm_params": algorithm_params,
            }
            if goals_per_region is not None:
                config_kwargs["goals_per_region"] = goals_per_region
            config = PlannerConfig(**config_kwargs)
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
