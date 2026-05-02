"""Core deterministic mining logic for boosted data collection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from namo.planners import get_region_snapshot
from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter

from .schema import build_cell_chain_csr


def deterministic_state_hash(state: Any) -> int:
    """Stable hash for simulator state using qpos/qvel raw bytes."""

    qpos = np.asarray(list(state.qpos), dtype=np.float64)
    qvel = np.asarray(list(state.qvel), dtype=np.float64)
    h = hashlib.blake2b(digest_size=16)
    h.update(qpos.tobytes())
    h.update(qvel.tobytes())
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


@dataclass(frozen=True)
class WavefrontSnapshot:
    free_mask: np.ndarray
    reachable_mask: np.ndarray
    reachable_edges: Tuple[int, ...]
    resolution: float
    bounds: Tuple[float, float, float, float]
    grid_shape: Tuple[int, int]
    source: str


@dataclass
class TransitionResult:
    success: bool
    invalid: bool
    stuck: bool
    collision: bool
    prune_deeper: bool
    failure_reason: str
    goal_pose: Tuple[float, float, float]
    child_state_hash: Optional[int] = None
    child_state: Any = None
    child_snapshot: Optional[WavefrontSnapshot] = None
    opened_cell_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))


@dataclass
class StateNode:
    state_hash: int
    state: Any
    snapshot: WavefrontSnapshot
    chain_ids: List[int]


@dataclass
class TransitionMemo:
    cache: Dict[Tuple[int, int, int], TransitionResult] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: Tuple[int, int, int]) -> Optional[TransitionResult]:
        out = self.cache.get(key)
        if out is None:
            self.misses += 1
            return None
        self.hits += 1
        return out

    def put(self, key: Tuple[int, int, int], value: TransitionResult) -> None:
        self.cache[key] = value


class ChainPool:
    """Compact parent-pointer chain storage."""

    def __init__(self) -> None:
        self.parent_id: List[int] = [-1]
        self.edge_idx: List[int] = [-1]
        self.depth_idx: List[int] = [-1]
        self.goal_x: List[float] = [math.nan]
        self.goal_y: List[float] = [math.nan]
        self.goal_theta: List[float] = [math.nan]

    def add(self, parent_id: int, edge_idx: int, depth_idx: int, goal_pose: Tuple[float, float, float]) -> int:
        cid = len(self.parent_id)
        self.parent_id.append(int(parent_id))
        self.edge_idx.append(int(edge_idx))
        self.depth_idx.append(int(depth_idx))
        self.goal_x.append(float(goal_pose[0]))
        self.goal_y.append(float(goal_pose[1]))
        self.goal_theta.append(float(goal_pose[2]))
        return cid

    def to_arrays(self) -> Dict[str, np.ndarray]:
        n = len(self.parent_id)
        return {
            "chain_id": np.arange(n, dtype=np.int32),
            "parent_id": np.asarray(self.parent_id, dtype=np.int32),
            "edge_idx": np.asarray(self.edge_idx, dtype=np.int16),
            "depth_idx": np.asarray(self.depth_idx, dtype=np.int16),
            "goal_x": np.asarray(self.goal_x, dtype=np.float32),
            "goal_y": np.asarray(self.goal_y, dtype=np.float32),
            "goal_theta": np.asarray(self.goal_theta, dtype=np.float32),
        }


class EarliestHorizonIndex:
    """Tracks earliest horizon ownership for each cell id."""

    def __init__(self, num_cells: int) -> None:
        self.earliest_horizon = np.zeros((num_cells,), dtype=np.uint8)

    def record_cells(
        self,
        horizon: int,
        cell_ids: np.ndarray,
        chain_id: int,
        horizon_cell_to_chain_ids: MutableMapping[int, Set[int]],
    ) -> None:
        if cell_ids.size == 0:
            return

        for raw_cell_id in cell_ids.tolist():
            cell_id = int(raw_cell_id)
            current = int(self.earliest_horizon[cell_id])
            if current == 0:
                self.earliest_horizon[cell_id] = np.uint8(horizon)
                horizon_cell_to_chain_ids.setdefault(cell_id, set()).add(chain_id)
            elif current == horizon:
                horizon_cell_to_chain_ids.setdefault(cell_id, set()).add(chain_id)


def _parse_observation_pose(obs: Mapping[str, Sequence[float]], object_id: str) -> Tuple[float, float, float]:
    key = f"{object_id}_pose"
    if key in obs and len(obs[key]) >= 3:
        pose = obs[key]
        return float(pose[0]), float(pose[1]), float(pose[2])

    if object_id in obs and len(obs[object_id]) >= 3:
        pose = obs[object_id]
        return float(pose[0]), float(pose[1]), float(pose[2])

    if "robot_pose" in obs and len(obs["robot_pose"]) >= 3:
        pose = obs["robot_pose"]
        return float(pose[0]), float(pose[1]), float(pose[2])

    return 0.0, 0.0, 0.0


def _resolve_library_target_pose(
    env: Any,
    object_id: str,
    edge_idx: int,
    depth_idx: int,
    boosted_config: Mapping[str, Any],
) -> Tuple[float, float, float]:
    """Resolve target pose for direct primitive execution from primitive library."""

    method_name = "get_primitive_library_target_pose"
    require_library = bool(boosted_config.get("boosted_require_primitive_library", True))

    if hasattr(env, method_name):
        raw = getattr(env, method_name)(object_id, int(edge_idx), int(depth_idx))
        if isinstance(raw, (tuple, list)) and len(raw) >= 3:
            return float(raw[0]), float(raw[1]), float(raw[2])
        raise ValueError(
            f"{method_name} returned invalid payload for object={object_id} "
            f"edge={edge_idx} depth={depth_idx}: {raw!r}"
        )

    if require_library:
        raise RuntimeError(
            f"Environment missing {method_name}; cannot execute boosted direct primitives "
            "with primitive-library targets"
        )

    # Backward-compatibility fallback for test doubles that do not expose C++ helper.
    obs = env.get_observation()
    return _parse_observation_pose(obs, object_id)


def _bfs_reachable_mask(
    free_mask: np.ndarray,
    robot_xy: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """8-connected BFS reachability on a boolean free mask."""

    grid_w, grid_h = int(free_mask.shape[0]), int(free_mask.shape[1])
    reachable = np.zeros((grid_w, grid_h), dtype=np.bool_)
    if grid_w <= 0 or grid_h <= 0 or resolution <= 0.0:
        return reachable

    start_x = int(math.floor((robot_xy[0] - bounds[0]) / resolution))
    start_y = int(math.floor((robot_xy[1] - bounds[2]) / resolution))
    if not (0 <= start_x < grid_w and 0 <= start_y < grid_h):
        return reachable

    q: deque[Tuple[int, int]] = deque()
    q.append((start_x, start_y))
    reachable[start_x, start_y] = True

    directions = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    while q:
        cx, cy = q.popleft()
        for dx, dy in directions:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h:
                continue
            if reachable[nx, ny]:
                continue
            if not bool(free_mask[nx, ny]):
                continue
            reachable[nx, ny] = True
            q.append((nx, ny))

    return reachable


def get_wavefront_snapshot(
    env: Any,
    object_id: str,
    *,
    use_cpp_grid_fastpath: bool,
    fallback_resolution: Optional[float] = None,
) -> WavefrontSnapshot:
    """Get free/reachable masks + reachable edges for current state."""

    if use_cpp_grid_fastpath and hasattr(env, "get_wavefront_snapshot_for_object"):
        raw = env.get_wavefront_snapshot_for_object(object_id)
        free_mask = np.asarray(raw["free_mask"], dtype=np.uint8).astype(np.bool_, copy=False)
        reachable_mask = np.asarray(raw["reachable_mask"], dtype=np.uint8).astype(np.bool_, copy=False)

        grid_shape_raw = raw.get("grid_shape")
        if isinstance(grid_shape_raw, (tuple, list)) and len(grid_shape_raw) == 2:
            grid_shape = (int(grid_shape_raw[0]), int(grid_shape_raw[1]))
        else:
            grid_shape = (int(free_mask.shape[0]), int(free_mask.shape[1]))

        return WavefrontSnapshot(
            free_mask=free_mask,
            reachable_mask=reachable_mask,
            reachable_edges=tuple(sorted(int(v) for v in raw.get("reachable_edges", []))),
            resolution=float(raw.get("resolution", 0.0)),
            bounds=tuple(float(v) for v in raw.get("bounds", [0.0, 0.0, 0.0, 0.0])),
            grid_shape=grid_shape,
            source="cpp_fastpath",
        )

    exporter = WavefrontSnapshotExporter(env, resolution=fallback_resolution)

    xml_path_fn = getattr(env, "get_xml_path", None)
    config_path_fn = getattr(env, "get_config_path", None)
    xml_path = xml_path_fn() if callable(xml_path_fn) else ""
    config_path = config_path_fn() if callable(config_path_fn) else ""

    snap = exporter.build_snapshot(
        xml_path=xml_path,
        config_path=config_path,
        use_current_state=True,
        goals_per_region=0,
    )

    free_mask = np.asarray(snap.dynamic_grid != -1, dtype=np.bool_)
    obs = env.get_observation()
    robot_pose = _parse_observation_pose(obs, "robot")
    reachable_mask = _bfs_reachable_mask(
        free_mask=free_mask,
        robot_xy=(robot_pose[0], robot_pose[1]),
        bounds=(float(snap.bounds[0]), float(snap.bounds[1]), float(snap.bounds[2]), float(snap.bounds[3])),
        resolution=float(snap.resolution),
    )

    reachable_edges = tuple(sorted(int(v) for v in env.get_reachable_edges(object_id)))
    return WavefrontSnapshot(
        free_mask=free_mask,
        reachable_mask=reachable_mask,
        reachable_edges=reachable_edges,
        resolution=float(snap.resolution),
        bounds=(float(snap.bounds[0]), float(snap.bounds[1]), float(snap.bounds[2]), float(snap.bounds[3])),
        grid_shape=(int(free_mask.shape[0]), int(free_mask.shape[1])),
        source="python_fallback",
    )


def classify_step_result(step_result: Any) -> Dict[str, Any]:
    """Normalize status flags from ``env.step`` result."""

    info = dict(getattr(step_result, "info", {}) or {})
    failure_reason = str(info.get("failure_reason", ""))
    failure_reason_l = failure_reason.lower()

    invalid = ("not applicable" in failure_reason_l) or ("not reachable" in failure_reason_l)
    stuck = str(info.get("stuck", "")).lower() == "true" or ("stuck" in failure_reason_l)

    movable_collision_payload = str(info.get("movable_collisions", ""))
    collision = (
        bool(str(info.get("collision_object", "")).strip())
        or str(info.get("wall_collision", "")).lower() == "true"
        or bool(movable_collision_payload.strip())
        or ("collision" in failure_reason_l)
    )

    return {
        "success": bool(getattr(step_result, "done", False)),
        "invalid": bool(invalid),
        "stuck": bool(stuck),
        "collision": bool(collision),
        "failure_reason": failure_reason,
        "prune_deeper": bool(invalid or stuck),
    }


def _infer_primitive_depth_count(
    env: Any,
    object_id: str,
    boosted_config: Mapping[str, Any],
) -> int:
    override = boosted_config.get("boosted_primitive_depth_count")
    if override is not None:
        return max(1, int(override))

    try:
        summary = env.get_reachability_summary(False)
        objects = summary.get("objects", {}) if isinstance(summary, Mapping) else {}
        obj_summary = objects.get(object_id, {}) if isinstance(objects, Mapping) else {}
        total_edges = int(obj_summary.get("total_edges", 0))
        total_primitives = int(obj_summary.get("total_primitives", 0))
        if total_edges > 0 and total_primitives > 0 and total_primitives % total_edges == 0:
            return max(1, total_primitives // total_edges)
    except Exception:
        pass

    legacy_max_depth = boosted_config.get("max_depth")
    if legacy_max_depth is not None:
        return max(1, int(legacy_max_depth))

    return 10


def _resolve_depth_indices_for_edge(
    env: Any,
    object_id: str,
    edge_idx: int,
    fallback_depth_count: int,
) -> Tuple[int, ...]:
    """Return deterministic depth indices for this edge from primitive library when available."""

    method_name = "get_valid_primitive_depth_indices"
    if hasattr(env, method_name):
        try:
            raw = getattr(env, method_name)(object_id, int(edge_idx))
            if isinstance(raw, (tuple, list)):
                depth_indices = sorted({int(v) for v in raw if int(v) >= 0})
                if depth_indices:
                    return tuple(depth_indices)
        except Exception:
            pass

    return tuple(range(max(1, int(fallback_depth_count))))


def _evaluate_transition(
    env: Any,
    object_id: str,
    parent_state: Any,
    parent_snapshot: WavefrontSnapshot,
    edge_idx: int,
    depth_idx: int,
    boosted_config: Mapping[str, Any],
) -> TransitionResult:
    env.set_full_state(parent_state)

    fallback_pose = _parse_observation_pose(env.get_observation(), object_id)
    try:
        action_target_pose = _resolve_library_target_pose(
            env=env,
            object_id=object_id,
            edge_idx=int(edge_idx),
            depth_idx=int(depth_idx),
            boosted_config=boosted_config,
        )
    except Exception as exc:
        return TransitionResult(
            success=False,
            invalid=True,
            stuck=False,
            collision=False,
            prune_deeper=True,
            failure_reason=f"primitive target lookup failed: {exc}",
            goal_pose=fallback_pose,
        )

    action = env.Action() if hasattr(env, "Action") else None
    if action is None:
        # pybind exposes Action class at module level, but object instance method works via namo_rl.Action.
        import namo_rl  # local import to avoid hard dependency during pure-unit tests

        action = namo_rl.Action()

    action.object_id = object_id
    action.x = float(action_target_pose[0])
    action.y = float(action_target_pose[1])
    action.theta = float(action_target_pose[2])
    action.edge_idx = int(edge_idx)
    action.depth = int(depth_idx)

    step_result = env.step(action)
    status = classify_step_result(step_result)

    if not status["success"]:
        return TransitionResult(
            success=False,
            invalid=bool(status["invalid"]),
            stuck=bool(status["stuck"]),
            collision=bool(status["collision"]),
            prune_deeper=bool(status["prune_deeper"]),
            failure_reason=str(status["failure_reason"]),
            goal_pose=action_target_pose,
        )

    child_state = env.get_full_state()
    child_state_hash = deterministic_state_hash(child_state)
    child_snapshot = get_wavefront_snapshot(
        env,
        object_id,
        use_cpp_grid_fastpath=bool(boosted_config.get("boosted_use_cpp_grid_fastpath", True)),
        fallback_resolution=float(parent_snapshot.resolution) if parent_snapshot.resolution > 0.0 else None,
    )

    opened_mask = (~parent_snapshot.free_mask) & child_snapshot.free_mask & child_snapshot.reachable_mask
    opened_cell_ids = np.flatnonzero(opened_mask.reshape(-1, order="C")).astype(np.int32)

    return TransitionResult(
        success=True,
        invalid=False,
        stuck=False,
        collision=bool(status["collision"]),
        prune_deeper=False,
        failure_reason=str(status["failure_reason"]),
        goal_pose=action_target_pose,
        child_state_hash=child_state_hash,
        child_state=child_state,
        child_snapshot=child_snapshot,
        opened_cell_ids=opened_cell_ids,
    )


def select_candidate_blocking_objects(
    env: Any,
    boosted_config: Mapping[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Collect deterministic candidate blocking objects for robot-adjacent regions."""

    snapshot = get_region_snapshot(
        env,
        goals_per_region=0,
        local_info_only=True,
        seed=int(boosted_config.get("seed", 42)),
        use_cpp_unified=True,
        use_xml_goal=not bool(boosted_config.get("boosted_ignore_xml_goal", True)),
    )

    robot_label = str(snapshot.get("robot_label", ""))
    adjacency = snapshot.get("adjacency", {})
    edge_objects = snapshot.get("edge_objects", {})

    if not robot_label:
        return [], snapshot

    neighbours = sorted(str(v) for v in adjacency.get(robot_label, set()))

    candidate_objects: Set[str] = set()
    for neighbour in neighbours:
        forward = edge_objects.get(robot_label, {}).get(neighbour, set())
        backward = edge_objects.get(neighbour, {}).get(robot_label, set())
        candidate_objects.update(str(v) for v in forward)
        candidate_objects.update(str(v) for v in backward)

    reachable_objects = set(str(v) for v in env.get_reachable_objects())
    filtered = sorted(obj for obj in candidate_objects if obj in reachable_objects)
    return filtered, snapshot


def mine_object_manifest(
    env: Any,
    baseline_state: Any,
    object_id: str,
    boosted_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Mine earliest-horizon cell openings for one object."""

    if not bool(boosted_config.get("boosted_same_object_only", True)):
        raise ValueError("boosted_same_object_only=false is not supported in phase-1 implementation")
    cell_filter = str(boosted_config.get("boosted_cell_filter", "newly_free_and_reachable"))
    if cell_filter != "newly_free_and_reachable":
        raise ValueError(
            f"Unsupported boosted_cell_filter={cell_filter!r}. "
            "Phase-1 supports only 'newly_free_and_reachable'."
        )

    max_horizon = max(1, int(boosted_config.get("boosted_max_horizon", 1)))
    primitive_depth_count = _infer_primitive_depth_count(env, object_id, boosted_config)

    env.set_full_state(baseline_state)
    initial_snapshot = get_wavefront_snapshot(
        env,
        object_id,
        use_cpp_grid_fastpath=bool(boosted_config.get("boosted_use_cpp_grid_fastpath", True)),
        fallback_resolution=None,
    )

    num_cells = int(initial_snapshot.grid_shape[0] * initial_snapshot.grid_shape[1])
    earliest = EarliestHorizonIndex(num_cells=num_cells)
    horizon_cell_to_chain_ids: List[Dict[int, Set[int]]] = [dict() for _ in range(max_horizon)]

    chain_pool = ChainPool()
    transition_cache = TransitionMemo()

    baseline_hash = deterministic_state_hash(baseline_state)
    state_cache: Dict[int, StateNode] = {
        baseline_hash: StateNode(
            state_hash=baseline_hash,
            state=baseline_state,
            snapshot=initial_snapshot,
            chain_ids=[0],
        )
    }
    frontier: Dict[int, List[int]] = {baseline_hash: [0]}

    stats: Dict[str, Any] = {
        "states_expanded": 0,
        "transitions_evaluated": 0,
        "transition_cache_hits": 0,
        "transition_cache_misses": 0,
        "invalid_transitions": 0,
        "stuck_transitions": 0,
        "collision_transitions": 0,
        "successful_transitions": 0,
        "pruned_same_edge_depth": 0,
        "source_initial_snapshot": initial_snapshot.source,
        "primitive_depth_count": primitive_depth_count,
        "depth_slots_source": "primitive_library_or_fallback",
    }

    for horizon in range(1, max_horizon + 1):
        next_frontier: Dict[int, List[int]] = {}

        for state_hash in sorted(frontier.keys()):
            parent_chain_ids = frontier[state_hash]
            node = state_cache[state_hash]
            stats["states_expanded"] += 1

            edges_sorted = tuple(sorted(node.snapshot.reachable_edges))
            for edge_idx in edges_sorted:
                prune_edge_depths = False
                depth_indices = _resolve_depth_indices_for_edge(
                    env=env,
                    object_id=object_id,
                    edge_idx=int(edge_idx),
                    fallback_depth_count=primitive_depth_count,
                )

                for depth_idx in depth_indices:
                    if prune_edge_depths:
                        break

                    cache_key = (state_hash, int(edge_idx), int(depth_idx))
                    tr = transition_cache.get(cache_key)
                    if tr is None:
                        stats["transitions_evaluated"] += 1
                        tr = _evaluate_transition(
                            env=env,
                            object_id=object_id,
                            parent_state=node.state,
                            parent_snapshot=node.snapshot,
                            edge_idx=int(edge_idx),
                            depth_idx=int(depth_idx),
                            boosted_config=boosted_config,
                        )
                        transition_cache.put(cache_key, tr)

                        if tr.success and tr.child_state_hash is not None and tr.child_snapshot is not None:
                            if tr.child_state_hash not in state_cache:
                                state_cache[tr.child_state_hash] = StateNode(
                                    state_hash=tr.child_state_hash,
                                    state=tr.child_state,
                                    snapshot=tr.child_snapshot,
                                    chain_ids=[],
                                )

                    if tr.success:
                        stats["successful_transitions"] += 1
                    if tr.invalid:
                        stats["invalid_transitions"] += 1
                    if tr.stuck:
                        stats["stuck_transitions"] += 1
                    if tr.collision:
                        stats["collision_transitions"] += 1

                    if tr.prune_deeper:
                        prune_edge_depths = True
                        stats["pruned_same_edge_depth"] += 1

                    if not tr.success or tr.child_state_hash is None:
                        continue

                    for parent_chain_id in parent_chain_ids:
                        child_chain_id = chain_pool.add(
                            parent_id=int(parent_chain_id),
                            edge_idx=int(edge_idx),
                            depth_idx=int(depth_idx),
                            goal_pose=tr.goal_pose,
                        )

                        earliest.record_cells(
                            horizon=horizon,
                            cell_ids=tr.opened_cell_ids,
                            chain_id=child_chain_id,
                            horizon_cell_to_chain_ids=horizon_cell_to_chain_ids[horizon - 1],
                        )

                        next_frontier.setdefault(tr.child_state_hash, []).append(child_chain_id)

        # Deduplicate provenance chain IDs per state, deterministic order.
        frontier = {
            state_hash: sorted(set(chain_ids))
            for state_hash, chain_ids in sorted(next_frontier.items(), key=lambda x: x[0])
        }

    stats["transition_cache_hits"] = transition_cache.hits
    stats["transition_cache_misses"] = transition_cache.misses

    horizon_cell_ids: List[np.ndarray] = []
    horizon_csr: List[Dict[str, np.ndarray]] = []
    for h in range(max_horizon):
        csr = build_cell_chain_csr(horizon_cell_to_chain_ids[h])
        horizon_cell_ids.append(csr.cell_ids)
        horizon_csr.append(
            {
                "cell_ids": csr.cell_ids,
                "indptr": csr.indptr,
                "chain_ids": csr.chain_ids,
            }
        )

    opened_counts = [int(arr.shape[0]) for arr in horizon_cell_ids]
    stats["opened_cells_per_horizon"] = opened_counts
    stats["opened_cells_total"] = int(sum(opened_counts))
    stats["chain_count"] = int(len(chain_pool.parent_id))
    stats["unique_states_total"] = int(len(state_cache))

    return {
        "object_id": object_id,
        "max_horizon": max_horizon,
        "horizon_cell_ids": horizon_cell_ids,
        "horizon_cell_to_chain_csr": horizon_csr,
        "chain_pool": chain_pool.to_arrays(),
        "stats": stats,
    }


def mine_environment_manifest(
    env: Any,
    boosted_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Mine boosted manifest blocks for all candidate blocking objects in env."""

    baseline_state = env.get_full_state()
    object_ids, region_snapshot = select_candidate_blocking_objects(env, boosted_config)

    per_object = []
    for object_id in object_ids:
        env.set_full_state(baseline_state)
        per_object.append(
            mine_object_manifest(
                env=env,
                baseline_state=baseline_state,
                object_id=object_id,
                boosted_config=boosted_config,
            )
        )

    return {
        "candidate_object_ids": object_ids,
        "region_snapshot_source": region_snapshot.get("source", "unknown"),
        "region_snapshot_robot_label": region_snapshot.get("robot_label", ""),
        "objects": per_object,
    }


def serialize_for_metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return JSON-friendly config snapshot."""

    out: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [
                x if isinstance(x, (str, int, float, bool)) or x is None else str(x)
                for x in v
            ]
        elif isinstance(v, dict):
            try:
                out[str(k)] = json.loads(json.dumps(v, sort_keys=True, default=str))
            except Exception:
                out[str(k)] = str(v)
        else:
            out[str(k)] = str(v)
    return out
