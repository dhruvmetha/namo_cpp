"""Build sage-learning-compatible synthetic episodes from boosted manifests."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .schema import reconstruct_chain_records


def cell_id_to_world_pose(cell_id: int, grid_metadata: Mapping[str, Any]) -> List[float]:
    """Convert a flat cell id to the world-space center pose ``[x, y, theta]``."""

    grid_shape = grid_metadata.get("grid_shape")
    bounds = grid_metadata.get("bounds")
    resolution = float(grid_metadata.get("resolution", 0.0))

    if not isinstance(grid_shape, Sequence) or len(grid_shape) != 2:
        raise ValueError(f"Invalid grid_shape in boosted manifest: {grid_shape!r}")
    if not isinstance(bounds, Sequence) or len(bounds) != 4:
        raise ValueError(f"Invalid bounds in boosted manifest: {bounds!r}")
    if resolution <= 0.0:
        raise ValueError(f"Invalid resolution in boosted manifest: {resolution!r}")

    grid_height = int(grid_shape[1])
    x_idx = int(cell_id) // grid_height
    y_idx = int(cell_id) % grid_height
    xmin = float(bounds[0])
    ymin = float(bounds[2])
    return [
        xmin + (float(x_idx) + 0.5) * resolution,
        ymin + (float(y_idx) + 0.5) * resolution,
        0.0,
    ]


def _build_state_lookup(object_manifest: Mapping[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    renderer_state_table = object_manifest.get("renderer_state_table")
    if not isinstance(renderer_state_table, Mapping):
        raise ValueError("Object manifest missing renderer_state_table")

    state_keys = renderer_state_table.get("state_key")
    observations = renderer_state_table.get("observation")
    if not isinstance(state_keys, np.ndarray):
        raise ValueError("renderer_state_table.state_key must be a numpy array")
    if not isinstance(observations, Sequence):
        raise ValueError("renderer_state_table.observation must be a sequence")
    if len(observations) != int(state_keys.shape[0]):
        raise ValueError("renderer_state_table state_key and observation length mismatch")

    lookup: Dict[str, Dict[str, List[float]]] = {}
    for idx, raw_key in enumerate(state_keys.tolist()):
        obs = observations[idx]
        if not isinstance(obs, Mapping):
            raise ValueError(f"renderer_state_table observation[{idx}] must be a mapping")
        lookup[str(raw_key)] = {
            str(k): [float(v) for v in values]
            for k, values in obs.items()
            if isinstance(values, Sequence)
        }
    return lookup


def _csr_chain_ids_for_cell(csr: Mapping[str, Any], cell_id: int) -> List[int]:
    cell_ids = csr.get("cell_ids")
    indptr = csr.get("indptr")
    chain_ids = csr.get("chain_ids")
    if not isinstance(cell_ids, np.ndarray) or not isinstance(indptr, np.ndarray) or not isinstance(chain_ids, np.ndarray):
        raise ValueError("Invalid CSR payload in boosted manifest")

    idx = int(np.searchsorted(cell_ids, int(cell_id)))
    if idx >= int(cell_ids.shape[0]) or int(cell_ids[idx]) != int(cell_id):
        return []
    start = int(indptr[idx])
    end = int(indptr[idx + 1])
    return [int(v) for v in chain_ids[start:end].tolist()]


def _select_chain_ids(
    chain_ids: Sequence[int],
    sample_policy: str,
    max_samples_per_cell: Optional[int],
) -> List[int]:
    ordered = [int(v) for v in chain_ids]
    if not ordered:
        return []

    if sample_policy == "canonical":
        selected = [ordered[0]]
    elif sample_policy == "all":
        selected = ordered
    else:
        raise ValueError(f"Unsupported sample_policy={sample_policy!r}")

    if max_samples_per_cell is not None:
        return selected[: max(1, int(max_samples_per_cell))]
    return selected


def build_synthetic_episode(
    manifest: Mapping[str, Any],
    object_manifest: Mapping[str, Any],
    *,
    chain_id: int,
    cell_id: int,
    horizon: int,
    solution_count_for_cell: int,
) -> Dict[str, Any]:
    """Build one synthetic episode dictionary from a boosted chain/cell target."""

    env_meta = manifest.get("environment", {})
    summary = manifest.get("summary", {})
    grid_metadata = manifest.get("grid_metadata", {})
    chain_pool = object_manifest["chain_pool"]
    baseline_state_key = str(object_manifest["baseline_state_key"])
    state_lookup = _build_state_lookup(object_manifest)
    chain_records = reconstruct_chain_records(chain_pool, chain_id=chain_id)

    if baseline_state_key not in state_lookup:
        raise ValueError(f"Baseline state key {baseline_state_key!r} missing from renderer state table")

    state_observations: List[Dict[str, List[float]]] = [dict(state_lookup[baseline_state_key])]
    action_sequence: List[Dict[str, Any]] = []
    for record in chain_records:
        result_state_key = str(record.get("result_state_key", ""))
        if result_state_key not in state_lookup:
            raise ValueError(
                f"State key {result_state_key!r} from chain {chain_id} missing from renderer state table"
            )

        goal_pose = record["goal_pose"]
        action_sequence.append(
            {
                "object_id": str(object_manifest["object_id"]),
                "target": [float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])],
                "edge_idx": int(record["edge_idx"]),
                "depth": int(record["depth_idx"]),
            }
        )
        state_observations.append(dict(state_lookup[result_state_key]))

    target_pose = cell_id_to_world_pose(int(cell_id), grid_metadata)
    object_stats = object_manifest.get("stats", {})
    task_id = str(env_meta.get("task_id", "task"))
    object_id = str(object_manifest["object_id"])
    episode_id = f"{task_id}_boosted_{object_id}_h{horizon}_cell{int(cell_id)}_chain{int(chain_id)}"

    return {
        "episode_id": episode_id,
        "algorithm": "boosted_data_collection",
        "solution_found": True,
        "solution_depth": len(action_sequence),
        "search_time_ms": -1.0,
        "nodes_expanded": int(object_stats.get("states_expanded", -1)),
        "action_sequence": action_sequence,
        "state_observations": state_observations,
        "static_object_info": env_meta.get("static_object_info", {}),
        "robot_goal": list(target_pose),
        "region_goals_sampled": [list(target_pose)],
        "region_goal_used": list(target_pose),
        "reachable_objects_before_action": [
            list(summary.get("baseline_reachable_objects", []))
        ],
        "xml_file": str(env_meta.get("xml_path", "")),
        "algorithm_stats": {
            "solutions_found_for_neighbour": int(solution_count_for_cell),
            "solutions_total_for_neighbour": int(solution_count_for_cell),
            "pushes_total_for_neighbour": int(object_stats.get("transitions_evaluated", len(action_sequence))),
        },
        "target_cell_id": int(cell_id),
        "target_cell_pose": list(target_pose),
    }


def iter_training_episodes(
    manifest: Mapping[str, Any],
    *,
    sample_policy: str = "canonical",
    max_horizon: Optional[int] = 2,
    max_cells_per_object: Optional[int] = None,
    max_samples_per_cell: Optional[int] = 1,
) -> Iterator[Dict[str, Any]]:
    """Yield deterministic synthetic episodes from one boosted manifest."""

    objects = manifest.get("objects", [])
    if not isinstance(objects, Sequence):
        raise ValueError("Boosted manifest objects must be a sequence")

    for object_manifest in objects:
        if not isinstance(object_manifest, Mapping):
            continue

        object_limit = None if max_cells_per_object is None else max(1, int(max_cells_per_object))
        yielded_for_object = 0
        max_object_horizon = int(object_manifest.get("max_horizon", 0))
        if max_horizon is not None:
            max_object_horizon = min(max_object_horizon, max(1, int(max_horizon)))

        horizon_cell_ids = object_manifest.get("horizon_cell_ids", [])
        horizon_csr = object_manifest.get("horizon_cell_to_chain_csr", [])
        for horizon_idx in range(max_object_horizon):
            cell_ids = horizon_cell_ids[horizon_idx]
            csr = horizon_csr[horizon_idx]
            if not isinstance(cell_ids, np.ndarray):
                raise ValueError("Boosted manifest horizon_cell_ids entries must be numpy arrays")

            for cell_id in cell_ids.tolist():
                if object_limit is not None and yielded_for_object >= object_limit:
                    break

                chain_ids = _csr_chain_ids_for_cell(csr, int(cell_id))
                if not chain_ids:
                    continue

                selected_chain_ids = _select_chain_ids(
                    chain_ids=chain_ids,
                    sample_policy=sample_policy,
                    max_samples_per_cell=max_samples_per_cell,
                )
                for chain_id in selected_chain_ids:
                    yield build_synthetic_episode(
                        manifest,
                        object_manifest,
                        chain_id=int(chain_id),
                        cell_id=int(cell_id),
                        horizon=horizon_idx + 1,
                        solution_count_for_cell=len(chain_ids),
                    )

                yielded_for_object += 1

            if object_limit is not None and yielded_for_object >= object_limit:
                break
