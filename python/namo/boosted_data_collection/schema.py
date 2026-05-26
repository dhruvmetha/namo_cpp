"""Schema utilities for boosted deterministic cell-opening manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 2
SCHEMA_NAME = "boosted_cell_manifest_v2"
CELL_INDEXING_CONVENTION = (
    "x_major_flat_index=x*grid_height+y; cell_center_world=(xmin+(x+0.5)*res, ymin+(y+0.5)*res)"
)


@dataclass(frozen=True)
class CSRCellChains:
    """Compact CSR-like mapping from cell ids to chain ids."""

    cell_ids: np.ndarray
    indptr: np.ndarray
    chain_ids: np.ndarray



def build_cell_chain_csr(cell_to_chain_ids: Mapping[int, Iterable[int]]) -> CSRCellChains:
    """Freeze a mutable ``cell -> chain_ids`` map to CSR arrays.

    The output is deterministic:
    - cells sorted ascending
    - chain ids for each cell sorted ascending and deduplicated
    """

    if not cell_to_chain_ids:
        return CSRCellChains(
            cell_ids=np.zeros((0,), dtype=np.int32),
            indptr=np.zeros((1,), dtype=np.int64),
            chain_ids=np.zeros((0,), dtype=np.int32),
        )

    sorted_cells = sorted(int(cell_id) for cell_id in cell_to_chain_ids.keys())
    indptr: List[int] = [0]
    flat_chain_ids: List[int] = []

    for cell_id in sorted_cells:
        chain_ids_sorted = sorted(set(int(v) for v in cell_to_chain_ids[cell_id]))
        flat_chain_ids.extend(chain_ids_sorted)
        indptr.append(len(flat_chain_ids))

    return CSRCellChains(
        cell_ids=np.asarray(sorted_cells, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.int64),
        chain_ids=np.asarray(flat_chain_ids, dtype=np.int32),
    )



def reconstruct_chain(
    chain_pool: Mapping[str, np.ndarray],
    chain_id: int,
) -> List[Tuple[int, int, float, float, float]]:
    """Reconstruct ``[(edge_idx, depth_idx, goal_x, goal_y, goal_theta), ...]`` from root."""

    parent = chain_pool["parent_id"]
    edge = chain_pool["edge_idx"]
    depth = chain_pool["depth_idx"]
    gx = chain_pool["goal_x"]
    gy = chain_pool["goal_y"]
    gt = chain_pool["goal_theta"]

    out_rev: List[Tuple[int, int, float, float, float]] = []
    cursor = int(chain_id)

    while cursor >= 0:
        if cursor == 0:
            break
        out_rev.append(
            (
                int(edge[cursor]),
                int(depth[cursor]),
                float(gx[cursor]),
                float(gy[cursor]),
                float(gt[cursor]),
            )
        )
        cursor = int(parent[cursor])

    out_rev.reverse()
    return out_rev


def reconstruct_chain_records(
    chain_pool: Mapping[str, np.ndarray],
    chain_id: int,
) -> List[Dict[str, object]]:
    """Reconstruct chain records from root to ``chain_id``.

    Each record includes the primitive parameters and, when present in the schema,
    the renderer-ready resulting state key for that chain node.
    """

    parent = chain_pool["parent_id"]
    edge = chain_pool["edge_idx"]
    depth = chain_pool["depth_idx"]
    gx = chain_pool["goal_x"]
    gy = chain_pool["goal_y"]
    gt = chain_pool["goal_theta"]
    result_state_key = chain_pool.get("result_state_key")

    out_rev: List[Dict[str, object]] = []
    cursor = int(chain_id)

    while cursor >= 0:
        if cursor == 0:
            break

        record: Dict[str, object] = {
            "chain_id": cursor,
            "edge_idx": int(edge[cursor]),
            "depth_idx": int(depth[cursor]),
            "goal_pose": (
                float(gx[cursor]),
                float(gy[cursor]),
                float(gt[cursor]),
            ),
        }
        if isinstance(result_state_key, np.ndarray):
            record["result_state_key"] = str(result_state_key[cursor])
        out_rev.append(record)
        cursor = int(parent[cursor])

    out_rev.reverse()
    return out_rev



def validate_manifest_schema(manifest: Mapping[str, object]) -> List[str]:
    """Return a list of schema validation errors (empty list means valid)."""

    errors: List[str] = []

    def require_key(container: Mapping[str, object], key: str, ctx: str) -> None:
        if key not in container:
            errors.append(f"Missing key '{ctx}.{key}'")

    top = manifest
    for key in [
        "schema_version",
        "schema_name",
        "producer_version",
        "run_metadata",
        "grid_metadata",
        "environment",
        "objects",
    ]:
        require_key(top, key, "manifest")

    grid = top.get("grid_metadata")
    if isinstance(grid, Mapping):
        for key in ["grid_shape", "resolution", "bounds", "cell_indexing_convention"]:
            require_key(grid, key, "grid_metadata")
    else:
        errors.append("grid_metadata must be a mapping")

    env_meta = top.get("environment")
    if isinstance(env_meta, Mapping):
        for key in ["task_id", "xml_path", "config_file", "static_object_info"]:
            require_key(env_meta, key, "environment")
    else:
        errors.append("environment must be a mapping")

    objs = top.get("objects")
    if not isinstance(objs, Sequence):
        errors.append("objects must be a sequence")
        return errors

    for idx, obj in enumerate(objs):
        ctx = f"objects[{idx}]"
        if not isinstance(obj, Mapping):
            errors.append(f"{ctx} must be a mapping")
            continue

        for key in [
            "object_id",
            "max_horizon",
            "baseline_state_key",
            "renderer_state_table",
            "horizon_cell_ids",
            "horizon_cell_to_chain_csr",
            "chain_pool",
            "stats",
        ]:
            require_key(obj, key, ctx)

        chain_pool = obj.get("chain_pool")
        if isinstance(chain_pool, Mapping):
            expected_chain_keys = [
                "chain_id",
                "parent_id",
                "edge_idx",
                "depth_idx",
                "goal_x",
                "goal_y",
                "goal_theta",
                "result_state_key",
            ]
            for key in expected_chain_keys:
                require_key(chain_pool, key, f"{ctx}.chain_pool")

            chain_lengths: List[int] = []
            for key in expected_chain_keys:
                value = chain_pool.get(key)
                if not isinstance(value, np.ndarray):
                    errors.append(f"{ctx}.chain_pool.{key} must be np.ndarray")
                    continue
                chain_lengths.append(int(value.shape[0]))
            if chain_lengths and len(set(chain_lengths)) != 1:
                errors.append(f"{ctx}.chain_pool arrays must share same length")
        else:
            errors.append(f"{ctx}.chain_pool must be a mapping")

        renderer_state_table = obj.get("renderer_state_table")
        if isinstance(renderer_state_table, Mapping):
            for key in ["state_key", "observation"]:
                require_key(renderer_state_table, key, f"{ctx}.renderer_state_table")

            state_keys = renderer_state_table.get("state_key")
            observations = renderer_state_table.get("observation")
            if not isinstance(state_keys, np.ndarray):
                errors.append(f"{ctx}.renderer_state_table.state_key must be np.ndarray")
            if not isinstance(observations, Sequence):
                errors.append(f"{ctx}.renderer_state_table.observation must be a sequence")
            elif isinstance(state_keys, np.ndarray) and len(observations) != int(state_keys.shape[0]):
                errors.append(
                    f"{ctx}.renderer_state_table.state_key and observation length mismatch"
                )

            if isinstance(observations, Sequence):
                for obs_idx, obs in enumerate(observations):
                    if not isinstance(obs, Mapping):
                        errors.append(
                            f"{ctx}.renderer_state_table.observation[{obs_idx}] must be a mapping"
                        )
        else:
            errors.append(f"{ctx}.renderer_state_table must be a mapping")

        horizon_cell_ids = obj.get("horizon_cell_ids")
        horizon_csr = obj.get("horizon_cell_to_chain_csr")
        if not isinstance(horizon_cell_ids, Sequence):
            errors.append(f"{ctx}.horizon_cell_ids must be a sequence")
        if not isinstance(horizon_csr, Sequence):
            errors.append(f"{ctx}.horizon_cell_to_chain_csr must be a sequence")

        if isinstance(horizon_cell_ids, Sequence) and isinstance(horizon_csr, Sequence):
            if len(horizon_cell_ids) != len(horizon_csr):
                errors.append(f"{ctx} horizon sequences length mismatch")
            for h, (cell_ids, csr) in enumerate(zip(horizon_cell_ids, horizon_csr), start=1):
                hctx = f"{ctx}.h{h}"
                if not isinstance(cell_ids, np.ndarray):
                    errors.append(f"{hctx}.cell_ids must be np.ndarray")
                if not isinstance(csr, Mapping):
                    errors.append(f"{hctx}.csr must be mapping")
                    continue
                for key in ["cell_ids", "indptr", "chain_ids"]:
                    require_key(csr, key, f"{hctx}.csr")
                csr_cell_ids = csr.get("cell_ids")
                if isinstance(cell_ids, np.ndarray) and isinstance(csr_cell_ids, np.ndarray):
                    if cell_ids.shape != csr_cell_ids.shape or not np.array_equal(cell_ids, csr_cell_ids):
                        errors.append(f"{hctx}.cell_ids and csr.cell_ids mismatch")

    return errors



def ensure_schema_or_raise(manifest: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if manifest is invalid."""

    errors = validate_manifest_schema(manifest)
    if errors:
        raise ValueError("Manifest schema validation failed:\n" + "\n".join(errors))
