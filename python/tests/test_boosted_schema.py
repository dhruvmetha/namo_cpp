import numpy as np

from namo.boosted_data_collection.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_cell_chain_csr,
    reconstruct_chain,
    reconstruct_chain_records,
    validate_manifest_schema,
)


def test_build_cell_chain_csr_sorted_and_deduped():
    csr = build_cell_chain_csr(
        {
            7: [5, 2, 2, 1],
            3: [9, 4],
        }
    )

    assert csr.cell_ids.tolist() == [3, 7]
    assert csr.indptr.tolist() == [0, 2, 5]
    assert csr.chain_ids.tolist() == [4, 9, 1, 2, 5]


def test_reconstruct_chain_parent_pointers():
    chain_pool = {
        "chain_id": np.asarray([0, 1, 2], dtype=np.int32),
        "parent_id": np.asarray([-1, 0, 1], dtype=np.int32),
        "edge_idx": np.asarray([-1, 4, 6], dtype=np.int16),
        "depth_idx": np.asarray([-1, 0, 2], dtype=np.int16),
        "goal_x": np.asarray([np.nan, 1.0, 2.0], dtype=np.float32),
        "goal_y": np.asarray([np.nan, 3.0, 4.0], dtype=np.float32),
        "goal_theta": np.asarray([np.nan, 0.1, 0.2], dtype=np.float32),
        "result_state_key": np.asarray(["baseline", "s1", "s2"], dtype="U32"),
    }

    chain = reconstruct_chain(chain_pool, chain_id=2)
    assert len(chain) == 2
    assert chain[0][:2] == (4, 0)
    assert chain[1][:2] == (6, 2)
    assert np.allclose(chain[0][2:], (1.0, 3.0, 0.1))
    assert np.allclose(chain[1][2:], (2.0, 4.0, 0.2))


def test_reconstruct_chain_records_includes_result_state_keys():
    chain_pool = {
        "chain_id": np.asarray([0, 1, 2], dtype=np.int32),
        "parent_id": np.asarray([-1, 0, 1], dtype=np.int32),
        "edge_idx": np.asarray([-1, 4, 6], dtype=np.int16),
        "depth_idx": np.asarray([-1, 0, 2], dtype=np.int16),
        "goal_x": np.asarray([np.nan, 1.0, 2.0], dtype=np.float32),
        "goal_y": np.asarray([np.nan, 3.0, 4.0], dtype=np.float32),
        "goal_theta": np.asarray([np.nan, 0.1, 0.2], dtype=np.float32),
        "result_state_key": np.asarray(["baseline", "s1", "s2"], dtype="U32"),
    }

    records = reconstruct_chain_records(chain_pool, chain_id=2)
    assert [record["result_state_key"] for record in records] == ["s1", "s2"]
    assert records[0]["edge_idx"] == 4
    assert records[1]["depth_idx"] == 2


def test_manifest_schema_validator_accepts_array_first_layout():
    obj = {
        "object_id": "box_1",
        "max_horizon": 2,
        "baseline_state_key": "state0",
        "renderer_state_table": {
            "state_key": np.asarray(["state0", "state1"], dtype="U32"),
            "observation": [
                {"robot_pose": [0.0, 0.0, 0.0], "box_1_pose": [1.0, 2.0, 0.1]},
                {"robot_pose": [0.0, 0.0, 0.0], "box_1_pose": [1.5, 2.0, 0.1]},
            ],
        },
        "horizon_cell_ids": [
            np.asarray([1, 3], dtype=np.int32),
            np.asarray([4], dtype=np.int32),
        ],
        "horizon_cell_to_chain_csr": [
            {
                "cell_ids": np.asarray([1, 3], dtype=np.int32),
                "indptr": np.asarray([0, 2, 3], dtype=np.int64),
                "chain_ids": np.asarray([1, 2, 4], dtype=np.int32),
            },
            {
                "cell_ids": np.asarray([4], dtype=np.int32),
                "indptr": np.asarray([0, 1], dtype=np.int64),
                "chain_ids": np.asarray([8], dtype=np.int32),
            },
        ],
        "chain_pool": {
            "chain_id": np.asarray([0, 1], dtype=np.int32),
            "parent_id": np.asarray([-1, 0], dtype=np.int32),
            "edge_idx": np.asarray([-1, 2], dtype=np.int16),
            "depth_idx": np.asarray([-1, 0], dtype=np.int16),
            "goal_x": np.asarray([np.nan, 0.1], dtype=np.float32),
            "goal_y": np.asarray([np.nan, 0.2], dtype=np.float32),
            "goal_theta": np.asarray([np.nan, 0.3], dtype=np.float32),
            "result_state_key": np.asarray(["state0", "state1"], dtype="U32"),
        },
        "stats": {"opened_cells_total": 3},
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "producer_version": "1.0.0",
        "run_metadata": {"seed": 42},
        "grid_metadata": {
            "grid_shape": [10, 20],
            "resolution": 0.02,
            "bounds": [0.0, 1.0, 0.0, 2.0],
            "cell_indexing_convention": "x_major_flat_index=x*H+y",
        },
        "environment": {
            "task_id": "env_0001",
            "xml_path": "env.xml",
            "config_file": "config.yaml",
            "static_object_info": {"robot": {"size_x": 0.2, "size_y": 0.2}},
        },
        "objects": [obj],
    }

    assert validate_manifest_schema(manifest) == []
