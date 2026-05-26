import gzip
import pickle

import h5py
import numpy as np

from namo.boosted_data_collection.episode_builder import iter_training_episodes
from namo.boosted_data_collection.export_sage_hdf5 import export_manifests_to_hdf5
from namo.boosted_data_collection.schema import SCHEMA_NAME, SCHEMA_VERSION


def _sample_manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "producer_version": "1.0.0",
        "run_metadata": {"seed": 42},
        "grid_metadata": {
            "grid_shape": [2, 2],
            "resolution": 0.5,
            "bounds": [0.0, 1.0, 0.0, 1.0],
            "cell_indexing_convention": "x_major_flat_index=x*grid_height+y",
        },
        "environment": {
            "task_id": "env_0001",
            "xml_path": "env.xml",
            "config_file": "config.yaml",
            "static_object_info": {
                "robot": {"size_x": 0.2, "size_y": 0.2},
                "box_1": {"size_x": 0.5, "size_y": 0.25},
            },
        },
        "summary": {
            "baseline_reachable_objects": ["box_1"],
            "candidate_object_ids": ["box_1"],
            "region_snapshot_source": "cpp",
            "region_snapshot_robot_label": "robot",
        },
        "objects": [
            {
                "object_id": "box_1",
                "max_horizon": 2,
                "baseline_state_key": "s0",
                "renderer_state_table": {
                    "state_key": np.asarray(["s0", "s1", "s2"], dtype="U32"),
                    "observation": [
                        {"robot_pose": [0.0, 0.0, 0.0], "box_1_pose": [0.0, 0.0, 0.0]},
                        {"robot_pose": [0.0, 0.0, 0.0], "box_1_pose": [0.5, 0.0, 0.0]},
                        {"robot_pose": [0.0, 0.0, 0.0], "box_1_pose": [0.5, 0.5, 0.0]},
                    ],
                },
                "horizon_cell_ids": [
                    np.asarray([1], dtype=np.int32),
                    np.asarray([2], dtype=np.int32),
                ],
                "horizon_cell_to_chain_csr": [
                    {
                        "cell_ids": np.asarray([1], dtype=np.int32),
                        "indptr": np.asarray([0, 1], dtype=np.int64),
                        "chain_ids": np.asarray([1], dtype=np.int32),
                    },
                    {
                        "cell_ids": np.asarray([2], dtype=np.int32),
                        "indptr": np.asarray([0, 1], dtype=np.int64),
                        "chain_ids": np.asarray([2], dtype=np.int32),
                    },
                ],
                "chain_pool": {
                    "chain_id": np.asarray([0, 1, 2], dtype=np.int32),
                    "parent_id": np.asarray([-1, 0, 1], dtype=np.int32),
                    "edge_idx": np.asarray([-1, 4, 6], dtype=np.int16),
                    "depth_idx": np.asarray([-1, 0, 1], dtype=np.int16),
                    "goal_x": np.asarray([np.nan, 0.25, 0.75], dtype=np.float32),
                    "goal_y": np.asarray([np.nan, 0.25, 0.75], dtype=np.float32),
                    "goal_theta": np.asarray([np.nan, 0.0, 0.0], dtype=np.float32),
                    "result_state_key": np.asarray(["s0", "s1", "s2"], dtype="U32"),
                },
                "stats": {
                    "states_expanded": 2,
                    "transitions_evaluated": 3,
                    "opened_cells_total": 2,
                },
            }
        ],
    }


def test_iter_training_episodes_builds_renderer_ready_samples():
    episodes = list(iter_training_episodes(_sample_manifest(), sample_policy="canonical", max_horizon=2))
    assert len(episodes) == 2

    ep0 = episodes[0]
    assert ep0["solution_depth"] == 1
    assert ep0["action_sequence"][0]["object_id"] == "box_1"
    assert ep0["state_observations"][0]["box_1_pose"] == [0.0, 0.0, 0.0]
    assert ep0["state_observations"][1]["box_1_pose"] == [0.5, 0.0, 0.0]
    assert ep0["robot_goal"] == [0.25, 0.75, 0.0]


def test_export_manifests_to_hdf5_writes_sage_keys(tmp_path, monkeypatch):
    manifest_path = tmp_path / "env_0001_boosted_manifest.pkl.gz"
    with gzip.open(manifest_path, "wb") as f:
        pickle.dump(_sample_manifest(), f, protocol=pickle.HIGHEST_PROTOCOL)

    class _DummyVisualizer:
        def __init__(self, *args, **kwargs):
            pass

    class _TestHDF5Writer:
        def __init__(self, output_path):
            self.output_path = output_path
            self.h5f = None
            self.current_idx = 0
            self.datasets = {}

        def __enter__(self):
            self.h5f = h5py.File(self.output_path, "w")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.h5f.attrs["n_samples"] = self.current_idx
            self.h5f.close()
            return False

        def add_sample(self, masks, metadata):
            if not self.datasets:
                for key, arr in masks.items():
                    self.datasets[key] = self.h5f.create_dataset(
                        key,
                        shape=(0,) + arr.shape,
                        maxshape=(None,) + arr.shape,
                        dtype=arr.dtype,
                    )
                self.datasets["solution_depth"] = self.h5f.create_dataset(
                    "solution_depth", shape=(0,), maxshape=(None,), dtype=np.int32
                )
                self.datasets["robot_goal"] = self.h5f.create_dataset(
                    "robot_goal", shape=(0, 3), maxshape=(None, 3), dtype=np.float32
                )

            for key, arr in masks.items():
                ds = self.datasets[key]
                ds.resize(self.current_idx + 1, axis=0)
                ds[self.current_idx] = arr

            self.datasets["solution_depth"].resize(self.current_idx + 1, axis=0)
            self.datasets["solution_depth"][self.current_idx] = int(metadata["solution_depth"])
            self.datasets["robot_goal"].resize(self.current_idx + 1, axis=0)
            self.datasets["robot_goal"][self.current_idx] = metadata["robot_goal"]
            self.current_idx += 1

    def _fake_process_episode(episode, _visualizer, **_kwargs):
        zeros = np.zeros((8, 8), dtype=np.float32)
        masks = {
            "local_static": zeros,
            "local_movable": zeros,
            "local_target_object": zeros,
            "local_robot_region": zeros,
            "local_goal_sample_region": zeros,
            "local_goal_mask_a1": zeros,
            "local_goal_mask_a2": zeros,
        }
        metadata = {
            "episode_id": episode["episode_id"],
            "task_id": "env_0001",
            "algorithm": episode["algorithm"],
            "solution_depth": episode["solution_depth"],
            "search_time_ms": episode["search_time_ms"],
            "nodes_expanded": episode["nodes_expanded"],
            "robot_goal": episode["robot_goal"],
            "xml_file": episode["xml_file"],
            "solutions_found": 1,
            "solutions_total": 1,
            "pushes_total": len(episode["action_sequence"]),
        }
        return masks, metadata

    monkeypatch.setattr(
        "namo.boosted_data_collection.export_sage_hdf5._load_visualization_components",
        lambda: (_DummyVisualizer, _TestHDF5Writer, _fake_process_episode),
    )

    output_h5 = tmp_path / "data.h5"
    summary = export_manifests_to_hdf5(
        [manifest_path],
        str(output_h5),
        local_only=True,
        local_crop_size=5.0,
        sample_policy="canonical",
        max_horizon=2,
        max_cells_per_object=None,
        max_samples_per_cell=1,
        verbose=False,
    )

    assert summary["episodes_written"] == 2
    with h5py.File(output_h5, "r") as h5f:
        assert int(h5f.attrs["n_samples"]) == 2
        for key in [
            "local_static",
            "local_movable",
            "local_target_object",
            "local_robot_region",
            "local_goal_sample_region",
            "local_goal_mask_a1",
            "local_goal_mask_a2",
            "solution_depth",
            "robot_goal",
        ]:
            assert key in h5f
