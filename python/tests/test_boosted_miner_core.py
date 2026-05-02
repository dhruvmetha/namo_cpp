from dataclasses import dataclass

import numpy as np

from namo.boosted_data_collection.miner import (
    EarliestHorizonIndex,
    TransitionMemo,
    classify_step_result,
    mine_object_manifest,
)


@dataclass
class _FakeState:
    sid: int

    @property
    def qpos(self):
        return [float(self.sid)]

    @property
    def qvel(self):
        return [0.0]


@dataclass
class _FakeStepResult:
    done: bool
    info: dict


class _FakeEnv:
    class Action:
        def __init__(self):
            self.object_id = ""
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
            self.edge_idx = -1
            self.depth = -1

    def __init__(self):
        self.state_id = 0

    def get_full_state(self):
        return _FakeState(self.state_id)

    def set_full_state(self, state):
        self.state_id = int(state.sid)

    def get_observation(self):
        # Keep target pose always valid for direct primitive execution.
        return {"box_1_pose": [0.0, 0.0, 0.0], "robot_pose": [0.0, 0.0, 0.0]}

    def get_primitive_library_target_pose(self, _object_id, edge_idx, depth_idx):
        # Deterministic stub for primitive-library target lookup.
        return [0.1 * float(edge_idx), 0.01 * float(depth_idx), 0.0]

    def get_valid_primitive_depth_indices(self, _object_id, _edge_idx):
        return [0, 1]

    def get_reachability_summary(self, _analysis_mode=False):
        return {
            "objects": {
                "box_1": {
                    "total_edges": 2,
                    "total_primitives": 4,
                }
            }
        }

    def get_wavefront_snapshot_for_object(self, _object_id):
        if self.state_id == 0:
            free = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
            reachable = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
            edges = [0, 1]
        elif self.state_id == 1:
            free = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
            reachable = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
            edges = []
        else:
            free = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
            reachable = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
            edges = []

        return {
            "free_mask": free,
            "reachable_mask": reachable,
            "reachable_edges": edges,
            "resolution": 0.1,
            "bounds": [0.0, 1.0, 0.0, 1.0],
            "grid_shape": [2, 2],
        }

    def step(self, action):
        # state=0 transitions
        if self.state_id == 0 and action.edge_idx == 0 and action.depth == 0:
            return _FakeStepResult(
                done=False,
                info={
                    "failure_reason": "collision with wall",
                    "wall_collision": "true",
                    "stuck": "false",
                },
            )

        if self.state_id == 0 and action.edge_idx == 0 and action.depth == 1:
            self.state_id = 1
            return _FakeStepResult(done=True, info={"failure_reason": "", "stuck": "false"})

        if self.state_id == 0 and action.edge_idx == 1 and action.depth == 0:
            return _FakeStepResult(
                done=False,
                info={
                    "failure_reason": "Controller-level stuck",
                    "stuck": "true",
                },
            )

        if self.state_id == 0 and action.edge_idx == 1 and action.depth == 1:
            self.state_id = 2
            return _FakeStepResult(done=True, info={"failure_reason": ""})

        return _FakeStepResult(done=False, info={"failure_reason": "Action not applicable"})


def test_earliest_horizon_index_invariants():
    idx = EarliestHorizonIndex(num_cells=10)
    h1 = {}
    h2 = {}

    idx.record_cells(horizon=1, cell_ids=np.asarray([1, 2], dtype=np.int32), chain_id=4, horizon_cell_to_chain_ids=h1)
    idx.record_cells(horizon=2, cell_ids=np.asarray([2, 3], dtype=np.int32), chain_id=6, horizon_cell_to_chain_ids=h2)

    # Cell 2 remains owned by horizon 1 and must not be inserted at horizon 2.
    assert sorted(h1.keys()) == [1, 2]
    assert sorted(h2.keys()) == [3]


def test_transition_memo_hit_miss_accounting():
    memo = TransitionMemo()
    key = (10, 2, 1)

    assert memo.get(key) is None
    memo.put(key, "value")
    assert memo.get(key) == "value"
    assert memo.misses == 1
    assert memo.hits == 1


def test_classify_step_result_pruning_policy():
    coll = _FakeStepResult(done=False, info={"failure_reason": "collision with wall", "wall_collision": "true"})
    stuck = _FakeStepResult(done=False, info={"failure_reason": "Controller-level stuck", "stuck": "true"})

    coll_status = classify_step_result(coll)
    stuck_status = classify_step_result(stuck)

    assert coll_status["collision"] is True
    assert coll_status["prune_deeper"] is False
    assert stuck_status["stuck"] is True
    assert stuck_status["prune_deeper"] is True


def test_mine_object_manifest_collision_does_not_prune_but_stuck_does():
    env = _FakeEnv()
    baseline = env.get_full_state()

    cfg = {
        "boosted_max_horizon": 1,
        "boosted_same_object_only": True,
        "boosted_use_cpp_grid_fastpath": True,
        "boosted_primitive_depth_count": 2,
    }
    out = mine_object_manifest(env, baseline_state=baseline, object_id="box_1", boosted_config=cfg)

    # One newly opened cell at horizon-1 from edge=0 depth=1.
    assert out["horizon_cell_ids"][0].tolist() == [1]

    stats = out["stats"]
    # Evaluated transitions:
    # edge0 depth0 (collision fail), edge0 depth1 (success), edge1 depth0 (stuck fail, prune depth1)
    assert stats["transitions_evaluated"] == 3
    assert stats["collision_transitions"] >= 1
    assert stats["pruned_same_edge_depth"] >= 1
