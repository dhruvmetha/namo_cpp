"""Regression for twohop_four_00001_v6, trial 3, planning call 6.

The wall leaves one isolated removable cell and a 185-cell patch on obj_4b.
Only the larger patch joins the robot region to region_4. Flooding the first
cell misses that edge; combining all patch boundaries invents region_3 edges.
"""
import math

from conftest import REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()

import namo_rl


def test_wall_split_footprint_connects_only_regions_touching_the_same_patch():
    env = namo_rl.RLEnvironment(
        str(REPO_ROOT / 'python/tests/data/split_movable_footprint_fixture.xml'),
        str(REPO_ROOT / 'config/margin_1mm/namo_config_complete_skill15_car_1x.yaml'),
        False, True,
    )
    env.set_robot_pose(.3636293319275133, .1330359131197642,
                       math.radians(168.80890479022423))
    env.warm_up()
    before = env.get_full_state().qpos
    for _ in range(2):
        snapshot = env.get_region_snapshot()
        assert 'region_4' in snapshot['adjacency']['robot']
        assert 'robot' in snapshot['adjacency']['region_4']
        assert snapshot['edge_objects']['robot']['region_4'] == {'obstacle_3_movable'}
        assert snapshot['edge_objects']['region_4']['robot'] == {'obstacle_3_movable'}
        assert snapshot['adjacency']['region_3'] == set()
        assert not snapshot['goal_reachable']
        assert env.get_full_state().qpos == before
