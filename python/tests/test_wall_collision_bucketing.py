"""A block brushing a wall is not a robot crashing into one.

`wall_collision` comes back true from two sites in namo_push_controller.cpp that mean opposite
things: line 636, a robot body hitting a static object, which returns false so the push never
happened; and line 645, the pushed object touching a static object, which that site's own comment
calls "a normal outcome, not a failed push".

region_opening bucketed both as `push_collided_with_wall`. That made the counter unreadable and,
worse, marked a perfectly good push as sim-failed so it never reached the opened / did-not-open
decision. The hardware side found it on a real full_namo run: 100 of 132 attempts read as wall
collisions with no way to tell how many were a block brushing a wall on its way to a working push.

Only the fatal path writes a failure_reason, which is the discriminator these tests pin.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from namo.planners.opening.region_opening import classify_sim_outcome  # noqa: E402


def test_robot_hitting_a_wall_is_a_failure():
    outcome, benign = classify_sim_outcome({
        "wall_collision": "true",
        "failure_type": "2",
        "failure_reason": "Robot collision during push with static object: wall_left (via base)",
    })
    assert outcome == "robot_collided_with_wall"
    assert benign is False


def test_block_brushing_a_wall_falls_through_to_the_region_check():
    outcome, benign = classify_sim_outcome({
        "wall_collision": "true",
        "failure_type": "0",
        "failure_reason": "",
    })
    assert outcome is None, "a benign touch must not claim an outcome bucket"
    assert benign is True


def test_a_clean_push_claims_nothing():
    assert classify_sim_outcome({"failure_type": "0", "failure_reason": ""}) == (None, False)


def test_unreachable_edge_still_outranks_a_wall_flag():
    outcome, benign = classify_sim_outcome({
        "failure_type": "4",
        "failure_reason": "Requested edge 12 not reachable",
        "wall_collision": "true",
    })
    assert outcome == "edge_unreachable"
    assert benign is False


def test_stuck_still_outranks_a_wall_flag():
    outcome, _ = classify_sim_outcome({
        "stuck": "true", "failure_type": "3", "wall_collision": "true",
        "failure_reason": "controller stuck",
    })
    assert outcome == "controller_stuck"


def test_the_old_bucket_name_is_gone():
    """The rename is the point: `push_collided_with_wall` meant two things and must not come back."""
    for info in ({"wall_collision": "true", "failure_reason": "Robot collision during push"},
                 {"wall_collision": "true", "failure_reason": ""}):
        assert classify_sim_outcome(info)[0] != "push_collided_with_wall"
