"""Jam pruning must not cross object boundaries.

The search remembers the shallowest depth that jammed on an edge and skips deeper pushes there,
because push_steps = depth+1 and one continuous push means depth k+1 retraces k's trajectory. That
argument is about ONE object on ONE edge. It says nothing about a different block that happens to
carry the same edge number.

Edge indices are per-object and both movables in a doorway number theirs over the same range, so
keying the jam table on the bare edge silently deletes untried pushes on the other block. Measured on
the two-movable pool before the fix: 988 of 1548 scenes have the two blocks sharing an edge number,
median 7 shared, and in 732 of those a SOLVING push sits on a shared number. Nearly half the pool
could lose its solution to a jam recorded against the other block.

This never mattered while a board held one object. It became reachable when the region graph started
naming both blocks on a doorway, which put two objects in one candidate pool.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from namo.planners.opening.best_first_search import (  # noqa: E402
    _record_state_local_jam,
    _state_local_live_candidates,
)


class _Goal:
    def __init__(self, edge, depth):
        self.edge_idx = edge
        self.depth = depth


class _Step:
    def __init__(self, failed):
        self.info = {"failure_reason": "stuck"} if failed else {}


def _pool():
    """Both blocks offering edge 7 at depths 0..2, which is the collision that used to bite."""
    return [(obj, _Goal(7, d), 1.0) for obj in ("block_a", "block_b") for d in range(3)]


def test_a_jam_on_one_block_leaves_the_other_blocks_same_edge_alive():
    jam_at = {}
    _record_state_local_jam(jam_at, "block_a", _Goal(7, 1), _Step(True), True)

    live = _state_local_live_candidates(_pool(), banned=set(), jam_at=jam_at,
                                        prune_jam_depth=True)
    survivors = {(obj, g.depth) for obj, g, _ in live}

    # block_a loses depth 1 and deeper on edge 7, which is the whole point of the pruning.
    assert ("block_a", 1) not in survivors
    assert ("block_a", 2) not in survivors
    assert ("block_a", 0) in survivors, "a SHORTER push may stop before the obstruction"

    # block_b is a different object. Its edge 7 was never tried and must still be offered.
    assert ("block_b", 0) in survivors
    assert ("block_b", 1) in survivors, "block_b's edge 7 was cancelled by block_a's jam"
    assert ("block_b", 2) in survivors


def test_pruning_still_works_within_one_object():
    """The fix must not disable the saving it is guarding."""
    jam_at = {}
    _record_state_local_jam(jam_at, "block_a", _Goal(7, 1), _Step(True), True)
    live = _state_local_live_candidates(
        [("block_a", _Goal(7, d), 1.0) for d in range(3)],
        banned=set(), jam_at=jam_at, prune_jam_depth=True)
    assert {g.depth for _, g, _ in live} == {0}


def test_a_push_that_did_not_fail_records_nothing():
    jam_at = {}
    _record_state_local_jam(jam_at, "block_a", _Goal(7, 1), _Step(False), True)
    assert jam_at == {}


def test_the_switch_still_turns_it_off():
    jam_at = {}
    _record_state_local_jam(jam_at, "block_a", _Goal(7, 1), _Step(True), False)
    assert jam_at == {}
    jam_at = {("block_a", 7): 1}
    live = _state_local_live_candidates(_pool(), banned=set(), jam_at=jam_at,
                                        prune_jam_depth=False)
    assert len(live) == 6, "with pruning off every candidate survives"
