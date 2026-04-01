from dataclasses import dataclass

from namo.planners.opening.region_opening import _sort_candidates_sync


@dataclass
class _DummyGoal:
    score: float


def test_sort_candidates_depth_first_then_score():
    # Format: [edge_idx, depth_idx, goal]
    candidates = [
        [0, 2, _DummyGoal(score=6.0)],   # deep but high score
        [1, 0, _DummyGoal(score=1.0)],   # shallow low score
        [2, 0, _DummyGoal(score=6.0)],   # shallow high score
        [3, 1, _DummyGoal(score=6.0)],   # mid high score
        [4, 1, _DummyGoal(score=1.0)],   # mid low score
    ]

    _sort_candidates_sync(candidates, depth_first=True)

    got = [(c[0], c[1], c[2].score) for c in candidates]
    assert got == [
        (2, 0, 6.0),
        (1, 0, 1.0),
        (3, 1, 6.0),
        (4, 1, 1.0),
        (0, 2, 6.0),
    ]


def test_sort_candidates_score_first_then_depth():
    candidates = [
        [0, 2, _DummyGoal(score=6.0)],   # deep but high score
        [1, 0, _DummyGoal(score=1.0)],   # shallow low score
        [2, 0, _DummyGoal(score=6.0)],   # shallow high score
        [3, 1, _DummyGoal(score=6.0)],   # mid high score
        [4, 1, _DummyGoal(score=1.0)],   # mid low score
    ]

    _sort_candidates_sync(candidates, depth_first=False)

    got = [(c[0], c[1], c[2].score) for c in candidates]
    assert got == [
        (2, 0, 6.0),
        (3, 1, 6.0),
        (0, 2, 6.0),
        (1, 0, 1.0),
        (4, 1, 1.0),
    ]

