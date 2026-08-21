"""What best_first picks, pinned to exact chains and simulation counts.

These chains were recorded while the search still lived in
scripts/sandbox/eval_bestfirst.py, reached by inserting that directory on
sys.path at call time, and they are what proved the move into
namo.planners.opening.best_first_search changed nothing. They stay as the
tripwire: fixed scene, fixed seed, exact chain and simulation count.

test_best_first_protocol_defaults.py pins the search parameters to the protocol
the registry numbers were measured under. This file pins the answers.

A failure here is not automatically a bug. It means the search now behaves
differently, and someone has to decide whether that was intended. If it was,
re-record the values below in the same commit that changed the search, and say
in the message which numbers moved and why. Silently re-recording them defeats
the point of the file.

Re-record with the _solve() call below, printing the chain and sims.

Uniform prior deliberately, not model. It exercises the same search loop,
ordering and budget accounting without needing a torch checkpoint, so the
tripwire runs anywhere the binding does.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# The one scene on this box where robot and goal sit in different regions.
SEPARATED_SCENE = REPO_ROOT / "data" / "benchmark_env.xml"
SEPARATED_CONFIG = REPO_ROOT / "config" / "namo_config_complete_skill15.yaml"

# The boundary select_boundary_from_xml chooses on this scene, pinned because
# the search is only comparable across runs when its inputs are identical.
EXPECTED_TARGET_LABEL = "goal"
EXPECTED_BLOCKERS = ["obstacle_9_movable"]
EXPECTED_SAMPLE_COUNT = 100

# (seed, simulations, chain) recorded 2026-08-19 against build_python at 42de63c.
# Different seeds must give different work, otherwise the seed is not reaching
# the search and a "deterministic" pass would mean nothing.
#
# Seed 1234 re-recorded 2026-08-20, in the commit that made a push keep its
# motion when the object jams instead of rolling back. It moved from a two-push
# chain at 8 simulations, ((10, 0), (24, 4)), to a single push at 2, (58, 4):
# push 58 wedges the object and that now counts as progress, so the search stops
# there rather than searching on for a chain that ends clear. Seeds 42 and 7 are
# untouched, chain and simulation count both, because neither of their pushes
# ends against anything.
RECORDED_RUNS = [
    (42, 3, (("obstacle_9_movable", 2, 0), ("obstacle_9_movable", 52, 1))),
    (7, 7, (("obstacle_9_movable", 40, 0), ("obstacle_9_movable", 54, 0))),
    (1234, 2, (("obstacle_9_movable", 58, 4),)),
]

pytestmark = pytest.mark.skip(
    reason="recorded against the retired point-robot fixture; needs a car 1x d5 re-recording",
)


def _scene_goal():
    match = re.search(
        r'<site name="goal".*?pos="([-\d.eE ]+)"', SEPARATED_SCENE.read_text(), re.S
    )
    x, y = (float(v) for v in match.group(1).split()[:2])
    return (x, y, 0.0)


@pytest.fixture(scope="module")
def service():
    from namo.services import NAMOPlanningService

    return NAMOPlanningService(
        config_path=str(SEPARATED_CONFIG), primitive_data_dir=str(REPO_ROOT / "data")
    )


@pytest.fixture(scope="module")
def selection(service):
    """The boundary and points every solve below is graded against."""
    return service.select_boundary_from_xml(str(SEPARATED_SCENE), _scene_goal())


def _solve(service, selection, seed):
    result = service.solve_boundary_from_xml(
        str(SEPARATED_SCENE),
        _scene_goal(),
        target_points=selection.target_points,
        blocking_objects=selection.blocking_objects,
        local_search="best_first",
        best_first_prior="uniform",
        shuffle_seed=seed,
    )
    chain = tuple((a.object_id, a.edge_idx, a.depth) for a in result.actions)
    return result, chain


def test_the_scene_still_presents_the_boundary_these_runs_were_recorded_against(
    selection,
):
    """A changed scene or sampler invalidates every chain below, loudly."""
    assert selection.found
    assert selection.target_label == EXPECTED_TARGET_LABEL
    assert selection.blocking_objects == EXPECTED_BLOCKERS
    assert len(selection.target_points) == EXPECTED_SAMPLE_COUNT


@pytest.mark.parametrize("seed,sims,chain", RECORDED_RUNS)
def test_the_search_still_picks_the_pushes_it_was_recorded_picking(
    service, selection, seed, sims, chain
):
    result, actual = _solve(service, selection, seed)

    assert result.success is True
    assert result.failure_reason == "success"
    assert actual == chain
    assert result.simulations_used == sims


def test_the_same_seed_twice_in_one_process_is_identical(service, selection):
    """Without this, a matching chain could be luck rather than the seed."""
    first, first_chain = _solve(service, selection, 42)
    second, second_chain = _solve(service, selection, 42)

    assert first_chain == second_chain
    assert first.simulations_used == second.simulations_used


def test_different_seeds_do_different_work(service, selection):
    """Proves the seed reaches the search, so determinism is not just a constant."""
    chains = {seed: chain for seed, _sims, chain in RECORDED_RUNS}

    assert len(set(chains.values())) == len(chains)
    assert len({sims for _seed, sims, _chain in RECORDED_RUNS}) == len(RECORDED_RUNS)
