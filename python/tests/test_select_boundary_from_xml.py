"""Choosing which region boundary to open next, without solving it.

An executor that runs one physical push at a time needs the boundary choice as
data it can freeze: which points define success, and which objects block it.
FullNAMOPlanner makes that choice internally (path[1] of a BFS to the goal
region) and never reports the points, so there was nothing to hold onto.

select_boundary_from_xml returns the choice. The rule itself is shared with the
planner via find_region_path, so the two cannot disagree about which boundary
is next.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import REAL_NAMO_RL
from namo.planners.full_namo.full_namo_planner import boundary_key, find_region_path
from namo.services.planning_service import _stale_boundaries

REPO_ROOT = Path(__file__).resolve().parents[2]
# A one-boundary scene whose movable blocker overlaps the divider walls' inflated
# footprints. The overlap is intentional: removing the blocker must retain wall-owned cells.
SEPARATED_SCENE = REPO_ROOT / "python" / "tests" / "data" / "region_boundary_overlap_fixture.xml"
SEPARATED_CONFIG = REPO_ROOT / "config" / "namo_config_complete_skill15_1x.yaml"

ROBOT, MIDDLE, GOAL = "robot", "region_3", "goal"
LINEAR = {ROBOT: {MIDDLE}, MIDDLE: {ROBOT, GOAL}, GOAL: {MIDDLE}}


# --- carrying a blocklist across a push --------------------------------------

def test_a_blocklist_naming_a_live_edge_is_not_stale():
    assert _stale_boundaries(LINEAR, [(ROBOT, MIDDLE)]) == []


def test_an_edge_that_renumbered_away_is_reported_stale():
    """Labels are ordinal, so a push can leave a blocklist naming nothing."""
    stale = _stale_boundaries(LINEAR, [(ROBOT, "region_9")])

    assert stale == [boundary_key(ROBOT, "region_9")]


def test_staleness_reads_the_edge_in_either_direction():
    """Adjacency is stored per endpoint; a one-sided entry is still a live edge."""
    one_sided = {ROBOT: {MIDDLE}, MIDDLE: set()}

    assert _stale_boundaries(one_sided, [(MIDDLE, ROBOT)]) == []


def test_no_blocklist_reports_nothing_stale():
    assert _stale_boundaries(LINEAR, None) == []


# --- the shared path rule ----------------------------------------------------

def test_path_is_the_shortest_region_chain():
    assert find_region_path(LINEAR, ROBOT, GOAL) == [ROBOT, MIDDLE, GOAL]


def test_same_region_is_a_single_hop_path():
    assert find_region_path(LINEAR, ROBOT, ROBOT) == [ROBOT]


def test_unknown_start_has_no_path():
    assert find_region_path(LINEAR, "nowhere", GOAL) is None


def test_disconnected_goal_has_no_path():
    assert find_region_path({ROBOT: set(), GOAL: set()}, ROBOT, GOAL) is None


def test_blocked_boundary_is_avoided():
    blocked = {boundary_key(ROBOT, MIDDLE)}

    assert find_region_path(LINEAR, ROBOT, GOAL, blocked) is None


def test_blocked_boundary_reroutes_when_an_alternative_exists():
    diamond = {ROBOT: {"a", "b"}, "a": {ROBOT, GOAL}, "b": {ROBOT, GOAL}, GOAL: {"a", "b"}}

    path = find_region_path(diamond, ROBOT, GOAL, {boundary_key(ROBOT, "a")})

    assert path == [ROBOT, "b", GOAL]


def test_neighbour_order_is_deterministic():
    """Which boundary gets opened must not depend on set iteration order."""
    diamond = {ROBOT: {"b", "a"}, "a": {ROBOT, GOAL}, "b": {ROBOT, GOAL}, GOAL: {"a", "b"}}

    assert find_region_path(diamond, ROBOT, GOAL)[1] == "a"


def test_boundary_key_is_order_independent():
    assert boundary_key(ROBOT, GOAL) == boundary_key(GOAL, ROBOT)


@pytest.mark.parametrize("bad", [(ROBOT, ROBOT), (ROBOT, 4)])
def test_boundary_key_rejects_nonsense(bad):
    with pytest.raises((TypeError, ValueError)):
        boundary_key(*bad)


# --- the select/solve handshake, against the real binding --------------------

pytestmark_real = pytest.mark.skipif(
    not REAL_NAMO_RL or not SEPARATED_SCENE.exists(),
    reason="needs the compiled namo_rl binding and the boundary fixture",
)


def _scene_goal():
    match = re.search(
        r'<site name="goal".*?pos="([-\d.eE ]+)"', SEPARATED_SCENE.read_text(), re.S
    )
    x, y = (float(v) for v in match.group(1).split()[:2])
    return (x, y, 0.0)


def _scene_robot():
    root = ET.parse(SEPARATED_SCENE).getroot()
    robot_geom = root.find(".//body[@name='robot']/geom[@name='robot']")
    x, y = (float(v) for v in robot_geom.attrib["pos"].split()[:2])
    return (x, y, 0.0)


def _fully_overlapped_scene(tmp_path):
    tree = ET.parse(SEPARATED_SCENE)
    root = tree.getroot()
    divider_bottom = root.find(".//geom[@name='divider_bottom']")
    divider_top = root.find(".//geom[@name='divider_top']")
    divider_bottom.set("pos", "0 -0.46 0.3")
    divider_bottom.set("size", "0.05 0.54 0.3")
    divider_top.set("pos", "0 0.46 0.3")
    divider_top.set("size", "0.05 0.54 0.3")
    movable = root.find(".//geom[@name='obstacle_1_movable']")
    movable.set("contype", "0")
    movable.set("conaffinity", "0")
    path = tmp_path / "fully_overlapped_boundary.xml"
    tree.write(path)
    return path


@pytestmark_real
def test_select_then_solve_grades_against_the_selected_points():
    """The whole handshake: the bar is chosen once and does not move."""
    from namo.services import NAMOPlanningService

    service = NAMOPlanningService(
        config_path=str(SEPARATED_CONFIG), primitive_data_dir=str(REPO_ROOT / "data")
    )
    goal = _scene_goal()

    selection = service.select_boundary_from_xml(str(SEPARATED_SCENE), goal)
    assert selection.found
    assert selection.target_points
    assert selection.blocking_objects

    # Deliberately no target_neighbor: the boundary must be re-found from the
    # blocking objects alone, because labels renumber after a push.
    result = service.solve_boundary_from_xml(
        str(SEPARATED_SCENE),
        goal,
        target_points=selection.target_points,
        blocking_objects=selection.blocking_objects,
        max_chain_depth=1,
    )

    assert result.resolved_target == selection.target_label
    assert result.graded_points == selection.target_points


@pytestmark_real
def test_a_reachable_goal_needs_no_boundary():
    from namo.services import NAMOPlanningService

    service = NAMOPlanningService(
        config_path=str(SEPARATED_CONFIG), primitive_data_dir=str(REPO_ROOT / "data")
    )

    selection = service.select_boundary_from_xml(str(SEPARATED_SCENE), _scene_robot())

    assert selection.goal_already_reachable
    assert not selection.found


@pytestmark_real
def test_blocking_the_only_boundary_exhausts_the_path():
    from namo.services import NAMOPlanningService

    service = NAMOPlanningService(
        config_path=str(SEPARATED_CONFIG), primitive_data_dir=str(REPO_ROOT / "data")
    )
    goal = _scene_goal()
    selection = service.select_boundary_from_xml(str(SEPARATED_SCENE), goal)

    blocked = service.select_boundary_from_xml(
        str(SEPARATED_SCENE),
        goal,
        blocked_boundaries=[(selection.region_path[0], selection.target_label)],
    )

    assert not blocked.found
    assert blocked.failure_reason == "region_path_exhausted"


@pytestmark_real
def test_removing_a_movable_does_not_erase_an_overlapping_wall(tmp_path):
    from namo.services import NAMOPlanningService

    service = NAMOPlanningService(
        config_path=str(SEPARATED_CONFIG), primitive_data_dir=str(REPO_ROOT / "data")
    )

    selection = service.select_boundary_from_xml(
        str(_fully_overlapped_scene(tmp_path)), _scene_goal()
    )

    assert not selection.found
    assert selection.failure_reason == "region_path_exhausted"
