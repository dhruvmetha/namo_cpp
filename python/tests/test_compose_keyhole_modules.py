from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

import compose_keyhole_modules as composer  # noqa: E402
import summarize_keyhole_modules as summarizer  # noqa: E402


K1 = "obstacle_0_movable"
K2 = "obstacle_1_movable"
CLUTTER = "obstacle_2_movable"


def _donor(index: int, edge: int) -> composer.Donor:
    return composer.Donor(
        xml_path=f"/donor_{index}.xml",
        object_id=f"source_{index}",
        region=f"region_{index}",
        object_center=(float(index), 0.0),
        object_theta=0.0,
        tier="easy",
        horizon="1push",
        template="set2/benchmark_5",
        valid_root=((edge, 0),),
    )


def _write_module(path: Path, object_id: str, *, goal_x: float) -> None:
    root = ET.Element("mujoco", {"model": object_id})
    worldbody = ET.SubElement(root, "worldbody")
    walls = ET.SubElement(worldbody, "body", {"name": "walls"})
    for name, pos, size in (
        ("west", (-1.0, 0.0, 0.1), (0.01, 1.0, 0.1)),
        ("east", (1.0, 0.0, 0.1), (0.01, 1.0, 0.1)),
        ("south", (0.0, -1.0, 0.1), (1.0, 0.01, 0.1)),
        ("north", (0.0, 1.0, 0.1), (1.0, 0.01, 0.1)),
    ):
        ET.SubElement(
            walls,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": " ".join(map(str, pos)),
                "size": " ".join(map(str, size)),
            },
        )
    ET.SubElement(worldbody, "body", {"name": "car", "pos": "-0.6 0 0", "euler": "0 0 0"})
    blocker = ET.SubElement(worldbody, "body", {"name": object_id})
    ET.SubElement(
        blocker,
        "geom",
        {
            "name": object_id,
            "type": "box",
            "pos": "0 0 0.05",
            "size": "0.1 0.1 0.05",
            "euler": "0 0 0",
        },
    )
    ET.SubElement(worldbody, "site", {"name": "goal", "pos": f"{goal_x} 0 0"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _append_movable(path: Path, object_id: str, *, x: float, y: float) -> None:
    tree = ET.parse(path)
    body = ET.SubElement(
        composer._worldbody(tree.getroot()),
        "body",
        {"name": object_id},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": object_id,
            "type": "box",
            "pos": f"{x} {y} 0.05",
            "size": "0.1 0.1 0.05",
        },
    )
    ET.SubElement(body, "joint", {"type": "free"})
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _xml_donor(path: Path, object_id: str, *, template: str) -> composer.Donor:
    return composer.Donor(
        xml_path=str(path),
        object_id=object_id,
        region="goal",
        object_center=(0.0, 0.0),
        object_theta=0.0,
        tier="medium",
        horizon="1push",
        template=template,
        valid_root=((0, 0),),
    )


def _state(*, goal: bool, k1: bool, k2: bool, k1_edges=(10,), k2_edges=(20,), counts=(0, 0)):
    objects = [object_id for object_id, reachable in ((K1, k1), (K2, k2)) if reachable]
    return {
        "goal": goal,
        "objects": objects,
        "edges": {K1: list(k1_edges), K2: list(k2_edges)},
        "counts": list(counts),
    }


class _FakeEnv:
    def __init__(self, states, *, k1_done=True, k2_done=True):
        self.states = states
        self.current = "initial"
        self.k1_done = k1_done
        self.k2_done = k2_done

    def set_robot_goal(self, *_goal):
        return None

    def get_full_state(self):
        return self.current

    def set_full_state(self, state):
        self.current = state

    def get_reachable_objects(self):
        return self.states[self.current]["objects"]

    def get_reachable_edges(self, object_id):
        return self.states[self.current]["edges"][object_id]

    def is_robot_goal_reachable(self):
        return self.states[self.current]["goal"]

    def get_observation(self):
        offset = {"initial": 0.0, "post_k1": 1.0, "post_k2": 2.0}[self.current]
        return {
            f"{K1}_pose": [offset, 0.0, 0.0],
            f"{K2}_pose": [offset, 1.0, 0.0],
        }

    def count_reachable_points(self, points):
        index = 0 if points[0][0] < 15.0 else 1
        return self.states[self.current]["counts"][index], -1

    def step(self, action):
        if action.object_id == K1:
            self.current = "post_k1"
            return SimpleNamespace(done=self.k1_done)
        if action.object_id == K2:
            self.current = "post_k2"
            return SimpleNamespace(done=self.k2_done)
        raise AssertionError(f"unexpected object {action.object_id}")


def _snapshot():
    def goals(x):
        return SimpleNamespace(goals=[SimpleNamespace(x=x, y=float(index)) for index in range(100)])

    return {
        "adjacency": {"robot": {"middle"}, "middle": {"robot", "goal"}, "goal": {"middle"}},
        "edge_objects": {
            "robot": {"middle": {K1}},
            "middle": {"robot": {K1}, "goal": {K2}},
            "goal": {"middle": {K2}},
        },
        "robot_label": "robot",
        "goal_label": "goal",
        "goal_in_free_space": True,
        "region_goals": {"middle": goals(10.0), "goal": goals(20.0)},
    }


def _run(monkeypatch, states, *, k1_done=True, k2_done=True, max_attempts=None):
    env = _FakeEnv(states, k1_done=k1_done, k2_done=k2_done)
    monkeypatch.setattr(composer.namo_rl, "RLEnvironment", lambda *_args: env)
    monkeypatch.setattr(composer, "extract_goal_from_xml", lambda _xml: (1.0, 2.0, 0.0))
    monkeypatch.setattr(composer, "get_region_snapshot", lambda *_args, **_kwargs: _snapshot())
    return composer.replay_two_keyhole_goal_chain(
        "/composed.xml",
        "/config.yaml",
        [_donor(0, 10), _donor(1, 20)],
        max_attempts=max_attempts,
    )


def _valid_states(counts=(0, 0)):
    return {
        "initial": _state(goal=False, k1=True, k2=False, counts=counts),
        "post_k1": _state(goal=False, k1=True, k2=True, counts=counts),
        "post_k2": _state(goal=True, k1=True, k2=True, counts=counts),
    }


def test_goal_chain_accepts_valid_progression_and_records_contract(monkeypatch):
    result = _run(monkeypatch, _valid_states())

    assert result["status"] == "solved"
    assert result["failure_reason"] is None
    assert result["component_path"] == ["robot", "middle", "goal"]
    assert result["boundary_objects"] == [[K1], [K2]]
    assert [state["goal_reachable"] for state in result["reachability_trace"]] == [False, False, True]
    assert K2 not in result["reachability_trace"][0]["reachable_objects"]
    assert result["reachability_trace"][1]["reachable_edges"][K2] == [20]
    assert result["actions"] == [[[10, 0]], [[20, 0]]]
    assert result["final_object_poses"] == result["object_pose_trace"][-1]


@pytest.mark.parametrize(
    ("states", "k1_done", "k2_done", "reason"),
    [
        (
            {
                **_valid_states(),
                "initial": _state(goal=False, k1=True, k2=True),
            },
            True,
            True,
            "k2_reachable_at_t0",
        ),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=True, k1=True, k2=True),
            },
            True,
            True,
            "k1_reached_goal",
        ),
        (_valid_states(), False, True, "k1_push_failed"),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=False, k1=True, k2=False),
            },
            True,
            True,
            "k1_did_not_expose_k2",
        ),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=False, k1=True, k2=True, k2_edges=()),
            },
            True,
            True,
            "k2_no_push_edges_after_k1",
        ),
        (_valid_states(), True, False, "k2_push_failed"),
        (
            {
                **_valid_states(),
                "post_k2": _state(goal=False, k1=True, k2=True),
            },
            True,
            True,
            "final_goal_unreachable",
        ),
    ],
)
def test_goal_chain_reports_specific_progression_failure(
    monkeypatch, states, k1_done, k2_done, reason
):
    result = _run(monkeypatch, states, k1_done=k1_done, k2_done=k2_done)

    assert result["status"] != "solved"
    assert result["failure_reason"] == reason


def test_diagnostic_point_counts_do_not_gate_valid_goal_progression(monkeypatch):
    result = _run(monkeypatch, _valid_states(counts=(0, 0)))

    assert result["status"] == "solved"
    assert result["target_point_thresholds"] == [20, 20]
    assert result["target_point_trace"] == [[0, 0], [0, 0], [0, 0]]


def test_goal_chain_can_cap_rejection_work_without_weakening_acceptance(monkeypatch):
    result = _run(monkeypatch, _valid_states(), max_attempts=1)

    assert result["status"] == "no_goal_chain"
    assert result["failure_reason"] == "replay_attempt_cap"
    assert result["attempts"] == 1


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ({"goal_in_free_space": False}, "goal_not_in_free_space"),
        ({"no_path": True}, "no_component_path"),
        ({"hop_mismatch": True}, "wrong_hop_count"),
        ({"no_reachable_blocker": True}, "k1_not_reachable"),
        ({"no_pushable_blocker": True}, "k1_no_push_edges"),
    ],
)
def test_static_acceptance_reports_specific_defect(defect, reason):
    row = {
        "goal_in_free_space": True,
        "boundaries": [{"objects": [K1]}, {"objects": [K2]}],
        **defect,
    }

    assert composer.static_acceptance(row, 2) == (False, reason)


def test_split_outer_wall_cuts_requested_portal(tmp_path):
    source = tmp_path / "module.xml"
    _write_module(source, "source", goal_x=0.6)
    root = ET.parse(source).getroot()

    composer._split_outer_wall(
        root,
        composer.PortalInterface("east", 0.2, "goal", 0.5),
        width=0.1,
    )

    segments = [
        geom
        for geom in composer._wall_body(root).findall("geom")
        if (geom.get("name") or "").startswith("east_portal_")
    ]
    assert len(segments) == 2
    bounds = sorted(
        (
            composer._numbers(segment.get("pos"))[1]
            - composer._numbers(segment.get("size"))[1],
            composer._numbers(segment.get("pos"))[1]
            + composer._numbers(segment.get("size"))[1],
        )
        for segment in segments
    )
    assert bounds[0] == pytest.approx((-1.0, 0.15))
    assert bounds[1] == pytest.approx((0.25, 1.0))


def test_room_stitch_preserves_two_modules_and_uses_second_goal(tmp_path, monkeypatch):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    output = tmp_path / "stitched.xml"
    _write_module(first, "source_0", goal_x=0.6)
    _write_module(second, "source_1", goal_x=0.7)
    donors = (
        _xml_donor(first, "source_0", template="set2/benchmark_3"),
        _xml_donor(second, "source_1", template="set2/benchmark_5"),
    )

    monkeypatch.setattr(
        composer,
        "_module_interface",
        lambda _donor, role, _config, _width: composer.PortalInterface(
            "east" if role == "exit" else "west",
            0.0,
            role,
            1.0,
        ),
    )
    metadata = composer.compose_room_stitch_xml(
        donors,
        output,
        "/config.yaml",
        portal_width=0.1,
        connector_length=0.2,
    )

    root = ET.parse(output).getroot()
    worldbody = composer._worldbody(root)
    assert root.get("model") == "stitched_two_keyhole_environment"
    assert [body.get("name") for body in worldbody.findall("body")].count("car") == 1
    assert composer._movable_body(root, K1) is not None
    assert composer._movable_body(root, K2) is not None
    assert worldbody.find("body[@name='module_1_walls']") is not None
    assert worldbody.find("body[@name='module_2_walls']") is not None
    assert worldbody.find("body[@name='connector_walls']") is not None
    global_walls = worldbody.find("body[@name='global_boundary_walls']")
    assert global_walls is not None
    assert [geom.get("name") for geom in global_walls.findall("geom")] == [
        "wall_1",
        "wall_2",
        "wall_3",
        "wall_4",
    ]
    assert composer._numbers(composer._goal_site(root).get("pos"))[:2] == pytest.approx([2.9, 0.0])
    assert metadata["mode"] == "room_stitch"
    assert [module["source_template"] for module in metadata["modules"]] == [
        "set2/benchmark_3",
        "set2/benchmark_5",
    ]


def test_same_template_keeps_one_host_room_and_transplants_only_blockers(tmp_path, monkeypatch):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    output = tmp_path / "composed.xml"
    _write_module(first, "source_0", goal_x=0.6)
    _write_module(second, "source_1", goal_x=0.7)
    donors = (
        _xml_donor(first, "source_0", template="set2/benchmark_5"),
        _xml_donor(second, "source_1", template="set2/benchmark_5"),
    )
    monkeypatch.setattr(composer, "geom_sig", lambda _path: ("full", "shared_walls"))

    metadata = composer.compose_same_template_xml(donors, output)

    root = ET.parse(output).getroot()
    worldbody = composer._worldbody(root)
    assert len([body for body in worldbody.findall("body") if body.get("name") == "walls"]) == 1
    assert worldbody.find("body[@name='connector_walls']") is None
    assert worldbody.find("body[@name='global_boundary_walls']") is None
    assert composer._movable_body(root, K1) is not None
    assert composer._movable_body(root, K2) is not None
    assert composer._numbers(composer._goal_site(root).get("pos"))[:2] == pytest.approx([0.7, 0.0])
    assert metadata == {
        "mode": "same_template",
        "template": "set2/benchmark_5",
        "wall_signature": "shared_walls",
        "host_xml": str(first.resolve()),
        "transplanted": "blockers_only",
    }


def test_same_template_rejects_different_named_templates(tmp_path):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    _write_module(first, "source_0", goal_x=0.6)
    _write_module(second, "source_1", goal_x=0.7)
    donors = (
        _xml_donor(first, "source_0", template="set2/benchmark_3"),
        _xml_donor(second, "source_1", template="set2/benchmark_5"),
    )

    with pytest.raises(composer.CompositionRejected, match="donor_template_mismatch"):
        composer.compose_same_template_xml(donors, tmp_path / "composed.xml")


def test_same_template_rejects_wall_signature_mismatch(tmp_path, monkeypatch):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    _write_module(first, "source_0", goal_x=0.6)
    _write_module(second, "source_1", goal_x=0.7)
    donors = (
        _xml_donor(first, "source_0", template="set2/benchmark_5"),
        _xml_donor(second, "source_1", template="set2/benchmark_5"),
    )
    monkeypatch.setattr(
        composer,
        "geom_sig",
        lambda path: ("full", "walls_a" if path == str(first) else "walls_b"),
    )

    with pytest.raises(composer.CompositionRejected, match="donor_wall_signature_mismatch"):
        composer.compose_same_template_xml(donors, tmp_path / "composed.xml")


def test_same_template_clutter_retains_one_renamed_host_object(tmp_path, monkeypatch):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    output = tmp_path / "composed.xml"
    _write_module(first, "obstacle_5_movable", goal_x=0.6)
    _append_movable(first, "obstacle_6_movable", x=-0.2, y=0.4)
    _append_movable(first, "obstacle_7_movable", x=0.2, y=0.4)
    _write_module(second, "obstacle_8_movable", goal_x=0.7)
    donors = (
        _xml_donor(first, "obstacle_5_movable", template="set2/benchmark_5"),
        _xml_donor(second, "obstacle_8_movable", template="set2/benchmark_5"),
    )
    monkeypatch.setattr(composer, "geom_sig", lambda _path: ("full", "shared_walls"))

    metadata = composer.compose_same_template_clutter_xml(
        donors, "obstacle_6_movable", output
    )

    root = ET.parse(output).getroot()
    movable_ids = [
        body.get("name")
        for body in composer._worldbody(root).findall("body")
        if composer.MOVABLE_RE.match(body.get("name") or "")
    ]
    assert movable_ids == [K1, K2, CLUTTER]
    assert composer._numbers(composer._movable_body(root, CLUTTER).find("geom").get("pos"))[
        :2
    ] == pytest.approx([-0.2, 0.4])
    assert metadata["mode"] == "same_template_clutter"
    assert metadata["clutter_object_ids"] == [CLUTTER]
    assert metadata["host_clutter_source_id"] == "obstacle_6_movable"


def test_passive_clutter_requires_stable_pose_and_decision_edge_effect():
    replay = {
        "object_pose_trace": [
            {CLUTTER: [0.0, 0.0, 0.0]},
            {CLUTTER: [0.001, 0.0, 0.0]},
            {CLUTTER: [0.0015, 0.0, 0.0]},
        ],
        "reachability_trace": [
            {"reachable_edges": {K1: [0, 1], K2: []}},
            {"reachable_edges": {K1: [0], K2: [2]}},
            {"reachable_edges": {K1: [0], K2: [2]}},
        ],
    }
    clean = {"status": "ok", "decision_edges": {"k1_t0": [0, 1, 2], "k2_t1": [2]}}

    assert composer.passive_clutter_motion_failure(replay, [CLUTTER]) is None
    failure, effect = composer.passive_clutter_edge_effect(replay, clean)
    assert failure is None
    assert effect["changed_decisions"] == ["k1_t0"]

    replay["object_pose_trace"][2][CLUTTER] = [0.003, 0.0, 0.0]
    assert composer.passive_clutter_motion_failure(replay, [CLUTTER]) == "passive_clutter_moved"

    no_effect = {
        "status": "ok",
        "decision_edges": {"k1_t0": [0, 1], "k2_t1": [2]},
    }
    assert composer.passive_clutter_edge_effect(replay, no_effect)[0] == (
        "passive_clutter_no_edge_effect"
    )


def test_clean_counterfactual_records_the_two_ranker_decision_points(monkeypatch):
    env = _FakeEnv(_valid_states())
    monkeypatch.setattr(composer.namo_rl, "RLEnvironment", lambda *_args: env)
    monkeypatch.setattr(composer, "extract_goal_from_xml", lambda _xml: (1.0, 2.0, 0.0))

    result = composer.clean_counterfactual_decision_edges(
        "/clean.xml",
        "/config.yaml",
        [[[10, 0]], [[20, 0]]],
    )

    assert result == {
        "status": "ok",
        "decision_edges": {"k1_t0": [10], "k2_t1": [20]},
    }


def test_mechanical_independence_rejects_cross_blocker_motion():
    stable = {
        "object_pose_trace": [
            {K1: [0.0, 0.0, 0.0], K2: [1.0, 0.0, 0.0]},
            {K1: [0.1, 0.0, 0.0], K2: [1.0, 0.0, 0.0]},
            {K1: [0.1, 0.0, 0.0], K2: [1.1, 0.0, 0.0]},
        ]
    }
    assert composer.mechanical_independence_failure(stable) is None

    k1_moves_k2 = {"object_pose_trace": [dict(row) for row in stable["object_pose_trace"]]}
    k1_moves_k2["object_pose_trace"][1] = {
        **k1_moves_k2["object_pose_trace"][1],
        K2: [1.01, 0.0, 0.0],
    }
    assert composer.mechanical_independence_failure(k1_moves_k2) == "k1_moved_k2"

    k2_moves_k1 = {"object_pose_trace": [dict(row) for row in stable["object_pose_trace"]]}
    k2_moves_k1["object_pose_trace"][2] = {
        **k2_moves_k1["object_pose_trace"][2],
        K1: [0.11, 0.0, 0.0],
    }
    assert composer.mechanical_independence_failure(k2_moves_k1) == "k2_moved_k1"


def test_scale_summary_separates_static_and_post_static_rejections():
    result = summarizer._aggregate(
        [
            {
                "attempted": 10,
                "accepted": 2,
                "rejections": {"wrong_hop_count": 7, "final_goal_unreachable": 1},
            }
        ]
    )

    assert result["static_passed"] == 3
    assert result["post_static_rejected"] == 1
    assert result["accepted"] == 2
