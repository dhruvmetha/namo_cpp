from namo.environment_selection import (
    analyze_environment_path_length,
    find_shortest_path_length,
)


def test_find_shortest_path_length_counts_edges():
    adjacency = {
        "robot": {"a"},
        "a": {"robot", "b"},
        "b": {"a", "goal"},
        "goal": {"b"},
    }

    assert find_shortest_path_length(adjacency, "robot", "goal") == 3
    assert find_shortest_path_length(adjacency, "robot", "robot") == 0
    assert find_shortest_path_length(adjacency, "robot", "missing") == -1


def test_analyze_environment_path_length_uses_unified_snapshot(monkeypatch):
    created = {}

    class FakeEnv:
        def __init__(self, xml_path, config_path, visualize):
            created["xml_path"] = xml_path
            created["config_path"] = config_path
            created["visualize"] = visualize

    def fake_snapshot(env, **kwargs):
        created["snapshot_kwargs"] = kwargs
        return {
            "adjacency": {
                "robot": {"a"},
                "a": {"robot", "goal"},
                "goal": {"a"},
            },
            "robot_label": "robot",
            "goal_label": "goal",
        }

    monkeypatch.setattr("namo.environment_selection.namo_rl.RLEnvironment", FakeEnv)
    monkeypatch.setattr("namo.environment_selection.get_region_snapshot", fake_snapshot)

    analysis = analyze_environment_path_length("scene.xml", "config.yaml", use_cpp_snapshot=False)

    assert analysis.path_length_n == 2
    assert analysis.robot_label == "robot"
    assert analysis.goal_label == "goal"
    assert analysis.selection_error is None
    assert analysis.adjacency["a"] == {"robot", "goal"}
    assert created["snapshot_kwargs"]["use_cpp_unified"] is False
    assert created["snapshot_kwargs"]["use_xml_goal"] is True


def test_analyze_environment_path_length_reports_missing_goal(monkeypatch):
    monkeypatch.setattr(
        "namo.environment_selection.namo_rl.RLEnvironment",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "namo.environment_selection.get_region_snapshot",
        lambda *_args, **_kwargs: {
            "adjacency": {"robot": {"a"}, "a": {"robot"}},
            "robot_label": "robot",
            "goal_label": "",
        },
    )

    analysis = analyze_environment_path_length("scene.xml", "config.yaml")

    assert analysis.path_length_n == -1
    assert analysis.selection_error == "missing_goal_region"
