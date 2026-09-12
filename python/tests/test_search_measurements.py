"""The optional collectors must not change the mandatory execution identity."""

from itertools import product
from types import SimpleNamespace

import pytest


def test_measurement_clocks_are_strictly_optional(monkeypatch):
    from namo.planners import search_measurements as measurement

    def forbidden_clock():
        raise AssertionError("disabled timing read the measurement clock")

    monkeypatch.setattr(measurement, "perf_counter", forbidden_clock)
    started = measurement.clock_start(None)
    measurement.clock_finish(None, "t_sim", started)
    assert started is None
    values = iter([10.0, 10.25, 11.0, 11.5])
    monkeypatch.setattr(measurement, "perf_counter", lambda: next(values))
    timing = {}
    for _ in range(2):
        measurement.clock_finish(timing, "t_sim", measurement.clock_start(timing))
    assert timing == {"t_sim": 0.75}


@pytest.mark.parametrize("key,value", [
    ("record_statistics", "false"), ("record_timing", 1), ("schema_version", True),
    ("schema_version", 2), ("unknown_switch", True),
])
def test_measurement_configuration_rejects_ambiguous_options(key, value):
    from namo.planners.search_measurements import measurement_options

    with pytest.raises(ValueError, match=key):
        measurement_options({key: value})


def test_identity_and_digest_do_not_depend_on_collectors():
    from namo.planners.search_measurements import SearchMeasurements, run_identity, state_record

    problem = {"xml_sha256": "scene", "target": [0.1, 0.2], "boundary_objects": ["A", "B"]}
    protocol = {"hmax": 2, "budget": 3000, "object_scope": "boundary_pool"}
    identity = run_identity(problem, protocol, "uniform", None, {"sampler": 42, "shuffle": 7000})
    assert identity != run_identity(problem, protocol, "uniform", None, {"sampler": 42, "shuffle": 8000})
    state = state_record(SimpleNamespace(qpos=[0.1, -0.0], qvel=[0.0, 0.0]))
    digests = []
    for statistics, timing in product((False, True), repeat=2):
        measured = SearchMeasurements(record_statistics=statistics, record_timing=timing)
        assert measured.next_attempt() == 0
        measured.event("simulation", object_id="B", chain_depth=1, target=[0.1, 0.2, 0.3])
        measured.append("attempts", {"attempt_id": 0, "offered_objects": ["A", "B"]})
        assert measured.next_commit() == 0
        measured.event("commit", state=state)
        digests.append(measured.execution_digest)
        assert (measured.statistics is not None) is statistics
        assert (measured.timing is not None) is timing
        if statistics:
            assert measured.statistics["attempts"][0]["offered_objects"] == ["A", "B"]
    assert len(set(digests)) == 1
