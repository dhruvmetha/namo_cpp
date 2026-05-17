import sys
import types


if "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    sys.modules["namo_rl"] = namo_rl_stub

from namo.planners.idfs.failure_codes import FailureCode, create_failure_info


def test_planner_invariant_violation_maps_to_dedicated_failure_code():
    info = create_failure_info("Planner invariant violation: target_not_immediate_neighbor")

    assert info["failure_code"] == int(FailureCode.PLANNER_INVARIANT_VIOLATION)
    assert info["failure_description"] == "Planner invariant violation (graph/reachability inconsistency)"
