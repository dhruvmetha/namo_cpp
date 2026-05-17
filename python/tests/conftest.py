import sys
import types


if "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    namo_rl_stub.StepResult = type("StepResult", (), {})
    sys.modules["namo_rl"] = namo_rl_stub
elif not hasattr(sys.modules["namo_rl"], "StepResult"):
    sys.modules["namo_rl"].StepResult = type("StepResult", (), {})
