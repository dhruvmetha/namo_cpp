from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from car_model.square_eval import evaluate_square_path


PYTHON = "/home/shanoriel/miniforge3/envs/leworldmodel/bin/python"


if __name__ == "__main__":
    result = evaluate_square_path(PROJECT_ROOT / "assets")
    print(f"Using Python: {PYTHON}")
    print("Square path evaluation: 4x [forward 0.10 m, left turn 90 deg]")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
