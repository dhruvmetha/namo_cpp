"""Small, simulator-free checks for the fixed benchmark selection."""
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "freeze_keyhole_random5.py"

class FreezeSelectionTest(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.exists(), "freeze packager is not implemented")
        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("freeze_keyhole_random5", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_difficulty_uses_all_five_runs_and_preserves_unresolved(self):
        module = self.module()
        cases = [
            ([1, 2, 3, 4, 5], ("easy", 3, 5)),
            ([10]*5, ("easy", 10, 5)),
            ([11]*5, ("medium", 11, 5)),
            ([50]*5, ("medium", 50, 5)),
            ([51]*5, ("hard", 51, 5)),
            ([12, 18, 30, None, None], ("medium", 30, 3)),
            ([None]*5, ("unresolved", None, 0)),
            ([9, None, None, None, None], ("unresolved", None, 1)),
            ([9, 12, None, None, None], ("unresolved", None, 2)),
        ]
        for values, expected in cases:
            runs = {str(seed): {"solved": value is not None,
                    "total_calls": value if value is not None else 7,
                    "failure_kind": None if value is not None else "region_path_exhausted"}
                    for seed, value in zip(module.SEEDS, values)}
            self.assertEqual(module.difficulty_for_runs(runs), expected)
        runs["7000"]["failure_kind"] = "planner_invariant_violation"
        self.assertEqual(module.difficulty_for_runs(runs)[0], "technical_error")

    def test_seeded_selection_is_order_independent_and_keeps_all_unresolved(self):
        module = self.module()
        pool = [{"geometry_id": f"{category}_{index}", "difficulty": category}
                for category in ("easy", "medium", "hard", "unresolved", "technical_error")
                for index in range(3)]
        selected = module.choose(pool, seed=20260908, per_tier=2)
        repeated = module.choose(list(reversed(pool)), seed=20260908, per_tier=2)
        self.assertEqual(selected, repeated)
        self.assertEqual(Counter(row["difficulty"] for row in selected),
                         {"easy": 2, "medium": 2, "hard": 2, "unresolved": 3})
        self.assertEqual(len({r["geometry_id"] for r in selected}), len(selected))

if __name__ == "__main__":
    unittest.main()
