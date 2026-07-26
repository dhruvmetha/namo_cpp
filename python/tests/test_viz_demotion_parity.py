"""Pins `_update_w_on_fail` (scripts/sandbox/eval_bestfirst.py) -- the per-board weight demotion rule that
viz/search/app.js's `demote()` reimplements in JavaScript to replay a recorded search's queue order.

This is the PYTHON reference only: these tests pin the Python implementation's behavior over time, they do
NOT cross-check it against app.js's `demote()`. The page's own runtime guard (verifyReconstruction() vs every
recorded pop) is the only thing that exercises the JS side, and only for whichever single discount mode a
given recorded trace actually used -- currently just one of the four ("off"/"gamma"/"conf"/"fitted") -- so a
JS-side drift in an untested discount mode on a trace that never uses it would NOT be caught by either of
these.

Direct import of eval_bestfirst.py (not a copy/reimplementation) -- it turned out importable without pulling
in the simulator: its own top-of-file sys.path setup is enough, no MJ_PATH/build_python bindings were touched
to reach `_update_w_on_fail`."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sandbox"))

from eval_bestfirst import _update_w_on_fail  # noqa: E402


def new_board(depth, w0=1.0, free_strikes=0):
    return {"depth": depth, "k_failed": 0, "w": w0, "w0": w0, "free_strikes": free_strikes}


def test_root_board_never_demotes():
    """depth 0 (root): w stays frozen at w0 regardless of discount mode or how many candidates fail."""
    board = new_board(depth=0)
    for q in (0.9, 0.5, 0.1):
        _update_w_on_fail(board, q, "conf", 0.65, 0.15, None, 0, 1e-3)
    assert board["w"] == 1.0
    assert board["k_failed"] == 3  # the lifetime log still counts failures on root boards


def test_off_discount_never_demotes():
    board = new_board(depth=1)
    _update_w_on_fail(board, 0.5, "off", 0.65, 0.15, None, 0, 1e-3)
    assert board["w"] == 1.0


def test_conf_multiplies_by_one_minus_q_to_the_tau_per_failure():
    board = new_board(depth=1)
    _update_w_on_fail(board, 0.4, "conf", 0.65, 0.15, None, 0, 1e-3)
    assert board["w"] == (1.0 - 0.4) ** 0.15
    w_after_1 = board["w"]
    _update_w_on_fail(board, 0.7, "conf", 0.65, 0.15, None, 0, 1e-3)
    assert board["w"] == w_after_1 * (1.0 - 0.7) ** 0.15


def test_gamma_multiplies_by_gamma_per_failure():
    board = new_board(depth=1)
    _update_w_on_fail(board, 0.3, "gamma", 0.65, 0.15, None, 0, 1e-3)
    assert board["w"] == 0.65
    _update_w_on_fail(board, 0.3, "gamma", 0.65, 0.15, None, 0, 1e-3)
    assert abs(board["w"] - 0.65 ** 2) < 1e-12


def test_fitted_reads_table_by_failure_count_and_clamps_at_kmax():
    g_table = {0: 1.0, 1: 0.8, 2: 0.5}
    gkmax = max(g_table)
    board = new_board(depth=1, w0=1.0)
    _update_w_on_fail(board, 0.5, "fitted", 0.65, 0.15, g_table, gkmax, 1e-3)  # k=1
    assert board["w"] == 1.0 * g_table[1]
    _update_w_on_fail(board, 0.5, "fitted", 0.65, 0.15, g_table, gkmax, 1e-3)  # k=2
    assert board["w"] == 1.0 * g_table[2]
    _update_w_on_fail(board, 0.5, "fitted", 0.65, 0.15, g_table, gkmax, 1e-3)  # k=3 -> clamps to kmax=2
    assert board["w"] == 1.0 * g_table[2]


def test_free_strikes_delays_the_first_demotion():
    """free_strikes=2: the first two failures are forgiven (patience), demotion starts on the third."""
    board = new_board(depth=1, free_strikes=2)
    _update_w_on_fail(board, 0.5, "gamma", 0.5, 0.15, None, 0, 1e-3)  # k_failed=1, k=1-2=-1 -> no demote
    assert board["w"] == 1.0
    _update_w_on_fail(board, 0.5, "gamma", 0.5, 0.15, None, 0, 1e-3)  # k_failed=2, k=0 -> no demote
    assert board["w"] == 1.0
    _update_w_on_fail(board, 0.5, "gamma", 0.5, 0.15, None, 0, 1e-3)  # k_failed=3, k=1 -> demotes
    assert board["w"] == 0.5


def test_eps_floor_holds():
    """w is floored at eps and never dips below it, however aggressive the discount."""
    board = new_board(depth=1)
    for _ in range(5):
        _update_w_on_fail(board, 0.9, "gamma", 0.01, 0.15, None, 0, 0.05)
        assert board["w"] >= 0.05
    assert board["w"] == 0.05
