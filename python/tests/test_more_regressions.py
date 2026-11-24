"""The 16 bug reproductions from the 2026-11-24 MORE audit, rewritten against the new code.

Each test states the behaviour MORE's paper and code define. On the 13763e54 port (more_mcts.py
and its training scripts) every one fails; on the vendored search plus the fixed training code
every one passes. `more_toy.collect` and `more_toy.backups` run whichever implementation is
importable, so the same file proves both:

  new:  PYTHONPATH=<this repo>/python:...           -> all pass
  old:  PYTHONPATH=<13763e54 tree>/python:...       -> all 16 fail (audit folder `original/`)

Audit: /common/users/dm1487/scratch_namo/more_audit_20261124/SPEC.md. Collection rollouts take
index 0 on both sides so the scenes are deterministic.
"""
import math
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

import more_toy as T
from more_toy import FakeEnv, PTS, goal_open_pts

SETUP, FINISH = ("obj", 1, 0), ("obj", 0, 0)


# -- BUG-1: an opening reached in a rollout keeps MORE's +1 ---------------------------------------
def test_bug1_rollout_found_finish_backs_055_into_the_setup(monkeypatch):
    """mcts/nodes.py:65-106 + mcts/push.py:60-69: max(0.2*0.02, 0.5*(1 + 0.2*0.5)) = 0.55."""
    env = FakeEnv(opens=[(SETUP, FINISH)])
    out = T.collect(monkeypatch, env, edges=2, iterations=1, calls=2)
    assert T.row(out["rows"], (), SETUP)["label"] == pytest.approx(0.55)


def test_bug1_finish_expanded_from_the_cache_keeps_its_plus_one(monkeypatch):
    """A finish first simulated by a rollout is labelled 1.1 once the tree expands it."""
    env = FakeEnv(opens=[(SETUP, FINISH)])
    out = T.collect(monkeypatch, env, edges=2, iterations=20, calls=10)
    assert T.row(out["rows"], (SETUP,), FINISH)["label"] == pytest.approx(1.1)


def test_bug1_a_reselected_terminal_opener_backs_up_its_full_reward(monkeypatch):
    """mcts/search.py:11-14: every selection of a terminal opener backs up push_result = 1.1."""
    env = FakeEnv(opens=[(("obj", 5, 0),)])
    with T.backups(monkeypatch) as seen:
        T.collect(monkeypatch, env, edges=6, iterations=30, calls=30)
    vals = [v for path, v in seen if path == (("obj", 5, 0),)]
    assert len(vals) >= 2, "the opener must be selected more than once"
    assert all(v == pytest.approx(1.1) for v in vals), vals


# -- BUG-2: one backup per iteration --------------------------------------------------------------
def test_bug2_one_iteration_backs_an_opener_up_once(monkeypatch):
    env = FakeEnv(opens=[(("obj", 1, 0),)])
    out = T.collect(monkeypatch, env, edges=2, iterations=1, calls=1)
    r = T.row(out["rows"], (), ("obj", 1, 0))
    assert r["label"] == pytest.approx(1.1) and r["n"] == 1


# -- BUG-3: collection always ends ----------------------------------------------------------------
class _Hang(Exception):
    pass


def test_bug3_collection_terminates_when_only_an_opener_is_left(monkeypatch):
    env = FakeEnv(opens=[(("obj", 1, 0),)])

    def alarm(_s, _f):
        raise _Hang()
    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(20)
    try:
        out = T.collect(monkeypatch, env, edges=2, iterations=300, calls=50)
    except _Hang:
        pytest.fail(f"still running after 20 s with {env.steps} calls spent")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert out["solved"]


# -- BUG-4: one threshold for the +1 and for the stop ---------------------------------------------
def test_bug4_a_state_credited_as_open_is_terminal(monkeypatch):
    """mcts/push.py:67-68 and :74 use the same test; at exactly 20/100 the state is open AND terminal."""
    at_bar = (("obj", 1, 0),)
    env = FakeEnv(reach={at_bar: 0.20})
    env.set_full_state(at_bar)
    assert goal_open_pts(env, PTS)
    env.set_full_state(())
    T.collect(monkeypatch, env, edges=2, iterations=50, calls=10,
              is_open=lambda e: goal_open_pts(e, PTS))
    assert at_bar not in [s for s, _k in env.log], "pushed again from a state credited as open"


# -- BUG-5: the true collection cost is reported --------------------------------------------------
def test_bug5_collection_reports_every_simulator_call(monkeypatch):
    env = FakeEnv(opens=[(SETUP, FINISH)])
    out = T.collect(monkeypatch, env, edges=2, iterations=20, calls=10)
    assert env.steps > 2, "the search kept going after the first opening"
    assert out["total_calls"] == env.steps


# -- training H5 helpers --------------------------------------------------------------------------
S, CROP_M = 224, 0.5
HW, HD, THETA = 0.10, 0.05, 0.4


def contact_px(edge, hw, hd, theta, crop_m, size):
    """Verbatim scripts/eval_scorer.py:45-56 (what live_scorer.contact_px_live calls)."""
    n = 15
    sl = lambda a, b, i: a + (b - a) * (i / (n - 1))  # noqa: E731
    if edge < 30:
        j = edge // 2; lx = sl(-hw, hw, j); ly = hd if edge % 2 == 0 else -hd  # noqa: E702
    else:
        k = (edge - 30) // 2; lx = hw if edge % 2 == 0 else -hw; ly = sl(-hd, hd, k)  # noqa: E702
    c, s = math.cos(theta), math.sin(theta)
    wx = c * lx - s * ly; wy = s * lx + c * ly  # noqa: E702
    res = crop_m / size; cx = size / 2.0  # noqa: E702
    return cx + wx / res, cx + wy / res


def board(size=S):
    return np.array([contact_px(e, HW, HD, THETA, CROP_M, size) for e in range(60)], np.float32)


def rect_mask(size=S):
    res, cx = CROP_M / size, size / 2.0
    rr, cc = np.mgrid[0:size, 0:size].astype(np.float64)
    x, y = (cc - cx) * res, (rr - cx) * res
    c, s = math.cos(THETA), math.sin(THETA)
    lx, ly = c * x + s * y, -s * x + c * y
    return ((np.abs(lx) <= HW) & (np.abs(ly) <= HD)).astype(np.float32)


def write_h5(path, xml, evidence, reach, weight, ctx=None, cpx=None):
    import h5py
    n = len(xml)
    with h5py.File(path, "w") as f:
        f.create_dataset("ctx", data=ctx if ctx is not None else np.zeros((n, 5, S, S), np.float16))
        f.create_dataset("contact_px", data=np.stack([cpx if cpx is not None else board()] * n))
        f.create_dataset("label", data=np.zeros((n, 60, 5), np.float32))
        f.create_dataset("weight", data=weight.astype(np.float32))
        f.create_dataset("reach_mask", data=reach.astype(np.float32))
        f.create_dataset("evidence", data=evidence.astype(np.int8))
        vlen = h5py.string_dtype("utf-8")
        f.create_dataset("xml", data=np.array(xml, dtype=object), dtype=vlen)
        f.create_dataset("object_id", data=np.array(["obj"] * n, dtype=object), dtype=vlen)
    return path


def dataset_of_main(monkeypatch, h5, out):
    """train_more.main() with its DEFAULT flags and zero epochs; the dataset it would train on."""
    import train_more
    seen = {}
    real = train_more.DataLoader

    def spy(ds, *a, **k):
        seen["ds"] = ds
        return real(ds, *a, **{**k, "num_workers": 0, "pin_memory": False})
    monkeypatch.setattr(train_more, "DataLoader", spy)
    monkeypatch.setattr("sys.argv", ["train_more.py", "--h5", str(h5), "--out", str(out), "--epochs", "0"])
    train_more.main()
    return seen["ds"]


def item_angle(ds, i):
    return float(ds[i][2])


# -- BUG-6: untried actions carry no loss ---------------------------------------------------------
def test_bug6_default_training_gives_untried_actions_no_loss(tmp_path, monkeypatch):
    n = 20
    ev = np.zeros((n, 60, 5), np.int8)
    reach = np.zeros((n, 60, 5), np.float32)
    w = np.zeros((n, 60, 5), np.float32)
    ev[:, 0, 0], reach[:, 0, 0], w[:, 0, 0] = 1, 1, 7
    reach[:, 1, 0] = reach[:, 2, 0] = 1                     # legal, never simulated
    w[:, 1, 0] = w[:, 2, 0] = 1.0                           # the old builder stamped weight 1 here
    h5 = write_h5(tmp_path / "t.h5", [f"room_{i}" for i in range(n)], ev, reach, w)
    ds = dataset_of_main(monkeypatch, h5, tmp_path / "run")
    cells = [tuple(map(int, c[1:])) for c in ds.index]
    assert (0, 0) in cells
    assert not any(c in {(1, 0), (2, 0)} for c in cells)


# -- BUG-7: contact_px is (column, row) ------------------------------------------------------------
PILOT_H5 = Path("/common/users/dm1487/scratch_namo/more_pilot_20260921/more_pilot_chunked.h5")


@pytest.mark.skipif(not PILOT_H5.exists(), reason="pilot corpus not on this filesystem")
def test_bug7_read_contact_values_lands_on_the_pushed_object():
    import h5py
    from scipy import ndimage
    from namo.rl_loop.more_net import read_contact_values
    on = tot = 0
    with h5py.File(PILOT_H5, "r") as f:
        for i in sorted(np.random.default_rng(0).choice(f["ctx"].shape[0], 40, replace=False)):
            m = f["ctx"][i][2].astype(np.float32) > 0.5                 # target_object channel
            d = np.where(m, ndimage.distance_transform_edt(m), ndimage.distance_transform_edt(~m))
            near = torch.tensor((d <= 3).astype(np.float32))[None, None]
            vals = read_contact_values(near, torch.tensor(f["contact_px"][i])[None], patch=0)
            on, tot = on + int(vals.sum()), tot + vals.numel()
    assert on / tot > 0.95, f"only {on/tot:.1%} of contacts read on the object boundary"


def _rotate_pairs():
    import train_more
    import verify_more_net
    from namo.rl_loop import more_scorer
    out = []
    for m in (train_more, more_scorer, verify_more_net):
        rot = getattr(m, "rotate_batch", None) or getattr(m, "_rotate")
        pix = getattr(m, "rotate_pixel", None) or getattr(m, "_rotate_pixel")
        out.append(pytest.param(rot, pix, id=m.__name__))
    return out


def _blob(row, col, size=S):
    img = torch.zeros(1, 1, size, size)
    img[0, 0, row - 1:row + 2, col - 1:col + 2] = 1.0
    return img


@pytest.mark.parametrize("rotate, rotate_pixel", _rotate_pairs())
def test_bug7_rotated_contact_follows_the_rotated_image(rotate, rotate_pixel):
    row, col, ang = 60, 150, torch.tensor([0.7])
    out = rotate(_blob(row, col), ang)[0, 0]
    r_true, c_true = np.unravel_index(int(out.argmax()), out.shape)
    c_pred, r_pred = rotate_pixel(torch.tensor([[float(col), float(row)]]), ang, S)[0].tolist()
    assert abs(c_pred - c_true) <= 1.5 and abs(r_pred - r_true) <= 1.5


def test_bug7_paint_puts_the_label_on_the_rotated_contact():
    import train_more
    rot = getattr(train_more, "rotate_batch")
    pix = getattr(train_more, "rotate_pixel")
    row, col, ang = 60, 150, torch.tensor([0.7])
    out = rot(_blob(row, col), ang)[0, 0]
    r_true, c_true = np.unravel_index(int(out.argmax()), out.shape)
    rpx = pix(torch.tensor([[float(col), float(row)]]), ang, S)
    target, wmap = torch.zeros(1, 5, S, S), torch.zeros(1, 5, S, S)
    train_more.paint(target, wmap, rpx, torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1.0]))
    r_p, c_p = np.argwhere(wmap[0, 0].numpy() > 0).mean(axis=0)
    assert abs(r_p - r_true) <= 1.5 and abs(c_p - c_true) <= 1.5


# -- BUG-8: rotate by each push's own direction; all 60 edges point right -------------------------
def _push_dir(e):
    """Face-normal push of edge e toward its mate (namo_push_controller.cpp:178-210, :258-260)."""
    if e < 30:
        lx, ly = (0.0, -1.0) if e % 2 == 0 else (0.0, 1.0)
    else:
        lx, ly = (-1.0, 0.0) if e % 2 == 0 else (1.0, 0.0)
    c, s = math.cos(THETA), math.sin(THETA)
    return c * lx - s * ly, s * lx + c * ly


def _dataset_all_edges(tmp_path, ctx=None):
    import train_more
    ev = np.zeros((1, 60, 5), np.int8)
    ev[0, :, 0] = 1
    reach = np.ones((1, 60, 5), np.float32)
    w = np.where(ev == 1, 1.0, 0.0).astype(np.float32)
    h5 = write_h5(tmp_path / "t.h5", ["room"], ev, reach, w, ctx=ctx)
    try:
        ds = train_more.MoreActionDataset(str(h5), invalid="more")       # new: rows only
    except TypeError:
        ds = train_more.MoreActionDataset(str(h5), include_unreachable=False)
    return [(int(ds.index[i][1]), i) for i in range(len(ds)) if int(ds.index[i][2]) == 0], ds


def test_bug8_training_rotation_is_each_push_direction(tmp_path):
    edges, ds = _dataset_all_edges(tmp_path)
    assert len(edges) == 60
    wrong = [e for e, i in edges
             if abs((item_angle(ds, i) - math.atan2(*_push_dir(e)[::-1]) + math.pi) % (2 * math.pi) - math.pi) > 1e-3]
    assert not wrong, f"{len(wrong)}/60 edges rotated by an angle that is not their push"


def test_bug8_every_edge_points_right_after_the_training_rotation(tmp_path):
    """MORE's canonical frame (dataset.py:69-87): after rotating, the object lies to the RIGHT of
    the contact and not to its left. Checked on all 60 edges with a rendered object mask."""
    import train_more
    rot = getattr(train_more, "rotate_batch")
    pix = getattr(train_more, "rotate_pixel")
    ctx = np.zeros((1, 5, S, S), np.float16)
    ctx[0, 2] = rect_mask()
    edges, ds = _dataset_all_edges(tmp_path, ctx=ctx)
    bad = []
    for e, i in edges:
        item = ds[i]
        img, ang = item[0][None], torch.tensor([float(item[2])])
        r = rot(img, ang)[0, 2]
        c0, r0 = pix(torch.tensor(board()[e])[None], ang, S)[0].round().long().tolist()
        right = r[max(r0 - 3, 0):r0 + 4, c0 + 3:c0 + 10].max().item()
        left = r[max(r0 - 3, 0):r0 + 4, max(c0 - 9, 0):c0 - 2].max().item()
        if not (right > 0.5 and left < 0.5):
            bad.append(e)
    assert not bad, f"{len(bad)}/60 edges do not point right after rotation: {bad[:10]}"


class _Shift(torch.nn.Module):
    """Stub net: the value at a pixel is the object mask `k` px to its right in the rotated frame."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        out = torch.zeros_like(x[:, :1])
        if self.k > 0:
            out[..., :-self.k] = x[:, 2:3, :, self.k:]
        else:
            out[..., -self.k:] = x[:, 2:3, :, :self.k]
        return out.expand(-1, 5, -1, -1)


def test_bug8_deploy_scoring_sees_every_push_pointing_right():
    from namo.rl_loop import more_scorer

    def scorer(net):
        sc = more_scorer.MoreScorer.__new__(more_scorer.MoreScorer)
        sc.device, sc.batch, sc.net = torch.device("cpu"), 60, net
        sc._angles = torch.tensor(2.0 * math.pi * np.arange(60) / 60, dtype=torch.float32)  # 13763e54 init
        return sc
    ctx = np.zeros((5, S, S), np.float32)
    ctx[2] = rect_mask()
    right = scorer(_Shift(6)).score_ctx(ctx, board())[:, 0]
    left = scorer(_Shift(-6)).score_ctx(ctx, board())[:, 0]
    wrong = [e for e in range(60) if not (right[e] > 0.5 and left[e] < 0.5)]
    assert not wrong, f"{len(wrong)}/60 edges are not presented as a push to the right"


# -- BUG-9: verify scores rooms training never saw ------------------------------------------------
def _verify_rooms(xml, frac=0.1, seed=0):
    """Verbatim selection rule of verify_more_net.py (13763e54 :70-74; now more_training.heldout_rooms)."""
    rooms = np.unique(xml)
    rng = np.random.default_rng(seed)
    return set(rng.choice(rooms, max(1, int(len(rooms) * frac)), replace=False))


def test_bug9_training_excludes_the_rooms_verify_scores(tmp_path, monkeypatch):
    xml = np.array([f"room_{i // 2}" for i in range(40)])
    ev = np.zeros((40, 60, 5), np.int8)
    reach = np.zeros((40, 60, 5), np.float32)
    w = np.zeros((40, 60, 5), np.float32)
    ev[:, 0, 0], reach[:, 0, 0], w[:, 0, 0] = 1, 1, 3
    h5 = write_h5(tmp_path / "t.h5", list(xml), ev, reach, w)
    ds = dataset_of_main(monkeypatch, h5, tmp_path / "run")
    trained = {xml[int(s)] for s in ds.index[:, 0]}
    assert trained and not (_verify_rooms(xml) & trained)
