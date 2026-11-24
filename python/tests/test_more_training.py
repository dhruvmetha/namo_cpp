"""MORE's training recipe as ported: schedule stepping, invalid-push labels, the EdgeCrossAttn path."""
import math

import h5py
import numpy as np
import pytest
import torch

from namo.rl_loop import more_training as mt


def _lrs_lifelong_trainer(iters_per_epoch, epochs, t0=50, t_mult=2, eta_min=1e-8, lr=1e-4):
    """lifelong_trainer.py:251-258, :276-378 as written: step() per iteration in epoch 0,
    step(epoch) after every later epoch; the `if epoch < 0` warmup never runs."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr, momentum=0.9, weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=t_mult, eta_min=eta_min)
    out = []
    for epoch in range(epochs):
        for _ in range(iters_per_epoch):
            out.append(opt.param_groups[0]["lr"])
            opt.step()
            if epoch == 0:
                sched.step()
        if epoch != 0:
            sched.step(epoch)
    return out


def test_more_schedule_steps_like_lifelong_trainer():
    p = torch.nn.Parameter(torch.zeros(1))
    opt, sched = mt.make_optimizer([p], mt.MORE_STAGE1)
    stepper = mt.Stepper(opt, sched, mt.MORE_STAGE1, iters_per_epoch=120)
    got = []
    for epoch in range(4):
        for step in range(120):
            got.append(opt.param_groups[0]["lr"])
            opt.step()
            stepper.after_iteration(epoch, step)
        stepper.after_epoch(epoch)
    assert got == pytest.approx(_lrs_lifelong_trainer(120, 4))
    assert max(got[:120]) == pytest.approx(1e-4) and got[50] == pytest.approx(1e-4)   # restart at 50


def test_more_schedule_has_the_finetune_stage():
    st = mt.stages("more")
    assert [s.name for s in st] == ["more", "more_ft"]
    assert (st[0].batch, st[0].lr, st[0].epochs, st[0].t0) == (8, 1e-4, 51, 50)
    assert (st[1].batch, st[1].lr, st[1].epochs, st[1].t0) == (28, 1e-5, 21, 20)
    assert all(s.warmup_iters == 0 and s.clip == 0 for s in st)


def test_overrides_touch_only_the_first_stage():
    st = mt.stages("more", epochs=2, batch=4)
    assert (st[0].epochs, st[0].batch) == (2, 4) and (st[1].epochs, st[1].batch) == (21, 28)


def test_heldout_rooms_matches_the_verify_rule():
    xml = [f"r{i}" for i in range(30)]
    rooms = np.unique(xml)
    rng = np.random.default_rng(0)
    assert mt.heldout_rooms(xml, 0.1, 0) == set(rng.choice(rooms, 3, replace=False))
    assert mt.heldout_rooms(xml, 0.0, 0) == set()


def test_push_angle_points_from_an_edge_to_its_mate():
    board = np.zeros((60, 2), np.float32)
    board[6], board[7] = (10.0, 20.0), (10.0, 40.0)        # mate below: push straight down (+row)
    assert mt.push_angle(board, 6) == pytest.approx(math.pi / 2)
    assert mt.push_angle(board, 7) == pytest.approx(-math.pi / 2)


def test_more_invalid_mode_paints_unreachable_cells_at_0001():
    import train_more
    S = 64
    target, wmap = torch.zeros(1, 5, S, S), torch.zeros(1, 5, S, S)
    rboard = torch.full((1, 60, 2), 32.0)
    rboard[0, 7] = torch.tensor([10.0, 50.0])               # (col, row)
    un = torch.zeros(1, 60, 5)
    un[0, 7, 3] = 1
    train_more.paint_invalid(target, wmap, rboard, un)
    assert wmap[0, 3, 50, 10] == pytest.approx(mt.MORE_INVALID_WEIGHT)
    assert target[0, 3, 49:52, 9:12].abs().sum() == 0 and wmap.sum() == pytest.approx(9 * 0.001)


def _tiny_h5(path, size, n=6):
    rng = np.random.default_rng(0)
    ev = np.zeros((n, 60, 5), np.int8)
    ev[:, :4, 0] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("ctx", data=rng.random((n, 5, size, size)).astype(np.float16))
        f.create_dataset("contact_px", data=(rng.random((n, 60, 2)) * (size - 1)).astype(np.float32))
        f.create_dataset("label", data=np.where(ev == 1, 0.55, 0).astype(np.float32))
        f.create_dataset("weight", data=np.where(ev == 1, 3, 0).astype(np.float32))
        f.create_dataset("reach_mask", data=np.where(np.arange(60)[None, :, None] < 20, 1.0, 0.0)
                         * np.ones((n, 60, 5), np.float32))
        f.create_dataset("evidence", data=ev)
        vlen = h5py.string_dtype("utf-8")
        f.create_dataset("xml", data=np.array([f"room_{i}" for i in range(n)], dtype=object), dtype=vlen)
        f.create_dataset("object_id", data=np.array(["obj"] * n, dtype=object), dtype=vlen)
    return path


@pytest.mark.parametrize("invalid", ["ours", "more"])
def test_edgecrossattn_trains_on_more_labels_and_loads_as_a_scorer(tmp_path, monkeypatch, invalid):
    """train_more_eca writes a checkpoint eval_scorer.load_scorer reads, with a scalar head whose
    raw output is MORE's units (no sigmoid)."""
    import train_more_eca
    from eval_scorer import load_scorer
    h5 = _tiny_h5(tmp_path / "t.h5", 64)
    monkeypatch.setattr("sys.argv", ["train_more_eca.py", "--h5", str(h5), "--out", str(tmp_path / "run"),
                                     "--epochs", "1", "--batch", "2", "--workers", "0", "--invalid", invalid])
    train_more_eca.main()
    ck = sorted((tmp_path / "run").glob("*.ckpt"))[-1]
    model = load_scorer(str(ck), 5, "cpu", "edge_crossattn")
    with h5py.File(h5) as f:
        x = torch.tensor(f["ctx"][0:1].astype(np.float32))
        c = torch.tensor(f["contact_px"][0:1])
    out = model(x, c)
    assert out.shape == (1, 60, 5)
    net = train_more_eca.make_net()
    net.load_state_dict({k[len("network."):]: v for k, v in torch.load(ck, weights_only=False)["state_dict"].items()})
    net.eval()
    assert torch.allclose(net(x, c), out)


def test_edgecrossattn_loss_is_more_smoothl1_times_weight_over_the_batch():
    import train_more_eca
    pred = torch.tensor([[[0.0, 1.0]]]).expand(2, 1, 2)
    target = torch.tensor([[[1.0, 1.0]]]).expand(2, 1, 2)
    weight = torch.tensor([[[3.0, 5.0]]]).expand(2, 1, 2)
    # SmoothL1(beta 0.8) of |1| is 1 - 0.4 = 0.6; summed over the batch of 2 then / 2.
    assert train_more_eca.more_loss(pred, target, weight).item() == pytest.approx(3 * 0.6)
