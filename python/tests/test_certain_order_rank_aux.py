import importlib.util
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_q2_rankaux_test", REPO / "scripts/rl_loop/train_q2_rankaux.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
certain_order_rank_aux_losses = MODULE.certain_order_rank_aux_losses
weighted_rank_aux = MODULE.weighted_rank_aux


def _loss(value, labels, mask, ceiling):
    return certain_order_rank_aux_losses(
        torch.tensor(value, dtype=torch.float32).reshape(1, 1, -1),
        torch.tensor(labels, dtype=torch.float32).reshape(1, 1, -1),
        torch.tensor(mask, dtype=torch.float32).reshape(1, 1, -1),
        torch.tensor(ceiling, dtype=torch.float32).reshape(1, 1, -1),
        temp=0.15,
    )


def test_exact_opener_ranks_above_lower_ceiling():
    good, opener, setup = _loss([0.9, 0.1], [1.0, 0.9], [1, 1], [0, 1])
    bad, _, _ = _loss([0.1, 0.9], [1.0, 0.9], [1, 1], [0, 1])
    assert good < bad
    assert torch.equal(good, opener)
    assert setup == 0


def test_exact_setup_ranks_above_strictly_lower_ceiling():
    good, opener, setup = _loss([0.9, 0.1], [0.9, 0.81], [1, 1], [0, 1])
    bad, _, _ = _loss([0.1, 0.9], [0.9, 0.81], [1, 1], [0, 1])
    assert good < bad
    assert opener == 0
    assert torch.equal(good, setup)


def test_equal_ceiling_is_not_an_opponent():
    low_equal, _, _ = _loss([0.5, 0.0, 0.1], [0.9, 0.9, 0.81], [1, 1, 1], [0, 1, 1])
    high_equal, _, _ = _loss([0.5, 1.0, 0.1], [0.9, 0.9, 0.81], [1, 1, 1], [0, 1, 1])
    assert torch.allclose(low_equal, high_equal)


def test_unknown_and_unreachable_cells_are_excluded():
    base, _, _ = _loss([0.5, 0.1, 0.1], [0.9, 0.81, -1.0], [1, 1, 0], [0, 1, 0])
    changed, _, _ = _loss([0.5, 0.1, 1.0], [0.9, 0.81, -1.0], [1, 1, 0], [0, 1, 0])
    assert torch.allclose(base, changed)


def test_row_without_strictly_lower_action_has_zero_loss():
    total, opener, setup = _loss([0.5, 1.0], [0.9, 0.9], [1, 1], [0, 1])
    assert total == 0
    assert opener == 0
    assert setup == 0


def test_multiple_exact_tiers_each_receive_ranking_pressure():
    value = torch.tensor([[[0.9, 0.7, 0.1]]], requires_grad=True)
    labels = torch.tensor([[[1.0, 0.9, 0.81]]])
    mask = torch.ones_like(labels)
    ceiling = torch.tensor([[[0.0, 0.0, 1.0]]])
    total, opener, setup = certain_order_rank_aux_losses(
        value, labels, mask, ceiling, temp=0.15)
    total.backward()
    assert opener > 0
    assert setup > 0
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_mixed_batch_with_different_valid_tiers_stays_finite():
    value = torch.tensor([
        [[0.8, 0.2, 0.5]],
        [[0.5, 0.7, 0.1]],
    ], requires_grad=True)
    labels = torch.tensor([
        [[1.0, 0.9, 0.9]],
        [[0.9, 0.9, 0.81]],
    ])
    mask = torch.ones_like(labels)
    ceiling = torch.tensor([
        [[0.0, 1.0, 1.0]],
        [[0.0, 1.0, 1.0]],
    ])
    total, opener, setup = certain_order_rank_aux_losses(
        value, labels, mask, ceiling, temp=0.15)
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(opener)
    assert torch.isfinite(setup)
    assert torch.isfinite(value.grad).all()


def test_split_weights_preserve_full_opener_and_add_lower_pool():
    opener = torch.tensor(2.0)
    lower = torch.tensor(4.0)
    assert torch.equal(weighted_rank_aux(opener, torch.tensor(0.0)), torch.tensor(0.2))
    assert torch.equal(weighted_rank_aux(opener, lower), torch.tensor(0.4))
