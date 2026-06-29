import math
import pytest
from levin_cost import softmax_logp, levin_cost


def test_softmax_logp_uniform():
    lp = softmax_logp([0.0, 0.0, 0.0])
    assert all(abs(x - math.log(1 / 3)) < 1e-9 for x in lp)


def test_softmax_logp_sums_to_one():
    lp = softmax_logp([1.0, 2.0, 3.0])
    assert abs(sum(math.exp(x) for x in lp) - 1.0) < 1e-9


def test_softmax_logp_order_preserved():
    lp = softmax_logp([3.0, 1.0, 2.0])
    assert lp[0] > lp[2] > lp[1]


def test_softmax_logp_tau_sharpens():
    hot = softmax_logp([1.0, 0.0], tau=0.5)
    cold = softmax_logp([1.0, 0.0], tau=2.0)
    assert math.exp(hot[0]) > math.exp(cold[0])   # lower tau -> sharper -> bigger top prob


def test_softmax_logp_empty():
    assert softmax_logp([]) == []


def test_softmax_logp_tau_nonpositive_raises():
    with pytest.raises(ValueError):
        softmax_logp([1.0], tau=0.0)


def test_levin_cost_depth1_uniform_of_two():
    lp = softmax_logp([0.0, 0.0])          # pi = 0.5
    assert abs(levin_cost(1, lp[0]) - 2.0) < 1e-9    # 1 / 0.5


def test_levin_cost_monotone_in_depth():
    assert levin_cost(2, math.log(0.5)) > levin_cost(1, math.log(0.5))


def test_levin_cost_lower_pi_higher_cost():
    assert levin_cost(1, math.log(0.1)) > levin_cost(1, math.log(0.9))


def test_levin_cost_bad_depth_raises():
    with pytest.raises(ValueError):
        levin_cost(0, math.log(0.5))
