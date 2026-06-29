"""Pure cost helpers for LevinTS search ordering — no env/model deps.

LevinTS expands nodes in non-decreasing order of cost = depth / pi(node), where pi(node) is the
product of action probabilities along the path to the node. We carry cumulative log-pi for stability.
"""
import math


def softmax_logp(scores, tau=1.0):
    """log-softmax of `scores` at temperature `tau`. Returns log-probs (<=0), same order. [] -> []."""
    if not scores:
        return []
    if tau <= 0:
        raise ValueError("tau must be > 0")
    z = [s / tau for s in scores]
    m = max(z)
    logden = m + math.log(sum(math.exp(zi - m) for zi in z))
    return [zi - logden for zi in z]


def levin_cost(depth, cum_logpi):
    """LevinTS cost = depth / pi = depth * exp(-cum_logpi). Lower = expand first. depth >= 1."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return depth * math.exp(-cum_logpi)
