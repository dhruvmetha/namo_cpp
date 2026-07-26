"""Index-table metrics. The headline quantity is the queue position of the best truly-good push."""


def _ordered(pool):
    return sorted(pool, key=lambda c: (-c["q"], c["edge"], c["depth"]))


def rank_of_best_green(pool, green):
    for i, c in enumerate(_ordered(pool), start=1):
        if (c["edge"], c["depth"]) in green:
            return i
    return None


def top1_truth(pool, openers, setups):
    if not pool:
        return "empty"
    c = _ordered(pool)[0]
    k = (c["edge"], c["depth"])
    if k in openers:
        return "opener"
    if k in setups:
        return "setup"
    return "dead"
