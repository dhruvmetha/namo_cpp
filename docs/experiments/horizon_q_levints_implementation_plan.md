# LevinTS / PHS* Adoption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Design spec: [horizon_q_levints_search_design.md](horizon_q_levints_search_design.md). Read it first.

**Goal:** Move Horizon-Q's search+learning onto LevinTS/PHS*, starting from a frozen-model priority swap
(`combine="levin"`), in gated one-change-at-a-time stages, each gated on **avg-sims-to-solve**.

**Architecture:** Stage 1 (this plan, fully detailed) adds a LevinTS `d/π` ordering as a new `--combine`
mode in the existing best-first search — same heap, different cost — plus solution-path logging, then an
A/B gate. Stages 2–4 (learning loss → sims cost-to-go `h` + PHS* → multi-push depth≥3) are a **gated
roadmap**: each becomes its own detailed plan once the prior gate's numbers land (their details depend on
those numbers — that is the experiment-driven constraint, not a placeholder).

**Tech Stack:** Python 3.11 (`/scratch/dm1487/envs/namo/bin/python`), `heapq` best-first search, the
`namo_rl` C++ bindings (MuJoCo 3.2.7) for sims, the EdgeCrossAttn scorer (sage_learning), pytest 9.

## Global Constraints

- **Python:** use `/scratch/dm1487/envs/namo/bin/python`; with `PYTHONPATH="$PWD/build_python:$PWD/python"`.
  On ilab (arrakis) the paths differ — `source env.ilab.sh` first (CLAUDE.ilab.md).
- **Frozen model for Stage 1:** the NoHz ranker ckpt — **read the path from
  [horizon_q_model_registry.md](horizon_q_model_registry.md) (NoHz-v3 row); NEVER glob/reconstruct it.**
- **No exhaustive ground truth (FOUNDATIONAL):** any learned `h`/value in Stages 2–3 trains on the
  search's OWN found solutions, NEVER on the global `(setup×finish)` pairmap. The current qboot `γ·V_GT`
  target is eval-luxury and must NOT be the deployable target.
- **Horizon is OFF, not deleted:** run single-Q (`budget_cond=False`); leave `budget_h` dormant.
- **Gate metric (every stage):** avg-sims-to-solve (primary) + solve-rate (secondary), on n≈1018
  episodes, **region** success criterion, **object-constrained** key (`pure2push.json`). 3 arms minimum.
- **One change at a time:** do not start a stage before the previous stage's gate is recorded.

---

# STAGE 1 — Frozen LevinTS search ordering (detailed)

### Task 1: Pure LevinTS cost helpers (`levin_cost.py`)

**Files:**
- Create: `scripts/sandbox/levin_cost.py`
- Test: `scripts/sandbox/test_levin_cost.py`

**Interfaces:**
- Produces: `softmax_logp(scores: list[float], tau: float = 1.0) -> list[float]` (log-probabilities, same
  order, each ≤ 0); `levin_cost(depth: int, cum_logpi: float) -> float` (= `depth * exp(-cum_logpi)`).

- [ ] **Step 1: Write the failing tests**

```python
# scripts/sandbox/test_levin_cost.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd $REPO && python -m pytest scripts/sandbox/test_levin_cost.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'levin_cost'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# scripts/sandbox/levin_cost.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd $REPO && python -m pytest scripts/sandbox/test_levin_cost.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sandbox/levin_cost.py scripts/sandbox/test_levin_cost.py
git commit -m "feat(levints): pure d/pi cost helpers + unit tests"
```

---

### Task 2: Wire `combine="levin"` into the search

**Files:**
- Modify: `scripts/sandbox/eval_bestfirst.py` (`solve_scene` `:57-83`, `priority` unchanged `:51-54`,
  arg parser `:96`/`:104`, caller `:142-144`)

**Interfaces:**
- Consumes: `softmax_logp`, `levin_cost` (Task 1); `candidates()` returning `(pool, V)` with
  `pool=[(obj,g,q)]`.
- Produces: `solve_scene(..., tau=1.0)` now returns a 4-tuple
  `(solved: bool, sims: int, plan_len: int|None, win_node: dict|None)`; nodes carry `"cum_logpi": float`.
  New CLI flags `--combine levin` and `--tau`.

- [ ] **Step 1: Add the import**

At `scripts/sandbox/eval_bestfirst.py:29` (after the `eval_m3` import) add:

```python
from levin_cost import softmax_logp, levin_cost  # noqa: E402
```

- [ ] **Step 2: Replace `solve_scene` with the LevinTS-aware version**

Replace the whole function (`:57-83`) with:

```python
def solve_scene(planner, env, goal, xml, s0, hmax, sim_budget, prior, agg, combine, rng, restrict_obj=None,
                is_open=lambda e: e.is_robot_goal_reachable(), raw=False, dive_bonus=0.0, tau=1.0):
    """Greedy best-first ON THE LABELED OBJECT (restrict_obj). Returns (solved, sims, plan_len|None, win|None).
    combine="levin": order the frontier by LevinTS cost depth/pi (min-heap), pi = softmax(q/tau) over the
    node's candidate pool, multiplied along the path (cum_logpi). depth = ndone+1. dive_bonus is IGNORED in
    levin mode (depth is baked into the cost). Other combines: unchanged (-priority max-heap + dive_bonus)."""
    heap = []; ctr = 0; sims = 0

    def push_pool(pool, V, from_state, parent_plan, parent_logpi, ndone):
        nonlocal ctr
        depth = ndone + 1                                  # this candidate's own push is #(ndone+1)
        logps = softmax_logp([q for (_o, _g, q) in pool], tau) if combine == "levin" else None
        for i, (obj, g, q) in enumerate(pool):
            cum_logpi = (parent_logpi + logps[i]) if combine == "levin" else 0.0
            sortkey = (levin_cost(depth, cum_logpi) if combine == "levin"
                       else -(priority(q, V, combine) + dive_bonus * ndone))   # min-heap on sortkey
            heapq.heappush(heap, (sortkey, ctr,
                                  {"obj": obj, "g": g, "from": from_state, "ndone": ndone,
                                   "plan": parent_plan + [(obj, g)], "cum_logpi": cum_logpi})); ctr += 1

    pool, V0 = candidates(planner, env, goal, xml, s0, hmax, prior, agg, rng, restrict_obj=restrict_obj, raw=raw)
    push_pool(pool, V0, s0, [], 0.0, 0)                    # roots: ndone=0, depth=1
    while heap and sims < sim_budget:
        _sk, _c, it = heapq.heappop(heap)
        env.set_full_state(it["from"]); env.step(make_action(it["obj"], it["g"])); sims += 1
        if is_open(env):
            return True, sims, len(it["plan"]), it
        ndone = it["ndone"] + 1
        if ndone < hmax:
            s_new = env.get_full_state()
            h = hmax - ndone
            pool, V = candidates(planner, env, goal, xml, s_new, h, prior, agg, rng, restrict_obj=restrict_obj, raw=raw)
            push_pool(pool, V, s_new, it["plan"], it["cum_logpi"], ndone)
    return False, sims, None, None
```

- [ ] **Step 3: Add the CLI flags**

At `scripts/sandbox/eval_bestfirst.py:96`, change the `--combine` choices to include `levin`:

```python
    ap.add_argument("--combine", default="blend", choices=["q", "blend", "product", "levin"])
```

After the `--dive-bonus` arg (`:104-107`) add:

```python
    ap.add_argument("--tau", type=float, default=1.0,
                    help="LevinTS softmax temperature for pi over the candidate pool (combine=levin only). "
                         "Lower=sharper policy=more aggressive dive. Swept in the Stage-1 gate.")
```

- [ ] **Step 4: Update the caller to unpack the 4-tuple**

At `scripts/sandbox/eval_bestfirst.py:142-144`, change the call to:

```python
                solved, sims, plen, _win = solve_scene(planner, env, goal, xml, s0, a.hmax, a.sim_budget,
                                                       a.prior, a.agg, a.combine, rng, restrict_obj=obj,
                                                       is_open=is_open, raw=a.raw, dive_bonus=a.dive_bonus,
                                                       tau=a.tau)
```

(`_win` is wired into the log in Task 3.)

- [ ] **Step 5: Smoke-test (env-coupled; this is the verification for Task 2)**

Get the frozen ckpt from the registry, then run a 3-room slice:

```bash
cd $REPO && CKPT=<NoHz-v3 path from horizon_q_model_registry.md>
python scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --combine levin --tau 1.0 \
    --hmax 2 --sim-budget 10 --start 0 --end 3 \
    --out /tmp/levin_smoke.json --leaf-out /tmp/levin_smoke.jsonl
```

Expected: prints a result JSON with non-null `avg_sims_to_solve` and `solve_rate`, no traceback; the
`combine=q` form on the same slice also still runs (regression check):

```bash
python scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --combine q \
    --hmax 2 --sim-budget 10 --start 0 --end 3 --out /tmp/q_smoke.json --leaf-out /tmp/q_smoke.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add scripts/sandbox/eval_bestfirst.py
git commit -m "feat(levints): combine=levin d/pi frontier ordering + --tau"
```

---

### Task 3: Log the solution path (for the future Levin-loss step)

**Files:**
- Modify: `scripts/sandbox/eval_bestfirst.py` (caller `:142`, leaf-log write `:146-147`)

**Interfaces:**
- Consumes: `win_node` dict from `solve_scene` (Task 2) with `"plan"=[(obj, g)]` and `"cum_logpi"`; goal
  primitives `g` expose `int(g.edge_idx)`, `int(g.depth)` (per `scorer_beam._candidates`).
- Produces: each solved leaf-log row gains `"solution": [{"obj","edge","depth"}]` and `"sol_logpi": float`.

- [ ] **Step 1: Add a plan serializer**

Above `def main():` (`:86`) add:

```python
def serialize_plan(plan):
    """[(obj, goal_primitive)] -> JSON-able [{obj, edge, depth}] for the solution-path log."""
    return [{"obj": o, "edge": int(g.edge_idx), "depth": int(g.depth)} for (o, g) in plan]
```

- [ ] **Step 2: Capture the win node and write it to the log**

At `:142` rename `_win` to `win`:

```python
                solved, sims, plen, win = solve_scene(planner, env, goal, xml, s0, a.hmax, a.sim_budget,
```

Replace the `lf.write(...)` (`:146-147`) with:

```python
                lf.write(json.dumps({"xml": xml, "object_id": obj, "region": rec.get("region"),
                                     "solved": solved, "sims": sims, "plan_len": plen,
                                     "solution": serialize_plan(win["plan"]) if win else None,
                                     "sol_logpi": (win["cum_logpi"] if win else None)}) + "\n")
```

- [ ] **Step 3: Verify the log captures solutions**

```bash
cd $REPO && CKPT=<NoHz-v3 path from registry>
python scripts/sandbox/eval_bestfirst.py --ckpt "$CKPT" --combine levin --tau 1.0 \
    --hmax 2 --sim-budget 10 --start 0 --end 5 --out /tmp/lev.json --leaf-out /tmp/lev.jsonl
python - <<'PY'
import json
rows = [json.loads(l) for l in open("/tmp/lev.jsonl")]
solved = [r for r in rows if r["solved"]]
assert solved, "expected at least one solved episode in the slice"
r = solved[0]
assert isinstance(r["solution"], list) and r["solution"], "solution path missing"
assert all({"obj", "edge", "depth"} <= set(s) for s in r["solution"])
assert r["sol_logpi"] is not None and r["sol_logpi"] <= 0.0
print("OK: solution-path logging works", r["solution"], r["sol_logpi"])
PY
```

Expected: prints `OK: solution-path logging works ...`.

- [ ] **Step 4: Commit**

```bash
git add scripts/sandbox/eval_bestfirst.py
git commit -m "feat(levints): log solution path + sol_logpi for the learning step"
```

---

### Task 4: STAGE-1 GATE — A/B `levin` vs `q` vs the `dive_bonus` hack

**Files:**
- Modify: [horizon_q_levints_search_design.md](horizon_q_levints_search_design.md) (fill §6 with numbers);
  [horizon_q_model_registry.md](horizon_q_model_registry.md) (add the eval rows + dirs).

**Interfaces:** consumes the full `eval_bestfirst.py` from Tasks 1–3, the frozen NoHz-v3 ckpt.

- [ ] **Step 1: Run the three arms (background, one per free GPU)**

On arrakis 5 GPUs are free — run arms in parallel. `CKPT` = NoHz-v3 from the registry.

```bash
cd $REPO && CKPT=<NoHz-v3 path from registry>; O=$NAMO_SCRATCH/eval/levints_gate
mkdir -p "$O"
COMMON="--ckpt $CKPT --hmax 2 --sim-budget 30 --start 0 --end 985 --success region"
# Arm A: baseline raw action-value
CUDA_VISIBLE_DEVICES=0 python scripts/sandbox/eval_bestfirst.py $COMMON --combine q \
    --out $O/q.json --leaf-out $O/q.jsonl &
# Arm B: the dive_bonus hack levin is meant to replace (use the value from the registry's best best-first row)
CUDA_VISIBLE_DEVICES=1 python scripts/sandbox/eval_bestfirst.py $COMMON --combine q --dive-bonus 1.0 \
    --out $O/q_dive.json --leaf-out $O/q_dive.jsonl &
# Arms C–E: LevinTS tau sweep
CUDA_VISIBLE_DEVICES=2 python scripts/sandbox/eval_bestfirst.py $COMMON --combine levin --tau 0.5 \
    --out $O/levin_t0.5.json --leaf-out $O/levin_t0.5.jsonl &
CUDA_VISIBLE_DEVICES=3 python scripts/sandbox/eval_bestfirst.py $COMMON --combine levin --tau 1.0 \
    --out $O/levin_t1.json --leaf-out $O/levin_t1.jsonl &
CUDA_VISIBLE_DEVICES=4 python scripts/sandbox/eval_bestfirst.py $COMMON --combine levin --tau 2.0 \
    --out $O/levin_t2.json --leaf-out $O/levin_t2.jsonl &
wait
```

(Each arm is ~1018 episodes × up to 30 sims ≈ a few hours. Use `run_in_background` / `nohup`; monitor the
per-20-room stderr progress lines.)

- [ ] **Step 2: Build the comparison table**

```bash
cd $REPO && O=$NAMO_SCRATCH/eval/levints_gate
python - "$O" <<'PY'
import json, sys, glob, os
O = sys.argv[1]
for f in sorted(glob.glob(f"{O}/*.json")):
    r = json.load(open(f))
    print(f"{os.path.basename(f):16s} combine={r['combine']:6s} dive={r['dive_bonus']} "
          f"solve%={r['solve_rate']:5.1f}  avg_sims_to_solve={r['avg_sims_to_solve']:.2f}  "
          f"avg_sims_all={r['avg_sims_all']:.2f}  n={r['n_episodes']}")
PY
```

- [ ] **Step 3: Apply the pre-registered gate and record the verdict**

Gate (spec §6): **ACCEPT-as-keeper iff** best `levin` arm's `avg_sims_to_solve` ≤ `combine=q` AND it
matches/beats `q --dive-bonus` **without** the dive knob. Expectation: `levin ≈ q` (within-node order is
identical; only the cross-depth dive differs). Write the table + ACCEPT/REJECT into the design spec §6 and
add the eval dir + headline numbers to the model registry. **Do not rationalize a loss into a win.**

```bash
git add docs/experiments/horizon_q_levints_search_design.md docs/experiments/horizon_q_model_registry.md
git commit -m "docs(levints): Stage-1 gate results (levin vs q vs dive_bonus)"
```

---

# STAGES 2–4 — Gated roadmap (detail each AFTER the prior gate)

> These are intentionally **not** bite-sized yet: each stage's design (loss weighting, `h` target shape,
> depth-≥3 labeling) depends on the previous gate's numbers. When a gate passes, run writing-plans again
> to expand that stage into TDD tasks. Listed here so the full arc + its gates are visible.

| Stage | Deliverable | Key files | Gate (avg-sims) | Depends on |
|---|---|---|---|---|
| **2 — Levin loss (learning half)** | Train π to MINIMIZE sims: minimize `L_k·log(1/π(n*))` on the Stage-1 logged solution paths (replace/augment the InfoNCE/HL-Gauss classification). The real lever. | `sage_learning/src/model/classifier_module.py` (`_compute_masked_loss`), `sage_learning/src/data/scorer_data.py`, new build that turns `*.jsonl` solution paths → training targets | retrained π under `combine=levin` beats the Stage-1 frozen number, 3 seeds | Stage 1 (logs) + a GPU train run |
| **3 — sims cost-to-go `h` + PHS\*** | A cost-to-go in SIMS trained on **found solutions** (remaining-sims observed), NOT `γ·V_GT`; add `combine="phs"` = `(g+h)/π`. **De-oracles the qboot.** | new `build_*_setup.py` (target = remaining-sims from leaf-logs, replacing `build_bootstrap_setup.py`'s pairmap read), `eval_bestfirst.py` (`combine="phs"`), value/`h` head | PHS\* `avg_sims` < LevinTS-only AND `h` trained with **zero** pairmap reads | Stage 2 (calibrated π) + Stage 1 logs |
| **4 — multi-push depth ≥ 3** | The real LevinTS payoff: search+learning past H=2. The `d/π` ordering already generalizes (verify with `--hmax 3`); the work is depth-≥3 finish-labeling/collection + candidate gen at deep nodes. | `eval_bestfirst.py` (`--hmax>2` verification), `scripts/pipeline/exit_collect.py` (depth-≥3 labeling), candidate gen | solves depth-3 problems the depth-2 system cannot, at bounded sims | Stages 1–3 |

---

## Self-review (done)

- **Spec coverage:** spec §1 decision context → Global Constraints; §2/§3/§4 (frozen `d/π` wiring) →
  Tasks 1–2; §5.3 logging → Task 3; §6 gate → Task 4; §7 deferreds → Stages 2–4 roadmap. No gaps.
- **Placeholder scan:** the only `<...>` is the registry ckpt path, deliberately fetched-not-globbed per
  the Global Constraints (a hardcoded path would violate the registry rule). No TBD/TODO in any step.
- **Type consistency:** `softmax_logp`/`levin_cost` signatures match between Task 1 and their Task 2 call
  sites; `solve_scene` 4-tuple return is produced in Task 2 and consumed in Task 3; node key `cum_logpi`
  is set in Task 2 and read in Task 3.
