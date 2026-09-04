# Combined Region-Opening Ridgeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate overall-test-set simulator-push and wall-time ridgelines by pooling the existing 1,310 one-push and 973 two-push common episodes.

**Architecture:** Keep loading and across-seed aggregation unchanged, then add a pure pooling function that horizon-tags every episode key before concatenation. Reuse the existing panel renderer for one new combined panel per metric while retaining the existing two-panel outputs.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, pytest.

**Engineering Standards:** Keep aggregation single-pass over the two small horizon mappings, document new functions, reuse the existing renderer and named plotting constants, preserve environment-provided data roots, emit contextual population errors through the existing loader, and require focused tests plus real-output validation before completion. No new configuration surface, secrets, or security-sensitive behavior is introduced.

---

### Task 1: Specify combined population behavior

**Files:**
- Create: `python/tests/test_plot_region_opening_ridgelines.py`
- Modify: `scripts/experiments/plot_region_opening_ridgelines.py`

- [ ] **Step 1: Write the failing pooling tests**

Create a small two-horizon fixture in which the same episode key occurs in both horizons. Assert that `combine_horizons` returns two horizon-tagged observations per method and metric rather than overwriting one. Add a second assertion that `successful_costs_and_unsolved_percentage` returns three successful values and a 25% unsolved rate for four pooled observations.

```python
def test_combine_horizons_tags_keys_and_preserves_all_observations():
    data = synthetic_horizon_data()
    combined = plot.combine_horizons(data)
    costs = combined["costs"]["HY5U"]["sims"]
    assert len(costs) == 4
    assert ("1push", ("shared.xml", "box", "goal")) in costs
    assert ("2push", ("shared.xml", "box", "goal")) in costs


def test_combined_unsolved_percentage_uses_pooled_denominator():
    combined = plot.combine_horizons(synthetic_horizon_data())
    solved, unsolved_pct = plot.successful_costs_and_unsolved_percentage(
        combined, "HY5U", "sims"
    )
    assert solved == [1.0, 2.0, 4.0]
    assert unsolved_pct == pytest.approx(25.0)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest -q python/tests/test_plot_region_opening_ridgelines.py
```

Expected: collection fails because `combine_horizons` and `successful_costs_and_unsolved_percentage` do not exist.

- [ ] **Step 3: Implement minimal pooling helpers**

Add `COMBINED_HORIZONS = ("1push", "2push")`. Implement `combine_horizons(data)` by building the existing `{"costs": {method: {metric: mapping}}}` shape and storing every observation under `(horizon, episode_key)`. Implement `successful_costs_and_unsolved_percentage(data, method, metric)` and make `panel` call it instead of duplicating success filtering.

```python
def combine_horizons(data: dict) -> dict:
    """Pool horizon populations while preserving each episode as one observation."""
    costs = {
        method: {metric: {} for metric in METRICS}
        for method in METHODS
    }
    for horizon in COMBINED_HORIZONS:
        for method in METHODS:
            for metric in METRICS:
                costs[method][metric].update({
                    (horizon, key): observation
                    for key, observation in data[horizon]["costs"][method][metric].items()
                })
    return {"costs": costs}
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: both pooling tests pass.

- [ ] **Step 5: Commit pooling behavior**

```bash
git add python/tests/test_plot_region_opening_ridgelines.py scripts/experiments/plot_region_opening_ridgelines.py
git commit -m "feat: pool region-opening ridgeline populations"
```

### Task 2: Render combined outputs

**Files:**
- Modify: `python/tests/test_plot_region_opening_ridgelines.py`
- Modify: `scripts/experiments/plot_region_opening_ridgelines.py`

- [ ] **Step 1: Write the failing rendering test**

Render the synthetic combined fixture to a temporary stem and assert that both PNG and PDF files exist and are nonempty.

```python
def test_plot_combined_metric_writes_png_and_pdf(tmp_path):
    combined = plot.combine_horizons(synthetic_horizon_data())
    output = tmp_path / "region_opening_cost_ridgelines_combined"
    plot.plot_combined_metric("sims", combined, output)
    for suffix in (".png", ".pdf"):
        assert output.with_suffix(suffix).stat().st_size > 0
```

- [ ] **Step 2: Run the rendering test and verify RED**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest -q python/tests/test_plot_region_opening_ridgelines.py::test_plot_combined_metric_writes_png_and_pdf
```

Expected: failure because `plot_combined_metric` does not exist.

- [ ] **Step 3: Implement the single-panel renderer and CLI outputs**

Add a named panel title mapping including `"combined": "Overall test set"`. Implement `plot_combined_metric` as a 4.2-by-3.0-inch one-axis figure that calls the existing `panel`, and add main calls for `<stem>_combined` and `<stem>_wall_time_combined` after the existing outputs.

```python
combined = combine_horizons(data)
plot_combined_metric("sims", combined, args.out.with_name(args.out.name + "_combined"))
plot_combined_metric(
    "t_wall",
    combined,
    args.out.with_name(args.out.name + "_wall_time_combined"),
)
```

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
source env.robotlearning.sh
"$NAMO_PYTHON" -m pytest -q python/tests/test_plot_region_opening_ridgelines.py
"$NAMO_PYTHON" -m py_compile scripts/experiments/plot_region_opening_ridgelines.py scripts/experiments/tabulate_region_opening_costs.py
```

Expected: all focused tests pass and both scripts compile.

- [ ] **Step 5: Commit combined rendering**

```bash
git add python/tests/test_plot_region_opening_ridgelines.py scripts/experiments/plot_region_opening_ridgelines.py
git commit -m "viz: render combined region-opening ridgelines"
```

### Task 3: Generate and inspect the real figures

**Files:**
- Generate: `docs/experiments/plots/region_opening_cost_ridgelines_combined.{png,pdf}`
- Generate: `docs/experiments/plots/region_opening_cost_ridgelines_wall_time_combined.{png,pdf}`

- [ ] **Step 1: Run the production plotter where registered raw results are readable**

```bash
source env.ilab.sh
"$NAMO_PYTHON" scripts/experiments/plot_region_opening_ridgelines.py
```

Expected: the loader verifies common populations of 1,310 and 973 before writing all eight existing and combined files.

- [ ] **Step 2: Validate generated artifacts**

Verify all four combined files are nonempty, inspect both PNGs, and confirm three ridges, complete annotations, one shared log-cost axis, the `Overall test set` title, and no one-push/two-push split.

- [ ] **Step 3: Commit the real combined outputs**

```bash
git add docs/experiments/plots/region_opening_cost_ridgelines_combined.* docs/experiments/plots/region_opening_cost_ridgelines_wall_time_combined.*
git commit -m "figures: add combined region-opening cost ridgelines"
```
