"""Tests for combined region-opening ridgeline aggregation and rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

import plot_region_opening_ridgelines as plot  # noqa: E402


SHARED_KEY = ("shared.xml", "box", "goal")
ONE_PUSH_KEY = ("one.xml", "box", "goal")
TWO_PUSH_KEY = ("two.xml", "box", "goal")


def synthetic_horizon_data() -> dict:
    """Return two small horizon populations with one intentionally reused key."""
    one_push = {
        "sims": {
            SHARED_KEY: (True, 1.0),
            ONE_PUSH_KEY: (True, 2.0),
        },
        "t_wall": {
            SHARED_KEY: (True, 0.1),
            ONE_PUSH_KEY: (True, 0.2),
        },
    }
    two_push = {
        "sims": {
            SHARED_KEY: (False, None),
            TWO_PUSH_KEY: (True, 4.0),
        },
        "t_wall": {
            SHARED_KEY: (False, None),
            TWO_PUSH_KEY: (True, 0.4),
        },
    }
    return {
        "1push": {
            "costs": {
                method: {metric: dict(values) for metric, values in one_push.items()}
                for method in plot.METHODS
            }
        },
        "2push": {
            "costs": {
                method: {metric: dict(values) for metric, values in two_push.items()}
                for method in plot.METHODS
            }
        },
    }


def test_combine_horizons_tags_keys_and_preserves_all_observations():
    combined = plot.combine_horizons(synthetic_horizon_data())

    costs = combined["costs"]["HY5U"]["sims"]

    assert len(costs) == 4
    assert ("1push", SHARED_KEY) in costs
    assert ("2push", SHARED_KEY) in costs


def test_combined_unsolved_percentage_uses_pooled_denominator():
    combined = plot.combine_horizons(synthetic_horizon_data())

    solved, unsolved_pct = plot.successful_costs_and_unsolved_percentage(
        combined, "HY5U", "sims"
    )

    assert solved == [1.0, 2.0, 4.0]
    assert unsolved_pct == pytest.approx(25.0)


def test_plot_combined_metric_writes_png_and_pdf(tmp_path):
    combined = plot.combine_horizons(synthetic_horizon_data())
    output = tmp_path / "region_opening_cost_ridgelines_combined"

    plot.plot_combined_metric("sims", combined, output)

    for suffix in (".png", ".pdf"):
        assert output.with_suffix(suffix).stat().st_size > 0


def test_combined_panel_has_no_title():
    combined = plot.combine_horizons(synthetic_horizon_data())
    figure, axis = plot.plt.subplots()

    plot.panel(axis, "combined", "sims", combined, show_method_labels=True)

    assert axis.get_title() == ""
    plot.plt.close(figure)
