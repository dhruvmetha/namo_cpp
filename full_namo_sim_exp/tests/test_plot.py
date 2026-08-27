from __future__ import annotations

from pathlib import Path

import pytest

from full_namo_sim_exp import plot
from full_namo_sim_exp.experiment_io import load_experiment

from .conftest import write_aggregate


def test_plot_uses_green_log_axes_and_tail_fractions(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)
    write_aggregate(experiment.aggregate_root(experiment.model), {0, 1, 2})
    for arm in experiment.random_arms:
        write_aggregate(experiment.aggregate_root(arm), {0, 1})

    figure = plot.create_figure(experiment)

    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert axis.get_xscale() == "log"
        assert [line.get_color() for line in axis.lines] == ["#009E73", "#999999"]
        assert [text.get_text() for text in axis.texts] == ["3/4", "10/20"]
        assert len(axis.collections) == 0
    plot.plt.close(figure)


def test_render_writes_pdf_png_and_caption_statistics(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)
    write_aggregate(experiment.aggregate_root(experiment.model), {0, 1, 2})
    for arm in experiment.random_arms:
        write_aggregate(experiment.aggregate_root(arm), {0, 1})

    outputs = plot.render(experiment)

    assert {path.suffix for path in outputs} == {".pdf", ".png"}
    assert all(path.stat().st_size > 1000 for path in outputs)
