from __future__ import annotations

from pathlib import Path

import pytest

from full_namo_sim_exp.pipeline import main


def test_launch_command_pins_partition_and_hardware(
    experiment_path: Path,
    capsys,
) -> None:
    assert main(["validate", "--experiment", str(experiment_path)]) == 0
    status = main(["launch-command", "--experiment", str(experiment_path)])

    output = capsys.readouterr().out
    assert status == 0
    assert "--partition=main-redhat" in output
    assert "--constraint=icelake" in output
    assert "--exclusive" in output


@pytest.mark.parametrize("command", ["aggregate", "stats", "plot", "all"])
def test_postprocessing_commands_verify_frozen_provenance(
    experiment_path: Path,
    command: str,
) -> None:
    assert main(["validate", "--experiment", str(experiment_path)]) == 0
    experiment_path.write_text(experiment_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config SHA-256 changed"):
        main([command, "--experiment", str(experiment_path)])
