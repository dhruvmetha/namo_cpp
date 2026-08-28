"""Tests for the frozen Full NAMO population builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPT_DIR))

import build_full_namo_population as builder  # noqa: E402
from verify_geom_disjoint import geom_sig  # noqa: E402


def write_scene(path: Path, *, wall_x: float, obstacle_x: float) -> Path:
    """Write the geometry fields consumed by the canonical room signature."""
    path.write_text(
        "<mujoco><worldbody>"
        f'<geom name="wall_1" pos="{wall_x} 0 0" size="0.1 1 0.1"/>'
        f'<geom name="obstacle_0_movable" pos="{obstacle_x} 0 0" '
        'size="0.1 0.1 0.1"/>'
        "</worldbody></mujoco>\n",
        encoding="utf-8",
    )
    return path


def probe_row(scene: Path, **overrides: object) -> dict[str, object]:
    """Return one structurally valid zero-simulation probe row."""
    row: dict[str, object] = {
        "xml_path": str(scene),
        "error": None,
        "goal_in_free_space": True,
        "no_path": False,
        "hop_count": 2,
        "hop_mismatch": False,
        "no_blocking_objects": False,
        "no_reachable_blocker": False,
        "no_pushable_blocker": False,
    }
    row.update(overrides)
    return row


def write_inputs(
    tmp_path: Path,
    scenes: list[Path],
    probes: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Write one candidate manifest and its probe JSONL."""
    manifest = tmp_path / "candidates.txt"
    manifest.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    probe = tmp_path / "probe.jsonl"
    probe.write_text("".join(json.dumps(row) + "\n" for row in probes), encoding="utf-8")
    return manifest, probe


def test_build_population_keeps_only_structurally_valid_geometry_disjoint_scenes(
    tmp_path: Path,
) -> None:
    valid = write_scene(tmp_path / "valid.xml", wall_x=1.0, obstacle_x=1.0)
    structural_drop = write_scene(
        tmp_path / "structural_drop.xml",
        wall_x=2.0,
        obstacle_x=2.0,
    )
    leaked = write_scene(tmp_path / "leaked.xml", wall_x=3.0, obstacle_x=3.0)
    train = write_scene(tmp_path / "train.xml", wall_x=3.0, obstacle_x=3.0)
    train_manifest = tmp_path / "train.txt"
    train_manifest.write_text(f"{train}\n", encoding="utf-8")
    manifest, probe = write_inputs(
        tmp_path,
        [valid, structural_drop, leaked],
        [
            probe_row(valid, solved=False),
            probe_row(structural_drop, no_pushable_blocker=True),
            probe_row(leaked),
        ],
    )
    out_dir = tmp_path / "population"

    audit = builder.build_population(
        manifest_path=manifest,
        probe_jsonl=probe,
        train_specs=[train_manifest],
        name="heldout-two-boundary-v1",
        expect_hop=2,
        out_dir=out_dir,
    )

    population = json.loads((out_dir / "population.json").read_text())
    assert population["name"] == "heldout-two-boundary-v1"
    assert population["scenes"] == [
        {
            "xml_path": str(valid.resolve()),
            "cluster_id": f"floorplan:{geom_sig(valid)[1]}",
        }
    ]
    dropped = [
        json.loads(line)
        for line in (out_dir / "dropped_scenes.jsonl").read_text().splitlines()
    ]
    assert {row["xml_path"]: row["reasons"] for row in dropped} == {
        str(leaked.resolve()): ["training_geometry_leak"],
        str(structural_drop.resolve()): ["no_pushable_blocker"],
    }
    assert audit["counts"] == {
        "input_scenes": 3,
        "accepted_scenes": 1,
        "dropped_scenes": 2,
        "training_scene_leaks": 1,
    }
    assert audit["drop_reasons"] == {
        "no_pushable_blocker": 1,
        "training_geometry_leak": 1,
    }
    assert (out_dir / "accepted_scenes.txt").read_text() == f"{valid.resolve()}\n"


@pytest.mark.parametrize(
    ("manifest_count", "probe_count", "message"),
    [
        (2, 1, "probe population mismatch: 1 missing, 0 extra"),
        (1, 2, "probe population mismatch: 0 missing, 1 extra"),
    ],
)
def test_build_population_requires_exact_probe_population(
    tmp_path: Path,
    manifest_count: int,
    probe_count: int,
    message: str,
) -> None:
    scenes = [
        write_scene(tmp_path / f"scene_{index}.xml", wall_x=index + 1, obstacle_x=1.0)
        for index in range(2)
    ]
    manifest, probe = write_inputs(
        tmp_path,
        scenes[:manifest_count],
        [probe_row(scene) for scene in scenes[:probe_count]],
    )

    with pytest.raises(ValueError, match=message):
        builder.build_population(
            manifest_path=manifest,
            probe_jsonl=probe,
            train_specs=[],
            name="population",
            expect_hop=2,
            out_dir=tmp_path / "out",
        )


def test_build_population_rejects_duplicate_canonical_input_paths(tmp_path: Path) -> None:
    scene = write_scene(tmp_path / "scene.xml", wall_x=1.0, obstacle_x=1.0)
    manifest, probe = write_inputs(
        tmp_path,
        [scene, scene],
        [probe_row(scene)],
    )
    with pytest.raises(ValueError, match="duplicate manifest scene"):
        builder.build_population(
            manifest_path=manifest,
            probe_jsonl=probe,
            train_specs=[],
            name="population",
            expect_hop=2,
            out_dir=tmp_path / "out_manifest",
        )

    manifest, probe = write_inputs(
        tmp_path,
        [scene],
        [probe_row(scene), probe_row(scene)],
    )
    with pytest.raises(ValueError, match="duplicate probe scene"):
        builder.build_population(
            manifest_path=manifest,
            probe_jsonl=probe,
            train_specs=[],
            name="population",
            expect_hop=2,
            out_dir=tmp_path / "out_probe",
        )


def test_build_population_refuses_to_overwrite_frozen_outputs(tmp_path: Path) -> None:
    scene = write_scene(tmp_path / "scene.xml", wall_x=1.0, obstacle_x=1.0)
    manifest, probe = write_inputs(tmp_path, [scene], [probe_row(scene)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "population.json").write_text("frozen\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite.*population.json"):
        builder.build_population(
            manifest_path=manifest,
            probe_jsonl=probe,
            train_specs=[],
            name="population",
            expect_hop=2,
            out_dir=out_dir,
        )
