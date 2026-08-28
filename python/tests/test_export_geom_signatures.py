"""Reproducibility tests for compact training-geometry references."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPT_DIR))

import export_geom_signatures as exporter  # noqa: E402
from verify_geom_disjoint import geom_sig  # noqa: E402


def write_scene(path: Path, *, wall_x: float, obstacle_x: float) -> Path:
    path.write_text(
        "<mujoco><worldbody>"
        f'<geom name="wall_1" pos="{wall_x} 0 0" size="0.1 1 0.1"/>'
        f'<geom name="obstacle_0_movable" pos="{obstacle_x} 0 0" '
        'size="0.1 0.1 0.1"/>'
        "</worldbody></mujoco>\n",
        encoding="utf-8",
    )
    return path


def test_export_signatures_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    scene_b = write_scene(tmp_path / "b.xml", wall_x=2.0, obstacle_x=2.0)
    scene_a = write_scene(tmp_path / "a.xml", wall_x=1.0, obstacle_x=1.0)
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{scene_b}\n{scene_a}\n{scene_a}\n", encoding="utf-8")
    artifact = tmp_path / "training_geometry.json"

    exporter.export_signatures(
        train_specs=[manifest],
        out_path=artifact,
        workers=1,
    )

    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result == {
        "schema": "namo-room-geometry-signatures-v1",
        "sources": [
            {
                "path": str(manifest.resolve()),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        ],
        "counts": {
            "xml_paths": 3,
            "unique_xml_paths": 2,
            "unique_room_signatures": 2,
            "unique_floorplan_signatures": 2,
        },
        "full_signatures": sorted([geom_sig(scene_a)[0], geom_sig(scene_b)[0]]),
        "wall_signatures": sorted([geom_sig(scene_a)[1], geom_sig(scene_b)[1]]),
    }


def test_export_signatures_refuses_an_incomplete_geometry_export(tmp_path: Path) -> None:
    valid = write_scene(tmp_path / "valid.xml", wall_x=1.0, obstacle_x=1.0)
    invalid = tmp_path / "invalid.xml"
    invalid.write_text("<mujoco/>\n", encoding="utf-8")
    manifest = tmp_path / "train.txt"
    manifest.write_text(f"{valid}\n{invalid}\n", encoding="utf-8")
    artifact = tmp_path / "training_geometry.json"

    with pytest.raises(ValueError, match="1 of 2 unique training XMLs are unparseable"):
        exporter.export_signatures(
            train_specs=[manifest],
            out_path=artifact,
            workers=1,
        )

    assert not artifact.exists()
