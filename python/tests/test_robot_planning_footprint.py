import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def car_env():
    try:
        import namo_rl  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"namo_rl not importable: {e}")

    repo_root = Path(__file__).resolve().parents[2]
    xml_path = repo_root / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml"
    config_path = repo_root / "config" / "namo_config_car.yaml"

    if not xml_path.exists():
        pytest.skip(f"Missing XML fixture: {xml_path}")
    if not config_path.exists():
        pytest.skip(f"Missing config: {config_path}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    prev_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        env = namo_rl.RLEnvironment(str(xml_path), str(config_path), visualize=False)
        env.reset()
        return env
    finally:
        os.chdir(prev_cwd)


def test_robot_footprint_export_matches_geometry_and_snapshot(car_env):
    try:
        from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    except Exception as e:  # pragma: no cover
        pytest.skip(f"namo package not importable: {e}")

    object_info = car_env.get_object_info()
    robot_info = object_info["robot"]

    assert robot_info["size_x"] == pytest.approx(0.035, abs=1e-6)
    assert robot_info["size_y"] == pytest.approx(0.038, abs=1e-6)
    assert robot_info["size_z"] == pytest.approx(0.07, abs=1e-6)

    exporter = WavefrontSnapshotExporter(car_env)
    assert exporter.robot_half_extent == pytest.approx((robot_info["size_x"], robot_info["size_y"]), abs=1e-6)
