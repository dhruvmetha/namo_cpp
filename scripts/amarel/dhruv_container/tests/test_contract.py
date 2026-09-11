"""Small contract checks; no cluster, image download, or simulator required."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


RECIPES = Path(__file__).resolve().parents[1]


def identity_module():
    spec = importlib.util.spec_from_file_location("native_identity", RECIPES / "native_identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_pins_reference_toolchain():
    definition = (RECIPES / "ubuntu20.def").read_text()
    for pin in ("From: ubuntu:20.04", "gcc-9=9.4.0-1ubuntu1~20.04.2",
                "g++-9=9.4.0-1ubuntu1~20.04.2", "libc6=2.31-0ubuntu9.18",
                "libstdc++6=10.5.0-1ubuntu1~20.04", "binutils=2.34-6ubuntu1.11"):
        assert pin in definition


def test_missing_environment_fails_before_build(tmp_path):
    config = tmp_path / "empty.env"
    config.write_text("")
    result = subprocess.run(["bash", str(RECIPES / "build.sh"), str(config)],
                            text=True, capture_output=True, env={"PATH": os.environ["PATH"]})
    assert result.returncode != 0
    assert "CONTAINER_ROOT" in result.stderr


@pytest.mark.parametrize("existing", ["dhruv-focal.sif", "mujoco", "code/build_python"])
def test_build_refuses_existing_artifacts(tmp_path, existing):
    root = tmp_path / "run"
    (root / existing).mkdir(parents=True)
    config = tmp_path / "run.env"
    config.write_text(f"CONTAINER_ROOT='{root}'\nFROZEN_ROOT='{tmp_path / 'frozen'}'\n"
                      f"PYTHON_ENV='{tmp_path / 'python'}'\n")
    result = subprocess.run(["bash", str(RECIPES / "build.sh"), str(config)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "refusing existing artifact" in result.stderr
    assert (root / existing).exists()


def test_identity_comparison_ignores_paths_but_not_versions_or_hashes():
    module = identity_module()
    reference = {"compiler": "9.4.0", "glibc": "2.31", "packages": {"libc6": "pinned"},
                 "libraries": {"libc-2.31.so": {"path": "/a/libc", "sha256": "abc"}}}
    actual = json.loads(json.dumps(reference))
    actual["libraries"]["libc-2.31.so"]["path"] = "/b/libc"
    assert module.compare(reference, actual) == []
    actual["libraries"]["libc-2.31.so"]["sha256"] = "changed"
    assert module.compare(reference, actual) == ["library:libc-2.31.so"]
    actual["compiler"] = "14.3.0"
    assert "compiler" in module.compare(reference, actual)


def test_identity_missing_library_is_a_mismatch():
    module = identity_module()
    assert module.compare({"libraries": {"libm.so": {"sha256": "x"}}},
                          {"libraries": {}}) == ["library:libm.so"]


def test_native_build_uses_frozen_target_and_validation_reuses_existing_runner():
    build = (RECIPES / "native-build.sh").read_text()
    assert "dhruv-native.flags" in build
    assert "NAMO_MARCH=native" not in build
    assert "-march=x86-64-v3" not in build
    validate = (RECIPES / "validate.sh").read_text()
    assert "validate_full_namo_alignment.py" in validate
    assert "--phase fixed" in validate and "--phase search" in validate
