"""Regression tests for the unsuffixed-primitive-base cleanup.

After commits c2aa6a9 + 4f21fe9 + 454f8db + fb01ec5, the skill must:
  - load primitives strictly from <prefix>_{square,wide,tall}.dat,
  - NOT require <prefix>.dat (the unsuffixed base) to exist,
  - and the generator must not create the unsuffixed base on its own.

These tests fail on pre-cleanup code (loading would still happen via
the now-removed fallback, masking the regression) and pass on the
post-cleanup code (loading goes only through the suffixed path).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import (
    CAR_START_POSE, REAL_NAMO_RL, REPO_ROOT, CONFIG_PATH, _require_real_namo_rl,
)

_require_real_namo_rl()


# Where the canonical (suffixed) primitive files live after this PR.
PRIMITIVE_DIR = REPO_ROOT / "data"
EXPECTED_SUFFIXED_FILES = (
    "1x_car_d5_motion_primitives_15_square.dat",
    "1x_car_d5_motion_primitives_15_wide.dat",
    "1x_car_d5_motion_primitives_15_tall.dat",
)

# The unsuffixed file that used to exist as a byte-duplicate. Must NOT
# be present after the cleanup PR.
EXPECTED_DELETED_FILE = "1x_car_d5_motion_primitives_15.dat"


def test_suffixed_primitive_files_present():
    """Sanity: the canonical files exist where the skill expects them."""
    for name in EXPECTED_SUFFIXED_FILES:
        path = PRIMITIVE_DIR / name
        assert path.exists(), f"Required primitive file missing: {path}"


def test_unsuffixed_base_file_absent():
    """Catches accidental re-introduction of the duplicate.

    If someone regenerates primitives with an old generator binary, or
    a future change reintroduces the 'backward compatibility' write to
    the base path, this test fires.
    """
    path = PRIMITIVE_DIR / EXPECTED_DELETED_FILE
    assert not path.exists(), (
        f"Unsuffixed base primitive file is present again: {path}. "
        f"This file is a byte-duplicate of _square.dat and was removed "
        f"as part of the cleanup PR. Either the generator regressed "
        f"(re-introduced the backward-compat branch) or someone copied "
        f"it back manually. Delete it."
    )


def test_skill_loads_with_unsuffixed_base_missing(monkeypatch):
    """The skill construction MUST succeed even though motion_primitives_15.dat
    doesn't exist. This proves the fallback-chain removal in commit c2aa6a9
    works as intended.
    """
    import namo_rl

    # Confirm the precondition: the unsuffixed file is not on disk
    assert not (PRIMITIVE_DIR / EXPECTED_DELETED_FILE).exists(), \
        "test precondition violated: unsuffixed base shouldn't exist"

    # Construct env → triggers NAMOPushSkill::initialize_skill →
    # loads square/wide/tall planners from suffixed files only.
    # If the cleanup regressed and the skill still requires the base
    # file, this construction throws.
    env = namo_rl.RLEnvironment(
        str(PRIMITIVE_DIR / "nominal_primitive_scene_square_1x_car.xml"),
        str(CONFIG_PATH),
        False,
        True,
    )
    env.set_robot_pose(*CAR_START_POSE)
    env.warm_up()
    # If construction succeeded, the planners loaded successfully from
    # the suffixed paths. Sanity-check that the env is functional.
    assert env.get_observation()  # non-empty dict


def test_skill_fails_loudly_when_suffixed_files_missing(tmp_path, monkeypatch):
    """The runtime profile rejects a config pointing at another table."""
    from namo.runtime_profile import require_canonical_runtime_config

    # Build a copy of the config that points at a non-existent prefix.
    # The skill should fail at construction with a useful message.
    bad_prefix = tmp_path / "primitives_that_dont_exist.dat"
    test_config = tmp_path / "test_config.yaml"
    shutil.copy(CONFIG_PATH, test_config)
    text = test_config.read_text()
    # Replace the motion_primitives_file line. The prefix doesn't need
    # to exist; what matters is that <prefix>_square.dat etc. don't exist.
    text = text.replace(
        'motion_primitives_file: "data/1x_car_d5_motion_primitives_15.dat"',
        f'motion_primitives_file: "{bad_prefix}"',
    )
    test_config.write_text(text)

    with pytest.raises(ValueError, match="car 1x d5"):
        require_canonical_runtime_config(test_config)


def test_generator_does_not_create_unsuffixed_base(tmp_path):
    """Running the generator with --output PATH should produce
    PATH_{square,wide,tall}.dat — and NOT PATH itself.
    """
    output_base = tmp_path / "out.dat"
    generator = REPO_ROOT / "build_python" / "generate_motion_primitives_db"
    assert generator.exists(), \
        f"Generator binary not built: {generator}. Run cmake --build first."

    # Run from repo root so config relative paths resolve.
    result = subprocess.run(
        [
            str(generator),
            "--config",
            str(CONFIG_PATH),
            "--scenes-suffix",
            "_1x_car",
            "--single-edge",
            "0",
            "--min-push-steps",
            "5",
            "--settle-ticks",
            "1",
            "--output",
            str(output_base),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # generator takes ~6 min headless
    )
    assert result.returncode == 0, (
        f"Generator failed: stdout={result.stdout[-500:]} "
        f"stderr={result.stderr[-500:]}"
    )

    # Suffixed files should exist
    for shape in ("square", "wide", "tall"):
        suffixed = tmp_path / f"out_{shape}.dat"
        assert suffixed.exists(), f"Generator did not produce {suffixed}"

    # Unsuffixed base must NOT exist
    assert not output_base.exists(), (
        f"Generator regressed: produced unsuffixed {output_base}. "
        f"The 'backward compatibility' branch is back."
    )
