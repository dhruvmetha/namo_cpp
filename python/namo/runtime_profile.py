"""The one supported NAMO runtime profile."""

from pathlib import Path

import yaml

CANONICAL_CONFIG = "config/namo_config_complete_skill15_car_1x.yaml"
CANONICAL_PRIMITIVE_PREFIX = "1x_car_d5_"
CANONICAL_NUM_DEPTHS = 5
CANONICAL_MOTION_PRIMITIVE_BASE = "1x_car_d5_motion_primitives_15.dat"


def require_canonical_runtime_config(config_path: str | Path) -> Path:
    """Reject robot and scale configurations outside car 1x d5."""
    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read NAMO runtime config {path}: {exc}") from exc

    planning = config.get("planning", {}) or {}
    skill = config.get("skill", {}) or {}
    primitives = config.get("motion_primitives", {}) or {}
    system = config.get("system", {}) or {}
    robot_size = planning.get("robot_size")
    primitive_base = Path(str(system.get("motion_primitives_file", ""))).name

    canonical = (
        planning.get("robot_type") == "diff_drive"
        and planning.get("high_level_resolution") == 0.01
        and planning.get("skill_level_resolution") == 0.005
        and robot_size == [0.035, 0.035]
        and skill.get("max_push_steps") == CANONICAL_NUM_DEPTHS
        and primitives.get("max_push_steps") == CANONICAL_NUM_DEPTHS
        and primitive_base == CANONICAL_MOTION_PRIMITIVE_BASE
    )
    if not canonical:
        raise ValueError(
            "NAMO supports only the canonical car 1x d5 runtime config: "
            "diff_drive, resolutions 0.01/0.005, robot_size [0.035, 0.035], "
            "five push depths, and the 1x_car_d5 primitive table"
        )
    return path


def require_canonical_primitive_profile(
    primitive_prefix: str,
    max_push_steps: int | None,
) -> int:
    """Reject action tables that do not match the car 1x d5 policy head."""
    if primitive_prefix != CANONICAL_PRIMITIVE_PREFIX:
        raise ValueError(
            "NAMO supports only the car 1x d5 primitive table: "
            f"primitive_prefix must be {CANONICAL_PRIMITIVE_PREFIX!r}, "
            f"got {primitive_prefix!r}"
        )
    if max_push_steps is not None and max_push_steps != CANONICAL_NUM_DEPTHS:
        raise ValueError(
            "NAMO supports exactly five push depths: "
            f"max_push_steps must be {CANONICAL_NUM_DEPTHS}, got {max_push_steps}"
        )
    return CANONICAL_NUM_DEPTHS
