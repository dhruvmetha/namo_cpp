import pytest

from namo.runtime_profile import (
    CANONICAL_CONFIG,
    CANONICAL_NUM_DEPTHS,
    CANONICAL_PRIMITIVE_PREFIX,
    require_canonical_primitive_profile,
    require_canonical_runtime_config,
)


def test_car_1x_d5_profile_is_accepted():
    require_canonical_runtime_config(CANONICAL_CONFIG)
    assert require_canonical_primitive_profile(
        CANONICAL_PRIMITIVE_PREFIX, CANONICAL_NUM_DEPTHS
    ) == CANONICAL_NUM_DEPTHS


@pytest.mark.parametrize("prefix", ["", "car_", "1x_car_"])
def test_non_d5_primitive_tables_are_rejected(prefix):
    with pytest.raises(ValueError, match="car 1x d5"):
        require_canonical_primitive_profile(prefix, CANONICAL_NUM_DEPTHS)


@pytest.mark.parametrize("depths", [1, 4, 6, 10])
def test_any_depth_count_other_than_five_is_rejected(depths):
    with pytest.raises(ValueError, match="exactly five"):
        require_canonical_primitive_profile(CANONICAL_PRIMITIVE_PREFIX, depths)


def test_point_robot_config_is_rejected():
    with pytest.raises(ValueError, match="car 1x d5"):
        require_canonical_runtime_config("config/namo_config_complete_skill15.yaml")
