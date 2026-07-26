import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))

from add_contact_px import contact_offsets_world, contact_px  # noqa: E402


def test_shape_and_dtype():
    off = contact_offsets_world(0.15, 0.10, 0.0)
    assert off.shape == (60, 2)
    assert off.dtype == np.float32


def test_zero_rotation_offsets_lie_on_the_object_rectangle():
    hw, hd = 0.15, 0.10
    off = contact_offsets_world(hw, hd, 0.0)
    on_edge = (np.isclose(np.abs(off[:, 0]), hw, atol=1e-6) |
               np.isclose(np.abs(off[:, 1]), hd, atol=1e-6))
    assert on_edge.all()


def test_rotation_is_a_rigid_transform():
    hw, hd, th = 0.15, 0.10, 0.7
    a = contact_offsets_world(hw, hd, 0.0)
    b = contact_offsets_world(hw, hd, th)
    c, s = np.cos(th), np.sin(th)
    expected = np.stack([a[:, 0] * c - a[:, 1] * s, a[:, 0] * s + a[:, 1] * c], axis=1)
    assert np.allclose(b, expected, atol=1e-6)


@pytest.mark.parametrize("edge", [0, 7, 15, 31, 44, 59])
def test_pixel_path_matches_the_factored_offsets(edge):
    hw, hd, th, crop_m, S = 0.15, 0.10, 0.7, 1.0, 64
    px = contact_px(edge, hw, hd, th, crop_m, S)
    wx, wy = contact_offsets_world(hw, hd, th)[edge]
    res = crop_m / S
    assert np.allclose(px, (S / 2.0 + wx / res, S / 2.0 + wy / res), atol=1e-4)
