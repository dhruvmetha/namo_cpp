import math

import pytest
import torch

from namo.rl_loop.action_motion import action_motion_from_contact_px, primitive_motion_tables


def _contact_px(hw, hd, theta, crop_m=0.5, size=64):
    points = []
    for edge in range(60):
        if edge < 30:
            j = edge // 2
            lx = -hw + 2 * hw * j / 14
            ly = hd if edge % 2 == 0 else -hd
        else:
            j = (edge - 30) // 2
            lx = hw if edge % 2 == 0 else -hw
            ly = -hd + 2 * hd * j / 14
        c, s = math.cos(theta), math.sin(theta)
        points.append((size / 2 + (c * lx - s * ly) / (crop_m / size),
                       size / 2 + (s * lx + c * ly) / (crop_m / size)))
    return torch.tensor(points, dtype=torch.float32)


@pytest.mark.parametrize("shape_i,hw,hd", [(0, 0.05, 0.05), (1, 0.08, 0.04), (2, 0.04, 0.08)])
def test_action_motion_recovers_shape_family_and_rotation(shape_i, hw, hd):
    theta = 0.37
    got = action_motion_from_contact_px(_contact_px(hw, hd, theta))
    local = primitive_motion_tables()[shape_i]
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.stack((
        (c * local[..., 0] - s * local[..., 1]) / 0.5,
        (s * local[..., 0] + c * local[..., 1]) / 0.5,
        local[..., 2] / math.pi,
    ), dim=-1)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
    assert got.shape == (60, 5, 3)
    assert not torch.equal(got[:, 0], got[:, 4])
