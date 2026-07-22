"""Exact pre-simulation motion features for the 60x5 car push grid."""
import math
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
SHAPES = ("square", "wide", "tall")
NUM_EDGES = 60
NUM_DEPTHS = 5
LEGACY_MOTION_DIM = 3
FINAL_POSE_DIM = 4


@lru_cache(maxsize=1)
def primitive_motion_tables() -> torch.Tensor:
    """Return object-local ``(dx,dy,dtheta)`` tables with shape ``(3,60,5,3)``."""
    tables = np.empty((len(SHAPES), NUM_EDGES, NUM_DEPTHS, 3), dtype=np.float32)
    for shape_i, shape in enumerate(SHAPES):
        path = REPO / "data" / f"1x_car_d5_motion_primitives_15_{shape}.dat"
        with path.open("rb") as fh:
            count = struct.unpack("I", fh.read(4))[0]
            for _ in range(count):
                dx, dy, dtheta, edge, push_steps = struct.unpack("fffBB", fh.read(14))
                tables[shape_i, edge, push_steps - 1] = (dx, dy, dtheta)
    return torch.from_numpy(tables)


def action_motion_from_contact_px(contact_px: torch.Tensor, crop_m: float = 0.5,
                                  feature_dim: int = FINAL_POSE_DIM) -> torch.Tensor:
    """Build the exact active primitive pose feature from stored contact geometry.

    Contact indices 0→28 span the object's local x axis and 30→58 span its local y axis. Their
    lengths therefore recover the same square/wide/tall choice used by ``PrimitiveGoalStrategy``;
    the x-axis direction recovers object yaw. The corrected feature describes the primitive's
    nominal final pose in the world-aligned object-centered crop:
    ``(2*world_dx/crop_m, 2*world_dy/crop_m, sin(theta+dtheta), cos(theta+dtheta))``. Thus the crop
    center is (0,0), either crop edge is +/-1, and orientation uses the same axes as the image.

    ``feature_dim=3`` preserves the original experiment's
    ``(world_dx/crop_m, world_dy/crop_m, dtheta/pi)`` encoding so its checkpoints remain evaluable.
    """
    unbatched = contact_px.ndim == 2
    cp = contact_px.unsqueeze(0) if unbatched else contact_px
    x_axis = cp[:, 28] - cp[:, 0]
    y_axis = cp[:, 58] - cp[:, 30]
    x_len = torch.linalg.vector_norm(x_axis, dim=-1)
    y_len = torch.linalg.vector_norm(y_axis, dim=-1)
    ratio = torch.maximum(x_len, y_len) / torch.minimum(x_len, y_len)
    shape = torch.where(ratio < 1.05, 0, torch.where(x_len > y_len, 1, 2)).long()

    local = primitive_motion_tables().to(device=cp.device, dtype=cp.dtype)[shape]
    c = x_axis[:, 0] / x_len
    s = x_axis[:, 1] / x_len
    dx, dy = local[..., 0], local[..., 1]
    world_dx = c[:, None, None] * dx - s[:, None, None] * dy
    world_dy = s[:, None, None] * dx + c[:, None, None] * dy
    if feature_dim == LEGACY_MOTION_DIM:
        out = torch.stack((world_dx / crop_m, world_dy / crop_m, local[..., 2] / math.pi), dim=-1)
    else:
        theta = torch.atan2(s, c)
        final_theta = theta[:, None, None] + local[..., 2]
        out = torch.stack((2.0 * world_dx / crop_m, 2.0 * world_dy / crop_m,
                           torch.sin(final_theta), torch.cos(final_theta)), dim=-1)
    return out[0] if unbatched else out
