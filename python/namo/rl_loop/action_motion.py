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


def action_motion_from_contact_px(contact_px: torch.Tensor, crop_m: float = 0.5) -> torch.Tensor:
    """Build the exact active primitive motion in crop coordinates from stored contact geometry.

    Contact indices 0→28 span the object's local x axis and 30→58 span its local y axis. Their
    lengths therefore recover the same square/wide/tall choice used by ``PrimitiveGoalStrategy``;
    the x-axis direction recovers object yaw. The returned dimensionless feature is
    ``(world_dx/crop_m, world_dy/crop_m, dtheta/pi)`` for every complete push.
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
    out = torch.stack((world_dx / crop_m, world_dy / crop_m, local[..., 2] / math.pi), dim=-1)
    return out[0] if unbatched else out
