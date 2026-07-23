"""Exact pre-simulation motion features for the 60x5 car push grid."""
import math
import os
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
CROP_RELATIVE_MOTION_DIM = 3
FINAL_POSE_DIM = 4
NO_MOTION_ENCODING = "none"
LEGACY_MOTION_ENCODING = "legacy"
CROP_RELATIVE_MOTION_ENCODING = "crop_relative"
FINAL_POSE_ENCODING = "final_pose"


def configured_action_motion_encoding() -> str:
    """Return the explicitly configured training encoding, or ``none`` for a baseline."""
    if os.environ.get("NAMO_ACTION_MOTION", "0") != "1":
        return NO_MOTION_ENCODING
    return os.environ.get("NAMO_ACTION_MOTION_ENCODING", CROP_RELATIVE_MOTION_ENCODING)


def action_motion_feature_dim(encoding: str) -> int:
    """Return the feature width for one named action-motion encoding."""
    return {
        NO_MOTION_ENCODING: 0,
        LEGACY_MOTION_ENCODING: LEGACY_MOTION_DIM,
        CROP_RELATIVE_MOTION_ENCODING: CROP_RELATIVE_MOTION_DIM,
        FINAL_POSE_ENCODING: FINAL_POSE_DIM,
    }[encoding]


def checkpoint_action_motion_encoding(checkpoint: dict, feature_dim: int) -> str:
    """Resolve semantics while keeping untagged historical checkpoints loadable."""
    tagged = checkpoint.get("action_motion_encoding")
    if tagged is not None:
        if action_motion_feature_dim(tagged) != feature_dim:
            raise ValueError(f"checkpoint action-motion tag {tagged!r} disagrees with dim {feature_dim}")
        return tagged
    if feature_dim == 0:
        return NO_MOTION_ENCODING
    if feature_dim == LEGACY_MOTION_DIM:
        return LEGACY_MOTION_ENCODING
    if feature_dim == FINAL_POSE_DIM:
        return FINAL_POSE_ENCODING
    raise ValueError(f"cannot infer action-motion encoding from feature dim {feature_dim}")


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
                                  encoding: str | None = None,
                                  feature_dim: int | None = None) -> torch.Tensor:
    """Build the exact active primitive pose feature from stored contact geometry.

    Contact indices 0→28 span the object's local x axis and 30→58 span its local y axis. Their
    lengths therefore recover the same square/wide/tall choice used by ``PrimitiveGoalStrategy``;
    the x-axis direction recovers object yaw. The corrected feature describes the primitive's
    nominal final pose in the world-aligned object-centered crop:
    ``(2*world_dx/crop_m, 2*world_dy/crop_m, sin(theta+dtheta), cos(theta+dtheta))``. Thus the crop
    center is (0,0), either crop edge is +/-1, and orientation uses the same axes as the image.

    ``encoding="crop_relative"`` is the corrected relative-motion feature:
    ``(2*world_dx/crop_m, 2*world_dy/crop_m, dtheta/pi)``. Translation is in the image crop's
    normalized coordinates, while rotation is the action's relative rotation rather than absolute yaw.

    ``feature_dim=3`` without a named encoding preserves the original experiment's
    ``(world_dx/crop_m, world_dy/crop_m, dtheta/pi)`` encoding so its checkpoints remain evaluable.
    """
    if encoding is None:
        encoding = (LEGACY_MOTION_ENCODING if feature_dim == LEGACY_MOTION_DIM
                    else FINAL_POSE_ENCODING)
    expected_dim = action_motion_feature_dim(encoding)
    if feature_dim is not None and feature_dim != expected_dim:
        raise ValueError(f"encoding {encoding!r} has dim {expected_dim}, not {feature_dim}")
    if encoding == NO_MOTION_ENCODING:
        raise ValueError("the no-motion baseline has no action-motion tensor")
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
    if encoding == LEGACY_MOTION_ENCODING:
        out = torch.stack((world_dx / crop_m, world_dy / crop_m, local[..., 2] / math.pi), dim=-1)
    elif encoding == CROP_RELATIVE_MOTION_ENCODING:
        out = torch.stack((2.0 * world_dx / crop_m, 2.0 * world_dy / crop_m,
                           local[..., 2] / math.pi), dim=-1)
    else:
        theta = torch.atan2(s, c)
        final_theta = theta[:, None, None] + local[..., 2]
        out = torch.stack((2.0 * world_dx / crop_m, 2.0 * world_dy / crop_m,
                           torch.sin(final_theta), torch.cos(final_theta)), dim=-1)
    return out[0] if unbatched else out
