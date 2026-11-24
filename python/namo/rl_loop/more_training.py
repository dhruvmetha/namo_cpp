"""What MORE's two trainers share: the room split, the schedules, and the invalid-push labels.

Reference: lifelong_trainer.py and dataset.py at arc-l/more revision 70402c0, and the paper's
section IV-C. Two choices are still open with the user and are configurable here; their defaults
are marked PROVISIONAL and keep the behaviour the pilot was trained with.

  invalid pushes   "ours" (PROVISIONAL default): an unreachable (edge, depth) cell is its own
                   sample, label -1, weight 1, at most 30 per state.
                   "more": MORE's dense negatives, dataset.py:423-430. Inside every sample, every
                   unreachable cell of the same state is painted label 0 at weight 0.001, MORE's
                   collision weight. MORE's 0.0001 "pushing thin air" class has no counterpart:
                   every contact in our library touches the object.
  schedule         "ours" (PROVISIONAL default): the pilot's recipe.
                   "more": lifelong_trainer.py exactly, then the paper's fine-tune stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np
import torch

LOSS_BETA = 0.8                 # lifelong_trainer.py:42, :242 SmoothL1Loss(beta)
VISIT_CAP = 50                  # dataset.py:191
MORE_INVALID_WEIGHT = 0.001     # dataset.py:426, collision
INVALID_MODES = ("ours", "more")
NUM_EDGES, NUM_DEPTHS = 60, 5


def heldout_rooms(xml: Iterable[str], frac: float = 0.1, seed: int = 0) -> set:
    """The rooms verify_more_net.py scores; training excludes exactly these."""
    rooms = np.unique(np.asarray(list(xml)))
    if frac <= 0 or len(rooms) == 0:
        return set()
    rng = np.random.default_rng(seed)
    return set(rng.choice(rooms, max(1, int(len(rooms) * frac)), replace=False))


def decode(strings) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in strings])


def push_angle(board: np.ndarray, edge: int) -> float:
    """Image angle of edge `edge`'s push, from its contact toward its mate's (namo_push_controller.cpp:201-210, :258-260).

    board is the state's (60, 2) contact_px in (column, row); the mate of edge e is e ^ 1.
    Rotating the image by this angle (rotate_batch below) makes the push point right, as
    MORE's dataset does for every sample (dataset.py:69-87).
    """
    d = board[edge ^ 1] - board[edge]
    return math.atan2(float(d[1]), float(d[0]))


def rotate_batch(ctx: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate each image by its own angle; a point moves by R(-angle) (see rotate_pixel)."""
    b = ctx.shape[0]
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(b, 2, 3, device=ctx.device, dtype=ctx.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = torch.nn.functional.affine_grid(theta, ctx.shape, align_corners=True)
    return torch.nn.functional.grid_sample(ctx, grid, mode="bilinear", align_corners=True)


def rotate_pixel(px: torch.Tensor, angle: torch.Tensor, size: int) -> torch.Tensor:
    """Where (column, row) pixels land after rotate_batch. px (..., 2), angle broadcastable to px[..., 0]."""
    centre = (size - 1) / 2.0
    x, y = px[..., 0] - centre, px[..., 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    # affine_grid samples input = R(angle) @ output, so a point moves by R(-angle).
    return torch.stack([x * cos + y * sin + centre, -x * sin + y * cos + centre], dim=-1)


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    batch: int
    lr: float
    t0: int
    t_mult: int
    eta_min: float
    warmup_iters: int       # 0 = none
    clip: float             # 0 = none
    more_stepping: bool     # lifelong_trainer.py:373-378 instead of fractional epochs


# lifelong_trainer.py:29-35 defaults; momentum 0.9 and weight decay 2e-5 at :243-248 for every stage.
MORE_STAGE1 = Stage("more", epochs=51, batch=8, lr=1e-4, t0=50, t_mult=2, eta_min=1e-8,
                    warmup_iters=0, clip=0.0, more_stepping=True)
# Paper IV-C fine-tune stage; the same numbers sit commented out at lifelong_trainer.py:36-41
# (epochs 21 there, 20 in the paper; the code is followed). MORE starts it as a new run that loads
# the weights only (lifelong_trainer.py:261-263), so the optimiser and schedule restart.
MORE_STAGE2 = Stage("more_ft", epochs=21, batch=28, lr=1e-5, t0=20, t_mult=2, eta_min=1e-8,
                    warmup_iters=0, clip=0.0, more_stepping=True)
# PROVISIONAL default: the pilot's recipe (train_more.sbatch passed --batch 32 --epochs 2).
OURS_STAGE = Stage("ours", epochs=4, batch=16, lr=1e-4, t0=2, t_mult=1, eta_min=1e-6,
                   warmup_iters=1000, clip=10.0, more_stepping=False)
SCHEDULES = {"ours": (OURS_STAGE,), "more": (MORE_STAGE1, MORE_STAGE2)}


def stages(schedule: str, epochs: Optional[int] = None, batch: Optional[int] = None,
           lr: Optional[float] = None):
    """The stage list, with explicit command-line overrides applied to the FIRST stage only."""
    out = list(SCHEDULES[schedule])
    over = {k: v for k, v in (("epochs", epochs), ("batch", batch), ("lr", lr)) if v is not None}
    if over:
        out[0] = replace(out[0], **over)
    return out


def make_optimizer(params, stage: Stage):
    """lifelong_trainer.py:243-258: SGD(momentum 0.9, wd 2e-5) + CosineAnnealingWarmRestarts."""
    opt = torch.optim.SGD([p for p in params if p.requires_grad], lr=stage.lr, momentum=0.9,
                          weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=stage.t0, T_mult=stage.t_mult, eta_min=stage.eta_min)
    return opt, sched


class Stepper:
    """When the LR changes.

    more_stepping: lifelong_trainer.py:373-378. In epoch 0 the scheduler steps after EVERY
    iteration (so T_0=50 means 50 iterations there); after each later epoch it steps once with
    the epoch index. Their warmup branch (:278 `if epoch < 0`) never runs, so there is none.
    Otherwise: an optional linear warmup, then the cosine stepped by fractional epoch.
    """

    def __init__(self, opt, sched, stage: Stage, iters_per_epoch: int):
        self.opt, self.sched, self.stage, self.n = opt, sched, stage, max(iters_per_epoch, 1)

    def after_iteration(self, epoch: int, step: int):
        st = self.stage
        if st.more_stepping:
            if epoch == 0:
                self.sched.step()
            return
        if epoch == 0 and step < st.warmup_iters:
            a = step / st.warmup_iters                    # torch_utils.warmup_lr_scheduler shape
            for g in self.opt.param_groups:
                g["lr"] = st.lr * (0.001 * (1 - a) + a)
        else:
            self.sched.step(epoch + step / self.n)

    def after_epoch(self, epoch: int):
        if self.stage.more_stepping and epoch != 0:
            self.sched.step(epoch)


def invalid_cells(reach: np.ndarray, weight: np.ndarray, mode: str, cap: int, rng) -> np.ndarray:
    """Boolean (states, 60, 5): unreachable cells that become their OWN samples ("ours" only)."""
    if mode != "ours":
        return np.zeros_like(reach, dtype=bool)
    un = reach == 0
    for s_i in range(un.shape[0]):
        cells = np.argwhere(un[s_i])
        if cap and len(cells) > cap:
            drop = rng.choice(len(cells), len(cells) - cap, replace=False)
            un[s_i][tuple(cells[drop].T)] = False
    return un
