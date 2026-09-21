"""MORE's trained network as a search prior, closing its System 2 to System 1 loop.

This is the piece that makes MORE a method rather than parts. Plain UCT collects, the
net learns from those returns, and then the net guides the search that replaces UCT.
Without it the guided search would still be reading our ranker, which is the mistake
that made an early keyhole run a search ablation wearing MORE's name.

`score_state` matches `LiveScorer.score_state` exactly, so `rank_first_pushes_h2` calls
it without knowing the difference and the two arms differ only by the model behind the
same interface.

Scoring follows `mcts_utils.py:911` `_sampled_prediction_precise`: rotate the scene by
EACH candidate's own push angle, batch the rotated copies through one forward pass,
un-rotate, and read `np.max` over a 7x7 window at that action's contact pixel. The
16-rotation loop in `models.py` is their other code path and would bin our 60 contact
directions into 22.5 degree buckets; this one handles them exactly.

The returned grid is already in MORE's reward units, 0 to 1.2, because the network was
trained to regress push_result. So the search wants `prior_scale="raw"`, not the
`minmax` rescale that exists only to squeeze our own ranker's out-of-range scores into
the [0, 1.2] clamp `best_child` applies.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from namo.rl_loop.more_net import MorePushNet, read_contact_values

NUM_EDGES, NUM_DEPTHS = 60, 5
RENDER_SIZE = 224
PATCH = 3                    # mcts_utils.py:929 reads a 7x7 window, so a half-width of 3


class MoreScorer:
    """MORE's PushNet behind the scorer interface the search already speaks."""

    def __init__(self, ckpt, render_config, device=None, batch=60):
        import live_scorer
        live_scorer.OUT = RENDER_SIZE            # must be set before the renderer reads it
        from namo.rl_loop.build_train_h5 import _Renderer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(ckpt, map_location="cpu")
        weights = state.get("model", state)
        in_ch = weights["body.conv1.weight"].shape[1]
        self.net = MorePushNet(in_channels=in_ch, num_depths=NUM_DEPTHS)
        self.net.load_state_dict(weights)
        self.net.to(self.device).eval()
        self.renderer = _Renderer(render_config)
        self.batch = batch
        # The angle of the push through each contact edge, matching the collector's
        # convention: 60 edges evenly around the object.
        angles = 2.0 * math.pi * np.arange(NUM_EDGES) / NUM_EDGES
        self._angles = torch.tensor(angles, dtype=torch.float32, device=self.device)

    def warmup(self, repeats=1):
        """Match LiveScorer's interface so the eval harness can warm the model."""
        x = torch.zeros(1, self.net.in_channels, RENDER_SIZE, RENDER_SIZE, device=self.device)
        with torch.no_grad():
            for _ in range(max(repeats, 1)):
                self.net(x)

    @torch.no_grad()
    def score_state(self, env, target_object, robot_goal, xml_file, region_samples=None,
                    h=1, raw=False):
        """(60,5) predicted push values for `target_object` at the live state.

        `h` and `raw` are accepted and ignored. MORE's network is not conditioned on a
        remaining budget and its output is already the raw predicted return, so there is
        nothing for either to switch. Keeping them in the signature is what lets this
        stand in for LiveScorer untouched.
        """
        ctx, _meta = self.renderer.render_ctx(env, target_object, robot_goal, xml_file,
                                              region_samples)
        cpx = self.renderer.contact_px_live(env, target_object)
        return self.score_ctx(ctx, cpx)

    @torch.no_grad()
    def score_ctx(self, ctx, contact_px):
        """One rotated copy per contact edge, batched, then read the value at its pixel."""
        base = torch.as_tensor(np.asarray(ctx), dtype=torch.float32, device=self.device)
        px = torch.as_tensor(np.asarray(contact_px), dtype=torch.float32, device=self.device)
        out = torch.zeros(NUM_EDGES, NUM_DEPTHS, device=self.device)
        for start in range(0, NUM_EDGES, self.batch):
            stop = min(start + self.batch, NUM_EDGES)
            idx = torch.arange(start, stop, device=self.device)
            ang = self._angles[idx]
            stack = base.unsqueeze(0).expand(len(idx), -1, -1, -1)
            rot = _rotate(stack, ang)
            pred = self.net(rot)                                   # (b, depths, H, W)
            rpx = _rotate_pixel(px[idx], ang, base.shape[-1])
            vals = read_contact_values(pred, rpx.unsqueeze(1), patch=PATCH)   # (b, 1, depths)
            out[start:stop] = vals[:, 0, :]
        return out.cpu().numpy()


def _rotate(x, angle):
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=True)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=True)


def _rotate_pixel(px, angle, size):
    centre = (size - 1) / 2.0
    y, x = px[:, 0] - centre, px[:, 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([y * cos - x * sin + centre, y * sin + x * cos + centre], dim=1)
