"""MORE's trained network as a search prior, closing its System 2 to System 1 loop.

This is the piece that makes MORE a method rather than parts. Plain UCT collects, the
net learns from those returns, and then the net guides the search that replaces UCT.
Without it the guided search would still be reading our ranker, which is the mistake
that made an early keyhole run a search ablation wearing MORE's name.

`score_state` matches `LiveScorer.score_state` exactly, so `rank_first_pushes_h2` calls
it without knowing the difference and the two arms differ only by the model behind the
same interface.

Scoring follows `mcts_utils.py:911` `_sampled_prediction_precise`: rotate the scene by
EACH candidate's own push angle so the push points right, batch the rotated copies through
one forward pass, and read `np.max` over a 7x7 window at that action's contact pixel in the
rotated frame. The push angle is the direction from edge e toward its mate e ^ 1
(more_net.push_angles; namo_push_controller.cpp:201-210, :258-260), and contact_px is
(column, row). Training rotates with the same functions (more_training.py).

The returned grid is in MORE's reward units, 0 to 1.2, because the network was trained to
regress max(child.q). MORE's search clamps it to [0, 1.2] itself (mcts_network/nodes.py:117).
"""

from __future__ import annotations


import numpy as np
import torch

from namo.rl_loop.more_net import MorePushNet, push_angles, read_contact_values
from namo.rl_loop.more_training import rotate_batch, rotate_pixel

NUM_EDGES, NUM_DEPTHS = 60, 5
RENDER_SIZE = 224           # PushNet's input size; each scorer passes its own size to its renderer
ECA_RENDER_SIZE = 64        # EdgeCrossAttn's input size
PATCH = 3                    # mcts_utils.py:929 reads a 7x7 window, so a half-width of 3


class MoreScorer:
    """MORE's PushNet behind the scorer interface the search already speaks."""

    def __init__(self, ckpt, render_config, device=None, batch=60):
        from namo.rl_loop.build_train_h5 import _Renderer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(ckpt, map_location="cpu")
        weights = state.get("model", state)
        in_ch = weights["body.conv1.weight"].shape[1]
        self.net = MorePushNet(in_channels=in_ch, num_depths=NUM_DEPTHS)
        self.net.load_state_dict(weights)
        self.net.to(self.device).eval()
        self.renderer = _Renderer(render_config, out=RENDER_SIZE)
        self.batch = batch

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
        angles = push_angles(px)                     # each edge's own push direction, per state
        out = torch.zeros(NUM_EDGES, NUM_DEPTHS, device=self.device)
        for start in range(0, NUM_EDGES, self.batch):
            stop = min(start + self.batch, NUM_EDGES)
            idx = torch.arange(start, stop, device=self.device)
            ang = angles[idx]
            stack = base.unsqueeze(0).expand(len(idx), -1, -1, -1)
            rot = rotate_batch(stack, ang)
            pred = self.net(rot)                                   # (b, depths, H, W)
            rpx = rotate_pixel(px[idx], ang, base.shape[-1])
            vals = read_contact_values(pred, rpx.unsqueeze(1), patch=PATCH)   # (b, 1, depths)
            out[start:stop] = vals[:, 0, :]
        return out.cpu().numpy()




class MoreEcaScorer:
    """Our EdgeCrossAttn, trained on MORE's labels and loss (train_more_eca.py), as MORE's prior.

    Same interface as LiveScorer and MoreScorer. The network was trained to regress MORE's
    push_result-unit labels (0 to 1.2), so its raw output IS the prior MORE's search reads
    (mcts_network/push.py:140-160, clamped to [0, 1.2] at mcts_network/nodes.py:117). LiveScorer
    would put a sigmoid on a scalar head (live_scorer.py:206); this does not. Rendering is
    LiveScorer's own, at its 64 px.
    """

    def __init__(self, ckpt, render_config=None, device=None):
        import live_scorer
        kw = {"ckpt": ckpt, "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
              "out": ECA_RENDER_SIZE}
        if render_config:
            kw["render_config"] = render_config
        self._live = live_scorer.LiveScorer(**kw)
        self.device = self._live.device

    def warmup(self, repeats=1):
        ctx = np.zeros((5, ECA_RENDER_SIZE, ECA_RENDER_SIZE), np.float32)
        cpx = np.full((NUM_EDGES, 2), ECA_RENDER_SIZE / 2, np.float32)
        for _ in range(max(repeats, 1)):
            self.score_ctx(ctx, cpx)

    @torch.no_grad()
    def score_state(self, env, target_object, robot_goal, xml_file, region_samples=None,
                    h=1, raw=False):
        """(60,5) predicted MORE returns; `h` and `raw` are accepted and ignored (see MoreScorer)."""
        ctx, _ = self._live.render_ctx(env, target_object, robot_goal, xml_file, region_samples)
        cpx = self._live.contact_px_live(env, target_object)
        return self.score_ctx(ctx, cpx)

    @torch.no_grad()
    def score_ctx(self, ctx, contact_px):
        ct = torch.as_tensor(np.asarray(ctx)[None], dtype=torch.float32, device=self.device)
        cp = torch.as_tensor(np.asarray(contact_px)[None], dtype=torch.float32, device=self.device)
        return self._live.model(ct, cp)[0].cpu().numpy()
