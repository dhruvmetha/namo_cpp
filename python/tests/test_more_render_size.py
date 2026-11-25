"""MORE's two priors each render at their own size, whichever one a process builds first.

live_scorer kept the render size in the module global OUT and read it on every render.
MoreScorer set it to 224 and MoreEcaScorer to 64 when built, so the scorer built last chose
the size for both. PushNet then rendered 64 px images with no error, and EdgeCrossAttn raised
a 3136-vs-256 token error (EXP-2026-11-24-more-faithful, first full eval). These tests build
both scorers in one process, in both orders, and render through a stand-in visualizer and env.
"""
import numpy as np
import pytest
import torch

import live_scorer
import train_more_eca
from namo.rl_loop import more_scorer
from namo.rl_loop.build_train_h5 import _Renderer
from namo.rl_loop.more_net import MorePushNet

GOAL = (0.5, 0.5, 0.0)
POSE = [0.1, -0.05, 0.4]
HALF = (0.05, 0.03)


class _Env:
    def get_observation(self):
        return {"robot_pose": [0.0, 0.0, 0.0], "obj_pose": list(POSE)}

    def get_object_info(self):
        return {"obj": {"size_x": HALF[0], "size_y": HALF[1]}}

    def get_reachable_objects(self):
        return ["obj"]


class _Viz:
    """The real visualizer hands back 224 px tight crops; LiveScorer resizes them."""

    def generate_all_masks_highres(self, ep, tight_crop_size_meters, fast_scorer):
        rng = np.random.default_rng(0)
        return {"local_tight": {k: rng.random((224, 224)).astype(np.float32) for k in live_scorer.TIGHT},
                "local_tight_metadata": {}}


class _SpyNet(torch.nn.Module):
    """Stands in for PushNet's ResNet-34 so the test records the image size it is fed."""

    in_channels = 5

    def __init__(self):
        super().__init__()
        self.sizes = []

    def forward(self, x):
        self.sizes.append(tuple(x.shape[-2:]))
        return torch.zeros(x.shape[0], 5, *x.shape[-2:])


def _pushnet(tmp_path):
    ck = tmp_path / "pushnet.pt"
    torch.save({"model": MorePushNet(in_channels=5, num_depths=5).state_dict()}, ck)
    sc = more_scorer.MoreScorer(str(ck), live_scorer.SKILL15_CFG, device="cpu")
    sc.renderer.viz = _Viz()
    sc.net = _SpyNet()
    return sc, sc.renderer


def _eca(tmp_path):
    ck = tmp_path / "eca.ckpt"
    train_more_eca.save_ckpt(train_more_eca.make_net(), ck, {})
    sc = more_scorer.MoreEcaScorer(str(ck), device="cpu")
    sc._live.viz = _Viz()
    return sc, sc._live


BUILD = {"pushnet": _pushnet, "eca": _eca}
SIZE = {"pushnet": 224, "eca": 64}


def _contact_px(size):
    return np.array([live_scorer.contact_px_fn(e, HALF[0], HALF[1], POSE[2], live_scorer.CROP_M, size)
                     for e in range(60)], dtype=np.float32)


@pytest.mark.parametrize("order", [("pushnet", "eca"), ("eca", "pushnet")], ids=["pushnet_first", "eca_first"])
def test_each_more_prior_renders_at_its_own_size_in_one_process(tmp_path, order):
    built = {name: BUILD[name](tmp_path) for name in order}
    env = _Env()
    for name, (scorer, renderer) in built.items():
        size = SIZE[name]
        ctx, _ = renderer.render_ctx(env, "obj", GOAL, "scene.xml")
        assert ctx.shape == (5, size, size), name
        np.testing.assert_array_equal(renderer.contact_px_live(env, "obj"), _contact_px(size))
        assert scorer.score_state(env, "obj", GOAL, "scene.xml").shape == (60, 5), name
    assert set(built["pushnet"][0].net.sizes) == {(224, 224)}


def test_our_live_scorer_renderer_stays_64_after_a_pushnet_scorer_is_built(tmp_path):
    _pushnet(tmp_path)
    ours = _Renderer(live_scorer.SKILL15_CFG)
    ours.viz = _Viz()
    ctx, _ = ours.render_ctx(_Env(), "obj", GOAL, "scene.xml")
    assert ctx.shape == (5, 64, 64)
    np.testing.assert_array_equal(ours.contact_px_live(_Env(), "obj"), _contact_px(64))
