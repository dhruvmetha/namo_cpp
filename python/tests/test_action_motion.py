import math

import pytest
import torch

from namo.rl_loop.action_motion import (
    CROP_RELATIVE_MOTION_DIM,
    CROP_RELATIVE_MOTION_ENCODING,
    FINAL_POSE_DIM,
    FINAL_POSE_ENCODING,
    LEGACY_MOTION_ENCODING,
    LEGACY_MOTION_DIM,
    action_motion_from_contact_px,
    checkpoint_action_motion_encoding,
    configured_action_motion_encoding,
    primitive_motion_tables,
)
from namo.rl_loop.sage_ext._sage import ClassifierModule, EdgeCrossAttn


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
        2.0 * (c * local[..., 0] - s * local[..., 1]) / 0.5,
        2.0 * (s * local[..., 0] + c * local[..., 1]) / 0.5,
        torch.sin(theta + local[..., 2]),
        torch.cos(theta + local[..., 2]),
    ), dim=-1)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
    assert got.shape == (60, 5, FINAL_POSE_DIM)
    assert not torch.equal(got[:, 0], got[:, 4])


def test_corrected_translation_matches_normalized_image_coordinates():
    crop_m = 0.5
    theta = -0.41
    got = action_motion_from_contact_px(_contact_px(0.08, 0.04, theta, crop_m=crop_m), crop_m=crop_m)
    local = primitive_motion_tables()[1]
    c, s = math.cos(theta), math.sin(theta)
    world_dx = c * local[..., 0] - s * local[..., 1]
    world_dy = s * local[..., 0] + c * local[..., 1]
    final_px = 32.0 + world_dx / (crop_m / 64.0)
    final_py = 32.0 + world_dy / (crop_m / 64.0)
    expected_u = 2.0 * final_px / 64.0 - 1.0
    expected_v = 2.0 * final_py / 64.0 - 1.0
    torch.testing.assert_close(got[..., 0], expected_u, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(got[..., 1], expected_v, atol=1e-6, rtol=1e-5)


def test_crop_relative_motion_uses_image_scale_and_relative_rotation():
    theta = 0.37
    got = action_motion_from_contact_px(
        _contact_px(0.08, 0.04, theta), encoding=CROP_RELATIVE_MOTION_ENCODING)
    local = primitive_motion_tables()[1]
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.stack((
        2.0 * (c * local[..., 0] - s * local[..., 1]) / 0.5,
        2.0 * (s * local[..., 0] + c * local[..., 1]) / 0.5,
        local[..., 2] / math.pi,
    ), dim=-1)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
    assert got.shape == (60, 5, CROP_RELATIVE_MOTION_DIM)


def test_final_orientation_is_continuous_across_angle_wrap():
    theta = math.pi - 0.01
    a = action_motion_from_contact_px(_contact_px(0.08, 0.04, theta))
    b = action_motion_from_contact_px(_contact_px(0.08, 0.04, theta - 2.0 * math.pi))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_legacy_three_vector_encoding_remains_available():
    theta = 0.37
    got = action_motion_from_contact_px(
        _contact_px(0.08, 0.04, theta), feature_dim=LEGACY_MOTION_DIM)
    local = primitive_motion_tables()[1]
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.stack((
        (c * local[..., 0] - s * local[..., 1]) / 0.5,
        (s * local[..., 0] + c * local[..., 1]) / 0.5,
        local[..., 2] / math.pi,
    ), dim=-1)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
    assert got.shape == (60, 5, LEGACY_MOTION_DIM)


def test_checkpoint_tag_disambiguates_same_width_encodings():
    assert checkpoint_action_motion_encoding({}, 3) == LEGACY_MOTION_ENCODING
    assert checkpoint_action_motion_encoding(
        {"action_motion_encoding": CROP_RELATIVE_MOTION_ENCODING}, 3
    ) == CROP_RELATIVE_MOTION_ENCODING
    assert checkpoint_action_motion_encoding({}, 4) == FINAL_POSE_ENCODING


def test_motion_enabled_defaults_to_crop_relative(monkeypatch):
    monkeypatch.setenv("NAMO_ACTION_MOTION", "1")
    monkeypatch.delenv("NAMO_ACTION_MOTION_ENCODING", raising=False)
    assert configured_action_motion_encoding() == CROP_RELATIVE_MOTION_ENCODING


def _small_action_token_model(depth_self_attn):
    return EdgeCrossAttn(
        img_size=16, patch=4, in_channels=5, dim=32, scene_depth=1, edge_depth=1,
        heads=1, num_depths=5, num_edges=60, pos_fourier=True, fourier_L=2,
        use_edge_embed=True, value_bins=7, action_motion_dim=3,
        action_motion_fourier=True, action_motion_fourier_L=2,
        action_depth_embed=True, action_depth_self_attn=depth_self_attn,
    )


def test_depth_self_attention_is_local_to_five_depths_and_backpropagates():
    model = _small_action_token_model(depth_self_attn=True)
    seen = []
    hook = model.action_depth_attn.attn.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape)))
    context = torch.randn(2, 5, 16, 16)
    contact_px = torch.rand(2, 60, 2) * 15.0
    action_motion = torch.randn(2, 60, 5, 3)

    logits = model(context, contact_px, action_motion=action_motion)
    hook.remove()
    assert logits.shape == (2, 60, 5, 7)
    assert seen == [(2 * 60, 5, 32)]
    assert model.action_depth_attn.attn.num_heads == 1

    loss = logits.square().mean()
    loss.backward()
    grad = model.action_depth_attn.attn.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_prior_motion_variant_strict_state_dict_is_unchanged():
    prior = _small_action_token_model(depth_self_attn=False)
    restored = _small_action_token_model(depth_self_attn=False)
    restored.load_state_dict(prior.state_dict(), strict=True)
    assert not hasattr(restored, "action_depth_attn")


def test_training_builder_enables_depth_local_attention_only_by_flag(monkeypatch):
    from namo.rl_loop.train_gen import _make_network

    monkeypatch.setenv("NAMO_ACTION_MOTION", "1")
    monkeypatch.setenv("NAMO_ACTION_MOTION_SHARP", "1")
    monkeypatch.setenv("NAMO_ACTION_DEPTH_SELF_ATTN", "1")
    treatment = _make_network(value_bins=51)
    assert treatment.action_motion_dim == CROP_RELATIVE_MOTION_DIM
    assert treatment.action_depth_self_attn is True
    assert treatment.action_depth_attn.attn.num_heads == 1

    monkeypatch.setenv("NAMO_ACTION_DEPTH_SELF_ATTN", "0")
    prior = _make_network(value_bins=51)
    assert prior.action_depth_self_attn is False
    assert not hasattr(prior, "action_depth_attn")


def test_training_builder_disables_inter_contact_attention_only_by_flag(monkeypatch):
    from namo.rl_loop.train_gen import _make_network

    monkeypatch.setenv("NAMO_EDGE_SELF_ATTN", "0")
    independent = _make_network(value_bins=51)
    assert all(not block.self_attn for block in independent.edge_blocks)
    assert all(not hasattr(block, "slf") for block in independent.edge_blocks)

    monkeypatch.setenv("NAMO_EDGE_SELF_ATTN", "1")
    full = _make_network(value_bins=51)
    assert all(block.self_attn for block in full.edge_blocks)
    assert all(hasattr(block, "slf") for block in full.edge_blocks)


def test_eval_loader_detects_depth_local_attention(tmp_path):
    from eval_auc import load_network
    from eval_scorer import load_scorer

    network = EdgeCrossAttn(
        img_size=64, patch=16, in_channels=5, dim=32, scene_depth=1, edge_depth=1,
        heads=1, num_depths=5, num_edges=60, value_bins=7,
        action_motion_dim=3, action_depth_embed=True, action_depth_self_attn=True,
    )
    module = ClassifierModule(
        network=network, head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0,
        dice_weight=0.0)
    checkpoint = tmp_path / "depth_local.ckpt"
    torch.save({
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(
            head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0),
        "action_motion_encoding": CROP_RELATIVE_MOTION_ENCODING,
    }, checkpoint)

    loaded = load_scorer(str(checkpoint), 5, "cpu", "edge_crossattn")
    assert loaded.network.action_depth_self_attn is True
    assert loaded.network.action_depth_attn.attn.num_heads == 1
    auc_network, hl = load_network(str(checkpoint), "cpu")
    assert auc_network.action_depth_self_attn is True
    assert auc_network.action_motion_encoding == CROP_RELATIVE_MOTION_ENCODING
    assert hl.num_bins == 7
