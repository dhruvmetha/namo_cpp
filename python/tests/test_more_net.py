"""MORE's push network, checked against `models.py` / `vision/backbone_utils.py`.

Pinned to revision 70402c001c30908e7a70b93f8de3c31abdd26fdb. The point of these is that
the port is MORE's architecture rather than something merely ResNet-shaped, because the
comparison is worth nothing if the baseline quietly became our own network again.
"""

import pytest

torch = pytest.importorskip("torch")

from namo.rl_loop.more_net import (  # noqa: E402
    FPN_OUT_CHANNELS,
    RESNET34_FPN_IN_CHANNELS,
    RETURN_LAYERS,
    MorePushNet,
    read_contact_values,
)


def test_fpn_wiring_matches_resnet_fpn_net():
    """resnet_fpn_net: layer1..4 -> '0'..'3', [64,128,256,512] for resnet34, 256 out."""
    assert RETURN_LAYERS == {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    assert RESNET34_FPN_IN_CHANNELS == [64, 128, 256, 512]
    assert FPN_OUT_CHANNELS == 256


def test_head_is_their_conv_stack():
    """BackboneWithFPNAndHeadPush: 256->256 k3, 256->128 k3, 128->32 k1, 32->out k1.

    The two 3x3 convolutions carry no padding in their code, which shrinks the map
    before the upsample. Padding them would be a quiet change to the receptive field.
    """
    net = MorePushNet(in_channels=5, num_depths=5)
    assert (net.conv0.in_channels, net.conv0.out_channels, net.conv0.kernel_size) == (256, 256, (3, 3))
    assert (net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size) == (256, 128, (3, 3))
    assert (net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size) == (128, 32, (1, 1))
    assert (net.conv3.in_channels, net.conv3.kernel_size) == (32, (1, 1))
    assert net.conv0.padding == (0, 0) and net.conv1.padding == (0, 0)
    for c in (net.conv0, net.conv1, net.conv2, net.conv3):
        assert c.bias is None


def test_no_imagenet_weights():
    """PushNet passes pretrained=False. Borrowed features would not be MORE's result."""
    a = MorePushNet(in_channels=2, num_depths=1)
    b = MorePushNet(in_channels=2, num_depths=1)
    w1 = a.body.layer1[0].conv1.weight
    w2 = b.body.layer1[0].conv1.weight
    assert not torch.allclose(w1, w2), "two fresh nets sharing weights means pretrained ones"


def test_output_is_one_dense_map_per_push_depth():
    """Their conv3 emits 1 channel because PUSH_DISTANCE is a single constant.

    Ours emits one per push length, the first of the two named deviations. A 1-channel
    head would leave the baseline unable to tell a short push from a long one.
    """
    net = MorePushNet(in_channels=5, num_depths=5)
    out = net(torch.zeros(2, 5, 224, 224))
    assert out.shape == (2, 5, 224, 224)
    assert net.conv3.out_channels == 5


def test_runs_at_both_render_resolutions():
    """224 is MORE's IMAGE_SIZE and the size our renderer crops to before downsampling."""
    net = MorePushNet(in_channels=5, num_depths=5)
    for size in (224, 64):
        assert net(torch.zeros(1, 5, size, size)).shape == (1, 5, size, size)


def test_input_width_is_configurable_for_the_goal_region_channel():
    """Their 2 masks carry no goal region; our task is defined by one."""
    for ch in (2, 5):
        net = MorePushNet(in_channels=ch, num_depths=5)
        assert net.body.conv1.in_channels == ch
        assert net(torch.zeros(1, ch, 64, 64)).shape == (1, 5, 64, 64)


def test_contact_reader_takes_the_max_over_their_window():
    """mcts_utils.py:929 reads np.max over a 7x7 window, not the single pixel."""
    vmap = torch.zeros(1, 2, 32, 32)
    vmap[0, 0, 12, 9] = 5.0          # 3 px away from (10, 10): inside a 7x7 window
    vmap[0, 1, 25, 25] = 7.0         # far away: must not leak in
    contacts = torch.tensor([[[10.0, 10.0]]])

    vals = read_contact_values(vmap, contacts, patch=3)
    assert vals.shape == (1, 1, 2)
    assert vals[0, 0, 0].item() == pytest.approx(5.0)
    assert vals[0, 0, 1].item() == pytest.approx(0.0)

    assert read_contact_values(vmap, contacts, patch=0)[0, 0, 0].item() == pytest.approx(0.0)


def test_contact_reader_clamps_pixels_at_the_border():
    """Contacts can sit on the crop edge, and an out-of-range index must not throw."""
    vmap = torch.zeros(1, 1, 16, 16)
    vmap[0, 0, 0, 0] = 3.0
    edge = torch.tensor([[[0.0, 0.0], [15.0, 15.0], [-4.0, 99.0]]])
    vals = read_contact_values(vmap, edge, patch=3)
    assert vals.shape == (1, 3, 1)
    assert vals[0, 0, 0].item() == pytest.approx(3.0)
