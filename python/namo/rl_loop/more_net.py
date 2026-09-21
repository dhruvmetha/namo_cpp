"""MORE's push-prediction network, ported to our action library.

`models.py:10` PushNet is `resnet_fpn_net("resnet34", trainable_layers=5, grasp=False,
input_channels=2)`, a fully-convolutional ResNet-34 with an FPN emitting a dense
per-pixel value map. Read at revision 70402c001c30908e7a70b93f8de3c31abdd26fdb.

The head, copied from `vision/backbone_utils.py:53` BackboneWithFPNAndHeadPush:

    body -> FPN -> take level "0"
    conv0 256->256 k3 (no padding), BN, ReLU
    conv1 256->128 k3 (no padding), BN, ReLU
    bilinear upsample to half the input
    conv2 128->32 k1, BN, ReLU
    bilinear upsample to the full input
    conv3 32->OUT k1

Wiring from `resnet_fpn_net`: return_layers layer1..layer4 -> "0".."3",
in_channels_list [64,128,256,512] for resnet34, FPN out_channels 256, a
LastLevelMaxPool extra block, FrozenBatchNorm2d, and no ImageNet pretraining.

TWO deviations, both forced by our action library and both named in the write-up:

  * `out_channels`. Their conv3 emits ONE channel because `PUSH_DISTANCE = 0.1` is a
    single constant: MORE has exactly one push length, so its head has nothing to say
    about distance. Our library has five. One channel would leave the baseline blind
    to a whole dimension of the action space, which is a strawman rather than a
    faithful port, so conv3 emits one channel per push depth.
  * `in_channels`. Theirs is 2, a target-object mask and an all-objects mask. Their
    task has no goal region; ours is defined by one. A network that cannot see where
    the goal is cannot know which way to open, so it gets the same channels our own
    renderer produces.

Everything else is theirs, including the unpadded 3x3 convolutions that shrink the map
before the upsample, and the frozen batch norm despite every layer being trainable.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

FPN_OUT_CHANNELS = 256
RESNET34_FPN_IN_CHANNELS = [64, 128, 256, 512]
RETURN_LAYERS = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}


def _resnet34_body(in_channels: int) -> nn.Module:
    """resnet34 with FrozenBatchNorm2d and a first conv widened to `in_channels`.

    MORE vendors a resnet that takes `input_channels` directly. torchvision's does not,
    so the first conv is rebuilt. Same layer, same stride, same kernel; only its input
    width differs, which is exactly what their fork changes too.
    """
    net = resnet34(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return net


class MorePushNet(nn.Module):
    """BackboneWithFPNAndHeadPush, with a depth channel per push length."""

    def __init__(self, in_channels: int = 5, num_depths: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths
        body = _resnet34_body(in_channels)
        self.body = IntermediateLayerGetter(body, return_layers=RETURN_LAYERS)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=RESNET34_FPN_IN_CHANNELS,
            out_channels=FPN_OUT_CHANNELS,
            extra_blocks=LastLevelMaxPool(),
        )
        self.conv0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, num_depths, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, H, W) -> (B, num_depths, H, W) of predicted push values."""
        half = (x.shape[-2] // 2, x.shape[-1] // 2)
        full = x.shape[-2:]
        x = self.fpn(self.body(x))["0"]
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, size=half, mode="bilinear", align_corners=True)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, size=full, mode="bilinear", align_corners=True)
        return self.conv3(x)


def read_contact_values(value_map: torch.Tensor, contact_px: torch.Tensor,
                        patch: int = 3) -> torch.Tensor:
    """Read one value per (contact, depth) out of a dense map.

    `mcts_utils.py:929` takes `np.max` over a 7x7 window centred on the action's start
    pixel rather than the single pixel, because the head is convolutional and the exact
    pixel is noisy. `patch` is the half-width, so the default 3 reproduces their 7x7.

    value_map  (B, D, H, W)
    contact_px (B, E, 2) pixel coordinates of each contact edge, row then column
    returns    (B, E, D)
    """
    b, d, h, w = value_map.shape
    e = contact_px.shape[1]
    rows = contact_px[..., 0].round().long().clamp(0, h - 1)
    cols = contact_px[..., 1].round().long().clamp(0, w - 1)
    out = value_map.new_empty((b, e, d))
    for i in range(e):
        r, c = rows[:, i], cols[:, i]
        vals = value_map.new_full((b, d), float("-inf"))
        for dr in range(-patch, patch + 1):
            rr = (r + dr).clamp(0, h - 1)
            for dc in range(-patch, patch + 1):
                cc = (c + dc).clamp(0, w - 1)
                vals = torch.maximum(vals, value_map[torch.arange(b), :, rr, cc])
        out[:, i, :] = vals
    return out
