"""Two-scale conditional 3D PatchGAN discriminator (pix2pixHD-style, adapted to 3D).

Not the official pix2pixHD code: a 3D adaptation whose LSGAN aggregation over scales and intermediate-feature layout
follow the NVIDIA pix2pixHD implementation (https://github.com/NVIDIA/pix2pixHD, BSD licence). The discriminator sees
the source (pre-contrast) concatenated with a real or generated target; scores are unbounded (no sigmoid).
Paper configuration: width 64, 3 layers, 2 scales, 1 source channel.
"""
import torch
from torch import nn
import torch.nn.functional as F


class PatchD(nn.Module):
    def __init__(self, source_channels=3, width=64, layers=3):
        super().__init__()
        blocks = [nn.Sequential(nn.Conv3d(source_channels + 1, width, 4, 2, 1),
                                nn.LeakyReLU(.2, inplace=False))]
        previous = width
        for stage in range(1, layers + 1):
            channels = width * min(2**stage, 8)
            blocks.append(nn.Sequential(
                nn.Conv3d(previous, channels, 4, 2 if stage < layers else 1, 1),
                nn.InstanceNorm3d(channels), nn.LeakyReLU(.2, inplace=False)))
            previous = channels
        self.blocks = nn.ModuleList(blocks)
        self.score = nn.Conv3d(previous, 1, 4, 1, 1)

    def forward(self, source, target):
        if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:] or target.shape[1] != 1:
            raise ValueError('Conditional inputs must share a grid and a single target channel')
        value = torch.cat((source, target), 1)
        features = []
        for block in self.blocks:
            value = block(value)
            features.append(value)
        return features, self.score(value)


class MultiScaleD(nn.Module):
    def __init__(self, source_channels=3, width=64, scales=2, layers=3):
        super().__init__()
        self.layers = layers
        self.discriminators = nn.ModuleList(
            [PatchD(source_channels, width, layers) for _ in range(scales)])

    def forward(self, source, target):
        outputs = []
        for index, network in enumerate(self.discriminators):
            outputs.append(network(source, target))
            if index + 1 < len(self.discriminators):
                source = F.avg_pool3d(source, 3, 2, 1, count_include_pad=False)
                target = F.avg_pool3d(target, 3, 2, 1, count_include_pad=False)
        return outputs


def build_discriminator(source_channels=1, width=64, scales=2, layers=3):
    return MultiScaleD(source_channels=source_channels, width=width, scales=scales, layers=layers)
