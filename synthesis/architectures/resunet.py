"""ResUNet generator (all ResUNet arms and the GAN generator).

``dynamic_network_architectures.architectures.unet.ResidualUNet`` from the Phyrise fork of dynamic-network-architectures
(tag ``isbi2027-results``, which adds the residual decoder with trilinear upsampling, ``decoder_type='trilinear'``),
configured by ``configs/architectures/resunet.json``: the 3d_fullres architecture block of the nnU-Net plans of the
generator family used before this study (6 stages, 32-64-128-256-320-320 features, 1-3-4-6-6-6 residual blocks, strides
1, 2, 2, 2, 2, (1, 2, 2), InstanceNorm, LeakyReLU, conv bias). One input and one output channel, deep supervision off.
"""
import importlib
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'architectures' / 'resunet.json'


def build_resunet(config=DEFAULT_CONFIG):
    arch = json.loads(Path(config).read_text())
    kw = dict(arch['arch_kwargs'])
    for key in arch['_kw_requires_import']:
        if kw[key]:
            module, name = kw[key].rsplit('.', 1)
            kw[key] = getattr(importlib.import_module(module), name)
    module, name = arch['network_class_name'].rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(input_channels=1, num_classes=1, deep_supervision=False, **kw)
