"""MedicalNet perceptual teacher: frozen MedicalNet 3D ResNet-50 (23-dataset checkpoint), MONAI perceptual recipe.

Recipe (MONAI ``PerceptualLoss`` with the MedicalNet network): layer4 features only, channel-L2-normalised, squared
difference summed over channels, spatial mean, batch mean. Adaptation: per-sample z-score of prediction and target
(MONAI z-scores the whole batch; identical for a batch of one). The teacher runs in fp32 with autocast disabled inside
it, whatever the caller's autocast. Inputs are zero-padded symmetrically to a multiple of 8.

Network re-typed from Project-MONAI/perceptual-models ``medicalnet/resnet.py`` (Apache-2.0), itself from
Tencent/MedicalNet (MIT). Checkpoint: ``resnet_50_23dataset.pth`` (sha256 ff48a622..., the file distributed with the
MONAI perceptual models); it is not redistributed here.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


MEDICALNET_R50_SHA256 = 'ff48a62219073fb977fd3f4ddfb8dc1367f0ec156c8d6f6c37e205bd683a246e'
HEAD_PREFIXES = ('conv_seg.',)  # Tencent segmentation head (absent from the MONAI R50 file)
NORMALISATIONS = ('batch', 'sample')
FEATURES = ('layer4',)
RECIPE = ('MONAI MedicalNet perceptual recipe: layer4 only, channel-L2-normalised features, squared difference summed '
          'over channels, spatial mean, batch mean')


# ---- network (Project-MONAI/perceptual-models medicalnet/resnet.py, Apache-2.0; Tencent/MedicalNet, MIT) ----
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # shortcut type B
            downsample = nn.Sequential(conv1x1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def resnet50():
    return MedicalNetResNet(Bottleneck, [3, 4, 6, 3])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def medicalnet_state_dict(checkpoint):
    """(state dict without 'module.' and without head keys, ignored head keys): weights_only load (downloaded file)."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = payload['state_dict'] if isinstance(payload, dict) and 'state_dict' in payload else payload
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
    ignored = sorted(k for k in state if k.startswith(HEAD_PREFIXES))
    return {k: v for k, v in state.items() if k not in ignored}, ignored


def medicalnet_r50(checkpoint, device='cpu', dtype=torch.float32):
    """Frozen, eval MedicalNet R50 loaded strictly + its provenance."""
    net = resnet50()
    state, ignored = medicalnet_state_dict(checkpoint)
    net.load_state_dict(state, strict=True)  # raises on any missing / unexpected key
    net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
    digest = sha256(checkpoint)
    prov = {'checkpoint': Path(checkpoint).name, 'checkpoint_sha256': digest,
            'checkpoint_sha256_matches_monai_file': digest == MEDICALNET_R50_SHA256, 'torch_load': 'weights_only',
            'architecture': 'MedicalNet 3D ResNet50 (Bottleneck [3,4,6,3], layer2 stride 2, layer3 dilation 2, layer4 '
                            'dilation 4, shortcut B; forward = layer4), re-typed from Project-MONAI/perceptual-models '
                            'medicalnet/resnet.py (Apache-2.0) / Tencent MedicalNet (MIT)',
            'keys_loaded': len(state), 'ignored_head_keys': ignored, 'strict_weights_loaded': True,
            'parameters': sum(p.numel() for p in net.parameters()), 'modalities': ['single channel (any MRI)']}
    return net, prov


def zscore(v, mode):
    """medicalnet_intensity_normalisation: 'batch' = over the whole tensor (MONAI), 'sample' = per sample."""
    if mode == 'batch':
        return (v - v.mean()) / v.std()
    if mode == 'sample':
        dims = tuple(range(1, v.ndim))
        return (v - v.mean(dims, keepdim=True)) / v.std(dims, keepdim=True)
    raise ValueError(f'normalisation {mode!r}: one of {NORMALISATIONS}')


def normalize_tensor(x, eps=1e-10):  # monai.losses.perceptual.normalize_tensor
    return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + eps)


class MedicalNetDistance(nn.Module):
    """D_M(prediction, target) of the MONAI recipe (see module docstring); per-sample values via per_sample()."""
    multiple = 8

    def __init__(self, network, normalisation='sample', features='layer4', teacher_autocast=False):
        super().__init__()
        self.teacher_autocast = bool(teacher_autocast)
        if normalisation not in NORMALISATIONS:
            raise ValueError(f'normalisation {normalisation!r}: one of {NORMALISATIONS}')
        if (features if isinstance(features, str) else tuple(features)) not in ('layer4', ('layer4',)):
            raise ValueError("MedicalNet recipe: features must be 'layer4' (single deep layer)")
        self.network, self.normalisation, self.selected = network, normalisation, FEATURES

    def _pad(self, x):
        pads = []
        for size in reversed(x.shape[2:]):
            missing = (-size) % self.multiple
            pads.extend((missing // 2, missing - missing // 2))
        return F.pad(x, pads) if any(pads) else x

    def features(self, v):
        p = next(self.network.parameters())
        with torch.autocast(v.device.type, enabled=self.teacher_autocast):  # fp32 teacher unless asked (see docstring)
            return self.network(self._pad(v).to(p.dtype)).float()

    def per_sample(self, prediction, target):
        if prediction.ndim != 5 or prediction.shape != target.shape or prediction.shape[1] != 1:
            raise ValueError('Expected matched B x 1 x D x H x W prediction and target')
        px = zscore(prediction.float(), self.normalisation)
        tx = zscore(target.detach().float(), self.normalisation)
        fx = normalize_tensor(self.features(px))
        with torch.no_grad():
            fy = normalize_tensor(self.features(tx))
        return (fx - fy).square().sum(1, keepdim=True).mean((2, 3, 4)).flatten()  # spatial_average_3d, one per sample

    def forward(self, prediction, target):
        return self.per_sample(prediction, target).mean()  # PerceptualLoss.forward: torch.mean(loss)


class MedicalNetTeacher(nn.Module):
    """Teacher interface: forward(prediction, target) -> D_M (batch mean); frozen and always in eval mode."""

    def __init__(self, checkpoint, device='cuda:0'):
        super().__init__()
        net, prov = medicalnet_r50(checkpoint, device=device, dtype=torch.float32)
        self.distance = MedicalNetDistance(net, normalisation='sample', features='layer4', teacher_autocast=False)
        self.provenance = dict(prov, features='layer4', feature_names=list(self.distance.selected), normalize_features=False,
                               dtype='float32', normalisation='sample', distance=RECIPE,
                               teacher_precision='fp32 (autocast disabled inside the teacher)')

    def train(self, mode=True):  # BatchNorm statistics stay frozen whatever the caller does
        return super().train(False)

    def forward(self, prediction, target):
        return self.distance(prediction, target)
