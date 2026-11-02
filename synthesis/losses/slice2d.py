"""2D slice teachers: frozen SAM 2.1 (Hiera-S trunk) or VGG16 applied to seeded axial slices of the 3D patch.

Extractors: the authors' TorchScript files of the published SAM / VGG perceptual losses (KonfAI 1.8.7 SAM_Perceptual;
https://huggingface.co/VBoussot/impact-torchscript-models, revision fd241efc). Their weights were checked identical to
the original releases: VGG16 = torchvision VGG16 IMAGENET1K_V1 ``features`` (26/26 tensors bit-identical to
vgg16-397923af.pth), SAM = the official SAM 2.1 Hiera-S ``image_encoder.trunk`` (202/202 tensors identical to
sam2.1_hiera_small.pt). Each file holds its own StandardizeImageNet layer:
x -> ((x - stats[0]) / (stats[1] - stats[0] + 1e-6) - ImageNet mean) / ImageNet std, on 3 identical channels.

Preprocessing: stats[0:2] = quantiles 0.5 % / 99.5 % of the valid voxels of the sample's 3D target patch (detached),
the same bounds for prediction and target, prediction not clipped.
Distance (= the AFP L1 distance = KonfAI SAM_Perceptual): sum over weighted layers of w_l * mean |phi_l(pred) - phi_l(target)|
(target features without gradient), mean over the K slices, then over the batch.
- VGG16: relu1_2 / relu2_2 / relu3_3 / relu4_3, weights 1, 1, 1, 1, native 160 x 160.
- SAM 2.1 Small: Hiera stage ends (blocks 0 / 2 / 13 / 15) weighted 0, 1, 1, 0; slices padded 160 -> 256 at the right /
  bottom with each slice's own minimum (KonfAI pads 320 -> 512 the same way).
Slices: K = 32 of the 96 axial slices (axis D, 0.7 x 0.7 mm in-plane), drawn without replacement from
numpy default_rng([20261021, batch step, sample]), stateless like the batch sampler (identical after a resume).
Precision: the caller's autocast (fp16 in training); features compared in fp32.

TorchScript compatibility: recent exports serialise every interpolate branch, including a Lanczos one that torch 2.8
does not have. ``register_torchscript_schema_compatibility`` declares its schema only, with no kernel: the file loads,
the unreachable branch would fail explicitly if ever executed, and the executed path (bicubic position embedding) is
unchanged.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SPEC = {
    'vgg': dict(filename='authors_VGG16.pt', sha256='8ba626577d0c95c494caef4d747f00e2551037c9f0157639afeb6afe7c590c92',
                weights=(1., 1., 1., 1.), tile=None, layers='VGG16 relu1_2, relu2_2, relu3_3, relu4_3',
                source='https://huggingface.co/VBoussot/impact-torchscript-models/resolve/fd241efc55787c5662b0e6444c9775dd9737a8b1/VGG/VGG16.pt',
                weights_identity='torchvision VGG16 IMAGENET1K_V1 features: 26/26 tensors bit-identical to vgg16-397923af.pth '
                                 '(sha256 397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0)'),
    'sam': dict(filename='authors_SAM2.1_Small.pt', sha256='b241bdcbc11ef5e39be3c4ef22e8a26a915eecce572ac3fce7f94feb2d84d761',
                weights=(0., 1., 1., 0.), tile=256, layers='SAM2.1 Hiera-S stage ends (blocks 0, 2, 13, 15), weighted 0,1,1,0',
                source='https://huggingface.co/VBoussot/impact-torchscript-models/resolve/fd241efc55787c5662b0e6444c9775dd9737a8b1/SAM2.1/SAM2.1_Small.pt',
                weights_identity='official SAM2.1 Hiera-S image_encoder.trunk: 202/202 tensors identical to sam2.1_hiera_small.pt'),
}
SLICES = 32
SLICE_SEED = 20261021
PERCENTILES = (0.005, 0.995)
_COMPAT = []


def register_torchscript_schema_compatibility():
    """Declare the schema of aten::_upsample_lanczos2d_aa (no kernel) so that the TorchScript files load on torch 2.8."""
    if not hasattr(torch.ops.aten, '_upsample_lanczos2d_aa'):
        lib = torch.library.Library('aten', 'FRAGMENT')
        lib.define('_upsample_lanczos2d_aa(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor')
        _COMPAT.append(lib)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def slice_indices(step, sample, depth, k=SLICES):
    rng = np.random.default_rng([SLICE_SEED, int(step), int(sample)])
    return np.sort(rng.choice(depth, size=k, replace=False))


def pad_min(x, tile):
    """x: N x 3 x H x W -> N x 3 x tile x tile, right / bottom filled with each slice's own minimum (differentiable)."""
    h, w = x.shape[-2:]
    if tile is None or (h, w) == (tile, tile):
        return x
    if h > tile or w > tile:
        raise ValueError('slice larger than the feature tile')
    low = x.amin(dim=(1, 2, 3), keepdim=True)
    out = F.pad(x, (0, tile - w, 0, tile - h))
    inside = torch.zeros(tile, tile, dtype=torch.bool, device=x.device)
    inside[:h, :w] = True
    return torch.where(inside, out, low.expand_as(out))


class SliceTeacher(nn.Module):
    """afp(prediction, target, valid=, step=) -> D (batch mean); frozen, always eval."""

    def __init__(self, kind, file, device='cuda:0', slices=SLICES):
        super().__init__()
        spec = SPEC[kind]
        register_torchscript_schema_compatibility()
        digest = sha256(file)
        if digest != spec['sha256']:
            raise ValueError(f'{kind}: {file} has sha256 {digest}, expected the authors file {spec["sha256"]}')
        self.network = torch.jit.load(file, map_location=device).eval()
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.kind, self.weights, self.tile, self.slices = kind, tuple(spec['weights']), spec['tile'], int(slices)
        self.provenance = {'kind': kind, 'checkpoint': Path(file).name, 'checkpoint_sha256': digest, 'source': spec['source'],
                           'weights_identity': spec['weights_identity'], 'layers': spec['layers'],
                           'layer_weights': list(self.weights), 'tile': self.tile or 'native 160',
                           'slices_per_sample': self.slices, 'slice_axis': 'D (axial, 0.7 x 0.7 mm in-plane)',
                           'slice_seed': SLICE_SEED, 'percentiles': list(PERCENTILES),
                           'preprocessing': 'authors StandardizeImageNet with stats[0:2] = target percentiles 0.5 / 99.5 '
                                            '(valid voxels of the 3D target patch, detached), same bounds for prediction '
                                            'and target, prediction not clipped, 3 identical channels, ImageNet mean / std',
                           'distance': 'sum_l w_l * mean |phi_l(pred) - phi_l(target)|, mean over slices and batch '
                                       '(= AFP L1 distance = KonfAI SAM_Perceptual)',
                           'precision': 'autocast of the caller (fp16 in training), features compared in fp32',
                           'parameters': sum(p.numel() for p in self.network.parameters()),
                           'torchscript_compat': 'schema-only aten::_upsample_lanczos2d_aa, never executed'}

    def train(self, mode=True):  # frozen extractor: eval whatever the caller does
        return super().train(False)

    def stats(self, target, valid):
        t = target.detach().float()
        keep = t[valid > 0] if valid is not None and bool((valid > 0).any()) else t.reshape(-1)
        lo, hi = torch.quantile(keep, torch.tensor(PERCENTILES, device=t.device))
        return torch.stack((lo, hi, keep.mean(), keep.std())).float()

    def per_sample(self, prediction, target, valid, step):
        if prediction.ndim != 5 or prediction.shape != target.shape or prediction.shape[1] != 1:
            raise ValueError('Expected matched B x 1 x D x H x W prediction and target')
        n = torch.tensor([len(self.weights)])
        out = []
        for b in range(prediction.shape[0]):
            idx = torch.as_tensor(slice_indices(step, b, prediction.shape[2], self.slices), device=prediction.device)
            st = self.stats(target[b], None if valid is None else valid[b])
            p = pad_min(prediction[b, 0].index_select(0, idx)[:, None].repeat(1, 3, 1, 1), self.tile)
            y = pad_min(target[b, 0].detach().index_select(0, idx)[:, None].repeat(1, 3, 1, 1), self.tile)
            fp = self.network(p, n, st)
            with torch.no_grad():
                fy = self.network(y, n, st)
            self.feature_dtype = str(fp[1].dtype)
            out.append(sum(w * (a.float() - c.float()).abs().mean() for w, a, c in zip(self.weights, fp, fy) if w))
        return torch.stack(out)

    def forward(self, prediction, target, valid=None, step=None):
        if step is None:
            raise ValueError('slice selection needs the batch step')
        return self.per_sample(prediction, target, valid, step).mean()
