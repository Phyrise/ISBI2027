"""LSGAN and discriminator feature matching for the two-scale PatchGAN (pix2pixHD form).

- ``adversarial``: least-squares GAN loss summed over the scales (as NVIDIA pix2pixHD ``GANLoss``), unbounded scores.
- ``feature_matching``: L1 between the discriminator's intermediate features on (source, fake) and (source, real),
  real features detached, each layer weighted 4 / (layers + 1), summed over layers, averaged over scales.
"""
import torch.nn.functional as F


def adversarial(outputs, real):
    return sum((score.float() - float(real)).square().mean() for _, score in outputs)


def feature_matching(fake, real, layers=3):
    if len(fake) != len(real) or not fake:
        raise ValueError('Feature lists must match')
    layer_weight = 4. / (layers + 1)
    return sum(layer_weight * sum(F.l1_loss(a.float(), b.detach().float())
               for a, b in zip(f[0], r[0])) for f, r in zip(fake, real)) / len(fake)
