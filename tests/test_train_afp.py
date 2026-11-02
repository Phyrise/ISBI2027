"""AFP distance value and input gradient vs the original teacher path (men_clinical FoundationAFP with
mae_weight = 0, afp_weight = 1, as built by the original common.teacher), with the real nnFoundationCNN checkpoint and
the real expert checkpoints, on a small random patch, fp32 on CPU (TF32 disabled)."""
import unittest

import torch

from synthesis.losses.afp import EncoderAFP
from tests.train_support import env_path, max_abs, need, original, report, root, strict_fp32

R11 = 'mama_3d_20261018_r11_arch'
SHAPE = (2, 1, 32, 64, 64)


def inputs(seed):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(SHAPE, generator=g) * 1.3 + .4
    pred = target + .5 * torch.randn(SHAPE, generator=g)
    return pred, target


def run(teacher, pred, target):
    p = pred.clone().requires_grad_(True)
    d = teacher(p, target)
    d.backward()
    return d.detach(), p.grad


@need('ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX', 'ISBI_ORIG_ROOT', 'ISBI_NNF_CKPT')
class TestAFP(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def compare(self, name, ckpt, trusted):
        pred, target = inputs(7)
        ours = EncoderAFP(ckpt, trusted=trusted, dtype='float32', device='cpu')
        v1, g1 = run(ours, pred, target)
        del ours
        with original(R11, 'common') as C:
            from men_clinical.foundation_afp import FoundationAFP
            theirs = FoundationAFP(checkpoint=ckpt, modality='first_postcontrast', features='encoder', device='cpu',
                                   dtype='float32')
            theirs.mae_weight, theirs.afp_weight = 0., 1.  # as common.teacher()
            v2, g2 = run(theirs, pred, target)
            names = list(theirs.distance.selected)
        report('afp', f'{name}: AFP value', abs(float(v1) - float(v2)), value=float(v2), features=names)
        report('afp', f'{name}: d AFP / d prediction', max_abs(g1, g2), grad_abs_max=float(g2.abs().max()))
        self.assertEqual(names, [f'encoder.{i}' for i in range(6)])
        self.assertEqual(float(v1), float(v2))
        self.assertEqual(max_abs(g1, g2), 0.)

    def test_nnf_teacher(self):
        self.compare('nnFoundationCNN', env_path('ISBI_NNF_CKPT'), False)

    def test_expert_scratch_teacher(self):
        self.compare('expert_scratch_c300 ep300', root() / 'results/expert_scratch_c300/checkpoint_ep300.pth', True)

    def test_expert_nnf_ft_teacher(self):
        self.compare('expert_c300 ep300', root() / 'results/expert_c300/checkpoint_ep300.pth', True)

    def test_zero_outside_valid(self):
        """The trainers pass where(valid, prediction, target): no AFP gradient reaches voxels outside the volume."""
        pred, target = inputs(3)
        valid = torch.ones(SHAPE)
        valid[..., :8, :, :] = 0
        teacher = EncoderAFP(env_path('ISBI_NNF_CKPT'), dtype='float32', device='cpu')
        p = pred.clone().requires_grad_(True)
        teacher(torch.where(valid.bool(), p, target), target).backward()
        outside = float(p.grad[valid == 0].abs().max())
        report('afp', 'gradient outside the valid volume (must be 0)', outside)
        self.assertEqual(outside, 0.)
