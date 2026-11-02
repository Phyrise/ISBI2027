"""Masked L1, PatchGAN discriminator, LSGAN, feature matching and the expert Dice + CE vs the originals (CPU, fp32);
and sha identity of the nnunetv2 / dynamic_network_architectures files imported here with those the paper runs used."""
import hashlib
import unittest
from pathlib import Path

import torch

from synthesis.architectures.patchgan3d import build_discriminator
from synthesis.losses.adversarial import adversarial, feature_matching
from synthesis.losses.pixel import masked_l1
from synthesis.losses.segmentation import segmentation_loss
from tests.train_support import env_path, max_abs, need, original, report, strict_fp32

R11 = 'mama_3d_20261018_r11_arch'
R14 = 'mama_3d_20261018_r14_hist'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@need('ISBI_ORIG_PROJECTS')
class TestLosses(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def test_masked_l1(self):
        g = torch.Generator().manual_seed(0)
        pred = torch.randn(2, 1, 16, 32, 32, generator=g).requires_grad_(True)
        target = torch.randn(2, 1, 16, 32, 32, generator=g)
        valid = (torch.rand(2, 1, 16, 32, 32, generator=g) > .3).float()
        a = masked_l1(pred, target, valid)
        ga, = torch.autograd.grad(a, pred)
        with original(R11, 'common') as C:
            b = C.pixel_l1(pred, target, valid)
            gb, = torch.autograd.grad(b, pred)
        report('losses', 'masked L1 value', abs(float(a) - float(b)))
        report('losses', 'masked L1 gradient', max_abs(ga, gb))
        self.assertEqual(float(a), float(b))
        self.assertEqual(max_abs(ga, gb), 0.)

    def test_discriminator_and_gan_losses(self):
        torch.manual_seed(20261022)
        ours = build_discriminator(1, 64, 2, 3)
        with original(R11, 'gan_parts') as G:
            torch.manual_seed(20261022)
            theirs = G.MultiScaleD(source_channels=1, width=64, scales=2, layers=3)
            self.assertEqual(max_abs(ours.state_dict(), theirs.state_dict()), 0.)
            g = torch.Generator().manual_seed(1)
            src, real = torch.randn(2, 1, 64, 64, 64, generator=g), torch.randn(2, 1, 64, 64, 64, generator=g)
            fake = (real + .3 * torch.randn(2, 1, 64, 64, 64, generator=g)).requires_grad_(True)
            o1, o2 = ours(src, fake), theirs(src, fake)
            out_gap = max(max_abs(s1, s2) for (_, s1), (_, s2) in zip(o1, o2))
            feat_gap = max(max_abs(a, b) for (f1, _), (f2, _) in zip(o1, o2) for a, b in zip(f1, f2))
            with torch.no_grad():
                r1, r2 = ours(src, real), theirs(src, real)
            l1 = adversarial(o1, True) + 10 * feature_matching(o1, r1, 3)
            l2 = G.adversarial(o2, True) + 10 * G.feature_matching(o2, r2, 3)
            g1, = torch.autograd.grad(l1, fake)
            g2, = torch.autograd.grad(l2, fake)
            d1 = .5 * (adversarial(ours(src, real), True) + adversarial(ours(src, fake.detach()), False))
            d2 = .5 * (G.adversarial(theirs(src, real), True) + G.adversarial(theirs(src, fake.detach()), False))
        report('losses', 'PatchGAN scores (2 scales)', out_gap)
        report('losses', 'PatchGAN intermediate features', feat_gap)
        report('losses', 'LSGAN(G) + 10 FM value', abs(float(l1) - float(l2)), value=float(l2))
        report('losses', 'LSGAN(G) + 10 FM gradient wrt the fake', max_abs(g1, g2))
        report('losses', 'LSGAN(D) value', abs(float(d1) - float(d2)))
        for gap in (out_gap, feat_gap, abs(float(l1) - float(l2)), max_abs(g1, g2), abs(float(d1) - float(d2))):
            self.assertEqual(gap, 0.)

    def test_segmentation_loss(self):
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(2, 2, 16, 32, 32, generator=g).requires_grad_(True)
        label = (torch.rand(2, 1, 16, 32, 32, generator=g) > .8).long()
        label[:, :, :3] = -1
        a = segmentation_loss()(logits, label)
        ga, = torch.autograd.grad(a, logits)
        with original(R14, 'train') as T:
            b = T.segmentation_loss()(logits, label)
            gb, = torch.autograd.grad(b, logits)
        report('losses', 'expert Dice + CE value', abs(float(a) - float(b)), value=float(b))
        report('losses', 'expert Dice + CE gradient', max_abs(ga, gb))
        self.assertEqual(float(a), float(b))
        self.assertEqual(max_abs(ga, gb), 0.)

    @need('ISBI_ORIG_TR')
    def test_pinned_dependencies_equal_paper_snapshot(self):
        import dynamic_network_architectures
        import nnunetv2
        tr = env_path('ISBI_ORIG_TR')
        nn_root = Path(nnunetv2.__file__).parent
        dna_root = Path(dynamic_network_architectures.__file__).parent
        files = [(nn_root / f, tr / 'nnunetv2' / f) for f in
                 ('training/loss/afp_features.py', 'training/loss/compound_losses.py', 'training/loss/dice.py',
                  'training/loss/robust_ce_loss.py', 'utilities/helpers.py', 'utilities/ddp_allgather.py')]
        files += [(p, tr / 'dynamic_network_architectures' / p.relative_to(dna_root)) for p in sorted(dna_root.rglob('*.py'))]
        differ = [str(a) for a, b in files if sha(a) != sha(b)]
        report('deps', f'files of the pinned tags differing from the paper snapshot ({len(files)} compared)',
               float(len(differ)), nnunetv2=str(nn_root), dynamic_network_architectures=str(dna_root))
        self.assertEqual(differ, [])
