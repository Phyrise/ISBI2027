"""MedicalNet and 2D slice teachers (SAM 2.1, VGG16): loss value and input gradient vs the original teacher modules
with the real weights, fp32 on CPU (TF32 disabled)."""
import unittest

import numpy as np
import torch

from synthesis.losses.medicalnet import MedicalNetTeacher
from synthesis.losses.slice2d import SliceTeacher, slice_indices
from tests.train_support import env_path, max_abs, need, original, report, strict_fp32


def inputs(shape, seed):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(shape, generator=g) * 1.3 + .4
    pred = target + .5 * torch.randn(shape, generator=g)
    valid = torch.ones(shape)
    valid[..., :2, :, :] = 0
    return pred, target, valid


def run(teacher, pred, target, **kw):
    p = pred.clone().requires_grad_(True)
    d = teacher(p, target, **kw)
    d.backward()
    return d.detach(), p.grad


@need('ISBI_ORIG_PROJECTS')
class TestTeachers(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    @need('ISBI_MEDICALNET_CKPT')
    def test_medicalnet(self):
        pred, target, _ = inputs((2, 1, 32, 64, 64), 11)
        ckpt = env_path('ISBI_MEDICALNET_CKPT')
        v1, g1 = run(MedicalNetTeacher(ckpt, device='cpu'), pred, target)
        with original('mama_3d_20261018_r5_mednet_v4', 'mednet_teacher') as M:
            v2, g2 = run(M.MedicalNetTeacher(checkpoint=ckpt, device='cpu'), pred, target)
        report('teachers', 'MedicalNet value', abs(float(v1) - float(v2)), value=float(v2))
        report('teachers', 'MedicalNet d D / d prediction', max_abs(g1, g2))
        self.assertEqual(float(v1), float(v2))
        self.assertEqual(max_abs(g1, g2), 0.)

    @need('ISBI_SLICE_WEIGHTS')
    def test_slice_teachers(self):
        pred, target, valid = inputs((2, 1, 12, 64, 64), 5)
        for kind, fname in (('vgg', 'authors_VGG16.pt'), ('sam', 'authors_SAM2.1_Small.pt')):
            ours = SliceTeacher(kind, env_path('ISBI_SLICE_WEIGHTS') / fname, device='cpu', slices=4)
            v1, g1 = run(ours, pred, target, valid=valid, step=1234)
            with original('mama_3d_20261018_r7_percep2d', 'slice_teacher') as T:
                theirs = T.SliceTeacher(kind, device='cpu', slices=4)
                v2, g2 = run(theirs, pred, target, valid=valid, step=1234)
                idx_gap = max(int(np.abs(slice_indices(s, b, 96) - T.slice_indices(s, b, 96)).max())
                              for s in range(0, 5000, 97) for b in range(2))
            report('teachers', f'{kind} slice teacher value', abs(float(v1) - float(v2)), value=float(v2), slices=4)
            report('teachers', f'{kind} slice teacher d D / d prediction', max_abs(g1, g2))
            report('teachers', f'{kind} slice indices (52 steps x 2 samples, 32 of 96)', float(idx_gap))
            self.assertEqual(float(v1), float(v2))
            self.assertEqual(max_abs(g1, g2), 0.)
            self.assertEqual(idx_gap, 0)
