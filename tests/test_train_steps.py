"""Training updates vs the original trainers: for each arm (or expert), the original ``prepare`` + ``step`` of the tree
that trained it and this repository's trainer take the same two real batches; the losses, gradient norms, parameters
and optimizer states are compared after every update.

CPU, fp32: autocast and the GradScaler disable themselves without CUDA on both sides, the AFP teacher network is fp32 on
both sides (fp16 on the GPU in the paper runs), TF32 off. Small patches cut by the paper sampler:
32 x 64 x 64, or 64 x 64 x 64 for the two-scale GAN score kernels."""
import copy
import gc
import unittest

import torch

from synthesis.data.sampler import Batches, split_cases
from tests.train_support import (EXPERTS, REPO, arm_config, cpu_fp32_teacher, cuda_calls_on_cpu, max_abs, need,
                                 original, report, root, slow, strict_fp32)
from synthesis.trainers.run import make_trainer

PATCH = (32, 64, 64)
GAN_PATCH = (64, 64, 64)  # two discriminator scales need at least 48 voxels per axis
UPDATES = (0, 1)
SGD_CASES = [  # (config, kind, original tree)
    ('l1', 'arms', 'l1', 'mama_3d_20261018_r2'),
    ('ssl', 'arms', 'ssl', 'mama_3d_20261018_r2'),
    ('mam_scratch300', 'arms', 'mam_scratch300', 'mama_3d_20261018_r4_ext'),
    ('mednet', 'arms', 'mednet', 'mama_3d_20261018_r5_mednet_v4'),
    ('vgg', 'arms', 'vgg', 'mama_3d_20261018_r7_percep2d'),
    ('ssl_nnf', 'arms', 'ssl_nnf', 'mama_3d_20261018_r6_nnfinit'),
    ('mam_scratch300_nnfoff', 'arms', 'mam_scratch300_nnfoff', 'mama_3d_20261018_r14_hist'),
    ('nnf_ft', 'experts', EXPERTS['nnf_ft'][0], EXPERTS['nnf_ft'][1]),
    ('scratch', 'experts', EXPERTS['scratch'][0], EXPERTS['scratch'][1]),
]
ADAM_CASES = ['swin_l1', 'swin_mam_scratch300', 'gan_nnf', 'gan_nnf_mam_scratch300']
R11 = 'mama_3d_20261018_r11_arch'


def cpu_config(name, kind):
    cfg = arm_config(name, kind)
    cfg['precision']['train_autocast'] = None
    cfg['precision']['generator_autocast'] = None
    if cfg.get('teacher') and cfg['teacher']['kind'] in ('nnf', 'expert'):
        cfg['teacher']['dtype'] = 'float32'
    return cfg


def batches(cfg, patch=PATCH):
    ds = Batches(root() / 'data', split_cases(REPO / 'configs/split_duke.json', 'train'), cfg['seed'], 0,
                 max(UPDATES) + 1, patch=patch)
    return [ds[u] for u in UPDATES]


def clone(batch):
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def details_gap(a, b):
    keys = [k for k in b if isinstance(b[k], float) and k in a]
    return max((abs(a[k] - b[k]) for k in keys), default=0.), {k: abs(a[k] - b[k]) for k in keys}


def opt_state(opt):
    return {f'{i}.{k}': v for i, row in opt.state_dict()['state'].items() for k, v in row.items() if torch.is_tensor(v)}


@need('ISBI_ORIG_ROOT', 'ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX', 'ISBI_NNF_CKPT', 'ISBI_MEDICALNET_CKPT',
      'ISBI_SLICE_WEIGHTS')
@slow()
class TestSteps(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def test_sgd_updates(self):
        for name, kind_dir, kind, tree_name in SGD_CASES:
            with self.subTest(name):
                cfg = cpu_config(name, kind_dir)
                ours = make_trainer(cfg, 'cpu').prepare()
                with original(tree_name, 'train') as T, cuda_calls_on_cpu():
                    T.teacher = cpu_fp32_teacher(T.teacher)
                    model, afp, lossfn, weight, opt, scaler, _ = T.prepare(kind)
                    self.assertEqual(max_abs(ours.model.state_dict(), model.state_dict()), 0.)
                    self.assertEqual((ours.weight, ours.opt.param_groups[0]['lr']), (weight, opt.param_groups[0]['lr']))
                    rows = []
                    for b in batches(cfg):
                        r1 = ours.step(clone(b))
                        r2 = T.step(kind, model, afp, lossfn, weight, opt, scaler, clone(b))
                        rows.append((details_gap(r1, r2), max_abs(ours.model.state_dict(), model.state_dict()), r2))
                    opt_gap = max_abs(opt_state(ours.opt), opt_state(opt))
                for u, ((dg, per), pg, r2) in zip(UPDATES, rows):
                    report('steps', f'{name} ({tree_name}) update {u}: loss terms / grad norm', dg, per_key=per,
                           loss=r2['loss'], grad_norm=r2['grad_norm'])
                    report('steps', f'{name} ({tree_name}) update {u}: parameters after the update', pg)
                report('steps', f'{name} ({tree_name}): SGD momentum buffers after {len(UPDATES)} updates', opt_gap)
                self.assertEqual(max(max(dg for (dg, _), _, _ in rows), max(pg for _, pg, _ in rows), opt_gap), 0.)
                del ours, model, afp, opt
                gc.collect()

    def test_adam_gan_updates(self):
        for name in ADAM_CASES:
            with self.subTest(name):
                cfg = cpu_config(name, 'arms')
                ours = make_trainer(cfg, 'cpu').prepare()
                with original(R11, 'train') as T, cuda_calls_on_cpu():
                    T.teacher = cpu_fp32_teacher(T.teacher)
                    r = T.prepare(name)
                    self.assertEqual(max_abs(ours.model.state_dict(), r.model.state_dict()), 0.)
                    if r.disc is not None:
                        self.assertEqual(max_abs(ours.disc.state_dict(), r.disc.state_dict()), 0.)
                    self.assertEqual(ours.weight, r.weight)
                    rows = []
                    for u, b in zip(UPDATES, batches(cfg, GAN_PATCH if name.startswith('gan_') else PATCH)):
                        r1 = ours.step(clone(b), u)
                        r2 = T.step(r, clone(b), u)
                        pg = max_abs(ours.model.state_dict(), r.model.state_dict())
                        if r.disc is not None:
                            pg = max(pg, max_abs(ours.disc.state_dict(), r.disc.state_dict()))
                        rows.append((details_gap(r1, r2), pg, r2))
                    opt_gap = max(max_abs(opt_state(a), opt_state(c)) for a, c in zip(ours.opts, r.opts))
                for u, ((dg, per), pg, r2) in zip(UPDATES, rows):
                    report('steps', f'{name} (r11) update {u}: loss terms / grad norms', dg, per_key=per,
                           loss=r2['loss'], grad_norm=r2['grad_norm'])
                    report('steps', f'{name} (r11) update {u}: G (and D) parameters after the update', pg)
                report('steps', f'{name} (r11): Adam moments (G and D) after {len(UPDATES)} updates', opt_gap)
                self.assertEqual(max(max(dg for (dg, _), _, _ in rows), max(pg for _, pg, _ in rows), opt_gap), 0.)
                del ours, r
                gc.collect()


if __name__ == '__main__':
    unittest.main()
