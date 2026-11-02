"""LR curve of every arm and expert vs the original lr_at (per epoch) / lr_r11 (per update), and the recipe values of
every config vs the paper design files and dose files."""
import json
import unittest

from tests.train_support import (ALL_ARMS, ARM_TREE, EXPERTS, REPO, arm_config, need, original, report, root)
from synthesis.trainers.schedules import epoch_lr, update_lr


@need('ISBI_ORIG_PROJECTS', 'ISBI_ORIG_ROOT')
class TestSchedules(unittest.TestCase):
    def test_arm_lr_curves(self):
        for arm in ALL_ARMS:
            cfg = arm_config(arm)
            for tree_name in ARM_TREE[arm]:
                with original(tree_name, 'common') as C:
                    if cfg['trainer'] == 'sgd':
                        n = cfg['schedule']['horizon']
                        ours = [epoch_lr(cfg, e) for e in range(n)]
                        theirs = [C.lr_at(e, arm) for e in range(n)]
                        unit = 'epoch'
                    else:
                        n = cfg['schedule']['horizon'] * cfg['train']['steps_per_epoch']
                        ours = [update_lr(cfg, u) for u in range(n)]
                        theirs = [C.lr_r11(u, arm) for u in range(n)]
                        unit = 'update'
                gap = max(abs(a - b) for a, b in zip(ours, theirs))
                report('schedules', f'lr {arm} vs {tree_name}', gap, points=n, unit=unit)
                self.assertEqual(gap, 0.)

    def test_expert_lr_curves(self):
        for name, (k100, t100, k300, t300) in EXPERTS.items():
            cfg = arm_config(name, 'experts')
            ours = [epoch_lr(cfg, e) for e in range(300)]
            for kind, tree_name, epochs in ((k100, t100, range(0, 100)), (k300, t300, range(100, 300))):
                with original(tree_name, 'common') as C:
                    gap = max(abs(ours[e] - C.lr_at(e, kind)) for e in epochs)
                report('schedules', f'lr expert {name} vs {kind} ({tree_name})', gap, points=len(epochs), unit='epoch')
                self.assertEqual(gap, 0.)

    def test_recipe_values(self):
        R = root()
        design = json.loads((R / 'design.json').read_text())
        r11 = json.loads((R / 'design_r11_arch.json').read_text())['kinds']
        doses = {'ssl': 'ssl', 'ssl_nnf': 'ssl', 'mam_scratch300': 'mam_scratch300', 'mam300': 'mam300',
                 'mednet': 'mednet', 'sam': 'sam', 'vgg': 'vgg', 'mam_scratch300_nnfoff': 'mam_scratch300',
                 'swin_mam_scratch300': 'swin_mam_scratch300', 'gan_nnf_mam_scratch300': 'mam_scratch300'}
        for arm in ALL_ARMS:
            cfg = arm_config(arm)
            lam = cfg['loss']['lambda']
            if arm in doses:
                paper = json.loads((R / 'doses' / f'{doses[arm]}.json').read_text())['lambda']
                report('configs', f'lambda {arm} vs doses/{doses[arm]}.json', abs(lam - paper))
                self.assertEqual(lam, paper)
                self.assertEqual(json.loads((REPO / cfg['loss']['dose_file']).read_text())['lambda'], paper)
            else:
                self.assertIsNone(lam)
            self.assertEqual(cfg['train']['steps_per_epoch'], design['synthesis']['steps_per_epoch'])
            self.assertEqual(cfg['train']['val_batches'], design['synthesis']['val_batches'])
            self.assertEqual(cfg['seed'], design['synthesis']['seed'])
            if arm in r11:
                k = r11[arm]
                self.assertEqual((cfg['epochs'], cfg['schedule']['horizon'], cfg['schedule']['constant_epochs'],
                                  cfg['schedule']['lr'], list(cfg['optimizer']['betas']), cfg['optimizer']['clip'],
                                  cfg['optimizer']['grad_scaler_init'], cfg['paper']['checkpoint_epoch']),
                                 (k['horizon'], k['horizon'], k['constant_epochs'], k['lr'], k['betas'], k['clip'],
                                  k['initial_scale'], k['horizon']))
                self.assertTrue(set(k['snapshots']) <= set(cfg['train']['snapshots']))
                for w in ('pixel_weight', 'adversarial_weight', 'fm_weight'):
                    if w in k:
                        self.assertEqual(cfg['loss'][w], k[w])
            else:
                o = design['optimizer']
                self.assertEqual((cfg['optimizer']['momentum'], cfg['optimizer']['nesterov'],
                                  cfg['optimizer']['weight_decay'], cfg['optimizer']['clip'], cfg['epochs']),
                                 (o['momentum'], o['nesterov'], o['weight_decay'], o['clip'], design['synthesis']['horizon']))
                self.assertTrue(set(design['snapshots']) <= set(cfg['train']['snapshots']))
        report('configs', 'recipe values of the 14 arms vs design.json / design_r11_arch.json / doses', 0.)
