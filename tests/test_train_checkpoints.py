"""Lambda calibration vs the original calibrate.py and the paper dose files; paper checkpoints restored into this
repository's trainers; the expert continuation fork vs the paper fork."""
import gc
import json
import shutil
import statistics
import tempfile
import unittest

import torch

from synthesis.data.sampler import Batches, split_cases
from synthesis.losses.calibrate import calibrate, norm_grad
from tests.train_support import (REPO, arm_config, cpu_fp32_teacher, cuda_calls_on_cpu, env_path, max_abs, need,
                                 original, report, root, slow, strict_fp32)
from synthesis.trainers import checkpoint as CK
from synthesis.trainers.run import make_trainer

R2 = 'mama_3d_20261018_r2'
DOSES = {'ssl': 'ssl', 'mam300': 'mam300', 'mam_scratch300': 'mam_scratch300', 'mednet': 'mednet', 'sam': 'sam',
         'vgg': 'vgg', 'swin_mam_scratch300': 'swin_mam_scratch300'}


@need('ISBI_ORIG_PROJECTS', 'ISBI_ORIG_ROOT', 'ISBI_ORIG_MX')
class TestCalibration(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def test_norm_grad_vs_original(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3, padding=1), torch.nn.LeakyReLU(), torch.nn.Conv3d(4, 1, 1))
        x = torch.randn(2, 1, 8, 16, 16)
        with original(R2, 'calibrate') as C:
            a, b = norm_grad(net(x).abs().mean(), net), C.norm_grad(net(x).abs().mean(), net)
        report('calibration', 'norm_grad vs original', abs(a - b), value=b)
        self.assertEqual(a, b)

    def test_lambda_rule_and_batches_vs_paper_doses(self):
        R = root()
        ds = Batches(R / 'data', split_cases(REPO / 'configs/split_duke.json', 'train'), 2026101801, 0, 64)
        plans = [json.loads(ds[i]['plan']) for i in range(64)]
        for name, f in DOSES.items():
            paper = json.loads((R / 'doses' / f'{f}.json').read_text())
            ours = json.loads((REPO / 'configs/doses' / f'{f}.json').read_text())
            l1 = [r['L1_norm'] for r in paper['rows']]
            d = [r['AFP_norm'] for r in paper['rows']]
            lam = statistics.median(l1) / statistics.median(d)
            rows = max(max(abs(a['L1_norm'] - b['L1_norm']), abs(a['AFP_norm'] - b['AFP_norm']))
                       for a, b in zip(ours['batch_norms'], paper['rows']))
            plan = sum(r['plan'] != p for r, p in zip(paper['rows'], plans) if 'plan' in r)
            report('calibration', f'{f}: lambda recomputed from the 64 paper rows vs stored lambda',
                   abs(lam - paper['lambda']), rows=len(paper['rows']))
            report('calibration', f'{f}: configs/doses rows vs paper rows', rows)
            report('calibration', f'{f}: calibration batch plans (seed 2026101801) vs paper rows: batches differing',
                   float(plan), compared=sum('plan' in r for r in paper['rows']))
            self.assertEqual((len(paper['rows']), lam, rows, plan, ours['lambda']), (64, paper['lambda'], 0., 0,
                                                                                     paper['lambda']))

    @need('ISBI_NNF_CKPT')
    def test_calibration_loop_vs_original(self):
        """ssl (nnF teacher): our calibrate() vs the loop of the original calibrate.py (r2) on 2 batches at a small
        patch, both fp32 on the CPU."""
        cfg = arm_config('ssl')
        cfg['calibration']['autocast'] = None
        cfg['teacher']['dtype'] = 'float32'
        ds = Batches(root() / 'data', split_cases(REPO / 'configs/split_duke.json', 'train'), 2026101801, 0, 2,
                     patch=(32, 64, 64))
        batches = [ds[0], ds[1]]
        ours = calibrate(cfg, 'cpu', batches=batches, n=2)['rows']
        with original(R2, 'calibrate') as C, cuda_calls_on_cpu():
            C.seed(20261018)
            net = C.generator().cuda().train()
            net.load_state_dict(torch.load(C.ROOT / 'generator_init.pth', map_location='cpu',
                                           weights_only=True)['network_weights'])
            afp = cpu_fp32_teacher(C.teacher)('ssl')
            theirs = []
            for batch in batches:  # verbatim loop body of the original main()
                x = batch['data'][:, 0:1].cuda(non_blocking=True)
                y = batch['data'][:, 1:2].cuda(non_blocking=True)
                valid = batch['valid'].cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = net(x)
                    l1 = C.pixel_l1(pred, y, valid)
                ng_l1 = C.norm_grad(l1, net)
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = net(x)
                    safe = torch.where(valid.bool(), pred.float(), y.detach())
                    distance = afp(safe, y)
                theirs.append((ng_l1, C.norm_grad(distance, net)))
        gap = max(max(abs(a['L1_norm'] - l), abs(a['AFP_norm'] - d)) for a, (l, d) in zip(ours, theirs))
        report('calibration', 'ssl calibration loop vs original (2 batches, 32x64x64, fp32)', gap,
               ours=[(r['L1_norm'], r['AFP_norm']) for r in ours], original=theirs)
        self.assertEqual(gap, 0.)


@need('ISBI_ORIG_ROOT', 'ISBI_NNF_CKPT', 'ISBI_TEST_TMP')
@slow()
class TestCheckpoints(unittest.TestCase):
    def test_restore_paper_checkpoints(self):
        R = root()
        for name, kind, ck_path in (('l1', 'arms', 'results/l1/checkpoint_ep300.pth'),
                                    ('swin_l1', 'arms', 'results/swin_l1/checkpoint_ep250.pth'),
                                    ('gan_nnf', 'arms', 'results/gan_nnf/checkpoint_ep200.pth'),
                                    ('scratch', 'experts', 'results/expert_scratch_c300/checkpoint_ep300.pth')):
            cfg = arm_config(name, kind)
            t = make_trainer(cfg, 'cpu').prepare()
            ck = torch.load(R / ck_path, map_location='cpu', weights_only=False)
            if cfg['trainer'] == 'sgd':
                epoch = CK.restore(ck, t.model, t.opt, t.scaler, torch.device('cpu'))
                gap = max_abs(t.opt.state_dict()['state'], ck['optimizer_state']['state'])
            else:
                epoch = CK.restore(ck, t.model, t.opts[0], t.scaler, torch.device('cpu'), t.disc,
                                   t.opts[1] if t.disc is not None else None, keep_step_on_cpu=True)
                gap = max_abs(t.opts[0].state_dict()['state'], ck['optimizer_state']['state'])
                if t.disc is not None:
                    gap = max(gap, max_abs(t.disc.state_dict(), ck['discriminator_weights']),
                              max_abs(t.opts[1].state_dict()['state'], ck['optimizer_d_state']['state']))
            gap = max(gap, max_abs(t.model.state_dict(), ck['network_weights']),
                      max_abs(torch.get_rng_state(), ck['torch_rng']))
            report('checkpoints', f'{name}: paper {ck_path} restored (weights, optimizer, D, RNG)', gap, epoch=epoch)
            self.assertEqual((epoch, gap), (int(ck['current_epoch']), 0.))
            del t, ck
            gc.collect()

    def test_expert_fork_vs_paper_fork(self):
        R = root()
        env_path('ISBI_TEST_TMP').mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix='fork_', dir=env_path('ISBI_TEST_TMP'))
        try:
            rec = CK.fork(R / 'results/expert_scratch', f'{tmp}/child', 100, 250)
            paper = json.loads((R / 'results/expert_scratch_c300/fork.json').read_text())
            plan = open(f'{tmp}/child/batch_plan.jsonl').read().splitlines()
            ref = (R / 'results/expert_scratch_c300/batch_plan.jsonl').read_text().splitlines()[:len(plan)]
            metrics = open(f'{tmp}/child/epoch_metrics.jsonl').read().splitlines()
            ref_m = (R / 'results/expert_scratch_c300/epoch_metrics.jsonl').read_text().splitlines()[:100]
            differ = sum(a != b for a, b in zip(plan, ref)) + sum(a != b for a, b in zip(metrics, ref_m))
            report('checkpoints', 'fork expert_scratch ep100: copy sha vs paper fork.json; plan / metrics lines differing',
                   float(differ), sha_equal=rec['copy_sha256'] == paper['parent_ep100_sha256'], plan_lines=len(plan))
            self.assertEqual((rec['copy_sha256'], len(plan), len(metrics), differ),
                             (paper['parent_ep100_sha256'], 25000, 100, 0))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
