"""Sliding-window inference, segmentation metrics and the expert quality gate vs the originals and the paper files."""
import json
import unittest

import numpy as np
import torch

from synthesis.experts.gate import gate, segmentation_metrics
from synthesis.inference.predict import predict
from tests.train_support import arm_config, cuda_calls_on_cpu, need, original, report, root, slow, strict_fp32

R11 = 'mama_3d_20261018_r11_arch'
R4 = 'mama_3d_20261018_r4_ext'


def tiny(out_channels, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3, padding=1), torch.nn.LeakyReLU(),
                               torch.nn.Conv3d(4, out_channels, 3, padding=1))


@need('ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX')
class TestInference(unittest.TestCase):
    def setUp(self):
        strict_fp32()

    def test_predict_vs_original(self):
        g = np.random.default_rng(0)
        for shape in ((120, 200, 170), (80, 150, 330)):  # one axis shorter than the 96 x 160 x 160 tile
            image = g.normal(size=shape).astype(np.float32)
            image[0, 0, 0] = -1.7
            for is_expert in (False, True):
                model = tiny(2 if is_expert else 1, 3)
                a, info = predict(model, image, is_expert, dtype=None, device='cpu')
                with original(R11, 'evaluate') as E, cuda_calls_on_cpu():
                    b, meta = E.predict(model, image, None, is_expert)
                gap = float(np.abs(a.astype(np.float64) - b).max())
                report('inference', f'sliding window {"expert" if is_expert else "generator"} {shape}', gap,
                       tiles=info['tiles'])
                self.assertEqual((info['tiles'], gap), (meta['tiles'], 0.))

    def test_segmentation_metrics_vs_original(self):
        g = np.random.default_rng(1)
        worst = 0.
        for _ in range(4):
            gt = g.random((40, 60, 60)) > .97
            pred = np.where(g.random((40, 60, 60)) > .5, gt, g.random((40, 60, 60)) > .98).astype(np.float32)
            a = segmentation_metrics(pred, gt)
            with original(R4, 'evaluate') as E:
                b = E.segmentation_metrics(pred, gt)
            self.assertEqual(set(a), set(b))
            worst = max(worst, max(abs(float(a[k]) - float(b[k])) for k in a if a[k] is not None))
        report('inference', 'segmentation metrics (Dice, HD95, volumes) vs original', worst)
        self.assertEqual(worst, 0.)


@need('ISBI_ORIG_ROOT')
class TestGate(unittest.TestCase):
    def test_gate_vs_paper_quality_files(self):
        R = root()
        for run, epoch, quality in (('expert', 100, 'expert_quality.json'),
                                    ('expert_scratch', 100, 'expert_scratch_quality.json'),
                                    ('expert_c300', 300, 'expert_c300_quality_ep300.json'),
                                    ('expert_scratch_c300', 300, 'expert_scratch_c300_quality_ep300.json')):
            ev = R / 'evaluations' / f'{run}_ep{epoch:03d}'
            cases = [r['patient'] for r in json.loads((R / 'manifest.json').read_text())['splits']['dev']]
            rows = [json.loads((ev / f'{c}.json').read_text())['metrics'] for c in cases]
            ours, paper = gate(rows), json.loads((R / quality).read_text())
            gap = abs(ours['mean_Dice'] - paper['mean_Dice'])
            report('gate', f'{run} ep{epoch} gate vs {quality}', gap, passed=ours['passed'], mean_Dice=ours['mean_Dice'],
                   n_empty=ours['n_empty'])
            self.assertEqual((ours['passed'], ours['n_empty'], gap), (paper['passed'], paper['n_empty'], 0.))


@need('ISBI_ORIG_ROOT', 'ISBI_ORIG_PROJECTS', 'ISBI_ORIG_MX', 'ISBI_NNF_CKPT')
class TestPaperPrediction(unittest.TestCase):
    @slow()
    def test_expert_dev_case_cpu_fp32_vs_paper_gpu_fp16(self):
        """Measures (does not hide) the precision gap: CPU fp32 here vs the paper's GPU fp16 inference, one dev case."""
        import nibabel as nib
        from synthesis.trainers.build import load_for_inference
        strict_fp32()
        R, case = root(), 'DUKE_059'
        cfg = arm_config('scratch', 'experts')
        model = load_for_inference(cfg, R / 'results/expert_scratch_c300/checkpoint_ep300.pth')
        data = np.load(R / 'data' / case / 'data.npy', mmap_mode='r')
        label = np.load(R / 'data' / case / 'label.npy')
        pred, info = predict(model, np.array(data[1], copy=True), True, dtype=None, device='cpu')
        ev = R / 'evaluations/expert_scratch_c300_ep300'
        paper = np.asarray(nib.load(ev / f'{case}_prediction.nii.gz').dataobj).transpose(2, 1, 0)
        row = json.loads((ev / f'{case}.json').read_text())
        ours = segmentation_metrics(pred, label)
        prob_gap = float(np.abs(pred.astype(np.float64) - paper).max())
        flips = int(((pred > .5) != (paper > .5)).sum())
        report('inference', f'expert_scratch_c300 ep300 {case}: softmax CPU fp32 vs paper GPU fp16', prob_gap,
               tiles=info['tiles'], voxels_flipped_at_0_5=flips, dice_ours=ours['Dice'], dice_paper=row['metrics']['Dice'])
        report('inference', f'expert_scratch_c300 ep300 {case}: Dice CPU fp32 vs paper GPU fp16',
               abs(ours['Dice'] - row['metrics']['Dice']))
        self.assertEqual(info['tiles'], row['inference']['tiles'])
        self.assertEqual(paper.shape, pred.shape)
