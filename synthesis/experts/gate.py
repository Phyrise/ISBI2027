"""Quality gate of a lesion expert, fixed before training: on the 40 dev cases, mean Dice >= 0.5 and at most 4 empty
predictions. An AFP arm may use an expert as teacher only if its gate passed (paper: both experts passed at epoch 300;
the scratch expert had failed at epoch 100, Dice 0.376, and was continued to 300).

Dev evaluation: full-volume sliding-window softmax of the expert on the z-scored post-contrast (synthesis.inference.predict,
float16), lesion = softmax > 0.5, compared with the resampled expert mask.

    python -m synthesis.experts.gate --config configs/experts/scratch.yaml --epoch 300 [--device cuda] [--out gate.json]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from synthesis.core.config import load_config, path, torch_dtype
from synthesis.core.io import save_json, sha256
from synthesis.data.sampler import split_cases
from synthesis.inference.predict import predict

VOXEL_ML = .7 * .7 * 1 / 1000
SPACING_DHW = (1., .7, .7)


def segmentation_metrics(pred, gt):
    p = pred > 0.5
    g = gt > 0
    tp, a, b = int((p & g).sum()), int(p.sum()), int(g.sum())
    dice = 2 * tp / (a + b) if a + b else 1.
    hd = None
    if a and b:
        pp = p ^ ndi.binary_erosion(p)
        gg = g ^ ndi.binary_erosion(g)
        d1 = ndi.distance_transform_edt(~gg, sampling=SPACING_DHW)[pp]
        d2 = ndi.distance_transform_edt(~pp, sampling=SPACING_DHW)[gg]
        hd = float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
    return {'Dice': dice, 'HD95_mm': hd, 'prediction_empty': a == 0, 'true_lesion_ml': b * VOXEL_ML,
            'predicted_lesion_ml': a * VOXEL_ML, 'false_positive_volume_ml': int((p & ~g).sum()) * VOXEL_ML,
            'false_negative_volume_ml': int((~p & g).sum()) * VOXEL_ML}


def gate(rows, min_dice=.5, max_empty=4, n_expected=40):
    dice = float(np.mean([r['Dice'] for r in rows]))
    empty = int(sum(r['prediction_empty'] for r in rows))
    return {'passed': len(rows) == n_expected and dice >= min_dice and empty <= max_empty,
            'rule': f'dev{n_expected}: mean Dice >= {min_dice} and <= {max_empty} empty predictions',
            'mean_Dice': dice, 'n_empty': empty, 'n': len(rows)}


def main():
    p = argparse.ArgumentParser(description='Dev evaluation and quality gate of a lesion expert.')
    p.add_argument('--config', required=True)
    p.add_argument('--epoch', type=int, default=300)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out')
    a = p.parse_args()
    from synthesis.trainers.build import load_for_inference
    cfg = load_config(a.config)
    run = path(cfg['paths']['runs_dir'], 'paths.runs_dir') / cfg['run_name']
    ck = run / f'checkpoint_ep{a.epoch:03d}.pth'
    model = load_for_inference(cfg, ck).to(a.device).eval()
    data_dir = path(cfg['paths']['data_dir'], 'paths.data_dir')
    dtype = torch_dtype(cfg['precision']['inference_autocast'])
    rows = []
    for case in split_cases(path(cfg['paths']['split'], 'paths.split'), 'dev'):
        t = time.perf_counter()
        d = data_dir / case
        data = np.load(d / 'data.npy', mmap_mode='r')
        pred, _ = predict(model, np.array(data[1], copy=True), True, dtype, a.device)
        m = segmentation_metrics(pred, np.load(d / 'label.npy', mmap_mode='r'))
        rows.append(dict(m, patient=case))
        print(json.dumps({'patient': case, 'Dice': m['Dice'], 'seconds': time.perf_counter() - t}), flush=True)
    g = cfg['gate']
    result = dict(gate(rows, g['min_mean_dice'], g['max_empty'], g['n_cases']), run=cfg['run_name'], epoch=a.epoch,
                  checkpoint_sha256=sha256(ck), rows=rows)
    save_json(Path(a.out) if a.out else run / f'quality_ep{a.epoch:03d}.json', result)
    print('GATE', json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    torch.set_num_threads(3)
    main()
