"""Lambda calibration (one rule for every teacher), run once per teacher before training; lambda is then fixed.

    lambda = median_b ||grad_theta L1_b|| / median_b ||grad_theta D_b||

over 64 fixed training batches (sampler seed 2026101801, updates 0..63), at the arm's common random init (the ResUNet
init for every ResUNet teacher, the SwinUNETR init for the Swin arm; never the nnF-encoder init), in train mode.
Gradient norms are taken with an autograd scale of 1024 (then divided out), separately for L1 and for D, each with
its own forward pass. Precision: the forward runs under ``calibration.autocast`` (float16 in the paper); the SwinUNETR
generator additionally runs under ``calibration.generator_autocast`` (bfloat16), as in its training.

In the paper, the dose was reused rather than recalibrated for ssl_nnf (ssl dose), M2 (mam_scratch300 dose) and the
GAN + AFP arm (mam_scratch300 dose); the MedicalNet dose was calibrated with an fp16 generator although that arm then
trained under bf16 (see docs/recipes.md).

Usage: python -m synthesis.losses.calibrate --config configs/arms/<arm>.yaml [--out dose.json] [--device cuda]
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from synthesis.core.amp import autocast
from synthesis.core.config import load_config, path, torch_dtype
from synthesis.core.io import save_json
from synthesis.data.sampler import loader, split_cases
from synthesis.losses.pixel import masked_l1
from synthesis.losses.teachers import build_teacher, needs_step
from synthesis.trainers.build import build_model


def norm_grad(loss, model, scale=1024.):
    grads = torch.autograd.grad(loss * scale, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None))) / scale
    del grads
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('nonfinite or zero calibration gradient')
    return norm


def calibration_model(cfg):
    """The arm's architecture at its common random init (nnF encoder never copied for the calibration)."""
    c = copy.deepcopy(cfg)
    c['init'] = dict(c['init'], nnf_encoder=False)
    return build_model(c)


def calibrate(cfg, device, net=None, teacher=None, batches=None, n=None, workers=2):
    """Returns the dose record. ``net`` / ``teacher`` / ``batches`` can be injected (tests)."""
    device = torch.device(device)
    cal = cfg['calibration']
    n = int(n or cal['batches'])
    record = None
    if net is None:
        net, record = calibration_model(cfg)
    net = net.to(device).train()
    teacher = teacher or build_teacher(cfg['teacher'], device)
    slice_step = needs_step(cfg['teacher'])
    outer, gen = torch_dtype(cal['autocast']), torch_dtype(cal.get('generator_autocast'))
    if batches is None:
        split = path(cfg['paths']['split'], 'paths.split')
        batches = loader(path(cfg['paths']['data_dir'], 'paths.data_dir'), split_cases(split, 'train'),
                         int(cal['seed']), 0, n, workers=workers, pin_memory=device.type == 'cuda')

    def forward(x):
        if gen is None:
            return net(x)
        with autocast(device, gen):
            return net(x).float()

    rows, start = [], time.perf_counter()
    for i, batch in enumerate(batches):
        x = batch['data'][:, 0:1].to(device, non_blocking=True)
        y = batch['data'][:, 1:2].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        with autocast(device, outer):
            pred = forward(x)
            l1 = masked_l1(pred, y, valid)
        ng_l1 = norm_grad(l1, net)
        del pred, l1
        with autocast(device, outer):
            pred = forward(x)
            safe = torch.where(valid.bool(), pred.float(), y.detach())
            if slice_step:
                distance = teacher(safe, y, valid=valid, step=json.loads(batch['plan'])['step'])
            else:
                distance = teacher(safe, y)
        ng_afp = norm_grad(distance, net)
        del pred, distance
        rows.append({'batch': i, 'L1_norm': ng_l1, 'AFP_norm': ng_afp})
        print(json.dumps(rows[-1]), flush=True)
    l1med = float(np.median([r['L1_norm'] for r in rows]))
    afpmed = float(np.median([r['AFP_norm'] for r in rows]))
    dose = l1med / afpmed
    return {'passed': len(rows) == n and bool(np.isfinite(dose)) and 0 < dose < 10000, 'run': cfg['run_name'],
            'lambda': dose, 'rule': 'median ||grad_theta L1|| / median ||grad_theta D||; common random init, '
                                    f'{n} fixed train batches (seed {cal["seed"]})',
            'median_L1_norm': l1med, 'median_AFP_norm': afpmed, 'amp_autograd_scale': 1024.,
            'autocast': cal['autocast'], 'generator_autocast': cal.get('generator_autocast'),
            'init': record, 'teacher': getattr(teacher, 'provenance', None), 'seconds': time.perf_counter() - start,
            'rows': rows}


def main():
    p = argparse.ArgumentParser(description='Calibrate lambda for one arm (median gradient-norm ratio rule).')
    p.add_argument('--config', required=True)
    p.add_argument('--out')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    cfg = load_config(a.config)
    if not cfg.get('teacher'):
        raise SystemExit('this arm has no feature teacher')
    if torch.device(a.device).type == 'cuda':
        torch.cuda.set_device(torch.device(a.device))
    torch.set_num_threads(2)
    out = Path(a.out) if a.out else path(cfg['paths']['runs_dir'], 'paths.runs_dir') / 'doses' / f'{cfg["run_name"]}.json'
    result = calibrate(cfg, a.device, workers=a.workers)
    save_json(out, result)
    print('DOSE', cfg['run_name'], result['lambda'], 'paper value:', cfg['loss'].get('lambda'), flush=True)
    if not result['passed']:
        raise SystemExit('calibration failed')


if __name__ == '__main__':
    main()
