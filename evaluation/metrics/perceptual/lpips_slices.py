"""LPIPS on axial slices (perceptual family): LPIPS_slice, LPIPS_body, LPIPS_lesion (Table 2 "LPIPS_L").

lpips 0.1.4, LPIPS(net='vgg', version='0.1', spatial=True): VGG16 ImageNet backbone with the v0.1 linear weights shipped
in the package; no download beyond the torchvision VGG16 weights.
Per patient: axial slices (axis D) kept if >= 5 % of their pixels are body; window = percentiles 0.5 / 99.5 of the
real post-contrast inside the body; real and synthetic clipped to the window and mapped to [-1, 1]; 3 replicated
channels, native slice size; the spatial LPIPS map (upsampled to the slice size by the package) gives
  LPIPS_slice  = mean over kept slices of the map mean (standard per-slice LPIPS),
  LPIPS_body   = mean of the map over the body pixels of the kept slices,
  LPIPS_lesion = mean of the map over the lesion pixels of the kept slices (None if there are none).
Lower is better. The VGG16 arm has VGG16 ImageNet as its feature-loss teacher: LPIPS-VGG is close to its teacher.

usage: python -m evaluation.metrics.perceptual.lpips_slices ENTRY... --data DIR --pred-dir DIR --out DIR [--device cuda|cpu]
"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from evaluation.metrics import io

BODY_FRAC = 0.05
BATCH = 8


def build(device):
    import lpips
    net = lpips.LPIPS(net='vgg', version='0.1', spatial=True, verbose=False).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    w = Path(lpips.__file__).parent / 'weights' / 'v0.1' / 'vgg.pth'
    prov = {'lpips_file': lpips.__file__, 'net': 'vgg', 'version': '0.1', 'spatial': True,
            'lin_weights_sha256': hashlib.sha256(w.read_bytes()).hexdigest()}
    return net, prov


def lpips_slices(net, pred, real, body, les, device):
    import torch
    lo, hi = np.percentile(real[body], [0.5, 99.5])
    keep = [z for z in range(real.shape[0]) if body[z].mean() >= BODY_FRAC]
    scal = lambda v: np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 2 - 1
    sl, bs, bn, ls_, ln = [], 0.0, 0, 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(keep), BATCH):
            zs = keep[i:i + BATCH]
            x = torch.from_numpy(scal(pred[zs])[:, None]).float().to(device).repeat(1, 3, 1, 1)
            y = torch.from_numpy(scal(real[zs])[:, None]).float().to(device).repeat(1, 3, 1, 1)
            m = net(x, y)[:, 0].double().cpu().numpy(); assert m.shape == pred[zs].shape
            sl += [float(v) for v in m.mean((1, 2))]
            bs += float(m[body[zs]].sum()); bn += int(body[zs].sum()); ls_ += float(m[les[zs]].sum()); ln += int(les[zs].sum())
    lz = [z for z in keep if les[z].any()]
    return {'window_z': [float(lo), float(hi)], 'slices_body': len(keep), 'slices_lesion': len(lz),
            'metrics': {'LPIPS_slice': float(np.mean(sl)), 'LPIPS_body': bs / bn, 'LPIPS_lesion': ls_ / ln if ln else None}}


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('entries', nargs='+'); p.add_argument('--data', required=True); p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', default='cuda'); p.add_argument('--cases', nargs='*')
    a = p.parse_args()
    torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device); net, prov = build(dev)
    for case in a.cases or io.cases(a.data):
        todo = [e for e in a.entries if not (Path(a.out) / e / f'{case}.json').exists()]
        if not todo:
            continue
        pre, post, body, label, _ = io.load_case(Path(a.data) / case)
        body, les = body > 0, label > 0
        for e in todo:
            t0 = time.perf_counter()
            row = lpips_slices(net, io.synthetic(e, case, pre, post, a.pred_dir), post, body, les, dev)
            io.save_json(Path(a.out) / e / f'{case}.json', {'patient': case, 'entry': e, **row, 'provenance': prov,
                                                           'seconds': time.perf_counter() - t0})
        print(case, todo, flush=True)


if __name__ == '__main__':
    main()
