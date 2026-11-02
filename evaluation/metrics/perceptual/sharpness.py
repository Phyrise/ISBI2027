"""Sharpness and fine structure of the synthetic post-contrast (perceptual family).

Voxels 1.0 x 0.7 x 0.7 mm (D, H, W); z units; real = post-contrast; body = body.npy > 0; lesion = label.npy > 0.
Gradients in z/mm (numpy.gradient with the spacing). d = signed distance to the lesion surface in mm (> 0 outside),
computed in the lesion bounding box + 12 mm (exact for |d| <= 12 mm).
1. body_gradient_ratio = mean |grad Y_hat| / mean |grad Y| over the body (1 = as sharp as the real image, < 1 = blur).
2. edge_sharpness_ratio = the same ratio on the band |d| <= 2 mm inside the body (Table 2 "|grad|", target 1);
   edge_sharpness_deficit = |1 - ratio|. Needs a lesion >= 64 voxels and a band >= 64 voxels, else None.
3. rim_gradient_error = mean over |d| <= 3 mm inside the body of (1/3) sum_a |D_a Y_hat - D_a Y| / h_a (forward
   differences, z/mm).
4. HFEN = ||LoG(Y_hat) - LoG(Y)||_2 / ||LoG(Y)||_2, LoG in mm (sigma = 1.5 mm per axis, sum_a d2_a / h_a^2), over the
   body (HFEN_body) and over {d <= 3 mm} inside the body = lesion dilated by 3 mm (HFEN_lesion).
5. hf_energy_ratio_body = ||LoG(Y_hat)||_2 / ||LoG(Y)||_2 over the body (< 1 = missing high frequencies).
Lower is closer to the real image for the deficit, the rim error and HFEN; target 1 for the ratios.

usage: python -m evaluation.metrics.perceptual.sharpness run ENTRY... --data DIR --pred-dir DIR --out DIR
       python -m evaluation.metrics.perceptual.sharpness selftest
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from evaluation.metrics import io

SP = np.array([1.0, 0.7, 0.7]); SIGMA_MM = 1.5; MARGIN_MM = 12.0
METRICS = ('body_gradient_ratio', 'edge_sharpness_ratio', 'edge_sharpness_deficit', 'rim_gradient_error',
           'HFEN_body', 'HFEN_lesion', 'hf_energy_ratio_body')


def grad_mag(v):
    g = np.gradient(v, *SP)
    return np.sqrt(sum(x * x for x in g))


def log_mm(v):
    s = SIGMA_MM / SP; out = np.zeros_like(v)
    for a in range(3):
        order = [0, 0, 0]; order[a] = 2
        out += ndi.gaussian_filter(v, s, order=order) / SP[a] ** 2
    return out


def context(real, lesion):
    """Quantities of the real image shared by every synthetic volume of a patient."""
    ctx = {'grad_real': grad_mag(real), 'log_real': log_mm(real)}
    if lesion.sum() >= 64:
        idx = np.argwhere(lesion); m = np.ceil(MARGIN_MM / SP).astype(int)
        lo = np.maximum(idx.min(0) - m, 0); hi = np.minimum(idx.max(0) + m + 1, lesion.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)); les = lesion[box]
        ctx['box'] = box
        ctx['sd'] = ndi.distance_transform_edt(~les, sampling=SP) - ndi.distance_transform_edt(les, sampling=SP)
    return ctx


def metrics(pred, real, mask, ctx):
    out = dict.fromkeys(METRICS)
    gp = grad_mag(pred); gr = ctx['grad_real']
    out['grad_pred_body_mean'] = float(gp[mask].mean()); out['grad_real_body_mean'] = float(gr[mask].mean())
    out['body_gradient_ratio'] = out['grad_pred_body_mean'] / max(out['grad_real_body_mean'], 1e-8)
    lp = log_mm(pred); lr = ctx['log_real']; nr = float(np.linalg.norm(lr[mask]))
    out['HFEN_body'] = float(np.linalg.norm((lp - lr)[mask]) / max(nr, 1e-8))
    out['hf_energy_ratio_body'] = float(np.linalg.norm(lp[mask]) / max(nr, 1e-8))
    if 'sd' in ctx:
        b = ctx['box']; sd = ctx['sd']; m = mask[b]
        band = (np.abs(sd) <= 2) & m
        if band.sum() >= 64:
            r = float(gp[b][band].mean() / max(gr[b][band].mean(), 1e-8))
            out['edge_sharpness_ratio'] = r; out['edge_sharpness_deficit'] = abs(1 - r)
        rim = (np.abs(sd) <= 3) & m
        if rim.sum() >= 64:
            pb, rb = pred[b], real[b]
            err = sum(np.abs(np.diff(pb, axis=a, append=pb.take([-1], axis=a)) - np.diff(rb, axis=a, append=rb.take([-1], axis=a)))
                      / SP[a] for a in range(3)) / 3.0
            out['rim_gradient_error'] = float(err[rim].mean())
        les3 = (sd <= 3) & m
        if les3.sum() >= 64:
            out['HFEN_lesion'] = float(np.linalg.norm((lp[b] - lr[b])[les3]) / max(np.linalg.norm(lr[b][les3]), 1e-8))
    return out


def run(entries, data, pred_dir, out):
    for case in io.cases(data):
        todo = [e for e in entries if not (Path(out) / e / f'{case}.json').exists()]
        if not todo:
            continue
        pre, post, body, label, _ = io.load_case(Path(data) / case)
        body, les = body > 0, label > 0; ctx = context(post, les)
        for e in todo:
            t0 = time.perf_counter(); m = metrics(io.synthetic(e, case, pre, post, pred_dir), post, body, ctx)
            io.save_json(Path(out) / e / f'{case}.json', {'patient': case, 'entry': e, 'metrics': {k: m[k] for k in METRICS},
                                                         'seconds': time.perf_counter() - t0})
        print(case, todo, flush=True)


def selftest():
    """Definition checks on synthetic volumes (identity, blur, noise, signed distance, LoG of a paraboloid)."""
    rng = np.random.default_rng(0)
    real = ndi.gaussian_filter(rng.normal(size=(48, 64, 64)).astype(np.float32), 1.0) * 5
    lesion = np.zeros(real.shape, bool); lesion[18:30, 24:40, 24:40] = True; real[lesion] += 3
    mask = np.ones(real.shape, bool); mask[:, :4] = False
    ctx = context(real, lesion)
    same = metrics(real.copy(), real, mask, ctx)
    assert abs(same['body_gradient_ratio'] - 1) < 1e-6 and same['HFEN_body'] < 1e-6 and same['rim_gradient_error'] < 1e-6
    assert abs(same['edge_sharpness_ratio'] - 1) < 1e-6 and same['HFEN_lesion'] < 1e-6 and abs(same['hf_energy_ratio_body'] - 1) < 1e-6
    blur = metrics(ndi.gaussian_filter(real, 1.5), real, mask, ctx)
    assert blur['body_gradient_ratio'] < 0.9 and blur['edge_sharpness_ratio'] < 0.9 and blur['hf_energy_ratio_body'] < 0.9
    assert blur['HFEN_body'] > 0.1 and blur['rim_gradient_error'] > 0
    noisy = metrics(real + rng.normal(scale=2, size=real.shape).astype(np.float32), real, mask, ctx)
    assert noisy['body_gradient_ratio'] > 1 and noisy['hf_energy_ratio_body'] > 1
    sd = ctx['sd']; b = ctx['box']
    assert (sd[lesion[b]] < 0).all() and (sd[~lesion[b]] > 0).all()
    assert abs(sd[lesion[b]].max() + 0.7) < 1e-6 and abs(sd[~lesion[b]].min() - 0.7) < 1e-6
    # LoG in mm: the Laplacian of x^2 + y^2 + z^2 (mm) is 6 away from the borders. Measured 5.975-5.987: ~0.4 % bias of
    # scipy's Gaussian kernel truncated at 4 sigma (same kernel for prediction and real image); tolerance 1 %.
    zz, yy, xx = np.meshgrid(*[np.arange(n) * h for n, h in zip((40, 40, 40), SP)], indexing='ij')
    q = ((zz - 20) ** 2 + (yy - 14) ** 2 + (xx - 14) ** 2).astype(np.float64)
    lap = log_mm(q)[15:25, 15:25, 15:25]
    assert np.allclose(lap, 6, rtol=0, atol=0.06), (lap.min(), lap.max())
    return {'identity': same, 'blur_sigma1.5vox': blur, 'noise_sd2': noisy, 'paraboloid_laplacian': [float(lap.min()), float(lap.max())]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['run', 'selftest']); p.add_argument('entries', nargs='*')
    p.add_argument('--data'); p.add_argument('--pred-dir'); p.add_argument('--out')
    a = p.parse_args()
    if a.cmd == 'selftest':
        r = selftest(); print('SELFTEST_PASSED paraboloid Laplacian', r['paraboloid_laplacian'])
    else:
        run(a.entries, a.data, a.pred_dir, a.out)


if __name__ == '__main__':
    main()
