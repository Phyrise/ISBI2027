"""Paired patient bootstrap (one implementation for every contrast of the paper).

For two arms A and B and one metric: d_i = A_i - B_i over the patients where both exist; the bootstrap resamples the
patients with replacement (`draws` resamples of size n), and the 95 % CI is the 2.5 / 97.5 percentiles of the
resampled means. The CI "excludes 0" when lo > 0 or hi < 0. One seed per model, no multiplicity correction:
the CI describes patient sampling only, not training variability.

Table 2 daggers (paired contrasts, configs/eval/table2.yaml): seed 20261029, 5000 draws. With `reseed: true` (the
paper) every cell draws from a fresh numpy.random.default_rng(seed); otherwise ONE generator is shared by all cells
and consumed in table order (row by row, then metric by metric), so the row and metric order changes the CIs in the
last digits. With `r_statistic: ratio_of_sums` (the paper), r is the difference of the two ratios of sums
(sum of predicted / sum of real lesion enhancement) recomputed on each resample, instead of a mean of per-patient
differences. A dagger marks a CI that excludes 0 in the `better` direction of the metric. Cells with fewer than
`min_n` paired patients are printed as \\pend and consume no draw.

The development readouts used another convention (a fresh default_rng(0) per comparison, 5000 draws) plus a FRAGILE
flag (`fragile()`: removing one patient changes whether the CI excludes 0; the patient stays in the number).

usage: bootstrap.py table2 --per-patient DIR [--config configs/eval/table2.yaml] [--style macro|plain|dagger]
       bootstrap.py pair A B METRIC --per-patient DIR [--seed 0] [--draws 5000]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]

# metric name -> (per-patient CSV, column)
SOURCES = {'MAE_lesion_z': ('lesion', 'MAE_lesion_z'), 'MAE_body_z': ('global', 'MAE_body_z'), 'SSIM_body': ('global', 'SSIM_body'),
           'PSNR_body_dB': ('global', 'PSNR_body_dB'), 'r': ('lesion', 'r'), 'MAE_ring_z': ('lesion', 'MAE_ring_z'),
           'foci': ('offtarget', 't2_FE_foci_ge_0.1ml'), 'FE_ml': ('offtarget', 't2_FE_ml'), 'Dice': ('segmenter', 'Dice'),
           'FP_ml': ('segmenter', 'false_positive_volume_ml'), 'DINO': ('perceptual', 'MedDINO_CS_448_normed'),
           'LPIPS_lesion': ('perceptual', 'LPIPS_lesion'), 'edge_ratio': ('perceptual', 'edge_sharpness_ratio')}


def paired_bootstrap(d, rng, draws=5000):
    """Mean of d and its percentile 95 % CI; `rng` is consumed (draws x n integers)."""
    d = np.asarray(d, dtype=float)
    means = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def excludes_zero(lo, hi):
    return lo > 0 or hi < 0


def fragile(d, seed=0, draws=5000):
    """Indices i such that dropping patient i changes whether the CI excludes 0 (fresh default_rng(seed) each time)."""
    d = np.asarray(d, dtype=float)
    ref = excludes_zero(*paired_bootstrap(d, np.random.default_rng(seed), draws)[1:])
    return [i for i in range(len(d))
            if excludes_zero(*paired_bootstrap(np.delete(d, i), np.random.default_rng(seed), draws)[1:]) != ref]


def per_patient(folder, metric):
    name, col = SOURCES[metric]
    out = {}
    with open(Path(folder) / f'{name}.csv') as f:
        for r in csv.DictReader(f):
            if r.get(col, '') != '':
                out.setdefault(r['arm'], {})[r['case']] = float(r[col])
    return out


def differences(folder, a, b, metric):
    v = per_patient(folder, metric)
    x, y = v.get(a, {}), v.get(b, {})
    cases = sorted(set(x) & set(y))
    return cases, np.array([x[c] - y[c] for c in cases])


def enhancements(folder, arm):
    """case -> (predicted, real) mean lesion enhancement (z) of one arm, from lesion.csv."""
    with open(Path(folder) / 'lesion.csv') as f:
        return {r['case']: (float(r['lesion_enhancement_prediction_z']), float(r['lesion_enhancement_real_z']))
                for r in csv.DictReader(f) if r['arm'] == arm}


def ratio_bootstrap(folder, a, b, rng, draws=5000):
    """Difference of the ratios of sums (paper r) of A and B and its percentile 95 % CI over resampled patients."""
    x, y = enhancements(folder, a), enhancements(folder, b)
    cases = sorted(set(x) & set(y))
    xa, ya = np.array([x[c] for c in cases]), np.array([y[c] for c in cases])
    idx = rng.integers(0, len(cases), (draws, len(cases)))
    est = xa[:, 0].sum() / xa[:, 1].sum() - ya[:, 0].sum() / ya[:, 1].sum()
    res = xa[idx, 0].sum(1) / xa[idx, 1].sum(1) - ya[idx, 0].sum(1) / ya[idx, 1].sum(1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return cases, (float(est), float(lo), float(hi))


def contrast(folder, a, b, metric, cfg, rng):
    """(n paired patients, (mean difference, lo, hi)) of one Table 2 cell; None when fewer than min_n patients."""
    if metric == 'r' and cfg.get('r_statistic') == 'ratio_of_sums':
        cases, out = ratio_bootstrap(folder, a, b, rng, cfg['draws'])
        return len(cases), out
    cases, d = differences(folder, a, b, metric)
    if len(cases) < cfg['min_n']:
        return len(cases), None
    return len(cases), paired_bootstrap(d, rng, cfg['draws'])


def table2(folder, cfg, style='macro'):
    shared = np.random.default_rng(cfg['seed'])
    lines = []
    for row in cfg['rows']:
        cells = []
        for m in cfg['metrics']:
            rng = np.random.default_rng(cfg['seed']) if cfg.get('reseed') else shared
            n, out = contrast(folder, row['a'], row['b'], m['name'], cfg, rng)
            if out is None:
                cells.append('\\pend'); continue
            mean, lo, hi = out
            f = f"{{:+.{m['decimals']}f}}"
            bold = excludes_zero(lo, hi)
            if style == 'dagger':
                better = (lo > 0) if m.get('better') == 'higher' else (hi < 0)
                cells.append('\\dagger' if better else '')
            elif style == 'macro':
                cells.append(f"\\{'cib' if bold else 'ci'}{{{f.format(mean)}}}{{{f.format(lo)}}}{{{f.format(hi)}}}")
            else:
                s = f'{f.format(mean)} [{f.format(lo)}, {f.format(hi)}]'
                cells.append(f'\\textbf{{{s}}}' if bold else s)
        lines.append(row['latex'] + ' & ' + ' & '.join(cells) + ' \\\\')
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('table2'); t.add_argument('--per-patient', required=True)
    t.add_argument('--config', default=str(REPO / 'configs' / 'eval' / 'table2.yaml'))
    t.add_argument('--style', choices=['macro', 'plain', 'dagger'], default='macro')
    q = sub.add_parser('pair'); q.add_argument('a'); q.add_argument('b'); q.add_argument('metric', choices=sorted(SOURCES))
    q.add_argument('--per-patient', required=True); q.add_argument('--seed', type=int, default=0)
    q.add_argument('--draws', type=int, default=5000); q.add_argument('--fragile', action='store_true')
    a = p.parse_args()
    if a.cmd == 'table2':
        print('\n'.join(table2(a.per_patient, yaml.safe_load(Path(a.config).read_text()), a.style)))
    else:
        cases, d = differences(a.per_patient, a.a, a.b, a.metric)
        mean, lo, hi = paired_bootstrap(d, np.random.default_rng(a.seed), a.draws)
        print(f'{a.a} - {a.b} {a.metric}: n={len(cases)} mean {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]'
              f" {'excludes 0' if excludes_zero(lo, hi) else 'includes 0'}; A<B {int((d < 0).sum())}/{len(d)}")
        if a.fragile:
            print('fragile:', [cases[i] for i in fragile(d, a.seed, a.draws)])


if __name__ == '__main__':
    main()
