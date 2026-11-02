"""PSNR over the body mask, paper definition.

PSNR_body = 10 log10(R^2 / MSE_body), MSE_body = mean over body voxels of (Y_hat - Y)^2, and
R = max(Y) - min(Y) over the body voxels of the REAL post-contrast (z units, per patient).
PSNR_volume = the same over the whole volume, with R from the whole real volume (descriptive).

This is the only PSNR of the paper (copy 26.05 dB, L1 28.85 dB on test91). An earlier test91 table (table.json /
READOUT.md of the evaluation driver) used a fixed range R = 20 z instead (copy 24.79, L1 27.59); that value is not
the paper's and is not reproduced here.

usage: python -m evaluation.metrics.global_.psnr --data DIR --pred-dir DIR --out DIR ENTRY... [--workers N]
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from evaluation.metrics import io


def psnr(pred, post, body):
    """dict(MSE_body, PSNR_body, MSE_volume, PSNR_volume, range_body, range_volume)."""
    body = body > 0
    rb = float(post[body].max() - post[body].min()); rv = float(post.max() - post.min())
    e2 = (pred - post) ** 2; mb = float(e2[body].mean()); mv = float(e2.mean())
    return {'MSE_body': mb, 'PSNR_body': 10 * np.log10(rb ** 2 / mb), 'MSE_volume': mv, 'PSNR_volume': 10 * np.log10(rv ** 2 / mv),
            'range_body': rb, 'range_volume': rv}


def one(case, data, pred_dir, out, entries):
    pre, post, body, _, _ = io.load_case(Path(data) / case)
    done = 0
    for e in entries:
        o = Path(out) / e / f'{case}.json'
        if o.exists():
            continue
        row = {'case': case, 'arm': e, **psnr(io.synthetic(e, case, pre, post, pred_dir), post, body)}
        row = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}
        io.save_json(o, row); done += 1
    return case, done


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('entries', nargs='+'); p.add_argument('--data', required=True); p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    with Pool(a.workers) as pool:
        for case, n in pool.imap_unordered(partial(one, data=a.data, pred_dir=a.pred_dir, out=a.out, entries=a.entries), io.cases(a.data)):
            print(case, n, flush=True)


if __name__ == '__main__':
    main()
