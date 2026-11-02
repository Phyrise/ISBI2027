"""False enhancement outside the lesion (off-target family).

Synthetic enhancement S = Y_hat - X, real enhancement E = Y - X (z units). Off-target region
O = body minus the expert lesion dilated by 5 voxels (6-connectivity iterations: the lesion ring of the lesion family is
excluded). For each threshold tau in (2, 3) z:
  FE = O & (S >= tau) & (E < tau)   false enhancement (absent from the real image)
  TE = O & (S >= tau) & (E >= tau)  true enhancement outside the lesion
  ME = O & (S < tau) & (E >= tau)   missed enhancement outside the lesion
  volumes in mL; FE connected components (26-connectivity) of at least 0.1 mL = false-enhancing foci:
  FE_foci_ge_0.1ml (count, Table 2 "FE" at tau = 2) and FE_foci_ml (their volume).
Inside the lesion: lesion_frac_S / lesion_frac_E = fraction of lesion voxels with S >= tau / E >= tau.
The real image as entry ('real', S = E) gives FE = 0 by construction (control). The pre-contrast copy has S = 0 and is
not measured. A threshold-defined proxy: components are not adjudicated tumours.

usage: python -m evaluation.metrics.offtarget.false_enhancement run ENTRY... --data DIR --pred-dir DIR --out DIR [--workers N]
       python -m evaluation.metrics.offtarget.false_enhancement summary ENTRY... --out DIR
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from evaluation.metrics import io

TAUS = (2.0, 3.0)
EXCLUSION_ITERATIONS = 5
MIN_FOCUS_ML = 0.1
CONNECTIVITY = ndi.generate_binary_structure(3, 3)   # 26-connectivity


def offtarget_region(body, lesion, iterations=EXCLUSION_ITERATIONS):
    return (body > 0) & ~ndi.binary_dilation(lesion > 0, iterations=iterations)


def false_enhancement(S, E, O, lesion, voxel_ml, taus=TAUS, min_ml=MIN_FOCUS_ML):
    """Per-threshold dict (keys 't2', 't3') of FE / TE / ME volumes, foci and lesion fractions."""
    les = lesion > 0; minvox = int(np.ceil(min_ml / voxel_ml)); out = {}
    for t in taus:
        s, e = S >= t, E >= t
        fe = O & s & ~e
        lab, n = ndi.label(fe, structure=CONNECTIVITY)
        sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
        big = sizes[sizes >= minvox]
        out[f't{t:g}'] = {'FE_ml': float(fe.sum() * voxel_ml), 'TE_ml': float((O & s & e).sum() * voxel_ml),
                          'ME_ml': float((O & ~s & e).sum() * voxel_ml), 'FE_foci_ge_0.1ml': int(big.size),
                          'FE_foci_ml': float(big.sum() * voxel_ml), 'lesion_frac_S': float(s[les].mean()),
                          'lesion_frac_E': float(e[les].mean())}
    return out


def one(case, data, pred_dir, out, entries):
    pre, post, body, label, props = io.load_case(Path(data) / case)
    E = post - pre
    vml = float(np.prod(props['spacing_DHW'])) / 1000.
    O = offtarget_region(body, label)
    done = []
    for e in entries:
        o = Path(out) / e / f'{case}.json'
        if o.exists():
            continue
        S = E if e == 'real' else io.load_prediction(pred_dir, e, case) - pre
        io.save_json(o, {'case': case, 'arm': e, 'voxel_ml': vml, **false_enhancement(S, E, O, label, vml)})
        done.append(e)
    return case, done


def summary(out, entries):
    """Per-entry means / medians over the patients present for every entry (paired)."""
    out = Path(out)
    cases = sorted(set.intersection(*[{p.stem for p in (out / e).glob('DUKE_*.json')} for e in entries]))
    X = {e: [json.loads((out / e / f'{c}.json').read_text()) for c in cases] for e in entries}
    res = {'n': len(cases), 'cases': cases, 'arms': {}}
    for t in [f't{x:g}' for x in TAUS]:
        for e in entries:
            g = lambda k: np.array([x[t][k] for x in X[e]], float)
            fe, te, me, nf, fml = g('FE_ml'), g('TE_ml'), g('ME_ml'), g('FE_foci_ge_0.1ml'), g('FE_foci_ml')
            prec = te / np.maximum(te + fe, 1e-9); rec = te / np.maximum(te + me, 1e-9)
            res['arms'].setdefault(e, {})[t] = {
                'FE_ml_mean': float(fe.mean()), 'FE_ml_median': float(np.median(fe)), 'FE_foci_mean': float(nf.mean()),
                'FE_foci_median': float(np.median(nf)), 'FE_foci_ml_mean': float(fml.mean()), 'TE_ml_mean': float(te.mean()),
                'ME_ml_mean': float(me.mean()), 'share_true_median': float(np.median(prec)), 'recall_median': float(np.median(rec)),
                'lesion_frac_S_median': float(np.median(g('lesion_frac_S'))),
                'lesion_frac_E_median': float(np.median(g('lesion_frac_E'))), 'cases_with_foci_pct': float((nf > 0).mean() * 100)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['run', 'summary']); p.add_argument('entries', nargs='+')
    p.add_argument('--data'); p.add_argument('--pred-dir'); p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.cmd == 'run':
        with Pool(a.workers) as pool:
            for case, done in pool.imap_unordered(partial(one, data=a.data, pred_dir=a.pred_dir, out=a.out, entries=a.entries),
                                                  io.cases(a.data)):
                print(case, done, flush=True)
    else:
        res = summary(a.out, a.entries)
        io.save_json(Path(a.out) / 'summary.json', res)
        for e in a.entries:
            s = res['arms'][e]['t2']
            print(f"{e:32s} n={res['n']} tau=2: FE {s['FE_ml_mean']:.2f} mL, foci {s['FE_foci_mean']:.2f} "
                  f"(patients with >= 1: {s['cases_with_foci_pct']:.0f}%)")


if __name__ == '__main__':
    main()
