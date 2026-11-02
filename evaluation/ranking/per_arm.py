"""Per-arm values (one row per arm, one column per metric of configs/eval/families.yaml) from the per-patient CSVs.

Aggregation per metric (families.yaml `aggregate`): mean | median | pct_gt1 | ratio_of_sums (sum of the `numerator`
column over patients divided by the sum of the `denominator` column, e.g. the paper's r), over the patients where the metric exists
(missing values are left out, e.g. LPIPS_lesion when no kept slice holds lesion voxels, or the edge ratio when the
lesion or its 2 mm band has fewer than 64 voxels). The number of patients used is written as n_<metric> when it is
below the cohort size.

usage: per_arm.py --per-patient DIR --out per_arm.csv [--families configs/eval/families.yaml] [--config configs/eval/test91.yaml]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def load_per_patient(folder):
    data = {}
    for f in sorted(Path(folder).glob('*.csv')):
        with f.open() as h:
            data[f.stem] = list(csv.DictReader(h))
    return data


def values(rows, arm, column):
    v = [r[column] for r in rows if r['arm'] == arm and column in r]
    return np.array([float(x) for x in v if x not in ('', None)], dtype=float), len(v)


def aggregate(v, how):
    if v.size == 0:
        return None
    if how == 'mean':
        return float(v.mean())
    if how == 'median':
        return float(np.median(v))
    if how == 'pct_gt1':
        return float((v > 1).mean() * 100)
    raise ValueError(how)


def ratio_of_sums(rows, arm, numerator, denominator):
    """Sum over patients of `numerator` / the same sum of `denominator`, over the patients where both exist."""
    v = [(r[numerator], r[denominator]) for r in rows if r['arm'] == arm and r.get(numerator) not in ('', None)
         and r.get(denominator) not in ('', None)]
    if not v:
        return None, 0
    v = np.array(v, dtype=float)
    return float(v[:, 0].sum() / v[:, 1].sum()), len(v)


def per_arm(per_patient, fam, cfg):
    rows = []
    for a in cfg['arms'] + cfg['references']:
        key = a['key']
        row = {'arm': key, 'label': a['label'], 'group': a.get('group', 'reference'), 'epoch': a.get('epoch', ''),
               'ranked': int(key not in {r['key'] for r in cfg['references']})}
        for m, spec in fam['metrics'].items():
            if spec['aggregate'] == 'ratio_of_sums':
                row[m], _ = ratio_of_sums(per_patient[spec['source']], key, spec['numerator'], spec['denominator'])
                continue
            v, n_rows = values(per_patient[spec['source']], key, spec['column'])
            row[m] = aggregate(v, spec['aggregate'])
            if 0 < v.size < n_rows:
                row[f'n_{m}'] = int(v.size)
        rows.append(row)
    return rows


def write(rows, path, fam):
    metric_cols = list(fam['metrics'])
    extra = sorted({k for r in rows for k in r if k.startswith('n_')})
    fields = ['arm', 'label', 'group', 'epoch', 'ranked'] + metric_cols + extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else (repr(r[k]) if isinstance(r.get(k), float) else r.get(k, ''))) for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--per-patient', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--families', default=str(REPO / 'configs' / 'eval' / 'families.yaml'))
    p.add_argument('--config', default=str(REPO / 'configs' / 'eval' / 'test91.yaml'))
    a = p.parse_args()
    fam = yaml.safe_load(Path(a.families).read_text()); cfg = yaml.safe_load(Path(a.config).read_text())
    rows = per_arm(load_per_patient(a.per_patient), fam, cfg)
    write(rows, a.out, fam)
    print(f'{a.out}: {len(rows)} rows, {len(fam["metrics"])} metrics')


if __name__ == '__main__':
    main()
