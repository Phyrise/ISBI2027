"""Family-weighted rank of the models (Table 2 "Rank" column).

Rule (paper, Sec. "Evaluation"):
  1. for each Table 1 metric, rank the ranked arms (the copy and the real image are not ranked); ties get the average
     rank; score = value for "lower", -value for "higher", |1 - value| for "target1";
  2. normalise each rank r to (r - 1) / (n - 1), so 0 = best and 1 = worst;
  3. family score = mean of the normalised positions of its metrics;
  4. global score = equal-weight mean of the family scores ("weighted": each family weighs the same whatever its
     number of metrics); global rank = rank of the global score (ties share the lowest rank).
The metric-equal variant (mean over all metrics, each metric weighing the same) is reported next to it.

Precision. `--round table1` first rounds every metric to the precision printed in Table 1 (families.yaml `decimals`,
r in whole percent), so that values that print the same tie. `--round none` ranks the full-precision values.
`--round mixed` reproduces how the 30/10 draft rank was computed by hand: the global, lesion, off-target and segmenter
metrics as printed, the three perceptual metrics at full precision.

usage: family_rank.py per_arm.csv [--round table1|none|mixed] [--families configs/eval/families.yaml]
                      [--out ranks.csv]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]


def table1_metrics(fam):
    return {m: s for m, s in fam['metrics'].items() if s.get('table1')}


def printed(value, spec):
    """Value as printed in Table 1 (r and other percent metrics in whole percent)."""
    if spec.get('percent'):
        return round(value * 100, spec['decimals'])
    return round(value, spec['decimals'])


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def score(v, direction):
    if direction == 'higher':
        return -v
    if direction == 'lower':
        return v
    if direction == 'target1':
        return abs(1 - v)
    raise ValueError(direction)


def rank(rows, fam, rounding='table1'):
    mets = table1_metrics(fam)
    arms = [r for r in rows if r['ranked'] == '1']
    names = [r['arm'] for r in arms]; n = len(names)
    pos, val = {}, {}
    for m, spec in mets.items():
        v = []
        for r in arms:
            x = float(r[m])
            if rounding == 'table1' or (rounding == 'mixed' and spec['family'] != 'perceptual'):
                x = printed(x, spec)
                if spec.get('percent'):
                    x /= 100
            v.append(x)
        val[m] = dict(zip(names, v))
        rk = rankdata([score(x, spec['direction']) for x in v], method='average')
        pos[m] = {k: (q - 1) / (n - 1) for k, q in zip(names, rk)}
    fams = [f for f in fam['families'] if any(s['family'] == f for s in mets.values())]
    fscore = {f: {k: float(np.mean([pos[m][k] for m, s in mets.items() if s['family'] == f])) for k in names} for f in fams}
    weighted = {k: float(np.mean([fscore[f][k] for f in fams])) for k in names}
    equal = {k: float(np.mean([pos[m][k] for m in mets])) for k in names}
    rk = lambda d: dict(zip(names, rankdata([d[k] for k in names], method='min').astype(int).tolist()))
    return {'arms': names, 'metrics': list(mets), 'families': fams, 'values': val, 'position': pos, 'family_score': fscore,
            'family_rank': {f: rk(fscore[f]) for f in fams}, 'weighted_score': weighted, 'weighted_rank': rk(weighted),
            'metric_equal_score': equal, 'metric_equal_rank': rk(equal)}


def write(res, path):
    names = res['arms']
    fields = (['arm', 'rank_weighted', 'score_weighted', 'rank_metric_equal', 'score_metric_equal']
              + [f'family_{f}' for f in res['families']] + [f'family_rank_{f}' for f in res['families']]
              + [f'pos_{m}' for m in res['metrics']])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader()
        for k in names:
            row = {'arm': k, 'rank_weighted': res['weighted_rank'][k], 'score_weighted': repr(res['weighted_score'][k]),
                   'rank_metric_equal': res['metric_equal_rank'][k], 'score_metric_equal': repr(res['metric_equal_score'][k])}
            row.update({f'family_{f}': repr(res['family_score'][f][k]) for f in res['families']})
            row.update({f'family_rank_{f}': res['family_rank'][f][k] for f in res['families']})
            row.update({f'pos_{m}': repr(res['position'][m][k]) for m in res['metrics']})
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('per_arm')
    p.add_argument('--families', default=str(REPO / 'configs' / 'eval' / 'families.yaml'))
    p.add_argument('--round', choices=['table1', 'none', 'mixed'], default='table1')
    p.add_argument('--out')
    a = p.parse_args()
    fam = yaml.safe_load(Path(a.families).read_text())
    res = rank(load(a.per_arm), fam, a.round)
    print(f"{'arm':24s} rank  score  | metric-equal | " + ' '.join(f'{f[:10]:>10s}' for f in res['families']))
    for k in sorted(res['arms'], key=res['weighted_score'].get):
        print(f"{k:24s} {res['weighted_rank'][k]:4d}  {res['weighted_score'][k]:.3f} | {res['metric_equal_rank'][k]:4d} "
              f"{res['metric_equal_score'][k]:.3f}  | " + ' '.join(f"{res['family_score'][f][k]:6.2f}(#{res['family_rank'][f][k]:<2d})"
                                                            for f in res['families']))
    if a.out:
        write(res, a.out)


if __name__ == '__main__':
    main()
