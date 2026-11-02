"""Results-table LaTeX rows (rank and family metrics) from per_arm.csv.

Columns: Rank, then the Table 1 metrics of configs/eval/families.yaml in family order, printed at their Table 1
precision. Per column, among the ranked (synthetic) arms only: \\fst{} (bold) on the best printed value, \\snd{}
(underline) on the second-best distinct printed value; arms whose printed values tie share the mark. "Best" follows
the metric direction (target1: closest to 1). The reference rows (pre-contrast copy, real image) are never marked.
The macros \\fst, \\snd and \\afp are those of the paper preamble.

usage: tables.py per_arm.csv [--round table1|none|mixed] [--out rows.tex]
"""
import argparse
from pathlib import Path

import yaml

try:
    from evaluation.ranking import family_rank as FR
except ImportError:  # run as a script from the ranking folder
    import family_rank as FR

REPO = Path(__file__).resolve().parents[2]

# What the reference rows print. The copy has no off-target row (no synthetic enhancement); for the real image only
# r, foci, Edge and the segmenter are shown, r = 100 % and Edge = 1 by definition.
REFERENCE_SHOWN = {'pre': None, 'real': ('r', 'foci_t2', 'edge_ratio', 'Dice', 'FP_ml')}
DEFINITIONAL = {'real': {'r': '100\\%', 'foci_t2': '0', 'edge_ratio': '1.00'}}


def fmt(v, spec):
    if v is None or v == '':
        return '--'
    v = float(v)
    if spec.get('percent'):
        return f"{v * 100:.{spec['decimals']}f}\\%"
    return f"{v:.{spec['decimals']}f}"


def marks(rows, m, spec):
    """{arm: '\\fst' | '\\snd'} from the printed values of the ranked arms."""
    shown = {r['arm']: fmt(r[m], spec) for r in rows if r['ranked'] == '1' and r[m] != ''}
    val = {k: float(s.rstrip('%').rstrip('\\')) for k, s in shown.items()}
    key = {'higher': lambda x: -x, 'lower': lambda x: x, 'target1': lambda x: abs(1 - x)}[spec['direction']]
    distinct = sorted(set(val.values()), key=key)
    out = {}
    for k, v in val.items():
        if v == distinct[0]:
            out[k] = '\\fst'
        elif len(distinct) > 1 and v == distinct[1]:
            out[k] = '\\snd'
    return out


def table1(rows, fam, cfg, rounding='table1'):
    mets = FR.table1_metrics(fam)
    order = [m for f in fam['families'] for m, s in mets.items() if s['family'] == f]
    res = FR.rank(rows, fam, rounding)
    mk = {m: marks(rows, m, mets[m]) for m in order}
    by = {r['arm']: r for r in rows}
    ncol = 2 + len(order)

    def cells(arm, ranked):
        r = by[arm]; out = []
        if ranked:
            q = res['weighted_rank'][arm]
            out.append({1: '\\fst{1}', 2: '\\snd{2}'}.get(q, str(q)))
        else:
            out.append('--')
        for m in order:
            shown = REFERENCE_SHOWN.get(arm)
            if not ranked and shown is not None and m not in shown:
                out.append('--'); continue
            s = DEFINITIONAL.get(arm, {}).get(m) or fmt(r[m], mets[m])
            out.append(f'{mk[m][arm]}{{{s}}}' if ranked and arm in mk[m] else s)
        return out

    lines = []
    for ref in cfg['references']:
        lines.append(' & '.join([ref['latex']] + cells(ref['key'], False)) + ' \\\\')
    group = None
    for a in cfg['arms']:
        if a['group'] != group:
            group = a['group']
            lines += ['\\midrule', f"\\multicolumn{{{ncol}}}{{l}}{{{cfg['groups_latex'][group]}}} \\\\"]
        lines.append(' & '.join([a['latex']] + cells(a['key'], True)) + ' \\\\')
    return lines, res


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('per_arm')
    p.add_argument('--families', default=str(REPO / 'configs' / 'eval' / 'families.yaml'))
    p.add_argument('--config', default=str(REPO / 'configs' / 'eval' / 'test91.yaml'))
    p.add_argument('--round', choices=['table1', 'none', 'mixed'], default='table1')
    p.add_argument('--out')
    a = p.parse_args()
    fam = yaml.safe_load(Path(a.families).read_text()); cfg = yaml.safe_load(Path(a.config).read_text())
    lines, _ = table1(FR.load(a.per_arm), fam, cfg, a.round)
    text = '\n'.join(lines) + '\n'
    if a.out:
        Path(a.out).write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
