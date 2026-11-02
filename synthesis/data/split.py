"""DUKE split of MAMA-MIA: the 91 official test cases are kept; the 200 official training cases are split 160 train /
40 dev, stratified by magnetic field strength, in a seeded hash order.

The paper split is shipped as ``configs/split_duke.json`` (patient IDs only). This module regenerates it from the
MAMA-MIA release (``train_test_splits.csv`` and ``patient_info_files/<case>.json``, field ``imaging_data.field_strength``)
and writes a manifest with the image / mask paths used by ``synthesis.data.preprocess``.

Usage:
    python -m synthesis.data.split --mama-root /path/to/MAMA-MIA --out split_manifest.json [--check configs/split_duke.json]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from synthesis.core.io import read_json, save_json

SEED = 20261018
N_TRAIN, N_DEV, N_TEST, N_OFFICIAL_TRAIN = 160, 40, 91, 200


def field_strength(mama_root, case):
    return read_json(Path(mama_root) / 'patient_info_files' / f'{case}.json')['imaging_data']['field_strength']


def split_ids(official_train, official_test, field_of, seed=SEED, n_dev=N_DEV):
    """official_train / official_test: DUKE case IDs; field_of: case -> field strength. Returns train, dev, test."""
    tr, te = sorted(official_train), sorted(official_test)
    if set(tr) & set(te):
        raise ValueError('official train and test overlap')
    groups = {}
    for case in tr:
        groups.setdefault(str(field_of(case)), []).append(case)
    dev, assigned, keys = [], 0, sorted(groups)
    for j, field in enumerate(keys):
        ids = sorted(groups[field], key=lambda c: hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest())
        n = n_dev - assigned if j == len(keys) - 1 else round(len(ids) * .2)
        dev.extend(ids[:n])
        assigned += n
    train, dev = sorted(set(tr) - set(dev)), sorted(dev)
    return train, dev, te


def official_duke(mama_root):
    rows = list(csv.DictReader((Path(mama_root) / 'train_test_splits.csv').open()))
    tr = [r['train_split'] for r in rows if r['train_split'].startswith('DUKE_')]
    te = [r['test_split'] for r in rows if r['test_split'].startswith('DUKE_')]
    return tr, te


def records(mama_root, ids, field_of):
    root = Path(mama_root)
    return [{'patient': c, 'pre': str(root / 'images' / c / f'{c}_0000.nii.gz'),
             'post': str(root / 'images' / c / f'{c}_0001.nii.gz'),
             'label': str(root / 'segmentations' / 'expert' / f'{c}.nii.gz'),
             'field_strength': field_of(c)} for c in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mama-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--check', help='split JSON (patient IDs) that the regenerated split must equal')
    a = p.parse_args()
    tr, te = official_duke(a.mama_root)
    if len(tr) != N_OFFICIAL_TRAIN or len(te) != N_TEST:
        raise ValueError(f'expected {N_OFFICIAL_TRAIN} / {N_TEST} official DUKE cases, got {len(tr)} / {len(te)}')
    field_of = lambda c: field_strength(a.mama_root, c)  # noqa: E731
    train, dev, test = split_ids(tr, te, field_of)
    assert len(train) == N_TRAIN and len(dev) == N_DEV
    if a.check:
        ref = read_json(a.check)['splits']
        if {'train': train, 'dev': dev, 'test': test} != {k: sorted(ref[k]) for k in ('train', 'dev', 'test')}:
            raise SystemExit('regenerated split differs from ' + a.check)
    manifest = {'seed': SEED, 'task': 'DUKE pre-contrast 0000 -> first post-contrast 0001, 3D',
                'split_method': 'official 91 test kept; official 200 train stratified by field strength into 160/40, '
                                'sha256(seed:case) order',
                'splits': {name: records(a.mama_root, ids, field_of)
                           for name, ids in (('train', train), ('dev', dev), ('test', test))}}
    save_json(a.out, manifest)
    print(json.dumps({k: len(v) for k, v in manifest['splits'].items()}))


if __name__ == '__main__':
    main()
