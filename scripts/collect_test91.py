"""Collect the per-patient outputs of every test91 stage into one CSV per metric family.

Each stage of scripts/eval_test91.sh writes one JSON file per entry and patient (<stage dir>/<entry>/<case>.json).
This script reads them, checks that every entry has all patients, and writes
    <out>/per_patient/{global,lesion,offtarget,segmenter,perceptual}.csv   (one row per entry and patient)
Then evaluation/ranking/per_arm.py turns them into per_arm.csv.

Entries: every arm of configs/eval/test91.yaml at its final epoch ("<key>_ep<epoch>"), the pre-contrast copy ("pre")
and, where a stage measures it, the real post-contrast ("real"). An extra arm (`extra_arms`) without image metrics is
skipped with a note; a Table 2 arm or a patient missing from any stage is an error.

The stage folders default to <eval_out>/<stage>/ (names from configs/eval/test91.yaml); each one can be overridden
on the command line, which is how outputs written under an earlier folder layout are read.

usage: collect_test91.py --eval-out DIR --out DIR [--image-dir DIR] [--image-copy-dir DIR] [--psnr-dir DIR]
                         [--offtarget-dir DIR] [--reader-dir DIR] [--sharpness-dir DIR] [--lpips-dir DIR]
                         [--meddino-dir DIR] [--vit-dir DIR] [--config configs/eval/test91.yaml]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# (per-patient CSV, stage, JSON sub-dictionary, fields, entries with a value)
GLOBAL_IMAGE = ('MAE_body_z', 'SSIM_body', 'body_bias_z')
LESION_IMAGE = ('MAE_lesion_z', 'MAE_ring_z', 'abs_error_mean_lesion_enhancement_z', 'lesion_enhancement_prediction_z',
                'lesion_enhancement_real_z', 'positive_error_outside_lesion_z')
OFFTARGET = ('FE_ml', 'TE_ml', 'ME_ml', 'FE_foci_ge_0.1ml', 'FE_foci_ml', 'lesion_frac_S', 'lesion_frac_E')
READER = ('Dice', 'HD95_mm', 'prediction_empty', 'true_lesion_ml', 'predicted_lesion_ml', 'false_positive_volume_ml')
SHARPNESS = ('body_gradient_ratio', 'edge_sharpness_ratio', 'edge_sharpness_deficit', 'rim_gradient_error', 'HFEN_body',
             'HFEN_lesion', 'hf_energy_ratio_body')
LPIPS = ('LPIPS_slice', 'LPIPS_body', 'LPIPS_lesion')
MEDDINO = ('MedDINO_CS_448_normed', 'MedDINO_CS_448_lesion_slices', 'MedDINO_CS_240_normed', 'MedDINO_CS_240_lesion_slices')
VIT = ('vit_body', 'vit_lesion', 'vit_healthy')


def read(p):
    return json.loads(Path(p).read_text())


def num(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return int(v)
    return repr(float(v)) if isinstance(v, float) else v


def entries(cfg, image_dir):
    arms = [(a['key'], f"{a['key']}_ep{a['epoch']:03d}", a['epoch']) for a in cfg['arms']]
    for a in cfg.get('extra_arms', []):
        ent = f"{a['key']}_ep{a['epoch']:03d}"
        if (Path(image_dir) / ent).is_dir():
            arms.append((a['key'], ent, a['epoch']))
        else:
            print(f'note: extra arm {ent} has no image metrics, skipped')
    return arms, [('pre', 'pre', ''), ('real', 'real', '')]


def case_files(folder, cases=None):
    files = sorted(Path(folder).glob('DUKE_*.json'))
    found = [f.stem for f in files]
    if cases is not None:
        missing = sorted(set(cases) - set(found))
        assert not missing, f'{folder}: {len(missing)} patients missing ({missing[:3]} ...)'
    return found


def collect(a, cfg):
    arms, refs = entries(cfg, a.image_dir)
    out = {k: [] for k in ('global', 'lesion', 'offtarget', 'segmenter', 'perceptual')}
    cases = case_files(a.image_copy_dir)
    assert len(cases) == cfg['n_cases'], (len(cases), cfg['n_cases'])

    def rows_for(folder_name, stage_dir, pick):
        folder = Path(stage_dir) / folder_name
        case_files(folder, cases)
        return {c: pick(read(folder / f'{c}.json')) for c in cases}

    table = {}   # (csv, key) -> {case: {field: value}}

    def put(csv_name, key, epoch, values):
        t = table.setdefault((csv_name, key), {'epoch': epoch, 'rows': {}})
        for c, v in values.items():
            t['rows'].setdefault(c, {}).update(v)

    for key, ent, ep in arms + refs:
        # image metrics of the gen stage (MAE, SSIM, lesion); the copy has its own folder
        if key != 'real':
            if key == 'pre':
                img = {c: read(Path(a.image_copy_dir) / f'{c}.json')['metrics'] for c in cases}
            else:
                img = rows_for(ent, a.image_dir, lambda j: j['metrics'])
            put('global', key, ep, {c: {k: m[k] for k in GLOBAL_IMAGE} for c, m in img.items()})
            les = {}
            for c, m in img.items():
                row = {k: m[k] for k in LESION_IMAGE}
                row['r'] = m['lesion_enhancement_prediction_z'] / m['lesion_enhancement_real_z']
                les[c] = row
            put('lesion', key, ep, les)
            ps = rows_for(ent, a.psnr_dir, lambda j: {'PSNR_body_dB': j['PSNR_body'], 'PSNR_volume_dB': j['PSNR_volume'],
                                                               'range_body_z': j['range_body']})
            put('global', key, ep, ps)
            sh = rows_for(ent, a.sharpness_dir, lambda j: j['metrics'])
            put('perceptual', key, ep, {c: {k: m[k] for k in SHARPNESS} for c, m in sh.items()})
            lp = rows_for(ent, a.lpips_dir, lambda j: j['metrics'])
            put('perceptual', key, ep, {c: {k: m[k] for k in LPIPS} for c, m in lp.items()})
        # off-target: arms and the real image (the copy has no synthetic enhancement, so no row)
        if key != 'pre':
            ot = rows_for(ent, a.offtarget_dir, lambda j: j)
            put('offtarget', key, ep, {c: dict({'voxel_ml': j['voxel_ml']},
                                               **{f'{t}_{k}': j[t][k] for t in ('t2', 't3') for k in OFFTARGET})
                                       for c, j in ot.items()})
        rd = rows_for(ent, a.reader_dir, lambda j: dict({'fold': j['fold']}, **{k: j['row'][k] for k in READER}))
        put('segmenter', key, ep, rd)
        md = rows_for(ent, a.meddino_dir, lambda j: j['row'])
        put('perceptual', key, ep, {c: {k: m[k] for k in MEDDINO} for c, m in md.items()})
        vt = rows_for(ent, a.vit_dir, lambda j: j['row'])
        put('perceptual', key, ep, {c: {k: m[k] for k in VIT} for c, m in vt.items()})

    order = [k for k, _, _ in arms + refs]
    for (csv_name, key), t in sorted(table.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
        for c in cases:
            out[csv_name].append(dict({'case': c, 'arm': key, 'epoch': t['epoch']}, **t['rows'][c]))
    return out, cases


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: num(r.get(k)) for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default=str(REPO / 'configs' / 'eval' / 'test91.yaml'))
    p.add_argument('--eval-out', required=True, help='root of the stage outputs (ISBI_EVAL_OUT)')
    p.add_argument('--out', required=True, help='folder that receives per_patient/*.csv')
    for s in ('image', 'image-copy', 'psnr', 'offtarget', 'reader', 'sharpness', 'lpips', 'meddino', 'vit'):
        p.add_argument(f'--{s}-dir')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    st = cfg['stages']; root = Path(a.eval_out)
    defaults = {'image_dir': root / st['image'], 'image_copy_dir': root / st['image'] / 'pre', 'psnr_dir': root / st['psnr'],
                'offtarget_dir': root / st['offtarget'], 'reader_dir': root / st['reader'], 'sharpness_dir': root / st['sharpness'],
                'lpips_dir': root / st['lpips'], 'meddino_dir': root / st['meddino'], 'vit_dir': root / st['nnf_vit']}
    for k, v in defaults.items():
        if getattr(a, k) is None:
            setattr(a, k, str(v))
    out, cases = collect(a, cfg)
    for name, rows in out.items():
        write_csv(Path(a.out) / 'per_patient' / f'{name}.csv', rows)
        print(f'{name}.csv: {len(rows)} rows ({len(rows) // len(cases)} entries x {len(cases)} patients)')
    bad = [(n, r['arm'], r['case'], k) for n, rows in out.items() for r in rows for k, v in r.items()
           if isinstance(v, float) and not math.isfinite(v)]
    assert not bad, bad[:5]


if __name__ == '__main__':
    main()
