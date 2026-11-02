"""Preprocessing of one DUKE case; the same function is used for train, dev and test.

Per case:
- canonical (RAS) orientation, linear resampling of pre and post to 0.7 x 0.7 x 1.0 mm (RAS axes), nearest for the mask
  and for the volume-validity bitmap;
- normalisation with the mean / std of the non-zero pre-contrast voxels (computed before the post-contrast is read),
  the same affine applied to pre and post, no clipping; voxels outside the resampled volume get the normalised value
  of a raw zero ("background");
- a heuristic body mask from the pre-contrast only (not a validated breast segmentation);
- arrays stored as D, H, W = superior, anterior, right.

Outputs in ``<out>/<case>/``: data.npy (2 x D x H x W float32: pre, post), label.npy, valid.npy, body.npy (uint8),
locations.npz (sparse body / lesion voxel positions for the sampler), properties.json.

Usage:
    python -m synthesis.data.preprocess --manifest split_manifest.json --splits train dev --out $ISBI_DATA [--workers 3]
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage as ndi

from synthesis.core.io import read_json, save_json, sha256

SPACING_RAS = (.7, .7, 1.)  # image grid, not a statement about the native resolution
PREPROCESS_VERSION = 'mama-duke-v1'


def normalization(pre):
    q = pre[np.isfinite(pre) & (pre != 0)]
    if q.size < 100 or not np.isfinite(pre).all():
        raise ValueError('invalid source')
    mean, std = float(q.mean(dtype=np.float64)), float(q.std(dtype=np.float64))
    if std < 1e-8:
        raise ValueError('constant source')
    return mean, std


def body_mask(pre, valid):
    """Heuristic body support from the pre-contrast only (Otsu-like threshold x 0.2, closing, hole filling, small
    components removed). Not a validated anatomical breast segmentation."""
    vals = pre[(pre > 0) & valid][::16]
    if vals.size < 100:
        raise ValueError('no positive source signal')
    high = float(np.percentile(vals, 99.8))
    hist, edges = np.histogram(vals, bins=256, range=(0, high))
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    s = np.cumsum(hist * centers)
    total = w[-1]
    score = (s[-1] * w - s * total) ** 2 / np.maximum(w * (total - w), 1)
    threshold = float(centers[np.argmax(score)]) * .2
    b = (pre > threshold) & valid
    b = ndi.binary_closing(b, iterations=2)
    b = ndi.binary_fill_holes(b) & valid
    lab, _ = ndi.label(b)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    if not counts.max():
        raise ValueError('empty body mask')
    keep = counts >= max(500, counts.max() * .01)
    keep[0] = False
    return keep[lab] & valid


def locations(mask, rng, n=20000):
    """Sparse deterministic positions (at most n voxels) instead of every voxel."""
    ix = np.flatnonzero(mask)
    if len(ix) > n:
        ix = rng.choice(ix, n, replace=False)
    if not len(ix):
        return np.empty((0, 3), np.int16)
    return np.column_stack(np.unravel_index(ix, mask.shape)).astype(np.int16)


def array_save(path, array):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, array)
    tmp.replace(path)


def prepare_case(record, out_root):
    """record: {'patient', 'pre', 'post', 'label'} (NIfTI paths). Writes <out_root>/<patient>/; skips a complete case."""
    start = time.perf_counter()
    case = record['patient']
    dest = Path(out_root) / case
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        old = read_json(dest / 'complete.json')
        if old.get('preprocess_version') != PREPROCESS_VERSION:
            raise ValueError('case prepared by another preprocessing version: ' + case)
        return {'patient': case, 'cached': True, 'seconds': 0}
    a, b, l = [nib.as_closest_canonical(nib.load(record[k])) for k in ('pre', 'post', 'label')]
    if a.shape != b.shape or a.shape != l.shape or not np.allclose(a.affine, b.affine, atol=1e-4) \
            or not np.allclose(a.affine, l.affine, atol=1e-4):
        raise ValueError('phase/mask geometry mismatch: ' + case)
    pre = a.get_fdata(dtype=np.float32)
    mean, std = normalization(pre)
    # Statistics are fixed before the target voxels are read.
    post = b.get_fdata(dtype=np.float32)
    lab = l.get_fdata(dtype=np.float32)
    if not np.isfinite(post).all() or not np.isfinite(lab).all() or not np.isin(lab, [0, 1]).all() or not lab.any():
        raise ValueError('invalid target or binary annotation: ' + case)
    grid = resample_to_output(a, voxel_sizes=SPACING_RAS, order=1)
    p = grid.get_fdata(dtype=np.float32)
    y = resample_from_to(b, grid, order=1).get_fdata(dtype=np.float32)
    seg = resample_from_to(l, grid, order=0).get_fdata(dtype=np.float32) > 0
    valid = resample_from_to(nib.Nifti1Image(np.ones(a.shape, dtype=np.uint8), a.affine), grid,
                             order=0).get_fdata(dtype=np.float32) > 0
    body = body_mask(p, valid)
    # Padding value in source-normalised units; every crop uses this constant.
    bg = -mean / std
    p = (p - mean) / std
    y = (y - mean) / std
    p[~valid] = bg
    y[~valid] = bg
    rng = np.random.default_rng(int(hashlib.sha256(case.encode()).hexdigest()[:8], 16))
    data = np.stack([p.transpose(2, 1, 0), y.transpose(2, 1, 0)]).astype(np.float32)
    seg = seg.transpose(2, 1, 0).astype(np.uint8)
    valid = valid.transpose(2, 1, 0).astype(np.uint8)
    body = body.transpose(2, 1, 0).astype(np.uint8)
    assert data.shape[1:] == seg.shape == body.shape == valid.shape and np.isfinite(data).all()
    array_save(dest / 'data.npy', data)
    array_save(dest / 'label.npy', seg)
    array_save(dest / 'valid.npy', valid)
    array_save(dest / 'body.npy', body)
    with (dest / 'locations.npz.tmp').open('wb') as f:
        np.savez(f, body=locations(body, rng), tumour=locations(seg, rng))
    (dest / 'locations.npz.tmp').replace(dest / 'locations.npz')
    props = {'case': case, 'shape_DHW': list(seg.shape), 'affine_RAS': grid.affine.tolist(),
             'spacing_DHW': [1., .7, .7], 'normalization': {'mean_source': mean, 'std_source': std, 'background': bg},
             'native_shape_XYZ': list(a.shape), 'native_spacing_XYZ': [float(x) for x in a.header.get_zooms()],
             'body_fraction': float(body.mean()), 'lesion_voxels': int(seg.sum()),
             'lesion_fraction_in_source_body': float(body[seg > 0].mean()),
             'real_enhancement_lesion_z': float((data[1] - data[0])[seg > 0].mean()),
             'real_mean_body_shift_z': float((data[1] - data[0])[body > 0].mean()),
             'sources': {k: {'file': Path(record[k]).name, 'sha256': sha256(record[k])} for k in ('pre', 'post', 'label')},
             'preprocess_version': PREPROCESS_VERSION,
             'data_bytes': sum(f.stat().st_size for f in dest.glob('*.npy')),
             'seconds': time.perf_counter() - start}
    save_json(dest / 'properties.json', props)
    save_json(dest / 'complete.json', {'patient': case, 'preprocess_version': PREPROCESS_VERSION,
                                       'completed': time.strftime('%F %T')})
    print(json.dumps({k: props[k] for k in ('case', 'shape_DHW', 'lesion_fraction_in_source_body', 'seconds')}),
          flush=True)
    return {'patient': case, 'cached': False, 'seconds': props['seconds']}


def main():
    p = argparse.ArgumentParser(description='Preprocess DUKE cases (same function for every split).')
    p.add_argument('--manifest', required=True, help='output of synthesis.data.split')
    p.add_argument('--splits', nargs='+', default=['train', 'dev'])
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    manifest = read_json(a.manifest)
    cases = [r for s in a.splits for r in manifest['splits'][s]]
    if a.limit:
        cases = cases[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    done = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for job in as_completed([pool.submit(prepare_case, r, a.out) for r in cases]):
            done.append(job.result())
    print('PREPROCESS_OK', len(done), flush=True)


if __name__ == '__main__':
    main()
