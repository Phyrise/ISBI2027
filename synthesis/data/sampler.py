"""Stateless paired patch sampler, shared by every arm, expert and calibration.

One dataset item is one whole batch of 2 patches, fully determined by (seed, global update index): the batch for update
``u`` is drawn from ``numpy.random.default_rng([seed, u])``, so a resumed run sees exactly the batches it would have seen.
Per patch: a case uniformly at random, a centre on a lesion voxel with probability 0.33 (else on a body voxel), a
96 x 160 x 160 crop (outside the image: the case background value for intensities, 0 for masks), lesion label -1
outside the valid volume, then paired mirrors (each axis with probability 0.5, the same for pre, post and masks). There
is no other augmentation.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from synthesis.core.io import read_json

PATCH = (96, 160, 160)
BATCH = 2
LESION_FRACTION = .33


def zscore(x):
    """Per-sample, per-channel z-score over the spatial dimensions."""
    dims = tuple(range(2, x.ndim))
    return (x - x.mean(dims, keepdim=True)) / (x.std(dims, keepdim=True) + 1e-8)


@lru_cache(maxsize=400)
def case_info(data_dir, case):
    d = Path(data_dir) / case
    props = read_json(d / 'properties.json')
    with np.load(d / 'locations.npz') as f:
        locs = {k: f[k] for k in ('body', 'tumour')}
    return props, locs


def crop(a, lo, shape, background=0):
    """Crop by lower corner over the last three axes; outside the array = background."""
    lo = np.asarray(lo)
    shape = np.asarray(shape)
    hi = lo + shape
    low = np.maximum(lo, 0)
    high = np.minimum(hi, np.asarray(a.shape[-3:]))
    if (high <= low).any():
        raise ValueError('crop has no overlap')
    out = np.full((*a.shape[:-3], *shape), background, dtype=a.dtype)
    src = tuple(slice(int(l), int(h)) for l, h in zip(low, high))
    dst = tuple(slice(int(l), int(h)) for l, h in zip(low - lo, high - lo))
    out[(..., *dst)] = a[(..., *src)]
    return out


class Batches(torch.utils.data.Dataset):
    """Items = whole batches for global updates start .. stop-1 (deterministic from seed and update index)."""

    def __init__(self, data_dir, cases, seed, start, stop, patch=PATCH, batch=BATCH, lesion_fraction=LESION_FRACTION):
        self.data_dir, self.ids = str(data_dir), list(cases)
        self.seed, self.start, self.stop, self.patch = int(seed), int(start), int(stop), tuple(patch)
        self.batch, self.lesion_fraction = int(batch), float(lesion_fraction)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        step = self.start + index
        rng = np.random.default_rng([self.seed, step])
        batch, plans = [], []
        for _ in range(self.batch):
            case = self.ids[int(rng.integers(len(self.ids)))]
            props, locs = case_info(self.data_dir, case)
            force = bool(rng.random() < self.lesion_fraction)
            points = locs['tumour' if force else 'body']
            if not len(points):
                points = locs['body']
            ctr = points[int(rng.integers(len(points)))].astype(int)
            lo = ctr - np.asarray(self.patch) // 2
            d = Path(self.data_dir) / case
            data = crop(np.load(d / 'data.npy', mmap_mode='r'), lo, self.patch, props['normalization']['background'])
            valid = crop(np.load(d / 'valid.npy', mmap_mode='r'), lo, self.patch, 0)
            label = crop(np.load(d / 'label.npy', mmap_mode='r'), lo, self.patch, 0).astype(np.int16)
            label[valid == 0] = -1
            flips = [axis for axis in range(3) if rng.random() < .5]  # paired mirrors only
            for axis in flips:
                data = np.flip(data, axis=axis + 1)
                valid = np.flip(valid, axis=axis)
                label = np.flip(label, axis=axis)
            batch.append((np.ascontiguousarray(data), np.ascontiguousarray(valid[None]), np.ascontiguousarray(label[None])))
            plans.append({'patient': case, 'lower_DHW': [int(v) for v in lo], 'forced_lesion': force, 'flips': flips})
        return {'data': torch.from_numpy(np.stack([x[0] for x in batch])),
                'valid': torch.from_numpy(np.stack([x[1] for x in batch])).float(),
                'label': torch.from_numpy(np.stack([x[2] for x in batch])).long(),
                'plan': json.dumps({'step': step, 'samples': plans}, sort_keys=True)}


def worker_init(_):
    torch.set_num_threads(1)


def loader(data_dir, cases, seed, start, stop, workers=2, patch=PATCH, pin_memory=True):
    ds = Batches(data_dir, cases, seed, start, stop, patch)
    opts = dict(batch_size=None, num_workers=workers, pin_memory=pin_memory, worker_init_fn=worker_init)
    if workers:
        opts.update(prefetch_factor=2, persistent_workers=True)
    return torch.utils.data.DataLoader(ds, **opts)


def split_cases(split_file, split):
    """Patient IDs of a split from a split JSON (configs/split_duke.json or a synthesis.data.split manifest)."""
    rows = read_json(split_file)['splits'][split]
    return [r if isinstance(r, str) else r['patient'] for r in rows]
