"""Full-volume inference of a generator (or of a lesion expert).

Sliding window over the whole preprocessed volume with 96 x 160 x 160 tiles, 50 % overlap, Gaussian blending
(sigma = tile / 8, weights floored at 1e-6), two tiles per forward, no mirroring and no target-dependent crop. Tiles
that extend past the volume are filled with the volume's background value (its corner voxel). Generators read the
z-scored pre-contrast and return the predicted post-contrast in the same z units; voxels that no tile covers keep the
pre-contrast value. Experts read the post-contrast (each tile z-scored) and return the lesion softmax.
Precision (``precision.inference_autocast`` of the arm): float16, except bfloat16 for MedicalNet, VGG16 and the four
SwinUNETR / GAN arms.

    python -m synthesis.inference.predict --config configs/arms/l1.yaml --checkpoint runs/l1/checkpoint_ep300.pth \
        --split test --data-dir $ISBI_TEST_DATA --out predictions/l1_ep300 [--device cuda]

writes ``<case>.npy`` (float16 by default, z units) and ``<case>.json`` (tiles, autocast, checkpoint sha256).
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from synthesis.core.amp import autocast
from synthesis.core.config import load_config, path, torch_dtype
from synthesis.core.io import save_json, sha256
from synthesis.data.sampler import PATCH, crop, split_cases, zscore


def windows(start, stop, size):
    if stop - start <= size:
        return [max(0, start - (size - (stop - start)) // 2)]
    out = list(range(start, stop - size + 1, size // 2))
    if out[-1] != stop - size:
        out.append(stop - size)
    return out


def predict(model, image, is_expert, dtype=torch.float16, device='cuda', patch=PATCH):
    """image: D x H x W array (z units). Returns (prediction, info)."""
    device = torch.device(device)
    low = np.zeros(3, dtype=int)
    high = np.array(image.shape)
    starts = list(itertools.product(*[windows(int(l), int(h), p) for l, h, p in zip(low, high, patch)]))
    sigma = np.array(patch) / 8
    grids = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) / s for n, s in zip(patch, sigma)], indexing='ij')
    weight = np.exp(-.5 * sum(g * g for g in grids)).astype(np.float32)
    weight = np.maximum(weight, 1e-6)
    acc = np.zeros(image.shape, dtype=np.float32)
    count = np.zeros_like(acc)
    background = float(image[0, 0, 0])
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(starts), 2):
            group = starts[i:i + 2]
            inp = np.stack([crop(image, lo, patch, background) for lo in group])[:, None]
            x = torch.from_numpy(inp).to(device)
            with autocast(device, dtype):
                out = model(zscore(x) if is_expert else x)
                out = torch.softmax(out.float(), 1)[:, 1] if is_expert else out[:, 0].float()
            pred = out.cpu().numpy()
            for y, lo in zip(pred, group):
                lo = np.asarray(lo)
                hi = lo + np.array(patch)
                lb = np.maximum(0, lo)
                ub = np.minimum(hi, image.shape)
                dst = tuple(slice(int(l), int(h)) for l, h in zip(lb, ub))
                src = tuple(slice(int(l), int(h)) for l, h in zip(lb - lo, ub - lo))
                acc[dst] += y[src] * weight[src]
                count[dst] += weight[src]
    result = np.zeros_like(image) if is_expert else image.copy()  # not covered: copy the pre-contrast
    inside = count > 0
    result[inside] = acc[inside] / count[inside]
    if not np.isfinite(result).all():
        raise RuntimeError('nonfinite prediction')
    return result, {'tiles': len(starts), 'autocast': str(dtype)}


def main():
    p = argparse.ArgumentParser(description='Predict a split with one checkpoint of an arm or expert.')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--data-dir', help='preprocessed cases (default: paths.data_dir of the config)')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save-dtype', default='float16', choices=['float16', 'float32'])
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    from synthesis.trainers.build import load_for_inference
    cfg = load_config(a.config)
    is_expert = cfg['task'] == 'segmentation'
    dtype = torch_dtype(cfg['precision']['inference_autocast'])
    data_dir = Path(a.data_dir) if a.data_dir else path(cfg['paths']['data_dir'], 'paths.data_dir')
    model = load_for_inference(cfg, a.checkpoint).to(a.device).eval()
    digest = sha256(a.checkpoint)
    cases = split_cases(path(cfg['paths']['split'], 'paths.split'), a.split)[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        start = time.perf_counter()
        data = np.load(data_dir / case / 'data.npy', mmap_mode='r')
        image = np.array(data[1] if is_expert else data[0], copy=True)
        pred, info = predict(model, image, is_expert, dtype, a.device)
        np.save(out / f'{case}.npy', pred.astype(a.save_dtype))
        save_json(out / f'{case}.json', dict(info, patient=case, run=cfg['run_name'], checkpoint_sha256=digest,
                                             seconds=time.perf_counter() - start))
        print(json.dumps({'patient': case, 'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    main()
