"""MedDINOv3 2D cosine similarity (perceptual family; Table 2 "DINO" = MedDINO_CS_448_normed).

Model: MedDINOv3 ViT-B/16 (CT3M checkpoint, teacher weights; code from github.com/ricklisz/MedDINOv3, whose
`dinov3` package must be importable: pass --repo or set ISBI_MEDDINO_REPO). Pre-trained on CT, not validated on breast MRI.
Per patient, for the real post-contrast and each synthetic volume (z units, written as NIfTI with the case affine):
  preprocess_volume: axes to (D, W, H); voxels > 0 clipped to [0, p99.5 of the positive voxels] and divided by p99.5
                     (each volume with its own p99.5); when positive voxels exist, negative values are clipped to zero;
  slices: axial slices where > 5 % of the preprocessed REAL slice is > 0 (same slices for every volume);
  features: 3 replicated channels, ImageNet mean / std, CLS output of the model; "448" resizes each slice to 448 x 448
            (antialias), "240" keeps the native slice size (the name comes from the 240 x 240 BraTS slices of the
            original script; no resize is applied here);
  score: mean over the slices of cos(f_real, f_synthetic) (eps 1e-6); *_lesion_slices: the mean over the slices that
         contain lesion voxels. Higher = closer to the real image; the real image scores 1.
TF32: matmul TF32 off, cuDNN TF32 on (torch defaults).

usage: python -m evaluation.metrics.perceptual.meddino ENTRY... --data DIR --pred-dir DIR --out DIR --checkpoint FILE --repo DIR
                                              [--device cuda|cpu] [--cases ...]
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluation.metrics import io

SIZES = (448, 240)
BATCH = 8
FOREGROUND_FRAC = 0.05


def nifti(path, vol, props):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0), dtype=np.float32), np.asarray(props['affine_RAS'])), str(path))


def load_frozen_meddinov3(checkpoint_path, device, repo):
    import torch
    sys.path.insert(0, str(Path(repo) / 'nnUNet' / 'nnunetv2' / 'training' / 'nnUNetTrainer' / 'dinov3'))
    from dinov3.models.vision_transformer import vit_base
    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['teacher']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'ibot' not in k and 'dino_head' not in k}
    model.load_state_dict(state_dict, strict=True)
    model.eval(); model.to(device)
    return model


def preprocess_volume(nifti_path):
    import nibabel as nib
    vol = nib.load(nifti_path).get_fdata()
    vol = np.transpose(vol, (2, 0, 1))
    mask = vol > 0
    if mask.any():
        p99 = np.percentile(vol[mask], 99.5)
        vol = np.clip(vol, 0, p99)
        vol = vol / (p99 + 1e-8)
    return vol


def extract_features(model, slices_np, target_size, batch_size, device, apply_norm):
    import torch
    from torchvision import transforms
    tensor = torch.from_numpy(slices_np).float().unsqueeze(1).repeat(1, 3, 1, 1)
    tl = []
    if target_size is not None:
        tl.append(transforms.Resize(target_size, antialias=True))
    if apply_norm:
        tl.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(tl) if tl else None
    out = []
    with torch.no_grad():
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i:i + batch_size].to(device)
            if transform is not None:
                batch = transform(batch)
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0)


def size_arg(s):
    """448 -> (448, 448) resize; 240 -> no resize (native slice size), as in the original script."""
    return (448, 448) if s == 448 else None


def meddino_scores(model, vols, names, lesion, props, tmp, device):
    """vols: {'real': post, name: synthetic, ...} in z units. Returns rows {name: {metric: value}}."""
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, True
    nifti(tmp / 'real.nii', vols['real'], props); real = preprocess_volume(str(tmp / 'real.nii'))
    assert real.shape[0] == lesion.shape[0], (real.shape, lesion.shape)
    idx = np.array([i for i in range(len(real)) if np.mean(real[i] > 0) > FOREGROUND_FRAC], dtype=int)
    on_lesion = lesion[idx].any(axis=(1, 2))
    ref = {s: extract_features(model, real[idx], size_arg(s), BATCH, device, True) for s in SIZES}
    rows = {}
    for n in names:
        nifti(tmp / 'x.nii', vols[n].astype(np.float32), props); x = preprocess_volume(str(tmp / 'x.nii'))
        r = {}
        for s in SIZES:
            cs = F.cosine_similarity(extract_features(model, x[idx], size_arg(s), BATCH, device, True), ref[s],
                                     dim=1, eps=1e-6).float().cpu().numpy()
            r[f'MedDINO_CS_{s}_normed'] = float(cs.mean())
            r[f'MedDINO_CS_{s}_lesion_slices'] = float(cs[on_lesion].mean()) if on_lesion.any() else None
        rows[n] = r
    return {'dino_slices': int(len(idx)), 'lesion_slices': int(on_lesion.sum()), 'rows': rows}


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('entries', nargs='+'); p.add_argument('--data', required=True); p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True); p.add_argument('--checkpoint', default=os.environ.get('ISBI_MEDDINO_CKPT'))
    p.add_argument('--repo', default=os.environ.get('ISBI_MEDDINO_REPO')); p.add_argument('--device', default='cuda')
    p.add_argument('--cases', nargs='*')
    a = p.parse_args()
    dev = torch.device(a.device); model = load_frozen_meddinov3(a.checkpoint, dev, a.repo)
    names = list(dict.fromkeys(['real'] + a.entries))
    with tempfile.TemporaryDirectory(prefix='meddino_') as tmp:
        for case in a.cases or io.cases(a.data):
            todo = [e for e in names if not (Path(a.out) / e / f'{case}.json').exists()]
            if not todo:
                continue
            t0 = time.perf_counter(); pre, post, body, label, props = io.load_case(Path(a.data) / case)
            vols = {e: io.synthetic(e, case, pre, post, a.pred_dir) for e in set(todo) | {'real'}}
            r = meddino_scores(model, vols, ['real'] + [e for e in todo if e != 'real'], label > 0, props, Path(tmp), dev)
            for e in todo:
                io.save_json(Path(a.out) / e / f'{case}.json', {'patient': case, 'entry': e, 'row': r['rows'][e],
                                                               'real_row': r['rows']['real'], 'dino_slices': r['dino_slices'],
                                                               'lesion_slices': r['lesion_slices'], 'seconds': time.perf_counter() - t0})
            print(case, todo, flush=True)


if __name__ == '__main__':
    main()
