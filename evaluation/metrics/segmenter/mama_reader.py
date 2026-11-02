"""Downstream reader: the nnU-Net released with MAMA-MIA, frozen, applied to synthetic post-contrast images.

Model: MAMA-MIA "full_image_dce_mri_tumor_segmentation" release (Dataset105_full_image, nnUNetPlans 3d_fullres,
5 folds, cross-validated by the dataset authors on real images of all 1506 cases). Run with STOCK nnunetv2 2.6.2
(nnUNetPredictor: tile step 0.5, Gaussian blending, no mirroring, checkpoint_final.pth). A modified nnunetv2 whose
preprocessing skips crop_to_nonzero changes the reader's input and must not be used.
Fold rule: each test patient is read by the single fold whose validation list (fold_k/nnunet_metrics_fold_k.csv of the
release) contains it; a patient in no or several validation lists is an error.
Input: the synthetic volume in z units back to scanner intensities (z * std_source + mean_source of the case's
pre-contrast normalisation), written as NIfTI with the case affine, then linearly resampled (SimpleITK, identity
transform, background 0) onto the native geometry of the original first post-contrast image (_0001), and given
to the predictor with the native spacing.
Metrics, against the native expert mask (segmentations/expert), native spacing:
  Dice = 2 |P & G| / (|P| + |G|) (1 if both empty); HD95_mm = max of the two directed 95th-percentile surface distances
  (None if P or G is empty; not used in the paper: the reader predicts small distant components even on real
  images); false_positive_volume_ml = |P & ~G| * voxel volume (Table 2 "FP mL"); predicted / true lesion volumes.
Not bit-reproducible on GPU: repeated reads differ by up to ~1.2e-4 in Dice.

usage: python -m evaluation.metrics.segmenter.mama_reader ENTRY... --data DIR --pred-dir DIR --out DIR --reader-dir DIR
                                                --manifest FILE [--device cuda|cpu] [--cases ...]
       (ENTRY = an arm entry, 'pre' or 'real'; the manifest gives each patient's native 'post' and 'label' NIfTI paths)
"""
import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluation.metrics import io

TILE_STEP = 0.5


def official_fold(reader_dir, cases):
    """{case: (fold, Dice in the release's validation CSV)}; each case must be in exactly one validation list."""
    fold = {}
    for f in range(5):
        for line in (Path(reader_dir) / f'fold_{f}' / f'nnunet_metrics_fold_{f}.csv').read_text().split('\n')[1:]:
            if line:
                fold.setdefault(line.split(',')[0].upper(), []).append((f, float(line.split(',')[1])))
    missing = [c for c in cases if len(fold.get(c, [])) != 1]
    assert not missing, f'patients without a unique validation fold: {missing}'
    return {c: fold[c][0] for c in cases}


def nifti(path, vol, props):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0), dtype=np.float32), np.asarray(props['affine_RAS'])), str(path))


def seg_native(p, g, spacing):
    """Segmentation metrics in native spacing (z, y, x) mm; distances in the bounding box of P | G + 1 voxel (exact)."""
    from scipy import ndimage as ndi
    p = p > 0; g = g > 0; tp = int((p & g).sum()); a = int(p.sum()); b = int(g.sum()); vv = float(np.prod(spacing)) / 1000
    hd = None
    if a and b:
        nz = np.nonzero(p | g); box = tuple(slice(max(0, int(i.min()) - 1), int(i.max()) + 2) for i in nz)
        p, g = p[box], g[box]
        pp = p ^ ndi.binary_erosion(p); gg = g ^ ndi.binary_erosion(g)
        d1 = ndi.distance_transform_edt(~gg, sampling=spacing)[pp]; d2 = ndi.distance_transform_edt(~pp, sampling=spacing)[gg]
        hd = float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
    return {'Dice': 2 * tp / (a + b) if a + b else 1., 'HD95_mm': hd, 'prediction_empty': a == 0, 'true_lesion_ml': b * vv,
            'predicted_lesion_ml': a * vv, 'false_positive_volume_ml': int((p & ~g).sum()) * vv}


def predictor(reader_dir, fold, device):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pr = nnUNetPredictor(tile_step_size=TILE_STEP, use_gaussian=True, use_mirroring=False,
                         perform_everything_on_device=device.type == 'cuda', device=device, verbose=False,
                         verbose_preprocessing=False, allow_tqdm=False)
    pr.initialize_from_trained_model_folder(str(reader_dir), use_folds=(fold,), checkpoint_name='checkpoint_final.pth')
    return pr


def read_volume(pred, vol_z, props, post_path, label_path, tmp):
    """Reader row of one volume (z units) of one patient."""
    import SimpleITK as sitk
    ref = sitk.ReadImage(str(post_path)); gt = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))) > 0
    assert gt.shape == tuple(reversed(ref.GetSize()))
    spacing = [float(s) for s in ref.GetSpacing()[::-1]]; nz = props['normalization']
    nifti(tmp, vol_z * nz['std_source'] + nz['mean_source'], props)
    x = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(tmp)), ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    seg = pred.predict_single_npy_array(x[None], {'spacing': spacing}, None, None, False)
    return seg_native(np.asarray(seg), gt, spacing)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('entries', nargs='+'); p.add_argument('--data', required=True); p.add_argument('--pred-dir', required=True)
    p.add_argument('--out', required=True); p.add_argument('--reader-dir', default=os.environ.get('ISBI_READER_DIR'))
    p.add_argument('--manifest', default=os.environ.get('ISBI_MANIFEST')); p.add_argument('--split', default='test')
    p.add_argument('--device', default='cuda'); p.add_argument('--cases', nargs='*')
    a = p.parse_args()
    dev = torch.device(a.device)
    man = {r['patient']: r for r in io.read_json(a.manifest)['splits'][a.split]}
    cases = a.cases or io.cases(a.data); folds = official_fold(a.reader_dir, cases); preds = {}
    with tempfile.TemporaryDirectory(prefix='reader_') as tmpdir:
        tmp = Path(tmpdir) / 'volume.nii'
        for case in cases:
            todo = [e for e in a.entries if not (Path(a.out) / e / f'{case}.json').exists()]
            if not todo:
                continue
            f, csv_dice = folds[case]
            if f not in preds:
                preds[f] = predictor(a.reader_dir, f, dev)
            pre, post, _, _, props = io.load_case(Path(a.data) / case)
            for e in todo:
                t0 = time.perf_counter()
                row = read_volume(preds[f], io.synthetic(e, case, pre, post, a.pred_dir), props, man[case]['post'], man[case]['label'], tmp)
                io.save_json(Path(a.out) / e / f'{case}.json', {'patient': case, 'entry': e, 'fold': f, 'official_csv_dice': csv_dice,
                                                               'mirroring': False, 'row': row, 'seconds': time.perf_counter() - t0})
                print(case, e, f'fold {f}', f"Dice {row['Dice']:.3f}", flush=True)


if __name__ == '__main__':
    main()
