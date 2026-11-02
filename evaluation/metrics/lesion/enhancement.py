"""Lesion metrics (z units, expert lesion mask L = label.npy > 0).

MAE_lesion_z   mean over L of |Y_hat - Y|
MAE_ring_z     mean over the ring R = dilate(L, 5 iterations, 6-connectivity) minus L, inside the body, of |Y_hat - Y|
enhancement    E_pred = mean_L(Y_hat - X), E_real = mean_L(Y - X)
|dE|           abs_error_mean_lesion_enhancement_z = |E_pred - E_real|
r              enhancement recovery: sum over patients of E_pred / the same sum of E_real (paper definition, target
               100 %; pooled_recovery, or aggregate ratio_of_sums in configs/eval/families.yaml). The per-patient
               ratio E_pred / E_real (recovery) is also kept; the real enhancement is >= 1.4 z in every test lesion.
positive_error_outside_lesion_z   mean over body minus L of max(Y_hat - Y, 0) (descriptive)
"""
import numpy as np
from scipy import ndimage as ndi

from evaluation.metrics.global_.image import mae

RING_ITERATIONS = 5


def ring(lesion, body, iterations=RING_ITERATIONS):
    lesion = lesion > 0
    return ndi.binary_dilation(lesion, iterations=iterations) & ~lesion & (body > 0)


def lesion_metrics(pred, pre, post, body, label):
    lesion = label > 0; mask = body > 0; err = pred - post
    enh_p = float((pred - pre)[lesion].mean()); enh_r = float((post - pre)[lesion].mean())
    return {'MAE_lesion_z': mae(err, lesion), 'MAE_ring_z': mae(err, ring(lesion, mask)),
            'lesion_enhancement_prediction_z': enh_p, 'lesion_enhancement_real_z': enh_r,
            'abs_error_mean_lesion_enhancement_z': abs(enh_p - enh_r),
            'positive_error_outside_lesion_z': float(np.maximum(err[mask & ~lesion], 0).mean())}


def recovery(row):
    """Per-patient r from a lesion_metrics() row."""
    return row['lesion_enhancement_prediction_z'] / row['lesion_enhancement_real_z']


def pooled_recovery(rows):
    """Paper r: sum of the predicted lesion enhancements over patients / the same sum of the real ones."""
    return (sum(r['lesion_enhancement_prediction_z'] for r in rows)
            / sum(r['lesion_enhancement_real_z'] for r in rows))


def summarise_r(r):
    """Median r and the percentage of patients with r > 1."""
    r = np.asarray(r, dtype=float)
    return {'r_median': float(np.median(r)), 'r_gt1_pct': float((r > 1).mean() * 100), 'n': int(r.size)}
