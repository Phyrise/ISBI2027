"""MAE and SSIM over the body mask.

MAE_body_z = mean over body voxels of |Y_hat - Y| (z units).
SSIM_body  = 3D SSIM map of skimage.metrics.structural_similarity(Y, Y_hat, data_range=20, win_size=7, full=True),
             averaged over the body voxels. The data range is fixed at 20 z for every model and patient.
body_bias_z = mean over body voxels of (Y_hat - Y) (signed; descriptive).
The body mask is the heuristic body mask of the preprocessing (body.npy > 0), not a breast segmentation.
"""
import numpy as np
from skimage.metrics import structural_similarity

SSIM_DATA_RANGE_Z = 20.
SSIM_WIN = 7


def mae(err, m):
    return float(np.abs(err[m]).mean()) if m.any() else None


def mae_body(pred, post, body):
    return mae(pred - post, body > 0)


def ssim_body(pred, post, body):
    _, ssim = structural_similarity(post, pred, data_range=SSIM_DATA_RANGE_Z, win_size=SSIM_WIN, full=True)
    return float(ssim[body > 0].mean())


def body_bias(pred, post, body):
    return float((pred - post)[body > 0].mean())
