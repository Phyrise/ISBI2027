"""Masked L1: mean absolute error over the valid voxels of the patch (fp32)."""


def masked_l1(pred, target, valid):
    return ((pred.float() - target.float()).abs() * valid).sum() / valid.sum().clamp_min(1)
