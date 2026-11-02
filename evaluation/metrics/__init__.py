"""Evaluation metrics of the paper, one sub-package per family.

global_     MAE, SSIM, PSNR over the body mask            (`global` is a Python keyword, hence the trailing underscore)
lesion      lesion MAE, ring MAE, |dE|, enhancement recovery r
offtarget   false enhancement outside the lesion (FE mL, foci)
perceptual  LPIPS, sharpness, MedDINOv3, nnFoundationViT
segmenter   MAMA-MIA nnU-Net reader (Dice, HD95, FP mL)

Every function works on the z-scored volumes of one prepared case (evaluation.metrics.io.load_case): pre-contrast X, real
post-contrast Y, synthetic post-contrast Y_hat, body mask and expert lesion mask, 1.0 x 0.7 x 0.7 mm, axes (D, H, W).
"""
