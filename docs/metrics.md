# Evaluation metrics

Every test patient is scored on the whole 3D volume. Ŷ is the synthetic post-contrast, Y the real one and X the
pre-contrast, all z-scored with the pre-contrast statistics at 1.0 × 0.7 × 0.7 mm. B is the body mask of the
preprocessing (`body.npy`), L the MAMA-MIA expert tumour segmentation (`label.npy`).

## The five families (Table 1 of the paper)

| Family | Metric | Definition | Better | Code (`evaluation/metrics/`) |
|---|---|---|:---:|---|
| Global | PSNR | 10 log10(R² / MSE over B), R = max − min of Y inside B | ↑ | `global_/psnr.py` |
| | MAE | mean \|Ŷ − Y\| over B (z) | ↓ | `global_/image.py` |
| | SSIM | 3D SSIM map (data range 20 z, window 7) averaged over B | ↑ | `global_/image.py` |
| Lesion | MAE | mean \|Ŷ − Y\| over L (z) | ↓ | `lesion/enhancement.py` |
| | r | Σ over patients of mean_L(Ŷ − X) / Σ over patients of mean_L(Y − X) (target 100 %) | ↑ | `lesion/enhancement.py` |
| Off-target | FE | number of false-enhancement regions per patient (below) | ↓ | `offtarget/false_enhancement.py` |
| Perceptual | DINO | MedDINOv3 CLS cosine similarity, axial slices of the whole image (448 × 448) | ↑ | `perceptual/meddino.py` |
| | LPIPS_L | LPIPS-VGG spatial map averaged over the tumour pixels of the axial slices | ↓ | `perceptual/lpips_slices.py` |
| | \|∇\| | mean \|∇Ŷ\| / mean \|∇Y\| within 2 mm of the tumour surface (target 1; sharpness around the tumour) | ↑ | `perceptual/sharpness.py` |
| Segmenter | Dice | MAMA-MIA nnU-Net on Ŷ vs the expert segmentation, native geometry | ↑ | `segmenter/mama_reader.py` |
| | FP mL | volume predicted outside the expert segmentation | ↓ | `segmenter/mama_reader.py` |

Per-patient values are averaged over the 91 test patients, except r, which is a ratio of sums
(`evaluation/ranking/per_arm.py`, `aggregate: ratio_of_sums`). The metric list, directions and aggregations are in
`configs/eval/families.yaml`.

**False enhancement.** Connected regions (26-connectivity) of at least 0.1 mL, outside the tumour and a 5-voxel ring,
where Ŷ − X ≥ 2 z but Y − X < 2 z. Besides invented lesions, they may include overestimated background parenchymal
enhancement, vessels or misregistration. The real image has FE = 0 by construction; the pre-contrast copy enhances
nothing and is not measured.

**Segmenter.** Ŷ is mapped back to the native geometry of the first post-contrast image and read by the released
MAMA-MIA nnU-Net (Dataset105_full_image, 3d_fullres, stock nnunetv2 2.6.2). Each patient is read by the fold model
that did not see it. This segmenter shares the tumour-segmentation task with the AFP experts, so segmenter gains of
AFP models are partly expected.

**Caveats.** MedDINOv3 is a CT model; neither it nor LPIPS is validated on breast MRI, and over the whole image DINO
scores the unenhanced copy above most models. The 2D perceptual metrics are computed slice-wise and averaged.

## Rank and statistics

- **Rank** (`evaluation/ranking/family_rank.py`): models are ranked on each metric (ties averaged), ranks are averaged
  within each family, then over the five families with equal weight. The copy and the real image are not ranked. The
  rank is exploratory.
- **Paired differences** (`evaluation/ranking/bootstrap.py`): mean difference A − B with a percentile 95 % CI from a
  5000-draw paired patient bootstrap (seed 20261029), without multiplicity correction. With one training seed per
  model, they capture test-patient but not training variance.
