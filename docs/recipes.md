# Training recipes

Every model has one file in `configs/arms/` (synthesis) or `configs/experts/` (segmentation experts), which extends
`configs/common.yaml`. The file name is the run name used by the scripts.

## Shared settings

- Data: MAMA-MIA DUKE, official split (160 train / 40 dev / 91 test, `configs/split_duke.json`), 1.0 × 0.7 × 0.7 mm,
  z-scored with the pre-contrast statistics.
- Patches 96 × 160 × 160, batch size 2, 250 updates per epoch, 33 % lesion-centred sampling, paired axis flips,
  seed 20261018 for every model.
- ResUNet: residual 3D U-Net of nnU-Net, stages 32-64-128-256-320-320, blocks 1-3-4-6-6-6, final stride (1, 2, 2).
- Pretrained = the nnFoundationCNN encoder copied over the common random initialisation (`init.nnf_encoder: true`);
  the decoder keeps the random initialisation.

## Recipes

| Recipe | Optimiser | Schedule | Used by |
|---|---|---|---|
| common | SGD, momentum 0.99, Nesterov, wd 3e-5, clip 12 | lr 1e-2, 5 warm-up epochs, poly (0.9) to 300 | ResUNet `l1`, `ssl`, `mam_scratch300`, `mam300`, `sam`, `vgg`, `mednet`, ... |
| nnF | same SGD | lr 1e-3, 15 warm-up epochs, poly (0.9) to 300 (official nnFoundationCNN fine-tuning) | ResUNet `*_off`, `*_nnfoff`, both experts |
| Adam (SwinUNETR) | Adam (0.9, 0.999), no wd, clip 12 | lr 1e-3, constant for 167 epochs, then linear decay to 250 | `swin_l1`, `swin_mam_scratch300` |
| Adam (GAN) | Adam (0.5, 0.999), no wd, no clip | lr 2e-4, constant for 100 epochs, then linear decay to 200 | `gan_nnf`, `gan_nnf_mam_scratch300` |

The GAN objective is 100 (L1 + λ AFP) + LSGAN + 10 feature matching, with a two-scale 3D PatchGAN; its generator
runs in bf16 and its discriminator in fp16.

## Models

| Config | Generator, start | Loss | λ | Recipe | Epoch | In Table 2 |
|---|---|---|---:|---|---:|:---:|
| `l1_off` | ResUNet, scratch | L1 | – | nnF | 300 | ✓ |
| `l1_nnfoff` | ResUNet, pretrained | L1 | – | nnF | 300 | ✓ |
| `mam_scratch300_off` | ResUNet, scratch | L1 + AFP_seg (scratch expert) | 0.1264 | nnF | 300 | ✓ |
| `mam_scratch300_nnfoff` | ResUNet, pretrained | L1 + AFP_seg | 0.1264 | nnF | 300 | ✓ |
| `ssl_off` | ResUNet, scratch | L1 + feat. nnF (frozen nnFoundationCNN encoder) | 11.78 | nnF | 300 | ✓ |
| `ssl_nnfoff` | ResUNet, pretrained | L1 + feat. nnF | 11.78 | nnF | 300 | text |
| `mam300` | ResUNet, scratch | L1 + AFP_seg-ft-nnF (expert fine-tuned from nnF) | 2.407 | common | 300 | ✓ |
| `sam` | ResUNet, scratch | L1 + SAM 2.1 slice features | 2.150 | common | 300 | ✓ |
| `vgg` | ResUNet, scratch | L1 + VGG16 slice features | 1.037 | common | 300 | ✓ |
| `mednet` | ResUNet, scratch | L1 + MedicalNet ResNet-50 features | 6.521 | common | 300 | ✓ |
| `l1`, `l1_nnf` | ResUNet, scratch / pretrained | L1 | – | common | 300 | recipe check |
| `mam_scratch300`, `mam_scratch300_nnf` | ResUNet, scratch / pretrained | L1 + AFP_seg | 0.1264 | common | 300 | recipe check |
| `ssl`, `ssl_nnf` | ResUNet, scratch / pretrained | L1 + feat. nnF | 11.78 | common | 300 | recipe check |
| `swin_l1` | SwinUNETR, scratch | L1 | – | Adam | 250 | ✓ |
| `swin_mam_scratch300` | SwinUNETR, scratch | L1 + AFP_seg | 0.1523 | Adam | 250 | ✓ |
| `gan_nnf` | ResUNet + PatchGAN, pretrained | L1 + adversarial | – | Adam | 200 | ✓ |
| `gan_nnf_mam_scratch300` | ResUNet + PatchGAN, pretrained | L1 + AFP_seg + adversarial | 0.1264 | Adam | 200 | ✓ |

The exact λ values are in each config and in `configs/doses/`. λ was calibrated for one start and recipe per loss and
generator (`scripts/calibrate.sh`) and reused, without recalibration, for the other starts and recipes of the same
loss, including the GAN (`dose_reused_from`).
ResUNet rows of Table 2 use the nnF recipe whenever it was trained, as it matched or beat the common recipe on the
development set; the common-recipe runs are kept for that comparison.

## Segmentation experts (AFP teachers)

| Config | Start | Loss | Dev Dice (ep 300) | Teacher of |
|---|---|---|---:|---|
| `experts/scratch` | random | Dice + CE | 0.561 | AFP_seg |
| `experts/nnf_ft` | nnFoundationCNN encoder | Dice + CE | 0.718 | AFP_seg-ft-nnF |

Both use the nnF recipe and must pass mean dev Dice ≥ 0.5 with at most four empty predictions (`synthesis/experts/gate.py`).
Only their six encoder stages are used as the AFP feature extractor.

## Reproducibility notes

- One training seed per model; `cudnn.benchmark` was on, so exact GPU bit reproduction is not claimed.
- MedicalNet λ was calibrated with an fp16 generator, whereas training and inference used bf16; VGG trained in fp16
  and inferred in bf16.
- Some runs were resumed from checkpoints (scratch-expert AFP ResUNet after epoch 100, SwinUNETR+AFP at epoch 61,
  GAN L1 at epoch 2); resuming restores optimiser, scaler and sampler states.
