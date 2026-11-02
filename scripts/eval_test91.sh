#!/usr/bin/env bash
# test91 evaluation of the models of configs/eval/test91.yaml, the pre-contrast copy and the real image (reference rows).
#
#   scripts/eval_test91.sh STAGE [STAGE ...]      STAGE in: gen read cpu lpips percep table  (or: all)
#
# Stages, their environment and their outputs (all under $ISBI_EVAL_OUT, one JSON per entry and patient):
#   gen     TRAINING environment ($PY_GEN): torch 2.8 + the training-side architectures (dynamic-network-architectures
#           fork with the trilinear residual decoder; MONAI 1.5.1 for SwinUNETR); GPU.
#           Per arm: the training side's inference (python -m synthesis.inference.predict with configs/arms/<key>.yaml and
#           $ISBI_RUNS/<key>/checkpoint_ep<epoch>.pth: full volume, 50 % overlap, Gaussian blending, no mirroring;
#           fp16, bf16 for mednet, vgg, Swin and GAN arms) writes a float32 prediction to pred_float32/<entry>/;
#           evaluation.metrics.image computes image/<entry>/<case>.json on it (as for the paper values) and stores the float16
#           copy pred/<entry>/<case>.npy read by every later stage; the float32 folder is then removed.
#           An extra arm without a training config is skipped. The copy's image metrics are written here too.
#   read    EVALUATION environment ($PY_EVAL) with STOCK nnunetv2 2.6.2 and no modified nnU-Net on PYTHONPATH; GPU.
#           MAMA-MIA reader -> reader/
#   cpu     EVALUATION environment, CPU: PSNR -> psnr/, false enhancement -> offtarget/, sharpness -> sharpness/
#   lpips   EVALUATION environment (lpips 0.1.4); GPU (CPU works, a few minutes per patient and entry) -> lpips/
#   percep  EVALUATION environment + MedDINOv3 code ($ISBI_MEDDINO_REPO); GPU -> meddino/, nnf_vit/
#   table   EVALUATION environment, CPU: per-patient CSVs, per_arm.csv, ranks and table rows -> tables/
#
# Required environment variables (no path is hard-coded in the repository):
#   ISBI_DATA_TEST     prepared test cases (<case>/data.npy, body.npy, label.npy, properties.json)
#   ISBI_EVAL_OUT      output root
#   ISBI_MANIFEST      split manifest (native NIfTI paths of the post-contrast and expert mask, for the reader)
#   ISBI_READER_DIR    MAMA-MIA nnU-Net release folder (full_image_dce_mri_tumor_segmentation)
#   ISBI_MEDDINO_CKPT  MedDINOv3 CT3M checkpoint;  ISBI_MEDDINO_REPO  clone of github.com/ricklisz/MedDINOv3
#   ISBI_NNF_VIT_DIR   nnFoundationViT folder (checkpoint_final.pth + PROVENANCE.json)
#   ISBI_RUNS          training run folders (<arm>/checkpoint_ep<epoch>.pth), gen stage only (same variable as the
#                      training side)
# Optional: PY_GEN, PY_EVAL (interpreters; default python), DEVICE (default cuda), WORKERS (default 4).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
PY_GEN="${PY_GEN:-python}"; PY_EVAL="${PY_EVAL:-python}"
DEVICE="${DEVICE:-cuda}"; WORKERS="${WORKERS:-4}"
: "${ISBI_DATA_TEST:?}" "${ISBI_EVAL_OUT:?}"
OUT="$ISBI_EVAL_OUT"; PRED="$OUT/pred"; DATA="$ISBI_DATA_TEST"

# Arms of configs/eval/test91.yaml: "<key> <epoch> <1 = Table 2 arm, 0 = extra arm>".
mapfile -t ARM_ROWS < <("$PY_EVAL" -c "
import yaml; c = yaml.safe_load(open('configs/eval/test91.yaml'))
for t1, arms in ((1, c['arms']), (0, c.get('extra_arms', []))):
    for a in arms: print(a['key'], f\"{a['epoch']:03d}\", t1)")

# ARMS = arm entries ("<key>_ep<epoch>") that have predictions; a Table 2 arm without predictions is an error.
load_entries() {
  ARMS=()
  local row key ep t1
  for row in "${ARM_ROWS[@]}"; do
    read -r key ep t1 <<< "$row"
    if [ -d "$PRED/${key}_ep$ep" ]; then ARMS+=("${key}_ep$ep")
    elif [ "$t1" = 1 ]; then echo "missing predictions: $PRED/${key}_ep$ep" >&2; exit 1
    else echo "note: extra arm ${key}_ep$ep has no predictions, skipped" >&2; fi
  done
}

stage_gen() {
  : "${ISBI_RUNS:?}"
  local row key ep t1 e
  for row in "${ARM_ROWS[@]}"; do
    read -r key ep t1 <<< "$row"; e="${key}_ep$ep"
    if [ ! -f "configs/arms/$key.yaml" ]; then echo "note: no training config for $key, not generated" >&2; continue; fi
    "$PY_GEN" -m synthesis.inference.predict --config "configs/arms/$key.yaml" --checkpoint "$ISBI_RUNS/$key/checkpoint_ep$ep.pth" \
      --split test --data-dir "$DATA" --out "$OUT/pred_float32/$e" --device "$DEVICE" --save-dtype float32
    "$PY_EVAL" -m evaluation.metrics.image run "$e" --data "$DATA" --pred-dir "$OUT/pred_float32" --out "$OUT/image" --store-float16 "$PRED"
    rm -r "$OUT/pred_float32/$e"
  done
  "$PY_EVAL" -m evaluation.metrics.image copy --data "$DATA" --out "$OUT/image"
}

stage_read() {
  load_entries
  : "${ISBI_READER_DIR:?}" "${ISBI_MANIFEST:?}"
  "$PY_EVAL" -m evaluation.metrics.segmenter.mama_reader real pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/reader" \
    --reader-dir "$ISBI_READER_DIR" --manifest "$ISBI_MANIFEST" --device "$DEVICE"
}

stage_cpu() {
  load_entries
  "$PY_EVAL" -m evaluation.metrics.global_.psnr pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/psnr" --workers "$WORKERS"
  "$PY_EVAL" -m evaluation.metrics.offtarget.false_enhancement run real "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/offtarget" --workers "$WORKERS"
  "$PY_EVAL" -m evaluation.metrics.offtarget.false_enhancement summary real "${ARMS[@]}" --out "$OUT/offtarget"
  "$PY_EVAL" -m evaluation.metrics.perceptual.sharpness run pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/sharpness"
}

stage_lpips() {
  load_entries
  "$PY_EVAL" -m evaluation.metrics.perceptual.lpips_slices pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/lpips" --device "$DEVICE"
}

stage_percep() {
  load_entries
  : "${ISBI_MEDDINO_CKPT:?}" "${ISBI_MEDDINO_REPO:?}" "${ISBI_NNF_VIT_DIR:?}"
  "$PY_EVAL" -m evaluation.metrics.perceptual.meddino pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/meddino" --device "$DEVICE"
  "$PY_EVAL" -m evaluation.metrics.perceptual.nnf_vit pre "${ARMS[@]}" --data "$DATA" --pred-dir "$PRED" --out "$OUT/nnf_vit" --device "$DEVICE"
}

stage_table() {
  local T="$OUT/tables"
  "$PY_EVAL" scripts/collect_test91.py --eval-out "$OUT" --out "$T"
  "$PY_EVAL" evaluation/ranking/per_arm.py --per-patient "$T/per_patient" --out "$T/per_arm.csv"
  "$PY_EVAL" evaluation/ranking/family_rank.py "$T/per_arm.csv" --out "$T/ranks.csv"
  "$PY_EVAL" evaluation/ranking/tables.py "$T/per_arm.csv" --out "$T/table1_rows.tex"
  "$PY_EVAL" evaluation/ranking/bootstrap.py table2 --per-patient "$T/per_patient" > "$T/table2_rows.tex"
}

[ $# -ge 1 ] || { sed -n '2,4p' "$0"; exit 1; }
[ "$1" = all ] && set -- gen read cpu lpips percep table
for s in "$@"; do
  case "$s" in
    gen|read|cpu|lpips|percep|table) echo "== stage $s"; "stage_$s" ;;
    *) echo "unknown stage: $s" >&2; exit 1 ;;
  esac
done
