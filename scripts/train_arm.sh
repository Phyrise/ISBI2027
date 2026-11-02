#!/usr/bin/env bash
# Train one model of configs/arms on one GPU, in the paper's slices (ResUNet arms: 0-100, 100-200, 200-300; the
# SwinUNETR / GAN arms in one go), resuming automatically.
#   scripts/train_arm.sh <arm> [cuda device, default cuda:0]
# <arm>: a file name of configs/arms without .yaml (l1, ssl, mam_scratch300, ..., gan_nnf_mam_scratch300).
# AFP arms with an expert teacher need that expert first (scripts/train_expert.sh).
set -euo pipefail
ARM=${1:?arm name}; DEVICE=${2:-cuda:0}
cd "$(dirname "$0")/.."
CFG=configs/arms/$ARM.yaml
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
TRAINER=$(python -c "from synthesis.core.config import load_config; print(load_config('$CFG')['trainer'])")
EPOCHS=$(python -c "from synthesis.core.config import load_config; print(load_config('$CFG')['epochs'])")
if [ "$TRAINER" = sgd ]; then SLICES="100 200 $EPOCHS"; else SLICES="$EPOCHS"; fi
for STOP in $SLICES; do
  RESUME=""; [ -f "$ISBI_RUNS/$ARM/checkpoint_latest.pth" ] && RESUME=--resume
  python -m synthesis.trainers.run --config "$CFG" --stop "$STOP" --device "$DEVICE" $RESUME
done
