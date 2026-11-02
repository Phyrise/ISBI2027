#!/usr/bin/env bash
# Train one AFP lesion expert (in-house SGD loop with the nnU-Net Dice + CE loss, not a stock nnUNetv2 trainer),
# in the paper's slices 0-100 and 100-300, with the dev40 quality gate after each slice.
#   scripts/train_expert.sh <scratch | nnf_ft> [cuda device, default cuda:0]
set -euo pipefail
EXPERT=${1:?scratch or nnf_ft}; DEVICE=${2:-cuda:0}
cd "$(dirname "$0")/.."
CFG=configs/experts/$EXPERT.yaml
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
RUN=$(python -c "from synthesis.core.config import load_config; print(load_config('$CFG')['run_name'])")
for STOP in 100 300; do
  RESUME=""; [ -f "$ISBI_RUNS/$RUN/checkpoint_latest.pth" ] && RESUME=--resume
  # Only the epoch-300 gate is fatal (the paper continued the scratch expert after its epoch-100 gate failed).
  python -m synthesis.experts.train_expert --config "$CFG" --stop "$STOP" --device "$DEVICE" $RESUME
done
