#!/usr/bin/env bash
# Calibrate lambda for one AFP arm (median gradient-norm ratio over 64 fixed batches at the common random init).
#   scripts/calibrate.sh <arm> [cuda device, default cuda:0]
# Writes $ISBI_RUNS/doses/<arm>.json; the paper values are in configs/doses/ and in each arm's loss.lambda.
# The other starts and recipes of a loss reuse its dose (dose_reused_from in the config, see docs/recipes.md).
set -euo pipefail
ARM=${1:?arm name}; DEVICE=${2:-cuda:0}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python -m synthesis.losses.calibrate --config "configs/arms/$ARM.yaml" --device "$DEVICE"
