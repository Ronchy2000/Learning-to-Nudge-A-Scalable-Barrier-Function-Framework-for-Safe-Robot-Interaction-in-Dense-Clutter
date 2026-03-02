#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="initial_dcbf_${TS}"
mkdir -p outputs/train

if [ -f outputs/data/LATEST_RUN ]; then
  LATEST_DATA_DIR="$(cat outputs/data/LATEST_RUN)"
  set -- --train_glob "${LATEST_DATA_DIR}/train_*.npz" --val_glob "${LATEST_DATA_DIR}/val_*.npz" "$@"
  echo "[run_train] use latest data dir: ${LATEST_DATA_DIR}"
fi

echo "[run_train] run_name=${RUN_NAME}"
python3 -m dcbf.training.train --config configs/train.yaml --run_name "${RUN_NAME}" "$@"
printf '%s\n' "outputs/train/${RUN_NAME}" > outputs/train/LATEST_RUN
echo "[run_train] latest -> $(cat outputs/train/LATEST_RUN)"
