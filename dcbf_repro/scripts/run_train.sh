#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p outputs/train

# Detect if user passed --run_name
RUN_NAME="initial_dcbf_${TS}"
for _arg in "$@"; do
  case "${_prev_arg:-}" in
    --run_name) RUN_NAME="$_arg" ;;
  esac
  _prev_arg="$_arg"
done

if [ -f outputs/data/LATEST_RUN ]; then
  LATEST_DATA_DIR="$(cat outputs/data/LATEST_RUN)"
  set -- --train_glob "${LATEST_DATA_DIR}/train_*.npz" --val_glob "${LATEST_DATA_DIR}/val_*.npz" "$@"
  echo "[run_train] use latest data dir: ${LATEST_DATA_DIR}"
fi

echo "[run_train] run_name=${RUN_NAME}"
python3 -m dcbf.training.train --config configs/train.yaml --run_name "${RUN_NAME}" "$@"
printf '%s\n' "outputs/train/${RUN_NAME}" > outputs/train/LATEST_RUN
echo "[run_train] latest -> $(cat outputs/train/LATEST_RUN)"
if [ -f "outputs/train/${RUN_NAME}/best.pt" ]; then
  printf '%s\n' "outputs/train/${RUN_NAME}/best.pt" > outputs/train/LATEST_CKPT
  echo "[run_train] latest_ckpt -> $(cat outputs/train/LATEST_CKPT)"
elif [ -f "outputs/train/${RUN_NAME}/latest.pt" ]; then
  printf '%s\n' "outputs/train/${RUN_NAME}/latest.pt" > outputs/train/LATEST_CKPT
  echo "[run_train] latest_ckpt (fallback) -> $(cat outputs/train/LATEST_CKPT)"
fi
