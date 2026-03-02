#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="outputs/refine/${TS}"
mkdir -p outputs/refine

if [ -f outputs/train/LATEST_RUN ]; then
  LATEST_TRAIN_DIR="$(cat outputs/train/LATEST_RUN)"
  set -- --checkpoint "${LATEST_TRAIN_DIR}/best.pt" "$@"
  echo "[run_refine] use latest checkpoint: ${LATEST_TRAIN_DIR}/best.pt"
fi
if [ -f outputs/train/LATEST_CKPT ]; then
  LATEST_TRAIN_CKPT="$(cat outputs/train/LATEST_CKPT)"
  set -- --checkpoint "${LATEST_TRAIN_CKPT}" "$@"
  echo "[run_refine] use latest checkpoint: ${LATEST_TRAIN_CKPT}"
fi

if [ -f outputs/data/LATEST_RUN ]; then
  LATEST_DATA_DIR="$(cat outputs/data/LATEST_RUN)"
  set -- --dataset_glob "${LATEST_DATA_DIR}/train_*.npz" "$@"
  echo "[run_refine] use latest data dir: ${LATEST_DATA_DIR}"
fi

echo "[run_refine] output_dir=${OUT_DIR}"
python3 -m dcbf.refinement.refine --config configs/refine.yaml --output_dir "${OUT_DIR}" --run_name "refined_dcbf_${TS}" "$@"
printf '%s\n' "${OUT_DIR}" > outputs/refine/LATEST_RUN
echo "[run_refine] latest -> $(cat outputs/refine/LATEST_RUN)"
LATEST_REFINE_CKPT="$(find "${OUT_DIR}" -maxdepth 3 -type f -name best.pt | sort | tail -n 1)"
if [ -n "${LATEST_REFINE_CKPT}" ]; then
  printf '%s\n' "${LATEST_REFINE_CKPT}" > outputs/refine/LATEST_CKPT
  echo "[run_refine] latest_ckpt -> $(cat outputs/refine/LATEST_CKPT)"
fi
