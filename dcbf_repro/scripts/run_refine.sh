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

if [ -f outputs/data/LATEST_RUN ]; then
  LATEST_DATA_DIR="$(cat outputs/data/LATEST_RUN)"
  set -- --dataset_glob "${LATEST_DATA_DIR}/train_*.npz" "$@"
  echo "[run_refine] use latest data dir: ${LATEST_DATA_DIR}"
fi

echo "[run_refine] output_dir=${OUT_DIR}"
python3 -m dcbf.refinement.refine --config configs/refine.yaml --output_dir "${OUT_DIR}" --run_name "refined_dcbf_${TS}" "$@"
printf '%s\n' "${OUT_DIR}" > outputs/refine/LATEST_RUN
echo "[run_refine] latest -> $(cat outputs/refine/LATEST_RUN)"
