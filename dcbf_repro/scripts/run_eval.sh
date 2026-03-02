#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  python3 -m dcbf.eval.evaluate --help
  exit 0
fi

TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="outputs/eval/${TS}"
mkdir -p outputs/eval

if [ -f outputs/train/LATEST_RUN ]; then
  LATEST_TRAIN_DIR="$(cat outputs/train/LATEST_RUN)"
  set -- --initial_checkpoint "${LATEST_TRAIN_DIR}/best.pt" "$@"
fi

if [ -f outputs/refine/LATEST_RUN ]; then
  LATEST_REFINE_DIR="$(cat outputs/refine/LATEST_RUN)"
  REFINE_TS="$(basename "${LATEST_REFINE_DIR}")"
  set -- --refined_checkpoint "${LATEST_REFINE_DIR}/refined_dcbf_${REFINE_TS}/best.pt" "$@"
fi

echo "[run_eval] output_dir=${OUT_DIR}"
python3 -m dcbf.eval.evaluate --config configs/eval.yaml --output_dir "${OUT_DIR}" "$@"
python3 -m dcbf.eval.plot --csv "${OUT_DIR}/metrics.csv" --output "${OUT_DIR}/metrics_plot.png"
printf '%s\n' "${OUT_DIR}" > outputs/eval/LATEST_RUN
echo "[run_eval] latest -> $(cat outputs/eval/LATEST_RUN)"
