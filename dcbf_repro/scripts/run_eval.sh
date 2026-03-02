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

resolve_best_ckpt() {
  SEARCH_DIR="$1"
  if [ -f "${SEARCH_DIR}/best.pt" ]; then
    printf '%s\n' "${SEARCH_DIR}/best.pt"
    return 0
  fi
  FOUND="$(find "${SEARCH_DIR}" -maxdepth 3 -type f -name best.pt | sort | tail -n 1)"
  if [ -n "${FOUND}" ]; then
    printf '%s\n' "${FOUND}"
    return 0
  fi
  return 1
}

if [ -f outputs/train/LATEST_CKPT ]; then
  TRAIN_CKPT="$(cat outputs/train/LATEST_CKPT)"
  echo "[run_eval] use initial_checkpoint=${TRAIN_CKPT}"
  set -- --initial_checkpoint "${TRAIN_CKPT}" "$@"
elif [ -f outputs/train/LATEST_RUN ]; then
  LATEST_TRAIN_DIR="$(cat outputs/train/LATEST_RUN)"
  if TRAIN_CKPT="$(resolve_best_ckpt "${LATEST_TRAIN_DIR}")"; then
    echo "[run_eval] use initial_checkpoint=${TRAIN_CKPT}"
    set -- --initial_checkpoint "${TRAIN_CKPT}" "$@"
  fi
fi

if [ -f outputs/refine/LATEST_CKPT ]; then
  REFINE_CKPT="$(cat outputs/refine/LATEST_CKPT)"
  echo "[run_eval] use refined_checkpoint=${REFINE_CKPT}"
  set -- --refined_checkpoint "${REFINE_CKPT}" "$@"
elif [ -f outputs/refine/LATEST_RUN ]; then
  LATEST_REFINE_DIR="$(cat outputs/refine/LATEST_RUN)"
  if REFINE_CKPT="$(resolve_best_ckpt "${LATEST_REFINE_DIR}")"; then
    echo "[run_eval] use refined_checkpoint=${REFINE_CKPT}"
    set -- --refined_checkpoint "${REFINE_CKPT}" "$@"
  fi
fi

echo "[run_eval] output_dir=${OUT_DIR}"
python3 -m dcbf.eval.evaluate --config configs/eval.yaml --output_dir "${OUT_DIR}" "$@"
python3 -m dcbf.eval.plot --csv "${OUT_DIR}/metrics.csv" --output "${OUT_DIR}/metrics_plot.png"
printf '%s\n' "${OUT_DIR}" > outputs/eval/LATEST_RUN
echo "[run_eval] latest -> $(cat outputs/eval/LATEST_RUN)"
