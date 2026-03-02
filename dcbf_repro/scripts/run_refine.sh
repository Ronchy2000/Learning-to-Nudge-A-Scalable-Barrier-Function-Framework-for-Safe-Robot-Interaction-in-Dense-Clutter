#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p outputs/refine

# Detect if user passed --output_dir or --run_name
OUT_DIR="outputs/refine/${TS}"
REFINE_RUN_NAME="refined_dcbf_${TS}"
for _arg in "$@"; do
  case "${_prev_arg:-}" in
    --output_dir) OUT_DIR="$_arg" ;;
    --run_name)   REFINE_RUN_NAME="$_arg" ;;
  esac
  _prev_arg="$_arg"
done

# Auto-detect latest checkpoint (only if user did not pass --checkpoint)
_has_checkpoint=false
for _arg in "$@"; do
  [ "$_arg" = "--checkpoint" ] && _has_checkpoint=true
done
if [ "$_has_checkpoint" = false ]; then
  if [ -f outputs/train/LATEST_CKPT ]; then
    LATEST_TRAIN_CKPT="$(cat outputs/train/LATEST_CKPT)"
    set -- --checkpoint "${LATEST_TRAIN_CKPT}" "$@"
    echo "[run_refine] use latest checkpoint: ${LATEST_TRAIN_CKPT}"
  elif [ -f outputs/train/LATEST_RUN ]; then
    LATEST_TRAIN_DIR="$(cat outputs/train/LATEST_RUN)"
    set -- --checkpoint "${LATEST_TRAIN_DIR}/best.pt" "$@"
    echo "[run_refine] use latest checkpoint: ${LATEST_TRAIN_DIR}/best.pt"
  fi
fi

if [ -f outputs/data/LATEST_RUN ]; then
  LATEST_DATA_DIR="$(cat outputs/data/LATEST_RUN)"
  # Only add if user did not pass --dataset_glob
  _has_dg=false
  for _arg in "$@"; do [ "$_arg" = "--dataset_glob" ] && _has_dg=true; done
  if [ "$_has_dg" = false ]; then
    set -- --dataset_glob "${LATEST_DATA_DIR}/train_*.npz" "$@"
    echo "[run_refine] use latest data dir: ${LATEST_DATA_DIR}"
  fi
fi

echo "[run_refine] output_dir=${OUT_DIR}  run_name=${REFINE_RUN_NAME}"
python3 -m dcbf.refinement.refine --config configs/refine.yaml \
  --output_dir "${OUT_DIR}" --run_name "${REFINE_RUN_NAME}" "$@"

printf '%s\n' "${OUT_DIR}" > outputs/refine/LATEST_RUN
echo "[run_refine] latest -> $(cat outputs/refine/LATEST_RUN)"
# Resolve best checkpoint; fallback to latest.pt if best.pt doesn't exist
LATEST_REFINE_CKPT="$(find "${OUT_DIR}" -maxdepth 3 -type f -name best.pt | sort | tail -n 1)"
if [ -z "${LATEST_REFINE_CKPT}" ]; then
  LATEST_REFINE_CKPT="$(find "${OUT_DIR}" -maxdepth 3 -type f -name latest.pt | sort | tail -n 1)"
fi
if [ -n "${LATEST_REFINE_CKPT}" ]; then
  printf '%s\n' "${LATEST_REFINE_CKPT}" > outputs/refine/LATEST_CKPT
  echo "[run_refine] latest_ckpt -> $(cat outputs/refine/LATEST_CKPT)"
fi
