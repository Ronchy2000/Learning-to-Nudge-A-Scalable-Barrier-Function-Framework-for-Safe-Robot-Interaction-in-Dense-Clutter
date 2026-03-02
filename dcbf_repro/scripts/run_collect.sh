#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="outputs/data/${TS}"
mkdir -p outputs/data
echo "[run_collect] output_dir=${OUT_DIR}"
python3 -m dcbf.data.collect collect --config configs/env.yaml --output_dir "${OUT_DIR}" "$@"
printf '%s\n' "${OUT_DIR}" > outputs/data/LATEST_RUN
echo "[run_collect] latest -> $(cat outputs/data/LATEST_RUN)"
