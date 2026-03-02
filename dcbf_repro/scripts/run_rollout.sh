#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
TS="${DCBF_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="outputs/eval/rollout/${TS}.json"
mkdir -p outputs/eval/rollout
echo "[run_rollout] output=${OUT_JSON}"
python3 -m dcbf.eval.rollout --env_config configs/env.yaml --output "${OUT_JSON}" "$@"
