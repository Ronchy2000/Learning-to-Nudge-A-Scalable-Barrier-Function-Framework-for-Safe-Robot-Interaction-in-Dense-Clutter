#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
python3 -m dcbf.eval.rollout --env_config configs/env.yaml "$@"
