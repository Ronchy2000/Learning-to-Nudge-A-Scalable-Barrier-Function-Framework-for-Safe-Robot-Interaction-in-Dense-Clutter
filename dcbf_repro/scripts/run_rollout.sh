#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m dcbf.eval.rollout --env_config configs/env.yaml "$@"
