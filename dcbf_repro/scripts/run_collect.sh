#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m dcbf.data.collect collect --config configs/env.yaml "$@"
