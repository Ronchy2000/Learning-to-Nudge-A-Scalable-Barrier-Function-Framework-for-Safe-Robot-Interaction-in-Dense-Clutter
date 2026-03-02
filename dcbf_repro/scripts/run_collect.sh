#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
python3 -m dcbf.data.collect collect --config configs/env.yaml "$@"
