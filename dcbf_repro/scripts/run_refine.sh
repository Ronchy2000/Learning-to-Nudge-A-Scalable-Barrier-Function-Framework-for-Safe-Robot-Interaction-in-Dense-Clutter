#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
python3 -m dcbf.refinement.refine --config configs/refine.yaml "$@"
