#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m dcbf.refinement.refine --config configs/refine.yaml "$@"
