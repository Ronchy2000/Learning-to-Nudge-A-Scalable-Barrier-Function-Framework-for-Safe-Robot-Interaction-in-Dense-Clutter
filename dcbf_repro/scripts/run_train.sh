#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m dcbf.training.train --config configs/train.yaml "$@"
