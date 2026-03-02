#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
python3 -m dcbf.training.train --config configs/train.yaml "$@"
