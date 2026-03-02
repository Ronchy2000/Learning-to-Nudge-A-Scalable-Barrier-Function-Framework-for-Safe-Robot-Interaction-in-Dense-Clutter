#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  python3 -m dcbf.eval.evaluate --help
  exit 0
fi
python3 -m dcbf.eval.evaluate --config configs/eval.yaml "$@"
python3 -m dcbf.eval.plot --csv outputs/eval/metrics.csv --output outputs/eval/metrics_plot.png
