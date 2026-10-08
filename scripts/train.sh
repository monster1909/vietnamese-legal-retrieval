#!/usr/bin/env bash
# Train the retriever end to end: chunk + BM25 -> fine-tune encoders -> tune fusion -> evaluate.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: bash scripts/train.sh [options]

Options:
  -d, --data DIR        folder with train.json and selected-contexts.zip (default: data)
  -a, --artifacts DIR   output folder for models, indexes and best_config.json (default: artifacts)
  -t, --trials N        Optuna trials for fusion tuning (default: 1500)
  -s, --steps LIST      comma-separated subset of train,tune,evaluate (default: all three)
  -h, --help            show this help

Examples:
  bash scripts/train.sh
  bash scripts/train.sh --data /mnt/legal --artifacts /mnt/out --trials 500
  bash scripts/train.sh --steps tune,evaluate   # re-tune on existing models
USAGE
}

DATA=data ARTIFACTS=artifacts TRIALS=1500 STEPS=train,tune,evaluate
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--data) DATA=$2; shift 2 ;;
    -a|--artifacts) ARTIFACTS=$2; shift 2 ;;
    -t|--trials) TRIALS=$2; shift 2 ;;
    -s|--steps) STEPS=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
PY=${PY:-python}
for step in ${STEPS//,/ }; do
  case "$step" in
    train) "$PY" -m src.pipeline train --data "$DATA" --artifacts "$ARTIFACTS" ;;
    tune) "$PY" -m src.pipeline tune --data "$DATA" --artifacts "$ARTIFACTS" --trials "$TRIALS" ;;
    evaluate) "$PY" -m src.pipeline evaluate --data "$DATA" --artifacts "$ARTIFACTS" ;;
    *) echo "unknown step: $step" >&2; exit 1 ;;
  esac
done
