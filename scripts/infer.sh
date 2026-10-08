#!/usr/bin/env bash
# Inference with trained artifacts: top-k documents for a question file, or an ad-hoc search.
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash scripts/infer.sh predict QUESTIONS_JSON [-o OUT_DIR] [-k N] [-a ARTIFACTS]
  bash scripts/infer.sh search "QUERY" [-k N] [-a ARTIFACTS]

Commands:
  predict   write OUT_DIR/predictions.json ({qid: [doc_id, ...]}) (default OUT_DIR: output)
  search    print the top-k documents for one question (default k: 5)

Options:
  -a, --artifacts DIR   trained artifacts folder (default: artifacts)
  -h, --help            show this help

Examples:
  bash scripts/infer.sh predict examples/data/questions.json -o output
  bash scripts/infer.sh search "Thủ tục đăng ký kết hôn với người nước ngoài?" -k 10
USAGE
}

[[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]] && { usage; exit 0; }
CMD=$1; shift
[[ $# -gt 0 && "$1" != -* ]] || { echo "missing argument for $CMD" >&2; usage >&2; exit 1; }
ARG=$1; shift

ARTIFACTS=artifacts OUT=output K=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--artifacts) ARTIFACTS=$2; shift 2 ;;
    -o|--out) OUT=$2; shift 2 ;;
    -k) K=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
PY=${PY:-python}
case "$CMD" in
  predict) "$PY" -m src.pipeline predict --questions "$ARG" --artifacts "$ARTIFACTS" --out "$OUT" -k "$K" ;;
  search) "$PY" -m src.pipeline search "$ARG" -k "$K" --artifacts "$ARTIFACTS" ;;
  *) echo "unknown command: $CMD" >&2; usage >&2; exit 1 ;;
esac
