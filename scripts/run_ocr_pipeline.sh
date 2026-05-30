#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PADDLEOCR_IMAGE="${PADDLEOCR_IMAGE:-ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120}"
OCR_EXECUTION_MODE="${OCR_EXECUTION_MODE:-docker}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ocr_pipeline.sh <command> [doc_id]

Commands:
  extract     Run PaddleOCR-VL extraction only.
  normalize   Build normalized documents and segments only.
  pretrain    Build per-document pretrain JSONL only.
  sft         Build per-document SFT JSONL only.
  all         Run extract -> normalize -> pretrain -> sft.

Environment:
  OCR_EXECUTION_MODE=docker|local   How to run OCR extraction. Default: docker
  PYTHON_BIN=python3                Python interpreter for non-OCR steps.
  PADDLEOCR_IMAGE=<image>           Container image for OCR extraction.

Examples:
  scripts/run_ocr_pipeline.sh all
  scripts/run_ocr_pipeline.sh all Build_a_Large_Language_Model_From_Scratch
  OCR_EXECUTION_MODE=local scripts/run_ocr_pipeline.sh extract ABCD
EOF
}

require_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
  }
}

run_extract() {
  local doc_id="${1:-}"
  local args=(python "/workspace/project/scripts/ocr_extract.py" --overwrite)
  local local_args=("$REPO_ROOT/scripts/ocr_extract.py" --overwrite)
  if [[ -n "$doc_id" ]]; then
    args+=(--doc-id "$doc_id")
    local_args+=(--doc-id "$doc_id")
  fi
  if [[ "$OCR_EXECUTION_MODE" == "docker" ]]; then
    docker run --rm \
      --gpus all \
      --network host \
      --user "$(id -u):$(id -g)" \
      -v "$REPO_ROOT:/workspace/project" \
      "$PADDLEOCR_IMAGE" \
      "${args[@]}"
  else
    require_python
    "$PYTHON_BIN" "${local_args[@]}"
  fi
}

run_normalize() {
  local doc_id="${1:-}"
  local args=("$REPO_ROOT/scripts/ocr_normalize.py")
  if [[ -n "$doc_id" ]]; then
    args+=(--doc-id "$doc_id")
  fi
  require_python
  "$PYTHON_BIN" "${args[@]}"
}

run_pretrain() {
  local doc_id="${1:-}"
  local args=("$REPO_ROOT/scripts/build_pretrain_jsonl.py")
  if [[ -n "$doc_id" ]]; then
    args+=(--doc-id "$doc_id")
  fi
  require_python
  "$PYTHON_BIN" "${args[@]}"
}

run_sft() {
  local doc_id="${1:-}"
  local args=("$REPO_ROOT/scripts/build_sft_jsonl.py")
  if [[ -n "$doc_id" ]]; then
    args+=(--doc-id "$doc_id")
  fi
  require_python
  "$PYTHON_BIN" "${args[@]}"
}

main() {
  local command="${1:-}"
  local doc_id="${2:-}"

  if [[ -z "$command" ]]; then
    usage
    exit 1
  fi

  case "$command" in
    extract)
      run_extract "$doc_id"
      ;;
    normalize)
      run_normalize "$doc_id"
      ;;
    pretrain)
      run_pretrain "$doc_id"
      ;;
    sft)
      run_sft "$doc_id"
      ;;
    all)
      run_extract "$doc_id"
      run_normalize "$doc_id"
      run_pretrain "$doc_id"
      run_sft "$doc_id"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
