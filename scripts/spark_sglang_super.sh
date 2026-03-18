#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-sglang}"
MODEL_PATH="${MODEL_PATH:-/home/hmbown/HF_Models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/nemotron-3-super-120b-a12b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-nemotron_3}"

if [[ ! -x "$VENV_DIR/bin/sglang" ]]; then
  echo "Missing SGLang runtime at $VENV_DIR/bin/sglang" >&2
  echo "Run 'nemo setup sglang' or install the server env first." >&2
  exit 1
fi

export PATH="$VENV_DIR/bin:$PATH"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
if [[ -z "${TRITON_PTXAS_PATH:-}" ]] && command -v ptxas >/dev/null 2>&1; then
  export TRITON_PTXAS_PATH
  TRITON_PTXAS_PATH="$(command -v ptxas)"
fi
export MAX_JOBS="${MAX_JOBS:-1}"

exec "$VENV_DIR/bin/sglang" serve \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --reasoning-parser "$REASONING_PARSER"
