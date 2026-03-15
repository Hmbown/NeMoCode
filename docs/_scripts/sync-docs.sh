#!/usr/bin/env bash
# sync-docs.sh — Check NVIDIA documentation URLs for availability and fetch latest content summaries.
# Run periodically to detect new products, changed URLs, or broken links.
#
# Usage: ./docs/_scripts/sync-docs.sh [--check-only] [--verbose]

set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_FILE="$DOCS_DIR/_scripts/sync-report.md"
CHECK_ONLY=false
VERBOSE=false

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=true ;;
    --verbose) VERBOSE=true ;;
  esac
done

# Canonical URLs to check
declare -A URLS=(
  # NeMo
  ["NeMo Hub"]="https://docs.nvidia.com/nemo/index.html"
  ["NeMo Framework Guide"]="https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html"
  ["NeMo Microservices"]="https://docs.nvidia.com/nemo/microservices/latest/index.html"
  ["NeMo Guardrails"]="https://docs.nvidia.com/nemo/guardrails/index.html"
  ["NeMo Agent Toolkit"]="https://docs.nvidia.com/nemo/agent-toolkit/latest/"

  # NIM
  ["NIM Hub"]="https://docs.nvidia.com/nim/index.html"
  ["NIM LLMs"]="https://docs.nvidia.com/nim/large-language-models/latest/index.html"
  ["NIM Speech"]="https://docs.nvidia.com/nim/speech/latest/"
  ["NIM Text Embedding"]="https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/"
  ["NIM Text Reranking"]="https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/"
  ["NIM NV-CLIP"]="https://docs.nvidia.com/nim/nvclip/latest/"
  ["NIM Cosmos"]="https://docs.nvidia.com/nim/cosmos/latest/"
  ["NIM Visual GenAI"]="https://docs.nvidia.com/nim/visual-genai/latest/"
  ["NIM Multimodal Safety"]="https://docs.nvidia.com/nim/multimodal-safety/latest/"

  # API
  ["API Catalog"]="https://build.nvidia.com/explore/discover"
  ["API Docs"]="https://docs.api.nvidia.com/"
  ["NIM API Reference"]="https://docs.api.nvidia.com/nim/reference"

  # Inference
  ["Triton Docs"]="https://docs.nvidia.com/deeplearning/triton-inference-server/"
  ["TensorRT-LLM Docs"]="https://nvidia.github.io/TensorRT-LLM/"
  ["TensorRT Docs"]="https://docs.nvidia.com/deeplearning/tensorrt/index.html"
  ["Dynamo Docs"]="https://docs.nvidia.com/dynamo/"

  # Enterprise
  ["AI Enterprise"]="https://docs.nvidia.com/ai-enterprise/index.html"

  # Catalogs
  ["NGC Catalog"]="https://catalog.ngc.nvidia.com"
  ["NVIDIA Learn"]="https://learn.nvidia.com"

  # GitHub repos
  ["GitHub NeMo"]="https://github.com/NVIDIA/NeMo"
  ["GitHub NeMo-Guardrails"]="https://github.com/NVIDIA/NeMo-Guardrails"
  ["GitHub NeMo-Curator"]="https://github.com/NVIDIA/NeMo-Curator"
  ["GitHub NeMo-Skills"]="https://github.com/NVIDIA/NeMo-Skills"
  ["GitHub NeMo-Run"]="https://github.com/NVIDIA/NeMo-Run"
  ["GitHub Agent Toolkit"]="https://github.com/NVIDIA/nemo-agent-toolkit"
  ["GitHub TensorRT-LLM"]="https://github.com/NVIDIA/TensorRT-LLM"
  ["GitHub Triton"]="https://github.com/triton-inference-server/server"
  ["GitHub Dynamo"]="https://github.com/ai-dynamo/dynamo"
  ["GitHub GenAI Examples"]="https://github.com/NVIDIA/GenerativeAIExamples"
)

echo "# NVIDIA Docs Sync Report" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

ok_count=0
fail_count=0
redirect_count=0

echo "| Resource | URL | Status |" >> "$REPORT_FILE"
echo "|---|---|---|" >> "$REPORT_FILE"

for name in $(echo "${!URLS[@]}" | tr ' ' '\n' | sort); do
  url="${URLS[$name]}"

  # Check URL status
  http_code=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 10 "$url" 2>/dev/null || echo "000")

  if [[ "$http_code" == "200" ]]; then
    status="OK"
    ((ok_count++))
  elif [[ "$http_code" =~ ^3[0-9]{2}$ ]]; then
    status="REDIRECT ($http_code)"
    ((redirect_count++))
  elif [[ "$http_code" == "000" ]]; then
    status="TIMEOUT"
    ((fail_count++))
  else
    status="FAILED ($http_code)"
    ((fail_count++))
  fi

  echo "| $name | $url | $status |" >> "$REPORT_FILE"

  if $VERBOSE; then
    printf "%-30s %s %s\n" "$name" "$http_code" "$url"
  fi
done

echo "" >> "$REPORT_FILE"
echo "## Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "- OK: $ok_count" >> "$REPORT_FILE"
echo "- Redirects: $redirect_count" >> "$REPORT_FILE"
echo "- Failed: $fail_count" >> "$REPORT_FILE"
echo "- Total: $((ok_count + fail_count + redirect_count))" >> "$REPORT_FILE"

echo ""
echo "=== Sync Report ==="
echo "OK: $ok_count | Redirects: $redirect_count | Failed: $fail_count"
echo "Report saved to: $REPORT_FILE"

if [[ $fail_count -gt 0 ]]; then
  echo ""
  echo "WARNING: $fail_count URLs failed. Check the report for details."
  echo "Failed URLs may indicate moved or removed documentation."
  exit 1
fi
