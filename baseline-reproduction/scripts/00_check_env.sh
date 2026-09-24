#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

echo "=== environment check ==="
echo "ROOT=${ROOT}"

ok=0
for c in git python3 curl; do
  if command -v "$c" >/dev/null 2>&1; then echo "OK  $c"; else echo "NO  $c"; ok=1; fi
done

if command -v uv >/dev/null 2>&1; then echo "OK  uv"; else echo "NO  uv  (needed to install OpenJarvis)"; ok=1; fi
if command -v ollama >/dev/null 2>&1; then echo "OK  ollama"; else echo "NO  ollama (needed for qwen3.5:9b)"; ok=1; fi
if command -v jarvis >/dev/null 2>&1; then echo "OK  jarvis"; else echo "NO  jarvis (will use uv run after clone)"; fi

if [[ -d "${THIRD_PARTY}/.git" ]]; then
  echo "OK  OpenJarvis clone at ${THIRD_PARTY}"
  git -C "${THIRD_PARTY}" rev-parse --short HEAD
else
  echo "NO  OpenJarvis clone — run scripts/01_clone_openjarvis.sh"
fi

if command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | grep -qi 'qwen3.5:9b\|qwen3.5'; then
    echo "OK  qwen3.5 model present"
    ollama list | grep -i qwen || true
  else
    echo "NO  qwen3.5:9b — run scripts/02_pull_model.sh"
  fi
fi

echo "disk:"
df -h "${ROOT}" | tail -1
exit "${ok}"
