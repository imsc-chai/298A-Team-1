#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

need_cmd ollama
log "Pulling qwen3.5:9b via Ollama (several GB)..."
ollama pull qwen3.5:9b
ollama list | grep -i qwen || true
log "Done. Next: bash scripts/03_smoke_eval.sh"
