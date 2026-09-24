#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

if [[ ! -d "${THIRD_PARTY}/.git" ]]; then
  echo "Clone first: bash scripts/01_clone_openjarvis.sh" >&2
  exit 1
fi

need_cmd ollama
PYTHON="${THIRD_PARTY}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "OpenJarvis venv missing. cd third_party/OpenJarvis && uv sync" >&2
  exit 1
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT="${RESULTS_RUNS}/smoke/${STAMP}"
LOG="${RESULTS_LOGS}/smoke_${STAMP}.log"
mkdir -p "${OUT}"

log "SMOKE: 2 PinchBench tasks, seed 42, qwen3.5:9b, tools auto-approved"
log "log=${LOG}"

"${PYTHON}" "${ROOT}/scripts/run_pinchbench.py" \
  --condition skills_on \
  --model qwen3.5:9b \
  --engine ollama \
  --seeds 42 \
  --max-samples 2 \
  --max-turns 40 \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG}"

log "Smoke finished. If that worked, run bash scripts/04_full_eval.sh"
