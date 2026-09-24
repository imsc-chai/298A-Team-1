#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

need_cmd ollama
PYTHON="${THIRD_PARTY}/.venv/bin/python"
STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT="${RESULTS_RUNS}/table1_23/${STAMP}"
LOG="${RESULTS_LOGS}/table1_23_${STAMP}.log"
mkdir -p "${OUT}"

log "Table 1 PinchBench: 23 tasks, 1 seed (42), qwen3.5:9b"
log "This is NOT guaranteed to hit 87.9%. Expect several hours. log=${LOG}"

"${PYTHON}" "${ROOT}/scripts/run_pinchbench.py" \
  --condition skills_on \
  --model qwen3.5:9b \
  --engine ollama \
  --seeds 42 \
  --max-turns 40 \
  --tasks-file "${ROOT}/configs/pinchbench_table1_23.txt" \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG}"

log "23-task run finished."
