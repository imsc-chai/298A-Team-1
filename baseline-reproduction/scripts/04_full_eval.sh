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
  echo "OpenJarvis venv missing." >&2
  exit 1
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT="${RESULTS_RUNS}/table1_c/${STAMP}"
LOG="${RESULTS_LOGS}/table1_c_${STAMP}.log"
mkdir -p "${OUT}"

echo "$(git -C "${THIRD_PARTY}" rev-parse HEAD)" > "${OUT}/openjarvis_commit.txt"
uname -a > "${OUT}/hardware.txt"
date -u > "${OUT}/started_utc.txt"

log "FULL Table 1 (c): PinchBench, qwen3.5:9b, seeds 42-46, tools auto-approved"
log "Published target: 87.9%"
log "This can take many hours. log=${LOG}"

"${PYTHON}" "${ROOT}/scripts/run_pinchbench.py" \
  --condition skills_on \
  --model qwen3.5:9b \
  --engine ollama \
  --seeds 42,43,44,45,46 \
  --max-samples 0 \
  --max-turns 40 \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG}"

date -u > "${OUT}/ended_utc.txt"
log "Full eval finished. Run: python3 scripts/05_summarize_results.py"
