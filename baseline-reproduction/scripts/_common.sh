#!/usr/bin/env bash
# Shared paths. Source from other scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export THIRD_PARTY="${ROOT}/third_party/OpenJarvis"
export RESULTS_RUNS="${ROOT}/results/runs"
export RESULTS_LOGS="${ROOT}/results/logs"
export CONFIG_FULL="${ROOT}/configs/table1_c_hermes_openjarvis_qwen3.5-9b.yaml"
export CONFIG_SMOKE="${ROOT}/configs/smoke.yaml"

mkdir -p "${RESULTS_RUNS}" "${RESULTS_LOGS}" "${ROOT}/data" "${ROOT}/third_party"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    return 1
  fi
}

jarvis_bin() {
  if [[ -x "${THIRD_PARTY}/.venv/bin/jarvis" ]]; then
    echo "${THIRD_PARTY}/.venv/bin/jarvis"
    return
  fi
  if command -v jarvis >/dev/null 2>&1; then
    command -v jarvis
    return
  fi
  echo ""
}
