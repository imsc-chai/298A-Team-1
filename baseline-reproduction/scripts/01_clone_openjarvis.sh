#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

mkdir -p "${ROOT}/third_party"
if [[ -d "${THIRD_PARTY}/.git" ]]; then
  log "OpenJarvis already cloned. Fetching latest main..."
  git -C "${THIRD_PARTY}" fetch --depth 1 origin main
  git -C "${THIRD_PARTY}" checkout --force FETCH_HEAD
else
  log "Cloning OpenJarvis (depth 1)..."
  git clone --depth 1 https://github.com/open-jarvis/OpenJarvis.git "${THIRD_PARTY}"
fi

log "Pinned commit: $(git -C "${THIRD_PARTY}" rev-parse HEAD)"
git -C "${THIRD_PARTY}" rev-parse HEAD > "${ROOT}/results/published/openjarvis_commit.txt"
log "Next: bash scripts/00_check_env.sh && bash scripts/02_pull_model.sh"
