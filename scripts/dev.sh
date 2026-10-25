#!/usr/bin/env bash
# Usage: scripts/dev.sh test | fast | api | ui | baseline
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  test)     python -m pytest -q ;;
  fast)     python -m pytest -q tests/test_api_contract.py ;;
  api)      uvicorn api.main:app --reload ;;
  ui)       streamlit run app.py --server.fileWatcherType none ;;
  baseline) python ~/baseline_latency.py --runs 20 ;;
  *)        echo "Usage: $0 test|fast|api|ui|baseline"; exit 1 ;;
esac
