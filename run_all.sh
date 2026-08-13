#!/usr/bin/env bash
# Full pipeline. Run from the project root.
set -euo pipefail

# Resolve paths from this file so the script can also be run elsewhere.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Prefer the project's virtual environment when it exists.
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

echo "[1/5] Generating test data ..."      && "$PYTHON" "$PROJECT_ROOT/make_testdata.py"
echo "[2/5] Training forecast model ..."   && "$PYTHON" "$PROJECT_ROOT/train_forecast.py"
echo "[3/5] Training anomaly model ..."    && "$PYTHON" "$PROJECT_ROOT/train_anomaly.py"
echo "[4/5] Evaluating sensor checks ..."  && "$PYTHON" "$PROJECT_ROOT/sensor_check.py"
echo "[5/5] Combined evaluation ..."       && "$PYTHON" "$PROJECT_ROOT/evaluate_combined.py"
echo
echo "Done. Results in results/, models in models/."
