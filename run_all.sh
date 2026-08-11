#!/usr/bin/env bash
# Full pipeline. Run from the project root.
set -e
echo "[1/5] Generating test data ..."      && python3 src/make_testdata.py
echo "[2/5] Training forecast model ..."   && python3 src/train_forecast.py
echo "[3/5] Training anomaly model ..."    && python3 src/train_anomaly.py
echo "[4/5] Evaluating sensor checks ..."  && python3 src/sensor_check.py
echo "[5/5] Combined evaluation ..."       && python3 src/evaluate_combined.py
echo
echo "Done. Results in results/, models in models/."
