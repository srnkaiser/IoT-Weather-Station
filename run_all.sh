#!/usr/bin/env bash
# Full training and evaluation pipeline. Run from the project root:
#   ./run_all.sh
#
# All scripts live in the project root, not in src/ - the paths in the README
# tree are aspirational. Live inference (predict_live.py) is NOT part of this;
# it runs separately once the models exist.
set -e

DATASET=$(python3 -c "import config; print(config.DATASET)")
echo "Dataset: $DATASET"
echo

if [ "$DATASET" = "synthetic" ]; then
  echo "[0/4] Generating synthetic test data ..."
  python3 make_testdata.py
  echo
fi

echo "[1/6] Training forecast model ..."   && python3 train_forecast.py
echo
echo "[2/6] Training fallback model ..."   && python3 train_fallback.py
echo
echo "[3/6] Training anomaly model ..."    && python3 train_anomaly.py
echo
echo "[4/6] Evaluating sensor checks ..."  && python3 sensor_check.py
echo
echo "[5/6] Combined evaluation ..."       && python3 evaluate_combined.py
echo
echo "[6/6] Contamination sensitivity ..." && python3 tune_contamination.py
echo
echo "Done. Results in results/, models in models/."
echo
echo "Live inference:"
echo "  python3 predict_live.py              # once"
echo "  python3 predict_live.py --watch 300  # every 5 minutes"
