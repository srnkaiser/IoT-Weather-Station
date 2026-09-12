#!/usr/bin/env bash
# Reproduce the conference-paper forecasting experiments.
#
# Deliberately separate from run_all.sh. That script reproduces the submitted
# student project and needs nothing but scikit-learn; adding TensorFlow steps
# to it would break the path a reviewer of that submission follows. This one
# is the BRAINCON work and needs the TensorFlow environment:
#
#   python3 -m venv .venv-tf && .venv-tf/bin/pip install tensorflow
#
# Expect roughly three hours on a MacBook Air M2 (CPU only). Every run writes
# results/bilstm_<tag>.json and models/forecast_bilstm_<tag>.keras, so an
# interrupted sweep can be resumed by deleting the finished lines below.
set -e

TF=.venv-tf/bin/python
if [ ! -x "$TF" ]; then
  echo "No TensorFlow environment at $TF - see the header of this file." >&2
  exit 1
fi

# The settled configuration. Tuning was measured and is worth 0.0004 degC, so
# these are not the product of a search - they are simply what the runs used.
# Sequence length is passed per run, not baked in here: "${CFG[@]/a/b}"
# substitutes element by element, and "--seq-minutes" and "60" are two
# elements, so a replacement would silently not match.
CFG=(--lr 3e-4 --units 96 --patience 12 --reduce-lr --epochs 60)
H60=(--seq-minutes 60)
H12H=(--seq-minutes 720)

echo "=== The comparison that matters: same inputs as the baseline ==="
# Five seeds, not one. This configuration spreads ~0.009 degC between seeds,
# about a third of the gap being reported, so a single run is not a result.
for s in 42 7 1 2 3; do
  tag=60min_calendar; [ "$s" != "42" ] && tag=60min_calendar_s$s
  $TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --calendar --seed $s --tag $tag
done

echo "=== Without the calendar features: what withholding them costs ==="
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --seed 42 --tag 60min_tuned
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --seed 7  --tag 60min_tuned_s7

echo "=== Does more history help? ==="
$TF train_bilstm.py "${H12H[@]}" "${CFG[@]}" --calendar --seed 42 \
    --tag 720min_calendar

echo "=== Ablations: does each half of 'Attention-BiLSTM' earn its name? ==="
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --calendar --seed 42 --no-attention   --tag 60min_noatt
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --calendar --seed 42 --unidirectional --tag 60min_unidir
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --calendar --seed 42 --no-attention --unidirectional \
    --tag 60min_plainlstm

echo "=== Architecture or inputs? Same engineered features as the baseline ==="
$TF train_bilstm.py "${H60[@]}" "${CFG[@]}" --seed 42 --features --tag 60min_features

echo "=== Aligned comparison, significance, regimes, attention, figures ==="
python3 compare_models.py --export-baseline
for f in results/bilstm_*.json; do
  tag=$(basename "$f" .json); tag=${tag#bilstm_}
  $TF compare_models.py --export-bilstm "$tag"
done
python3 compare_models.py --analyse
python3 compare_models.py --plots

echo
echo "Done. See results/model_comparison.md and the three PNGs beside it."
