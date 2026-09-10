"""
Attention-BiLSTM forecasting, evaluated against the existing baseline.

WHY THIS EXISTS

The conference paper proposes replacing the engineered-feature models with a
sequence model. That proposal rests on an assumption nobody has tested yet:
that a BiLSTM beats gradient boosting on this data. This script settles it
before the federated experiments are built on top.

The honest expectation, stated up front so a poor result is not mistaken for
a bug: at a 60-minute horizon on a single station, tree ensembles are very
strong on engineered lag features. Measured evidence from this project -
removing 50 minutes of history from the feature set cost only 0.7 percentage
points of skill - suggests most of the usable signal sits in the current
state and its short-term tendency, which is exactly the part a sequence
model has no advantage in extracting.

TWO SEQUENCE LENGTHS, DELIBERATELY

  60 min   identical information to the baseline. The fair comparison: does
           the architecture help, holding input information constant?
  12 h     what a sequence model would normally be given. Answers whether
           longer history is worth anything at this horizon.

Reporting only the second would confound two effects - a better architecture
and more input - and could not distinguish them.

CONSTRAINTS (from CLAUDE.md, do not change)
  - same dataset, same chronological split, same 60-minute horizon
  - scaling fitted on training data only
  - skill over persistence is the headline metric, not R2

Usage:  python3 train_bilstm.py [--seq-minutes 60] [--epochs 40]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data_loader import load_dataset

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise SystemExit(
        "TensorFlow is not installed.\n"
        "  .venv-tf/bin/pip install tensorflow\n"
        "then run this script with .venv-tf/bin/python")

CHANNELS = ["temperature", "humidity", "pressure"]


def make_sequences(arr: np.ndarray, target: np.ndarray, seq_len: int,
                   horizon: int):
    """
    Slide a window over the series -> (samples, timesteps, channels).

    Built with stride tricks rather than a Python loop: on 420k rows the loop
    version copies the data seq_len times over and is the slowest part of the
    script by a wide margin.
    """
    n = len(arr) - seq_len - horizon + 1
    if n <= 0:
        raise SystemExit(f"Not enough data for seq_len={seq_len}")
    stride = arr.strides[0]
    X = np.lib.stride_tricks.as_strided(
        arr, shape=(n, seq_len, arr.shape[1]),
        strides=(stride, stride, arr.strides[1]), writeable=False)
    y = target[seq_len + horizon - 1: seq_len + horizon - 1 + n]
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


def build_model(seq_len: int, n_channels: int, units: int = 64):
    """
    BiLSTM followed by additive attention over the time axis.

    The attention layer scores every timestep, normalises the scores with a
    softmax, and returns the weighted sum of the BiLSTM outputs. Two reasons
    for this design rather than a plain final-state readout: it lets the model
    weight recent and older observations differently, and the weights can be
    read out afterwards to show which part of the history the forecast relied
    on - the interpretability claim in the paper draft.
    """
    inp = keras.layers.Input(shape=(seq_len, n_channels), name="sequence")

    seq = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=True), name="bilstm")(inp)
    seq = keras.layers.Dropout(0.15)(seq)

    # Additive (Bahdanau-style) attention over timesteps.
    score = keras.layers.Dense(32, activation="tanh", name="att_hidden")(seq)
    score = keras.layers.Dense(1, name="att_score")(score)
    weights = keras.layers.Softmax(axis=1, name="attention_weights")(score)
    context = keras.layers.Multiply()([seq, weights])
    context = keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name="context")(context)

    x = keras.layers.Dense(32, activation="relu")(context)
    out = keras.layers.Dense(1, name="forecast")(x)
    return keras.Model(inp, out, name="attention_bilstm")


def metrics(y_true, y_pred, baseline_mae):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(1 - ss_res / ss_tot, 4),
        "Skill_vs_Persistence_%": round(100 * (1 - mae / baseline_mae), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-minutes", type=int, default=60,
                    help="history length in minutes (60 = same as baseline)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    seq_len = max(2, args.seq_minutes // cfg.STEP_MINUTES)
    horizon = cfg.FORECAST_HORIZON_MIN // cfg.STEP_MINUTES

    print("=" * 70)
    print(f"ATTENTION-BiLSTM  |  {args.seq_minutes} min history "
          f"({seq_len} steps)  |  {cfg.FORECAST_HORIZON_MIN} min horizon")
    print("=" * 70)

    df = load_dataset()[CHANNELS].dropna()
    values = df.to_numpy(dtype=np.float32)
    target = df[cfg.TARGET].to_numpy(dtype=np.float32)

    # Chronological split - never shuffled. Same 80/20 cut as the baseline.
    cut = int(len(values) * (1 - cfg.TEST_SIZE))

    # Scaling statistics from the training portion only. Using the whole
    # series would leak test information into the inputs.
    mu, sd = values[:cut].mean(0), values[:cut].std(0) + 1e-8
    scaled = (values - mu) / sd

    Xtr, ytr = make_sequences(scaled[:cut], target[:cut], seq_len, horizon)
    Xte, yte = make_sequences(scaled[cut:], target[cut:], seq_len, horizon)
    print(f"\nTrain {Xtr.shape}   Test {Xte.shape}")
    print(f"Memory: {(Xtr.nbytes + Xte.nbytes) / 1e9:.2f} GB")

    # Persistence on exactly these test samples: the last observed
    # temperature of each window. Recomputed here rather than reused from the
    # baseline run, because the sequence construction drops a different number
    # of leading rows and the sample sets must match exactly.
    persist = target[cut + seq_len - 1: cut + seq_len - 1 + len(yte)]
    base_mae = float(np.mean(np.abs(yte - persist)))
    print(f"Persistence on these samples: MAE {base_mae:.4f}")

    model = build_model(seq_len, len(CHANNELS), args.units)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=["mae"])
    print(f"Parameters: {model.count_params():,}")

    stop = keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=5, restore_best_weights=True, verbose=1)

    t0 = time.time()
    model.fit(Xtr, ytr, validation_split=0.15, epochs=args.epochs,
              batch_size=args.batch, callbacks=[stop], verbose=2)
    train_seconds = time.time() - t0

    pred = model.predict(Xte, batch_size=512, verbose=0).ravel()
    m = metrics(yte, pred, base_mae)
    m["persistence_MAE"] = round(base_mae, 4)
    m["seq_minutes"] = args.seq_minutes
    m["parameters"] = int(model.count_params())
    m["train_seconds"] = round(train_seconds, 1)
    m["test_samples"] = int(len(yte))

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)
    for k, v in m.items():
        print(f"  {k:<26} {v}")

    tag = f"{args.seq_minutes}min"
    out = cfg.RESULTS_DIR / f"bilstm_{tag}.json"
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    model.save(cfg.MODEL_DIR / f"forecast_bilstm_{tag}.keras")
    print(f"\nSaved -> results/bilstm_{tag}.json, "
          f"models/forecast_bilstm_{tag}.keras")

    # Compare against the committed baseline rather than a remembered number.
    base_path = cfg.RESULTS_DIR / "forecast_metrics.json"
    if base_path.exists():
        base = json.load(open(base_path))
        gb = base.get("GradientBoosting", {})
        if gb:
            print("\n" + "-" * 70)
            print("AGAINST THE EXISTING BASELINE")
            print("-" * 70)
            print(f"  Gradient Boosting   MAE {gb['MAE']:.4f}   "
                  f"skill {gb['Skill_vs_Persistence_%']:+.2f} %")
            print(f"  Attention-BiLSTM    MAE {m['MAE']:.4f}   "
                  f"skill {m['Skill_vs_Persistence_%']:+.2f} %")
            diff = m["MAE"] - gb["MAE"]
            verdict = "BiLSTM better" if diff < 0 else "Gradient Boosting better"
            print(f"  Difference          {diff:+.4f} degC   -> {verdict}")


if __name__ == "__main__":
    main()
