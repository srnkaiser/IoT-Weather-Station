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

  60 min   same history length as the baseline. Run it with --calendar to
           hold the input information genuinely constant: does the
           architecture help by itself?
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

@keras.utils.register_keras_serializable(package="weather")
class AttentionPool(keras.layers.Layer):
    """
    Sum a sequence along the time axis after attention weighting.

    This was a Lambda layer until it turned out that a Lambda saves as a
    pickled Python function with no declared output shape, so the saved model
    could not be loaded again - not by us, and not by anyone we send it to.
    A registered layer serialises by name and reloads cleanly.
    """

    def call(self, inputs):
        seq, weights = inputs
        return tf.reduce_sum(seq * weights, axis=1)

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[2])


BASE_CHANNELS = ["temperature", "humidity", "pressure"]

# The baseline feature set contains four calendar features (hour_sin/cos,
# doy_sin/cos) that a raw sequence does not carry. A sequence model can read
# the recent trend, but it cannot know whether 14 degC at a rising trend is a
# spring morning or an autumn afternoon - and at a 60-minute horizon the point
# of the daily cycle matters. Withholding them would make the comparison
# unfair to the sequence model and the result easy to dismiss, so they are
# available as extra input channels via --calendar.
CALENDAR_CHANNELS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Same cyclic encoding as features.py, so both models see it identically."""
    idx = df.index
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.dayofyear.to_numpy().astype(float)
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


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


def build_model(seq_len: int, n_channels: int, units: int = 64,
                attention: bool = True, bidirectional: bool = True):
    """
    BiLSTM followed by additive attention over the time axis.

    The attention layer scores every timestep, normalises the scores with a
    softmax, and returns the weighted sum of the BiLSTM outputs. Two reasons
    for this design rather than a plain final-state readout: it lets the model
    weight recent and older observations differently, and the weights can be
    read out afterwards to show which part of the history the forecast relied
    on - the interpretability claim in the paper draft.

    The two switches exist so the name can be taken apart. "Attention-BiLSTM"
    asserts that both halves earn their place; the only way to know is to
    remove one at a time and measure. Without attention the readout becomes
    the final hidden state, which is what a plain BiLSTM regressor does;
    without bidirectionality the layer is an ordinary forward LSTM.
    """
    inp = keras.layers.Input(shape=(seq_len, n_channels), name="sequence")

    core = keras.layers.LSTM(units, return_sequences=attention)
    seq = (keras.layers.Bidirectional(core, name="bilstm")(inp)
           if bidirectional else
           keras.layers.LSTM(units, return_sequences=attention,
                             name="lstm")(inp))
    seq = keras.layers.Dropout(0.15)(seq)

    if attention:
        # Additive (Bahdanau-style) attention over timesteps.
        score = keras.layers.Dense(32, activation="tanh", name="att_hidden")(seq)
        score = keras.layers.Dense(1, name="att_score")(score)
        weights = keras.layers.Softmax(axis=1, name="attention_weights")(score)
        context = AttentionPool(name="context")([seq, weights])
    else:
        # return_sequences=False already collapsed the time axis.
        context = seq

    x = keras.layers.Dense(32, activation="relu")(context)
    out = keras.layers.Dense(1, name="forecast")(x)
    name = ("attention_bilstm" if attention and bidirectional else
            "bilstm" if bidirectional else
            "attention_lstm" if attention else "lstm")
    return keras.Model(inp, out, name=name)


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
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="initial learning rate")
    ap.add_argument("--patience", type=int, default=5,
                    help="early-stopping patience in epochs")
    ap.add_argument("--reduce-lr", action="store_true",
                    help="halve the learning rate when validation stalls")
    ap.add_argument("--no-attention", action="store_true",
                    help="ablation: read out the final state instead of an "
                         "attention-weighted sum")
    ap.add_argument("--unidirectional", action="store_true",
                    help="ablation: forward LSTM instead of bidirectional")
    ap.add_argument("--features", action="store_true",
                    help="feed the baseline's engineered feature set as "
                         "channels instead of the three raw sensor series; "
                         "separates 'better architecture' from 'better inputs'")
    ap.add_argument("--calendar", action="store_true",
                    help="add the baseline's four calendar features as extra "
                         "input channels (makes the input sets comparable)")
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_STATE,
                    help="random seed; vary it to measure run-to-run spread")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for the output files")
    args = ap.parse_args()

    # Without this the weight initialisation differs on every run, so two
    # results cannot be compared at all - a 0.004 degC difference would be
    # indistinguishable from noise. Seeded, a repeat run reproduces exactly,
    # and varying the seed on purpose measures the spread.
    keras.utils.set_random_seed(args.seed)

    seq_len = max(2, args.seq_minutes // cfg.STEP_MINUTES)
    horizon = cfg.FORECAST_HORIZON_MIN // cfg.STEP_MINUTES

    print("=" * 70)
    print(f"ATTENTION-BiLSTM  |  {args.seq_minutes} min history "
          f"({seq_len} steps)  |  {cfg.FORECAST_HORIZON_MIN} min horizon")
    print("=" * 70)

    if args.features:
        # The decisive control. If the sequence model matches the baseline
        # once it is handed the same engineered inputs, then the gap was
        # never about the architecture and the federated design can keep a
        # sequence model without paying for it in accuracy. If it still
        # loses, the architecture itself is the weaker choice here.
        #
        # The feature set already contains lags, so a window over it is
        # redundant by construction. That is the point: it holds the input
        # information constant with the baseline, which is what is being
        # tested.
        from features import build_features
        base = load_dataset()
        feat = build_features(base).dropna()
        feat[cfg.TARGET] = base[cfg.TARGET].reindex(feat.index)
        df = feat.dropna()
        channels = [c for c in df.columns if c != cfg.TARGET] + [cfg.TARGET]
        df = df[channels]
        print(f"Input channels: {len(channels)} engineered features "
              f"(same set as the baseline, plus the target series)")
    else:
        channels = BASE_CHANNELS + (CALENDAR_CHANNELS if args.calendar else [])
        df = load_dataset()[BASE_CHANNELS].dropna()
        if args.calendar:
            df = add_calendar(df)
        df = df[channels]
        print(f"Input channels: {', '.join(channels)}")
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

    model = build_model(seq_len, len(channels), args.units,
                        attention=not args.no_attention,
                        bidirectional=not args.unidirectional)
    print(f"Architecture: {model.name}")
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="mse",
                  metrics=["mae"])
    print(f"Parameters: {model.count_params():,}")

    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=args.patience,
        restore_best_weights=True, verbose=1)]
    if args.reduce_lr:
        # Halving the rate when validation stalls gives the model a chance to
        # settle into a minimum instead of bouncing past it - the 12-hour run
        # peaked at epoch 3 and then degraded, which is what that looks like.
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=max(2, args.patience // 3),
            min_lr=1e-5, verbose=1))

    t0 = time.time()
    model.fit(Xtr, ytr, validation_split=0.15, epochs=args.epochs,
              batch_size=args.batch, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - t0

    pred = model.predict(Xte, batch_size=512, verbose=0).ravel()
    m = metrics(yte, pred, base_mae)
    m["persistence_MAE"] = round(base_mae, 4)
    m["seq_minutes"] = args.seq_minutes
    m["parameters"] = int(model.count_params())
    m["train_seconds"] = round(train_seconds, 1)
    m["test_samples"] = int(len(yte))
    m["learning_rate"] = args.lr
    m["units"] = args.units
    m["patience"] = args.patience
    m["reduce_lr_on_plateau"] = bool(args.reduce_lr)
    m["calendar_channels"] = bool(args.calendar)
    m["engineered_features"] = bool(args.features)
    m["n_channels"] = len(channels)
    m["seed"] = args.seed
    m["architecture"] = model.name
    m["attention"] = not args.no_attention
    m["bidirectional"] = not args.unidirectional

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)
    for k, v in m.items():
        print(f"  {k:<26} {v}")

    tag = args.tag or f"{args.seq_minutes}min"
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
