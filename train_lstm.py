"""
OPTIONAL: LSTM forecast model, for the comparison table in the paper.

Not required. The gradient boosting model in train_forecast.py is a genuine
trained model and satisfies the assignment. Run this only if you want a
neural network row in the results table.

An honest expectation before you start: on a single-station dataset with a
1 h horizon, an LSTM usually does NOT beat gradient boosting. Tree ensembles
are very strong on tabular lag features, and the LSTM has to learn from raw
sequences what the feature engineering already handed the trees for free.
If your LSTM comes out slightly worse, that is a normal and reportable
result, not a bug. Say so in the paper - a negative result that is explained
is worth more than a tuned number without justification.

Requires:  pip install tensorflow
Usage:     python src/train_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data_loader import load_dataset

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise SystemExit(
        "TensorFlow is not installed.\n"
        "  pip install tensorflow\n"
        "This script is optional - the project works without it.")

SEQ_LEN = 72          # 72 x 10 min = 12 h of history fed to the LSTM
CHANNELS = ["temperature", "humidity", "pressure"]


def make_sequences(arr: np.ndarray, target: np.ndarray, seq_len: int, horizon: int):
    """Slide a window over the series to build (samples, timesteps, channels)."""
    n = len(arr) - seq_len - horizon + 1
    X = np.empty((n, seq_len, arr.shape[1]), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        X[i] = arr[i:i + seq_len]
        y[i] = target[i + seq_len + horizon - 1]
    return X, y


def main():
    print("=" * 68)
    print("OPTIONAL: LSTM FORECAST MODEL")
    print("=" * 68)

    df = load_dataset()
    df = df[CHANNELS].dropna()
    horizon = cfg.FORECAST_HORIZON_MIN // cfg.STEP_MINUTES

    values = df.to_numpy(dtype=np.float32)
    target = df[cfg.TARGET].to_numpy(dtype=np.float32)

    cut = int(len(values) * (1 - cfg.TEST_SIZE))

    # Standardise using TRAINING statistics only - using the full series
    # would leak test information into the scaling.
    mu, sd = values[:cut].mean(0), values[:cut].std(0) + 1e-8
    scaled = (values - mu) / sd

    Xtr, ytr = make_sequences(scaled[:cut], target[:cut], SEQ_LEN, horizon)
    Xte, yte = make_sequences(scaled[cut:], target[cut:], SEQ_LEN, horizon)
    print(f"\nTrain sequences: {Xtr.shape}   Test: {Xte.shape}")

    # Persistence baseline on exactly the same test samples.
    persist = np.array([target[cut + i + SEQ_LEN - 1] for i in range(len(yte))])
    base_mae = float(np.mean(np.abs(yte - persist)))

    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, len(CHANNELS))),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.15),
        keras.layers.LSTM(32),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    stop = keras.callbacks.EarlyStopping(monitor="val_mae", patience=5,
                                         restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=40, batch_size=256,
              callbacks=[stop], verbose=2)

    pred = model.predict(Xte, verbose=0).ravel()
    mae = float(np.mean(np.abs(yte - pred)))
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    print("\n" + "-" * 68)
    print(f"Persistence MAE : {base_mae:.4f}")
    print(f"LSTM MAE        : {mae:.4f}")
    print(f"LSTM RMSE       : {rmse:.4f}")
    print(f"LSTM R2         : {r2:.4f}")
    print(f"Skill vs persistence: {100 * (1 - mae / base_mae):+.2f}%")

    model.save(cfg.MODEL_DIR / "forecast_lstm.keras")
    pd.DataFrame([{
        "Model": "LSTM", "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Skill_vs_Persistence_%": round(100 * (1 - mae / base_mae), 2),
    }]).to_csv(cfg.RESULTS_DIR / "lstm_metrics.csv", index=False)
    print("\nSaved -> models/forecast_lstm.keras, results/lstm_metrics.csv")


if __name__ == "__main__":
    main()
