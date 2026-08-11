"""
Feature engineering for the weather forecast model.

Design rule: every feature must be derivable at inference time from what the
DHT22 and BMP180 actually deliver plus the system clock. No wind, no solar
radiation, no station metadata - the ESP32 cannot measure those.

Feature groups
--------------
lags        value N minutes ago                -> short-term memory
tendencies  current minus value N minutes ago  -> direction of change
rolling     mean / std over a window           -> level and variability
dewpoint    Magnus formula from T and RH       -> physical humidity measure
spread      T - dewpoint                       -> how close to saturation
cyclic      sin/cos of hour and day-of-year    -> time without a discontinuity

Why the pressure tendency matters most: absolute pressure depends on station
altitude, so a model trained on it would not transfer to our station. The
3-6 h change does transfer, and it is the classic synoptic predictor -
falling pressure announces an approaching front.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def magnus_dewpoint(t_c, rh_pct):
    """Dew point in degC via the Magnus formula (Sonntag coefficients)."""
    a, b = 17.62, 243.12
    rh = np.clip(rh_pct, 1.0, 100.0)
    gamma = np.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * gamma) / (a - gamma)


def _steps(minutes: int, step_minutes: int) -> int:
    return max(1, int(round(minutes / step_minutes)))


def build_features(df: pd.DataFrame,
                   step_minutes: int | None = None,
                   lags_min: list[int] | None = None,
                   tendency_min: list[int] | None = None,
                   rolling_min: list[int] | None = None) -> pd.DataFrame:
    """Turn the raw time series into a model input matrix."""
    step_minutes = step_minutes or cfg.STEP_MINUTES
    lags_min = lags_min or cfg.LAGS_MIN
    tendency_min = tendency_min or cfg.TENDENCY_MIN
    rolling_min = rolling_min or cfg.ROLLING_MIN

    cols = [c for c in cfg.BASE_COLUMNS if c in df.columns]
    X = pd.DataFrame(index=df.index)

    # Current readings
    for c in cols:
        X[c] = df[c]

    # Lags, tendencies, rolling statistics
    for c in cols:
        s = df[c]
        for m in lags_min:
            X[f"{c}_lag{m}"] = s.shift(_steps(m, step_minutes))
        for m in tendency_min:
            X[f"{c}_tend{m}"] = s - s.shift(_steps(m, step_minutes))
        for m in rolling_min:
            k = _steps(m, step_minutes)
            r = s.rolling(k, min_periods=k // 2)
            X[f"{c}_mean{m}"] = r.mean()
            X[f"{c}_std{m}"] = r.std()
            # Range = max - min in the window. One feature covering both
            # failure directions: exactly 0 means the sensor is stuck,
            # an unusually large value means a spike occurred.
            X[f"{c}_range{m}"] = r.max() - r.min()

    # --- Altitude-invariant pressure --------------------------------------
    # Absolute pressure depends on station altitude. Jena sits at ~155 m, our
    # station will not. A model trained on absolute hPa would carry that
    # offset as a learned bias and misread every value at deployment.
    # Replace the absolute level by the anomaly against its own 24 h mean.
    # Tendencies and standard deviations are already differences and
    # therefore altitude-invariant by construction.
    if "pressure" in cols and cfg.ALTITUDE_INVARIANT_PRESSURE:
        ref = X["pressure_mean1440"] if "pressure_mean1440" in X else X["pressure"]
        X["pressure_anom"] = X["pressure"] - ref
        drop = [c for c in X.columns
                if c.startswith("pressure")
                and ("lag" in c or "mean" in c or c == "pressure")]
        X = X.drop(columns=drop)

    # Physical humidity features
    if "temperature" in cols and "humidity" in cols:
        dp = magnus_dewpoint(df["temperature"].to_numpy(), df["humidity"].to_numpy())
        X["dewpoint"] = dp
        X["dewpoint_spread"] = df["temperature"].to_numpy() - dp
        X["dewpoint_tend180"] = X["dewpoint"] - X["dewpoint"].shift(
            _steps(180, step_minutes))

    # Cyclic time encoding: 23:50 and 00:00 must be neighbours, not extremes.
    idx = df.index
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.dayofyear.to_numpy().astype(float)
    X["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return X


def build_supervised(df: pd.DataFrame,
                     horizon_min: int | None = None,
                     target: str | None = None,
                     step_minutes: int | None = None):
    """
    Build (X, y, persistence) for supervised forecasting.

    y            = target value horizon_min into the future
    persistence  = the current value, i.e. the naive "no change" forecast.
                   Any model that cannot beat this is worthless, so it is
                   carried through the whole evaluation as the baseline.
    """
    horizon_min = horizon_min or cfg.FORECAST_HORIZON_MIN
    target = target or cfg.TARGET
    step_minutes = step_minutes or cfg.STEP_MINUTES

    X = build_features(df, step_minutes=step_minutes)
    h = _steps(horizon_min, step_minutes)

    y = df[target].shift(-h)
    persistence = df[target].copy()

    mask = X.notna().all(axis=1) & y.notna() & persistence.notna()
    return X[mask], y[mask], persistence[mask]


def chronological_split(X, y, persistence, test_size: float | None = None):
    """
    Split by time, never randomly.

    A random split would let the model see the future during training and
    then be scored on the past, which inflates every metric. The chronological
    split is what makes the reported numbers defensible in the paper.
    """
    test_size = test_size or cfg.TEST_SIZE
    cut = int(len(X) * (1 - test_size))
    return (X.iloc[:cut], X.iloc[cut:],
            y.iloc[:cut], y.iloc[cut:],
            persistence.iloc[:cut], persistence.iloc[cut:])


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    X, y, p = build_supervised(df)
    print(f"\nFeature matrix: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"Target: {cfg.TARGET} at t+{cfg.FORECAST_HORIZON_MIN} min")
    print("\nFeatures:")
    for i, c in enumerate(X.columns):
        print(f"  {c}", end="\n" if (i + 1) % 4 == 0 else "")
    print()
