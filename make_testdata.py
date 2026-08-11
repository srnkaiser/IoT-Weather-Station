"""
Generator for synthetic weather test data.

PURPOSE: let the whole pipeline run end-to-end before the real dataset is
downloaded. The generated series is physically plausible (seasonal cycle,
daily cycle, weather fronts that couple pressure to temperature and humidity,
realistic sensor noise) but it is NOT real weather.

>>> Do not report metrics obtained on this data in the paper. <<<
Switch config.DATASET to "jena" once the real CSV is in place.

Usage:  python src/make_testdata.py [--years 2] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def magnus_dewpoint(t_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Dew point via the Magnus formula (Sonntag coefficients)."""
    a, b = 17.62, 243.12
    rh = np.clip(rh_pct, 1.0, 100.0)
    gamma = np.log(rh / 100.0) + (a * t_c) / (b + t_c)
    return (b * gamma) / (a - gamma)


def rh_from_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Inverse Magnus: relative humidity from temperature and dew point."""
    a, b = 17.62, 243.12
    num = np.exp((a * td_c) / (b + td_c))
    den = np.exp((a * t_c) / (b + t_c))
    return np.clip(100.0 * num / den, 5.0, 100.0)


def ornstein_uhlenbeck(n: int, tau_steps: float, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Mean-reverting noise - models synoptic (multi-day) weather variability."""
    theta = 1.0 / tau_steps
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (0.0 - x[i - 1]) + sigma * rng.normal()
    return x


def generate(years: float = 2.0, freq: str = "10min", seed: int = 42,
             start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    idx = pd.date_range(start=start, periods=int(years * 365.25 * 24 * 6),
                        freq=freq)
    n = len(idx)
    step_min = pd.Timedelta(freq).total_seconds() / 60.0
    steps_per_day = int(24 * 60 / step_min)

    doy = idx.dayofyear.to_numpy(dtype=float)
    hod = idx.hour.to_numpy(dtype=float) + idx.minute.to_numpy(dtype=float) / 60.0

    # --- Synoptic weather systems -----------------------------------------
    # One slow mean-reverting process drives pressure. Fronts last ~3 days.
    synoptic = ornstein_uhlenbeck(n, tau_steps=3 * steps_per_day, sigma=0.55, rng=rng)
    synoptic = synoptic / (np.std(synoptic) + 1e-9)

    # Pressure: seasonal mean plus the synoptic anomaly.
    pressure = 1013.0 + 2.0 * np.sin(2 * np.pi * (doy - 20) / 365.25) + 11.0 * synoptic

    # Cloudiness is anti-correlated with pressure (low pressure -> overcast).
    cloud = np.clip(0.5 - 0.42 * synoptic + 0.10 * rng.normal(size=n), 0.0, 1.0)

    # --- Temperature -------------------------------------------------------
    # Seasonal cycle, warmest around day 200.
    seasonal = 9.5 + 9.5 * np.sin(2 * np.pi * (doy - 110) / 365.25)

    # Daily cycle, peak around 15:00. Clouds damp the amplitude strongly.
    daily_amp = (3.2 + 2.6 * np.sin(2 * np.pi * (doy - 110) / 365.25)) * (1.0 - 0.65 * cloud)
    daily = daily_amp * np.sin(2 * np.pi * (hod - 9.0) / 24.0)

    # Warm sectors ahead of a front, cold air behind it.
    frontal = 3.4 * synoptic

    micro = ornstein_uhlenbeck(n, tau_steps=6, sigma=0.28, rng=rng)

    temperature_true = seasonal + daily + frontal + micro

    # --- Humidity ----------------------------------------------------------
    # Build humidity from a slowly varying dew point - this keeps the
    # temperature/humidity relationship physically sensible.
    dewpoint = (seasonal - 4.5) + 2.6 * synoptic + 3.0 * cloud \
        + ornstein_uhlenbeck(n, tau_steps=steps_per_day, sigma=0.30, rng=rng)
    dewpoint = np.minimum(dewpoint, temperature_true - 0.15)
    humidity_true = rh_from_dewpoint(temperature_true, dewpoint)

    # --- Sensor noise ------------------------------------------------------
    # DHT22 datasheet: +-0.5 degC, +-2..5 % RH, 0.1 resolution.
    temperature_dht = np.round(temperature_true + rng.normal(0, 0.28, n), 1)
    humidity_dht = np.round(np.clip(humidity_true + rng.normal(0, 1.6, n), 5, 100), 1)

    # BMP180: +-0.12 hPa, +-1.0 degC (its temperature sensor is coarser and
    # sits on the die, so it reads slightly warm).
    pressure_bmp = np.round(pressure + rng.normal(0, 0.12, n), 2)
    temperature_bmp = np.round(temperature_true + 0.35 + rng.normal(0, 0.45, n), 1)

    return pd.DataFrame({
        "timestamp": idx,
        "temperature": temperature_dht,
        "humidity": humidity_dht,
        "pressure": pressure_bmp,
        "temperature_bmp": temperature_bmp,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(cfg.DATASET_FILES["synthetic"]))
    args = ap.parse_args()

    df = generate(years=args.years, freq=cfg.RESAMPLE, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"Period: {df.timestamp.min()}  ->  {df.timestamp.max()}")
    print()
    print(df.drop(columns=["timestamp"]).describe().round(2))
    print()
    print("NOTE: synthetic data - for pipeline testing only, not for the paper.")


if __name__ == "__main__":
    main()
