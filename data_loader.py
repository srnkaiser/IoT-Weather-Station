"""
Loading and cleaning of weather time series.

Every supported source is mapped onto the same unified schema:
    index        : DatetimeIndex on a regular grid
    temperature  : degC
    humidity     : % relative humidity
    pressure     : hPa

Only these three columns are allowed to reach the model, because these are
the only quantities the DHT22 + BMP180 combination can deliver later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


# Physically plausible ranges. Values outside are measurement errors, not
# weather, and are removed before training.
VALID_RANGES = {
    "temperature": (-45.0, 55.0),
    "humidity": (0.0, 100.0),
    "pressure": (900.0, 1070.0),
    "temperature_bmp": (-45.0, 55.0),
}


def load_raw(dataset: str | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Read a raw dataset file and rename columns to the unified schema."""
    dataset = dataset or cfg.DATASET
    if dataset not in cfg.DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Known: {list(cfg.DATASET_SPECS)}")

    spec = cfg.DATASET_SPECS[dataset]
    path = Path(path) if path else cfg.DATASET_FILES[dataset]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"For 'synthetic' run:  python src/make_testdata.py\n"
            f"For 'jena' download the CSV and place it there (see README)."
        )

    df = pd.read_csv(path, sep=spec["sep"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    time_col = spec["time_col"]
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not in file. Found: {list(df.columns)[:12]}")

    if spec["time_format"]:
        ts = pd.to_datetime(df[time_col].astype(str).str.strip(),
                            format=spec["time_format"], errors="coerce")
    else:
        ts = pd.to_datetime(df[time_col], errors="coerce")

    rename = {src: dst for src, dst in spec["columns"].items() if src in df.columns}
    missing = set(spec["columns"].values()) - set(rename.values())
    missing.discard("temperature_bmp")  # optional cross-check channel
    if "temperature" in missing:
        raise KeyError("Dataset has no temperature column - cannot train.")

    out = df[list(rename)].rename(columns=rename)
    out.index = ts
    out.index.name = "timestamp"
    out = out[out.index.notna()].sort_index()

    for col in missing:
        print(f"  [warn] '{col}' missing in this dataset - will be unavailable as a feature")

    return out.astype(float)


def clean(df: pd.DataFrame, resample: str | None = None) -> pd.DataFrame:
    """Remove implausible values, put on a regular grid, close small gaps."""
    resample = resample or cfg.RESAMPLE
    df = df.copy()

    n_before = len(df)

    # 1) Sentinel values used by several weather services for "no measurement".
    df = df.replace([-999, -999.0, -9999, -9999.0], np.nan)

    # 2) Physically impossible readings.
    removed = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            removed[col] = int(bad.sum())
            df.loc[bad, col] = np.nan

    # 3) Duplicate timestamps -> keep the first.
    df = df[~df.index.duplicated(keep="first")]

    # 4) Regular grid. Averaging is the honest choice when downsampling.
    df = df.resample(resample).mean()

    # 5) Close only SHORT gaps. Long gaps stay NaN and their rows get dropped
    #    later - inventing hours of weather would corrupt the training set.
    max_gap = max(1, int(pd.Timedelta("1h") / pd.Timedelta(resample)))
    df = df.interpolate(method="time", limit=max_gap, limit_direction="forward")

    print(f"  rows: {n_before} raw -> {len(df)} on {resample} grid")
    for col, n in removed.items():
        if n:
            print(f"  removed {n} out-of-range values in '{col}'")
    print(f"  remaining gaps: {int(df.isna().any(axis=1).sum())} rows contain NaN")

    return df


def load_dataset(dataset: str | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: load + clean."""
    dataset = dataset or cfg.DATASET
    print(f"Loading dataset '{dataset}' ...")
    df = load_raw(dataset, path)
    df = clean(df)
    span = df.index.max() - df.index.min()
    print(f"  period: {df.index.min()}  ->  {df.index.max()}  ({span.days} days)")
    return df


if __name__ == "__main__":
    d = load_dataset()
    print()
    print(d.describe().round(2))
