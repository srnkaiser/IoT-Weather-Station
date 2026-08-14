"""
Layer 2: deterministic sensor fault detection.

The Isolation Forest in train_anomaly.py is a statistical detector. It is very
good at point anomalies (spikes, dropouts) and at unusual weather, but it is
structurally blind to two failure modes:

    stuck  a frozen value is not statistically extreme, it is just constant
    drift  a slowly growing offset stays inside the plausible range at every
           single point in time

Both are caught here instead, by exploiting hardware redundancy that the
statistical model cannot see: DHT22 and BMP180 measure temperature at the
same place at the same time. Weather affects both. A fault affects one.

Three checks
------------
flatline      rolling standard deviation is exactly zero -> sensor frozen
crosscheck    |T_DHT22 - T_BMP180| exceeds a robust threshold, sustained
range         value outside the datasheet operating range

The threshold is derived with median + k * MAD rather than mean + k * sigma.
MAD is robust: a drifting sensor inflates the standard deviation and would
raise a sigma-based threshold above the very fault it should detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

# Datasheet operating ranges.
SENSOR_RANGES = {
    "temperature": (-40.0, 80.0),    # DHT22
    "humidity": (0.0, 100.0),        # DHT22
    "pressure": (300.0, 1100.0),     # BMP180
    "temperature_bmp": (-40.0, 85.0),
}


def robust_threshold(diff: pd.Series, k: float, min_abs: float) -> float:
    """median + k * MAD, with a noise floor so we never flag normal offset."""
    med = float(np.nanmedian(np.abs(diff)))
    mad = float(np.nanmedian(np.abs(np.abs(diff) - med)))
    return max(med + k * mad * 1.4826, min_abs)


def detect_flatline(s: pd.Series, window: int) -> pd.Series:
    """True where the value has not changed at all over the whole window."""
    rng = s.rolling(window, min_periods=window).max() - \
        s.rolling(window, min_periods=window).min()
    flag = (rng == 0)
    # Mark the whole frozen stretch, not just its end.
    return flag[::-1].rolling(window, min_periods=1).max().astype(bool)[::-1]


def detect_crosscheck(t_primary: pd.Series, t_secondary: pd.Series,
                      k: float, min_abs: float, persistence: int,
                      reference: pd.Series | None = None,
                      expected_bmp_offset: float = 0.0) -> tuple[pd.Series, float]:
    """
    Flag sustained divergence between the two temperature sensors.

    reference: a clean stretch used to calibrate the threshold. When it is not
    available (as in live operation), compare against the configured expected
    BMP180 offset instead of calibrating from possibly faulty live readings.
    """
    diff = t_primary - t_secondary
    if reference is not None:
        thr = robust_threshold(reference, k, min_abs)
        exceed = diff.abs() > thr
    else:
        # BMP180 often reads slightly warmer because its die is near the
        # electronics. A persistent deviation beyond the allowed noise floor
        # is a sensor fault even if all recent values share that deviation.
        expected_diff = -expected_bmp_offset
        thr = min_abs
        exceed = (diff - expected_diff).abs() > thr
    sustained = exceed.rolling(persistence, min_periods=persistence).min().fillna(0).astype(bool)
    sustained = sustained[::-1].rolling(persistence, min_periods=1).max().astype(bool)[::-1]
    return sustained, thr


def detect_range(df: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df.index)
    for col, (lo, hi) in SENSOR_RANGES.items():
        if col in df.columns:
            flag |= (df[col] < lo) | (df[col] > hi)
    return flag


def run_checks(df: pd.DataFrame, reference_diff: pd.Series | None = None) -> pd.DataFrame:
    """Run all deterministic checks. Returns a DataFrame of boolean flags."""
    p = cfg.CROSSCHECK
    out = pd.DataFrame(index=df.index)

    out["range_fault"] = detect_range(df)

    for col in ("temperature", "humidity"):
        if col in df.columns:
            out[f"flatline_{col}"] = detect_flatline(df[col], p["flatline_window"])

    if "temperature_bmp" in df.columns and "temperature" in df.columns:
        cross, thr = detect_crosscheck(
            df["temperature"], df["temperature_bmp"],
            k=p["mad_k"], min_abs=p["min_abs_delta"],
            persistence=p["persistence"], reference=reference_diff,
            expected_bmp_offset=p.get("expected_bmp_offset", 0.0))
        out["crosscheck_fault"] = cross
        out.attrs["crosscheck_threshold"] = thr

    out["sensor_fault"] = out.any(axis=1)
    return out


# --------------------------------------------------------------------------
# Evaluation against the same injected faults used for the Isolation Forest
# --------------------------------------------------------------------------
def main():
    from data_loader import load_dataset
    from train_anomaly import inject_faults

    print("=" * 68)
    print("LAYER 2: DETERMINISTIC SENSOR FAULT DETECTION")
    print("=" * 68)

    df = load_dataset()
    cut = int(len(df) * (1 - cfg.TEST_SIZE))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    # Calibrate the threshold on clean commissioning data.
    ref_diff = train_df["temperature"] - train_df["temperature_bmp"]

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    corrupted, label, ftype = inject_faults(test_df, rng)

    flags = run_checks(corrupted, reference_diff=ref_diff)
    pred = flags["sensor_fault"].astype(int).to_numpy()
    lab = label.to_numpy()

    thr = flags.attrs.get("crosscheck_threshold", float("nan"))
    print(f"\nCross-check threshold (median + {cfg.CROSSCHECK['mad_k']} x MAD): "
          f"{thr:.2f} degC")
    print(f"Flagged {pred.sum():,} of {len(pred):,} samples")

    tp = int(((pred == 1) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    print("\n" + "-" * 68)
    print("OVERALL")
    print("-" * 68)
    print(f"  Precision  {prec:.4f}")
    print(f"  Recall     {rec:.4f}")
    print(f"  F1         {f1:.4f}")
    print(f"  FalseAlarmRate_%  {100 * fp / max(fp + tn, 1):.3f}")

    rows = {}
    for name in ["spike", "stuck", "drift", "dropout", "frontal"]:
        m = (ftype == name).to_numpy()
        if m.sum():
            rows[name] = {"n_samples": int(m.sum()),
                          "Recall": round(float(pred[m].mean()), 4)}
    rows["clean"] = {"n_samples": int((lab == 0).sum()),
                     "Recall": round(float(pred[lab == 0].mean()), 4)}
    per_type = pd.DataFrame(rows).T
    per_type.index.name = "FaultType"

    print("\n" + "-" * 68)
    print("DETECTION RATE BY FAULT TYPE")
    print("-" * 68)
    print(per_type.to_string())

    per_type.to_csv(cfg.RESULTS_DIR / "sensor_check_by_type.csv")
    (cfg.RESULTS_DIR / "sensor_check.md").write_text(
        f"# Layer 2: deterministic sensor fault detection\n\n"
        f"Cross-check threshold: {thr:.2f} degC\n\n"
        f"Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f}\n\n"
        + per_type.to_markdown() + "\n")

    print("\nSaved -> results/sensor_check_by_type.csv, results/sensor_check.md")


if __name__ == "__main__":
    main()
