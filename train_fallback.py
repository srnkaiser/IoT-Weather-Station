"""
Fallback forecast model with a 10-minute memory.

WHY THIS EXISTS

The main model in train_forecast.py uses windows up to 60 minutes, so it
needs an hour of uninterrupted history before it can predict anything. Wokwi
cannot reliably deliver that: the browser pauses the simulation whenever its
tab loses focus, so an hour of wall-clock time produces a feed riddled with
gaps.

This model uses a single 10-minute lag and nothing longer. On a 10-minute
grid that is two consecutive samples, roughly 20 minutes of simulation - a
demo can realistically produce that.

Absolute pressure stays out for the same reason as in the main model (Jena
averages 989 hPa, the Wokwi BMP180 reports 1013), so pressure enters only as
a 10-minute tendency.

Expect it to be clearly worse than the main model. That difference is a
result worth reporting: it quantifies what the longer history is actually
worth, rather than asserting that history helps.

predict_live.py uses this automatically when the main model cannot be fed.

Usage:  python3 train_fallback.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data_loader import load_dataset
from features import build_features, chronological_split, _steps

# Everything obtainable from two consecutive samples.
FAST_LAGS = [10]
FAST_TENDENCY = [10]
FAST_ROLLING = []          # no rolling statistics - they need a window


def build_fast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix reachable with a single 10-minute lag."""
    return build_features(df, lags_min=FAST_LAGS, tendency_min=FAST_TENDENCY,
                          rolling_min=FAST_ROLLING)


def main():
    print("=" * 68)
    print("FALLBACK FORECAST MODEL (10 min memory)")
    print("=" * 68)

    df = load_dataset()
    X = build_fast_features(df)

    h = _steps(cfg.FORECAST_HORIZON_MIN, cfg.STEP_MINUTES)
    y = df[cfg.TARGET].shift(-h)
    persistence = df[cfg.TARGET].copy()

    mask = X.notna().all(axis=1) & y.notna() & persistence.notna()
    X, y, persistence = X[mask], y[mask], persistence[mask]

    Xtr, Xte, ytr, yte, ptr, pte = chronological_split(X, y, persistence)
    print(f"\nSamples : {len(X):,}   Features: {X.shape[1]}")
    print(f"Features: {list(X.columns)}")
    print(f"Train   : {len(Xtr):,}   Test: {len(Xte):,}")

    base_mae = mean_absolute_error(yte, pte)
    print(f"\nPersistence baseline     MAE={base_mae:.3f}")

    t0 = time.time()
    model = HistGradientBoostingRegressor(**cfg.HGB_PARAMS)
    model.fit(Xtr, ytr)
    dt = time.time() - t0

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    r2 = float(r2_score(yte, pred))
    skill = 100 * (1 - mae / base_mae)

    print(f"GradientBoosting (fast)  MAE={mae:.3f}  RMSE={rmse:.3f}  "
          f"R2={r2:.4f}  Skill={skill:+.1f}%  ({dt:.1f}s)")

    metrics = {"MAE": round(float(mae), 4), "RMSE": round(rmse, 4),
               "R2": round(r2, 4), "Skill_vs_Persistence_%": round(skill, 2)}

    joblib.dump({
        "model": model,
        "model_name": "GradientBoosting (fast, 10 min memory)",
        "feature_names": list(X.columns),
        "target": cfg.TARGET,
        "horizon_min": cfg.FORECAST_HORIZON_MIN,
        "step_minutes": cfg.STEP_MINUTES,
        "lags_min": FAST_LAGS,
        "tendency_min": FAST_TENDENCY,
        "rolling_min": FAST_ROLLING,
        "dataset": cfg.DATASET,
        "metrics": metrics,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }, cfg.MODEL_DIR / "forecast_model_fast.joblib")

    with open(cfg.RESULTS_DIR / "fallback_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Comparison against the main model, if it has been trained.
    main_path = cfg.MODEL_DIR / "forecast_model.joblib"
    if main_path.exists():
        main_metrics = joblib.load(main_path)["metrics"]
        rows = pd.DataFrame({
            "Main model (60 min history)": main_metrics,
            "Fallback (10 min history)": metrics,
        }).T[["MAE", "RMSE", "R2", "Skill_vs_Persistence_%"]]
        print("\n" + "-" * 68)
        print("WHAT THE LONGER HISTORY BUYS")
        print("-" * 68)
        print(rows.to_string())
        (cfg.RESULTS_DIR / "fallback_comparison.md").write_text(
            "# Forecast with reduced history\n\n"
            "The fallback model uses a single 10-minute lag, so it can run "
            "after about 20 minutes of simulation instead of 60. The gap "
            "between the two rows is what the longer history is worth.\n\n"
            + rows.to_markdown() + "\n")
        print("\nSaved -> results/fallback_comparison.md")

    print(f"Saved model -> models/forecast_model_fast.joblib")


if __name__ == "__main__":
    main()
