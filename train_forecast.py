"""
Training and evaluation of the temperature forecast model.

Compares four approaches on an identical chronological split:

    Persistence        naive baseline, "temperature will not change"
    Ridge              linear model, scaled inputs
    Random Forest      bagged trees
    Gradient Boosting  HistGradientBoostingRegressor

The persistence baseline is the point of the whole table. Weather is highly
autocorrelated, so a 1 h temperature forecast is easy to make look good in
absolute terms. Only the improvement over persistence shows the model learned
anything. That comparison is what turns this into a defensible result.

Usage:  python src/train_forecast.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data_loader import load_dataset
from features import build_supervised, chronological_split


def metrics(y_true, y_pred, baseline_mae=None) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out = {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }
    if baseline_mae:
        # Skill score: fraction of the baseline error removed by the model.
        out["Skill_vs_Persistence_%"] = round(100 * (1 - mae / baseline_mae), 2)
    return out


def plot_predictions(idx, y_true, preds: dict, out_path: Path, hours: int = 96):
    n = min(len(y_true), int(hours * 60 / cfg.STEP_MINUTES))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(idx[:n], y_true[:n], label="Measured", color="#222", lw=2.0)
    colors = {"Persistence": "#bbbbbb", "Ridge": "#2a9d8f",
              "RandomForest": "#e76f51", "GradientBoosting": "#264653"}
    for name, p in preds.items():
        ax.plot(idx[:n], p[:n], label=name, lw=1.3, alpha=0.9,
                color=colors.get(name), ls="--" if name == "Persistence" else "-")
    ax.set_title(f"Temperature forecast, {cfg.FORECAST_HORIZON_MIN} min ahead "
                 f"(first {hours} h of the test set)")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(ncol=5, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_importance(model, feature_names, out_path: Path, top: int = 18):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(imp.index, imp.values, color="#e76f51")
    ax.set_title(f"Top {top} features - Random Forest")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(y_true, preds: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, p in preds.items():
        ax.hist(np.asarray(y_true) - np.asarray(p), bins=90, alpha=0.5,
                label=name, density=True)
    ax.set_title("Forecast error distribution on the test set")
    ax.set_xlabel("Error: measured - predicted (degC)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 68)
    print("FORECAST MODEL TRAINING")
    print("=" * 68)

    df = load_dataset()
    X, y, persistence = build_supervised(df)
    Xtr, Xte, ytr, yte, ptr, pte = chronological_split(X, y, persistence)

    print(f"\nSamples : {len(X):,}   Features: {X.shape[1]}")
    print(f"Train   : {len(Xtr):,}  ({Xtr.index.min().date()} -> {Xtr.index.max().date()})")
    print(f"Test    : {len(Xte):,}  ({Xte.index.min().date()} -> {Xte.index.max().date()})")
    print(f"Horizon : {cfg.FORECAST_HORIZON_MIN} min\n")

    results, preds = {}, {}

    # Baseline first - everything else is measured against it.
    preds["Persistence"] = pte.to_numpy()
    results["Persistence"] = metrics(yte, pte)
    base_mae = results["Persistence"]["MAE"]
    results["Persistence"]["Skill_vs_Persistence_%"] = 0.0
    results["Persistence"]["train_seconds"] = 0.0
    print(f"{'Persistence (baseline)':<24} MAE={base_mae:.3f}")

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(**cfg.RF_PARAMS),
        "GradientBoosting": HistGradientBoostingRegressor(**cfg.HGB_PARAMS),
    }

    fitted = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        dt = time.time() - t0
        p = model.predict(Xte)
        preds[name] = p
        results[name] = metrics(yte, p, base_mae)
        results[name]["train_seconds"] = round(dt, 1)
        fitted[name] = model
        print(f"{name:<24} MAE={results[name]['MAE']:.3f}  "
              f"RMSE={results[name]['RMSE']:.3f}  "
              f"R2={results[name]['R2']:.4f}  "
              f"Skill={results[name]['Skill_vs_Persistence_%']:+.1f}%  "
              f"({dt:.1f}s)")

    # --- Results table -----------------------------------------------------
    table = pd.DataFrame(results).T
    table.index.name = "Model"
    print("\n" + "-" * 68)
    print("RESULTS (test set, never seen during training)")
    print("-" * 68)
    print(table.to_string())

    best_name = min((n for n in results if n != "Persistence"),
                    key=lambda n: results[n]["MAE"])
    print(f"\nBest model: {best_name}")

    # --- Persist everything ------------------------------------------------
    bundle = {
        "model": fitted[best_name],
        "model_name": best_name,
        "feature_names": list(X.columns),
        "target": cfg.TARGET,
        "horizon_min": cfg.FORECAST_HORIZON_MIN,
        "step_minutes": cfg.STEP_MINUTES,
        "dataset": cfg.DATASET,
        "metrics": results[best_name],
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, cfg.MODEL_DIR / "forecast_model.joblib")

    table.to_csv(cfg.RESULTS_DIR / "forecast_metrics.csv")
    (cfg.RESULTS_DIR / "forecast_metrics.md").write_text(
        f"# Forecast results ({cfg.FORECAST_HORIZON_MIN} min horizon, "
        f"dataset: {cfg.DATASET})\n\n" + table.to_markdown() + "\n")
    with open(cfg.RESULTS_DIR / "forecast_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_predictions(Xte.index, yte.to_numpy(), preds,
                     cfg.RESULTS_DIR / "forecast_timeseries.png")
    plot_error_distribution(yte.to_numpy(),
                            {k: v for k, v in preds.items()
                             if k in ("Persistence", best_name)},
                            cfg.RESULTS_DIR / "forecast_errors.png")
    if "RandomForest" in fitted:
        plot_importance(fitted["RandomForest"], list(X.columns),
                        cfg.RESULTS_DIR / "feature_importance.png")

    print(f"\nSaved model   -> models/forecast_model.joblib")
    print(f"Saved metrics -> results/forecast_metrics.{{csv,md,json}}")
    print(f"Saved plots   -> results/*.png")

    if cfg.DATASET == "synthetic":
        print("\n" + "!" * 68)
        print("These numbers come from SYNTHETIC data and must not go in the")
        print("paper. Download the real dataset, set DATASET = 'jena' in")
        print("config.py and rerun to obtain reportable results.")
        print("!" * 68)


if __name__ == "__main__":
    main()
