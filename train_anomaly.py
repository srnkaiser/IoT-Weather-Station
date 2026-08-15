"""
Anomaly detection with an Isolation Forest.

The problem with unsupervised anomaly detection in a student project is that
it cannot be evaluated: there are no labels, so any contamination rate
"works" and nothing can be reported. This script solves that by injecting
synthetic faults of known type and position into the test set, then measuring
precision, recall and F1 against those known positions.

That gives a real quantitative results table instead of "the model found some
anomalies".

Injected fault types
--------------------
spike     single-sample outlier            (electrical interference, bad read)
stuck     sensor freezes at one value      (I2C hang, dead DHT22)
drift     slowly growing offset            (ageing, self-heating)
dropout   humidity collapses to near zero  (broken humidity element)
frontal   unusually fast temperature drop  (real environmental anomaly)

The first four are sensor faults, the last is a genuine weather event. A
useful detector should catch both, and the paper can discuss the distinction.

Usage:  python src/train_anomaly.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data_loader import load_dataset
from features import build_features


# Compact, altitude-invariant feature set for anomaly detection.
#
# Derived from the configured windows rather than written out by hand: these
# names have to exist in the matrix build_features() produces, so hard-coding
# them means every change to LAGS_MIN/TENDENCY_MIN/ROLLING_MIN breaks training
# with a KeyError that points at this list instead of at the actual cause.
_T_FAST = min(cfg.TENDENCY_MIN)   # fastest jump -> spikes
_T_SLOW = max(cfg.TENDENCY_MIN)   # medium-term change -> frontal passages
_ROLL = max(cfg.ROLLING_MIN)      # variability window

ANOMALY_FEATURES = list(dict.fromkeys([
    # Level and physical consistency
    "temperature", "humidity", "dewpoint_spread",
    # Short-term jumps -> spikes
    f"temperature_tend{_T_FAST}", f"humidity_tend{_T_FAST}",
    # Medium-term change -> frontal passages
    f"temperature_tend{_T_SLOW}", f"humidity_tend{_T_SLOW}",
    f"pressure_tend{_T_SLOW}",
    # Variability -> stuck sensors (near zero) and erratic readings (high)
    f"temperature_std{_ROLL}", f"humidity_std{_ROLL}", f"pressure_std{_ROLL}",
    f"temperature_range{_ROLL}", f"humidity_range{_ROLL}",
]))


def inject_faults(df: pd.DataFrame, rng: np.random.Generator,
                  n_each: int = 12) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return (corrupted_df, label, fault_type).

    label      1 where a fault was injected, else 0
    fault_type string name of the fault, "" where clean
    """
    d = df.copy()
    label = pd.Series(0, index=d.index, dtype=int)
    ftype = pd.Series("", index=d.index, dtype=object)
    n = len(d)
    margin = 400

    def mark(sl, name):
        label.iloc[sl] = 1
        ftype.iloc[sl] = name

    # 1) Spikes - single sample, large offset
    for _ in range(n_each):
        i = int(rng.integers(margin, n - margin))
        d.iloc[i, d.columns.get_loc("temperature")] += rng.choice([-1, 1]) * rng.uniform(9, 18)
        mark(slice(i, i + 1), "spike")

    # 2) Stuck sensor - value frozen for 1-4 h
    for _ in range(n_each):
        i = int(rng.integers(margin, n - margin))
        ln = int(rng.integers(6, 25))
        for col in ("temperature", "humidity"):
            d.iloc[i:i + ln, d.columns.get_loc(col)] = d.iloc[i, d.columns.get_loc(col)]
        mark(slice(i, i + ln), "stuck")

    # 3) Drift - offset growing linearly over 6-12 h
    for _ in range(n_each):
        i = int(rng.integers(margin, n - margin))
        ln = int(rng.integers(36, 73))
        ramp = np.linspace(0, rng.uniform(5, 10), ln)
        d.iloc[i:i + ln, d.columns.get_loc("temperature")] += ramp
        mark(slice(i, i + ln), "drift")

    # 4) Humidity dropout
    for _ in range(n_each):
        i = int(rng.integers(margin, n - margin))
        ln = int(rng.integers(6, 31))
        d.iloc[i:i + ln, d.columns.get_loc("humidity")] = rng.uniform(0.5, 3.0)
        mark(slice(i, i + ln), "dropout")

    # 5) Genuine environmental anomaly - very fast frontal passage
    for _ in range(n_each):
        i = int(rng.integers(margin, n - margin))
        ln = int(rng.integers(12, 31))
        drop = np.linspace(0, -rng.uniform(8, 14), ln)
        # A real front cools the air, so BOTH sensors must see it. This is
        # what makes the cross-check able to tell weather from hardware
        # faults: agreement means weather, divergence means a broken sensor.
        d.iloc[i:i + ln, d.columns.get_loc("temperature")] += drop
        if "temperature_bmp" in d.columns:
            d.iloc[i:i + ln, d.columns.get_loc("temperature_bmp")] += drop

        # Humidity rises towards saturation, asymptotically - never by a fixed
        # amount that then gets clipped at 100.
        #
        # The clipped version invalidated the evaluation: Jena's median
        # humidity is 79 %, so adding a flat 18 points pinned long stretches
        # to exactly 100. A constant value is precisely the signature the
        # flatline detector looks for, so Layer 2 reported a hardware fault on
        # a weather event - the one confusion this architecture exists to
        # avoid. The benchmark was manufacturing the error it then measured.
        hcol = d.columns.get_loc("humidity")
        h = d.iloc[i:i + ln, hcol].to_numpy()
        d.iloc[i:i + ln, hcol] = h + (100.0 - h) * 0.6

        mark(slice(i, i + ln), "frontal")

    return d, label, ftype


def evaluate(label: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "FalseAlarmRate_%": round(100 * fp / max(fp + tn, 1), 3),
    }


def main():
    print("=" * 68)
    print("ANOMALY DETECTION TRAINING")
    print("=" * 68)

    df = load_dataset()
    cut = int(len(df) * (1 - cfg.TEST_SIZE))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    print(f"\nTrain (assumed clean): {len(train_df):,}")
    print(f"Test  (faults injected): {len(test_df):,}")

    # --- Fit on clean training data ---------------------------------------
    Xtr = build_features(train_df)[ANOMALY_FEATURES].dropna()
    pipe = make_pipeline(StandardScaler(), IsolationForest(**cfg.IFOREST_PARAMS))
    pipe.fit(Xtr)
    print(f"\nIsolation Forest fitted on {len(Xtr):,} samples, "
          f"{len(ANOMALY_FEATURES)} features")
    print(f"contamination = {cfg.IFOREST_PARAMS['contamination']}")

    # --- Inject known faults into the test set ----------------------------
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    corrupted, label, ftype = inject_faults(test_df, rng)

    Xte = build_features(corrupted)[ANOMALY_FEATURES]
    valid = Xte.notna().all(axis=1)
    Xte, label, ftype = Xte[valid], label[valid], ftype[valid]

    pred = (pipe.predict(Xte) == -1).astype(int)
    score = -pipe.decision_function(Xte)  # higher = more anomalous

    print(f"\nInjected {int(label.sum()):,} anomalous samples "
          f"({100 * label.mean():.2f}% of test set)")
    print(f"Flagged  {int(pred.sum()):,} samples")

    # --- Overall metrics ---------------------------------------------------
    overall = evaluate(label.to_numpy(), pred)
    print("\n" + "-" * 68)
    print("OVERALL DETECTION PERFORMANCE")
    print("-" * 68)
    for k, v in overall.items():
        print(f"  {k:<20} {v}")

    # --- Per fault type ----------------------------------------------------
    rows = {}
    for name in ["spike", "stuck", "drift", "dropout", "frontal"]:
        m = (ftype == name).to_numpy()
        if m.sum() == 0:
            continue
        rows[name] = {
            "n_samples": int(m.sum()),
            "Recall": round(float(pred[m].mean()), 4),
            "MeanScore": round(float(np.mean(score[m])), 4),
        }
    rows["clean"] = {
        "n_samples": int((label == 0).sum()),
        "Recall": round(float(pred[(label == 0).to_numpy()].mean()), 4),
        "MeanScore": round(float(np.mean(score[(label == 0).to_numpy()])), 4),
    }
    per_type = pd.DataFrame(rows).T
    per_type.index.name = "FaultType"
    print("\n" + "-" * 68)
    print("DETECTION RATE BY FAULT TYPE  (Recall on 'clean' = false alarm rate)")
    print("-" * 68)
    print(per_type.to_string())

    # --- Save --------------------------------------------------------------
    joblib.dump({
        "model": pipe,
        "feature_names": ANOMALY_FEATURES,
        "contamination": cfg.IFOREST_PARAMS["contamination"],
        "metrics": overall,
        "dataset": cfg.DATASET,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }, cfg.MODEL_DIR / "anomaly_model.joblib")

    per_type.to_csv(cfg.RESULTS_DIR / "anomaly_by_type.csv")
    with open(cfg.RESULTS_DIR / "anomaly_metrics.json", "w") as f:
        json.dump({"overall": overall, "by_type": rows}, f, indent=2)
    (cfg.RESULTS_DIR / "anomaly_metrics.md").write_text(
        "# Anomaly detection results\n\n## Overall\n\n"
        + pd.Series(overall).to_frame("Value").to_markdown()
        + "\n\n## By injected fault type\n\n" + per_type.to_markdown() + "\n")

    # --- Plot --------------------------------------------------------------
    n = min(len(Xte), 3000)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    t = Xte.index[:n]
    axes[0].plot(t, Xte["temperature"].to_numpy()[:n], color="#264653", lw=1.0,
                 label="Temperature")
    flag = pred[:n].astype(bool)
    axes[0].scatter(t[flag], Xte["temperature"].to_numpy()[:n][flag], s=16,
                    color="#e63946", zorder=5, label="Flagged anomaly")
    axes[0].set_ylabel("degC")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Isolation Forest on the test set with injected faults")

    axes[1].plot(t, score[:n], color="#6a4c93", lw=0.9)
    axes[1].fill_between(t, score[:n], where=flag, color="#e63946", alpha=0.35)
    axes[1].set_ylabel("Anomaly score")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_DIR / "anomaly_detection.png", dpi=150)
    plt.close(fig)

    print("\nSaved model   -> models/anomaly_model.joblib")
    print("Saved metrics -> results/anomaly_metrics.{json,md}, anomaly_by_type.csv")
    print("Saved plot    -> results/anomaly_detection.png")


if __name__ == "__main__":
    main()
