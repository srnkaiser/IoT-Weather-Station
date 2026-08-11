"""
Combined evaluation of the two-layer detection architecture.

    Layer 1  Isolation Forest      statistical / environmental anomalies
    Layer 2  Deterministic checks  hardware faults via sensor redundancy
    Fused    Layer 1 OR Layer 2    what the deployed system actually reports

The interesting result is not the fused number on its own but the split: each
layer catches what the other one misses, and Layer 2 additionally tells the
two anomaly classes apart. A frontal passage is real weather and both
temperature sensors agree on it, so Layer 2 stays silent while Layer 1 fires.
A drifting DHT22 is a hardware fault, the sensors disagree, and Layer 2 fires.

Usage:  python src/evaluate_combined.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data_loader import load_dataset
from features import build_features
from sensor_check import run_checks
from train_anomaly import ANOMALY_FEATURES, evaluate, inject_faults


def main():
    print("=" * 72)
    print("COMBINED TWO-LAYER EVALUATION")
    print("=" * 72)

    bundle = joblib.load(cfg.MODEL_DIR / "anomaly_model.joblib")
    pipe = bundle["model"]

    df = load_dataset()
    cut = int(len(df) * (1 - cfg.TEST_SIZE))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    ref_diff = train_df["temperature"] - train_df["temperature_bmp"]

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    corrupted, label, ftype = inject_faults(test_df, rng)

    # Layer 1
    X = build_features(corrupted)[ANOMALY_FEATURES]
    valid = X.notna().all(axis=1)
    X = X[valid]
    l1 = pd.Series((pipe.predict(X) == -1).astype(int), index=X.index)

    # Layer 2
    flags = run_checks(corrupted, reference_diff=ref_diff)
    l2 = flags["sensor_fault"].astype(int).reindex(X.index).fillna(0).astype(int)

    label = label[valid]
    ftype = ftype[valid]
    fused = ((l1 == 1) | (l2 == 1)).astype(int)

    overall = {
        "Layer1_IsolationForest": evaluate(label.to_numpy(), l1.to_numpy()),
        "Layer2_SensorCrossCheck": evaluate(label.to_numpy(), l2.to_numpy()),
        "Fused_OR": evaluate(label.to_numpy(), fused.to_numpy()),
    }
    tbl = pd.DataFrame(overall).T[["Precision", "Recall", "F1", "FalseAlarmRate_%"]]
    tbl.index.name = "Detector"

    print("\n" + "-" * 72)
    print("OVERALL DETECTION PERFORMANCE")
    print("-" * 72)
    print(tbl.to_string())

    # Recall per fault type, per layer
    rows = []
    for name in ["spike", "stuck", "drift", "dropout", "frontal"]:
        m = (ftype == name).to_numpy()
        if not m.sum():
            continue
        rows.append({
            "FaultType": name,
            "Class": "environmental" if name == "frontal" else "sensor fault",
            "n": int(m.sum()),
            "Layer1": round(float(l1.to_numpy()[m].mean()), 3),
            "Layer2": round(float(l2.to_numpy()[m].mean()), 3),
            "Fused": round(float(fused.to_numpy()[m].mean()), 3),
        })
    m = (label == 0).to_numpy()
    rows.append({"FaultType": "clean", "Class": "-", "n": int(m.sum()),
                 "Layer1": round(float(l1.to_numpy()[m].mean()), 4),
                 "Layer2": round(float(l2.to_numpy()[m].mean()), 4),
                 "Fused": round(float(fused.to_numpy()[m].mean()), 4)})
    per_type = pd.DataFrame(rows).set_index("FaultType")

    print("\n" + "-" * 72)
    print("RECALL BY FAULT TYPE  (row 'clean' = false alarm rate)")
    print("-" * 72)
    print(per_type.to_string())

    print("\n" + "-" * 72)
    print("CLASSIFICATION OF DETECTED EVENTS")
    print("-" * 72)
    print("  Layer 2 silent, Layer 1 fires  -> environmental anomaly (weather)")
    print("  Layer 2 fires                  -> hardware fault, data untrustworthy")

    tbl.to_csv(cfg.RESULTS_DIR / "combined_overall.csv")
    per_type.to_csv(cfg.RESULTS_DIR / "combined_by_type.csv")
    (cfg.RESULTS_DIR / "combined_results.md").write_text(
        "# Two-layer anomaly detection results\n\n"
        f"Dataset: `{cfg.DATASET}`  |  Test samples: {len(label):,}  |  "
        f"Injected anomalies: {int(label.sum()):,}\n\n"
        "## Overall\n\n" + tbl.to_markdown() +
        "\n\n## Recall by fault type\n\n" + per_type.to_markdown() + "\n")
    with open(cfg.RESULTS_DIR / "combined_results.json", "w") as f:
        json.dump({"overall": overall, "by_type": rows}, f, indent=2)

    print("\nSaved -> results/combined_results.{md,json}, combined_*.csv")


if __name__ == "__main__":
    main()
