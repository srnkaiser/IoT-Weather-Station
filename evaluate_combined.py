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

    # --- Layer 2 measured against the job it actually has -----------------
    # The table above scores every detector against "was anything injected
    # here", which quietly penalises Layer 2 for its correct behaviour: a
    # frontal passage is a weather event, both sensors register it, and Layer 2
    # is supposed to stay silent. Counting that silence as a miss understates
    # a cross-check that is doing exactly what it was built for.
    #
    # Layer 2 detects hardware faults. Scored against hardware faults:
    is_sensor_fault = ftype.isin(["spike", "stuck", "drift", "dropout"]).to_numpy()
    is_env = (ftype == "frontal").to_numpy()
    sensor_only_label = is_sensor_fault.astype(int)

    l2_sensor = evaluate(sensor_only_label[~is_env], l2.to_numpy()[~is_env])
    print("\n" + "-" * 72)
    print("LAYER 2 SCORED AGAINST HARDWARE FAULTS ONLY")
    print("(weather events excluded - Layer 2 is meant to ignore those)")
    print("-" * 72)
    for k in ("Precision", "Recall", "F1", "FalseAlarmRate_%"):
        print(f"  {k:<20} {l2_sensor[k]}")

    # --- Can the pair tell the two classes apart? -------------------------
    # This is the architecture's actual claim, and nothing above tests it:
    # not "was something detected" but "was it correctly identified as
    # weather rather than a broken sensor". Only events that were detected
    # at all can be classified, so the rate is reported over those.
    detected = (fused == 1).to_numpy()
    n_env_det = int((is_env & detected).sum())
    n_hw_det = int((is_sensor_fault & detected).sum())
    env_correct = int((is_env & detected & (l2.to_numpy() == 0)).sum())
    hw_correct = int((is_sensor_fault & detected & (l2.to_numpy() == 1)).sum())

    print("\n" + "-" * 72)
    print("CLASSIFICATION OF DETECTED EVENTS")
    print("-" * 72)
    print("  Layer 2 silent, Layer 1 fires  -> environmental anomaly (weather)")
    print("  Layer 2 fires                  -> hardware fault, data untrustworthy")
    print()
    if n_env_det:
        print(f"  Weather events detected  : {n_env_det:>6}   "
              f"correctly called weather: {env_correct:>6} "
              f"({100 * env_correct / n_env_det:.1f} %)")
    if n_hw_det:
        print(f"  Hardware faults detected : {n_hw_det:>6}   "
              f"correctly called faults : {hw_correct:>6} "
              f"({100 * hw_correct / n_hw_det:.1f} %)")

    classification = {
        "weather_events_detected": n_env_det,
        "weather_classified_correctly": env_correct,
        "hardware_faults_detected": n_hw_det,
        "hardware_classified_correctly": hw_correct,
    }
    overall["Layer2_vs_hardware_faults_only"] = l2_sensor

    tbl.to_csv(cfg.RESULTS_DIR / "combined_overall.csv")
    per_type.to_csv(cfg.RESULTS_DIR / "combined_by_type.csv")

    l2_line = " | ".join(f"{k} {l2_sensor[k]}" for k in
                         ("Precision", "Recall", "F1", "FalseAlarmRate_%"))
    (cfg.RESULTS_DIR / "combined_results.md").write_text(
        "# Two-layer anomaly detection results\n\n"
        f"Dataset: `{cfg.DATASET}`  |  Test samples: {len(label):,}  |  "
        f"Injected anomalies: {int(label.sum()):,}\n\n"
        "## Overall\n\n" + tbl.to_markdown() +
        "\n\n## Recall by fault type\n\n" + per_type.to_markdown() +
        "\n\n## Layer 2 scored against hardware faults only\n\n"
        "The overall table scores every detector against 'was anything "
        "injected here'. That penalises Layer 2 for correct behaviour: a "
        "frontal passage is weather, both sensors see it, and Layer 2 is "
        "meant to stay silent. Scored against the faults it actually "
        "targets:\n\n"
        f"{l2_line}\n\n"
        "## Classification of detected events\n\n"
        f"- Weather events detected: {n_env_det}, correctly identified as "
        f"weather: {env_correct}\n"
        f"- Hardware faults detected: {n_hw_det}, correctly identified as "
        f"faults: {hw_correct}\n")
    with open(cfg.RESULTS_DIR / "combined_results.json", "w") as f:
        json.dump({"overall": overall, "by_type": rows,
                   "classification": classification}, f, indent=2)

    print("\nSaved -> results/combined_results.{md,json}, combined_*.csv")


if __name__ == "__main__":
    main()
