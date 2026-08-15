"""
Sensitivity analysis for the Isolation Forest's contamination parameter.

WHY THIS EXISTS

contamination tells the Isolation Forest what fraction of the data to treat
as anomalous. It is not learned - it is asserted, and it fixes the false
alarm rate almost by itself: set it to 0.03 and roughly 3 % of all readings
get flagged, whatever the data actually contains.

That is the parameter's weakness, and the honest way to report it is not to
pick one value and hope. It is to show the whole trade-off curve and then
justify the choice. Precision and recall move in opposite directions here,
and where you land on that curve is a deployment decision, not a statistical
one: a weather station that cries wolf every hour gets ignored.

READING THE RESULT

The dashed line marks the true anomaly rate in the test set. Setting
contamination above it forces false positives - the model has to flag its
quota even once it has run out of real anomalies to find. Below it, the
model necessarily misses some.

METHODOLOGICAL NOTE FOR THE PAPER

Selecting the value that maximises F1 on this curve means tuning against the
test set, which is exactly the kind of optimism the chronological split was
built to avoid. In deployment nobody knows the true anomaly rate. Pick from
the expected fault rate of the hardware, state that you did, and use this
curve to show what that choice costs.

Usage:  python3 tune_contamination.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data_loader import load_dataset
from features import build_features
from train_anomaly import ANOMALY_FEATURES, evaluate, inject_faults

# Spanning a factor of ten around the current setting of 0.03.
CONTAMINATION_GRID = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025,
                      0.03, 0.04, 0.05]


def main():
    print("=" * 72)
    print("CONTAMINATION SENSITIVITY ANALYSIS")
    print("=" * 72)

    df = load_dataset()
    cut = int(len(df) * (1 - cfg.TEST_SIZE))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    # Build the feature matrices once. They do not depend on contamination,
    # and on 420k rows they cost far more than fitting the forest.
    print("\nBuilding features ...")
    Xtr = build_features(train_df)[ANOMALY_FEATURES].dropna()

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    corrupted, label, ftype = inject_faults(test_df, rng)
    Xte = build_features(corrupted)[ANOMALY_FEATURES]
    valid = Xte.notna().all(axis=1)
    Xte, label, ftype = Xte[valid], label[valid], ftype[valid]

    true_rate = float(label.mean())
    print(f"Train: {len(Xtr):,} samples   Test: {len(Xte):,} samples")
    print(f"True anomaly rate in the test set: {100 * true_rate:.2f} %\n")

    rows = []
    for c in CONTAMINATION_GRID:
        params = dict(cfg.IFOREST_PARAMS)
        params["contamination"] = c
        pipe = make_pipeline(StandardScaler(), IsolationForest(**params))
        pipe.fit(Xtr)

        pred = (pipe.predict(Xte) == -1).astype(int)
        m = evaluate(label.to_numpy(), pred)

        # Recall per fault type - the split between what each layer can see
        # is the point of the whole architecture, so track it here too.
        per_type = {}
        for name in ["spike", "stuck", "drift", "dropout", "frontal"]:
            mask = (ftype == name).to_numpy()
            if mask.sum():
                per_type[f"recall_{name}"] = round(float(pred[mask].mean()), 4)

        rows.append({"contamination": c, "flagged_%": round(100 * pred.mean(), 3),
                     **m, **per_type})
        print(f"  contamination={c:<7} P={m['Precision']:.4f}  "
              f"R={m['Recall']:.4f}  F1={m['F1']:.4f}  "
              f"FAR={m['FalseAlarmRate_%']:.3f}%")

    table = pd.DataFrame(rows).set_index("contamination")

    print("\n" + "-" * 72)
    print("RESULT")
    print("-" * 72)
    print(table[["Precision", "Recall", "F1", "FalseAlarmRate_%",
                 "flagged_%"]].to_string())

    best = table["F1"].idxmax()
    print(f"\nHighest F1 at contamination = {best}")
    print("Do not simply adopt that value - see the note at the top of this "
          "file.\nIt is the best fit to THIS test set, which is not the same "
          "as the best\nchoice for deployment.")

    # --- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(table.index, table["Precision"], "o-", color="#2a9d8f",
            label="Precision", lw=2)
    ax.plot(table.index, table["Recall"], "s-", color="#e76f51",
            label="Recall", lw=2)
    ax.plot(table.index, table["F1"], "^-", color="#264653",
            label="F1", lw=2)
    ax.axvline(true_rate, ls="--", color="#888", lw=1.5)
    ax.annotate(f"true anomaly rate\n{100 * true_rate:.2f} %",
                xy=(true_rate, 0.92), xytext=(true_rate * 1.25, 0.92),
                fontsize=9, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888"))
    ax.set_xlabel("contamination")
    ax.set_ylabel("Score")
    ax.set_title("Isolation Forest: contamination controls the "
                 "precision-recall trade-off")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_DIR / "contamination_sensitivity.png", dpi=150)
    plt.close(fig)

    table.to_csv(cfg.RESULTS_DIR / "contamination_sensitivity.csv")
    (cfg.RESULTS_DIR / "contamination_sensitivity.md").write_text(
        "# Contamination sensitivity (Layer 1, Isolation Forest)\n\n"
        f"Dataset: `{cfg.DATASET}`  |  Test samples: {len(label):,}  |  "
        f"True anomaly rate: {100 * true_rate:.2f} %\n\n"
        + table[["Precision", "Recall", "F1", "FalseAlarmRate_%",
                 "flagged_%"]].to_markdown()
        + "\n\n## Recall by fault type\n\n"
        + table[[c for c in table.columns if c.startswith("recall_")]].to_markdown()
        + "\n")
    with open(cfg.RESULTS_DIR / "contamination_sensitivity.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved -> results/contamination_sensitivity.{png,csv,md,json}")


if __name__ == "__main__":
    main()
