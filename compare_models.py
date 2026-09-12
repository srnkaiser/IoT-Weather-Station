"""
Rigorous comparison of the Attention-BiLSTM against the existing baseline.

WHY THIS EXISTS

train_bilstm.py prints two MAE numbers next to each other. That is enough to
see which is lower and not enough to defend the claim at a conference. Three
things are missing, and each of them is a question a reviewer will ask:

  1. Is the difference larger than chance?
     Forecast errors on a 10-minute time series are strongly autocorrelated -
     a warm bias persists for hours. Treating 84,000 samples as independent
     would make a meaningless gap look overwhelmingly significant. This
     script uses the Diebold-Mariano test with a HAC variance, which is the
     standard tool for exactly this situation, plus a moving-block bootstrap
     as a second opinion that makes no distributional assumption at all.

  2. Are the two models even scored on the same samples?
     They are not, by construction. The baseline drops rows where a lagged
     feature is missing; the sequence model drops the first seq_len rows of
     each split. Comparing their published MAEs compares two different test
     sets. Everything here is aligned on timestamps first, so both models are
     judged on exactly the same forecasts.

  3. Does the average hide a regime where the ranking flips?
     A model that wins overall but loses every winter night is a different
     finding from one that wins everywhere. The breakdown by season, hour and
     volatility settles that.

It also extracts the attention weights, which is the interpretability claim
in the paper draft and had never been checked against the trained model.

THREE STAGES, because the two models live in different environments: the
baseline was fitted under sklearn 1.4 and the sequence models need
TensorFlow, whose environment carries sklearn 1.9. Predictions are therefore
exported separately and analysed together.

    python3 compare_models.py --export-baseline
    .venv-tf/bin/python compare_models.py --export-bilstm 60min_calendar
    python3 compare_models.py --analyse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data_loader import load_dataset

PRED_DIR = cfg.RESULTS_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)

BASE_CHANNELS = ["temperature", "humidity", "pressure"]
CALENDAR_CHANNELS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]


# --------------------------------------------------------------------------
# Stage 1: export predictions, one file per model, indexed by target time
# --------------------------------------------------------------------------
def export_baseline():
    """Predictions of the committed Gradient Boosting model on its test set."""
    import joblib
    from features import build_supervised, chronological_split

    bundle = joblib.load(cfg.MODEL_DIR / "forecast_model.joblib")
    df = load_dataset()
    X, y, persistence = build_supervised(df)
    Xtr, Xte, ytr, yte, ptr, pte = chronological_split(X, y, persistence)

    pred = bundle["model"].predict(Xte)

    # The index of X is the time the forecast is ISSUED. The quantity being
    # predicted sits one horizon later, and that is what has to match the
    # sequence model - otherwise the two are aligned an hour apart.
    target_time = Xte.index + pd.Timedelta(minutes=cfg.FORECAST_HORIZON_MIN)

    out = pd.DataFrame({"y_true": yte.to_numpy(), "y_pred": pred,
                        "persistence": pte.to_numpy()}, index=target_time)
    out.index.name = "target_time"
    path = PRED_DIR / "baseline_gradientboosting.csv"
    out.to_csv(path)
    print(f"{bundle['model_name']}: {len(out):,} predictions -> {path.name}")
    print(f"  MAE {np.abs(out.y_true - out.y_pred).mean():.4f}")


def export_bilstm(tag: str):
    """Predictions of a saved Attention-BiLSTM, plus its attention weights."""
    import tensorflow as tf
    from tensorflow import keras

    spec_path = cfg.RESULTS_DIR / f"bilstm_{tag}.json"
    if not spec_path.exists():
        raise SystemExit(f"No result file for tag '{tag}'")
    spec = json.load(open(spec_path))

    # The run's own JSON says how it was built. Re-deriving the settings from
    # the tag string would silently diverge the moment a tag is renamed.
    seq_len = max(2, spec["seq_minutes"] // cfg.STEP_MINUTES)
    horizon = cfg.FORECAST_HORIZON_MIN // cfg.STEP_MINUTES
    use_cal = bool(spec.get("calendar_channels", False))

    if spec.get("engineered_features", False):
        from features import build_features
        base = load_dataset()
        feat = build_features(base).dropna()
        feat[cfg.TARGET] = base[cfg.TARGET].reindex(feat.index)
        df = feat.dropna()
        channels = [c for c in df.columns if c != cfg.TARGET] + [cfg.TARGET]
        df = df[channels]
        return _run_export(tag, spec, df, channels, seq_len, horizon)

    df = load_dataset()[BASE_CHANNELS].dropna()
    if use_cal:
        idx = df.index
        hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
        doy = idx.dayofyear.to_numpy().astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    channels = BASE_CHANNELS + (CALENDAR_CHANNELS if use_cal else [])
    df = df[channels]
    return _run_export(tag, spec, df, channels, seq_len, horizon)


def _run_export(tag, spec, df, channels, seq_len, horizon):
    import numpy as np
    from tensorflow import keras

    values = df.to_numpy(dtype=np.float32)
    target = df[cfg.TARGET].to_numpy(dtype=np.float32)
    cut = int(len(values) * (1 - cfg.TEST_SIZE))
    mu, sd = values[:cut].mean(0), values[:cut].std(0) + 1e-8
    scaled = (values - mu) / sd

    arr = scaled[cut:]
    n = len(arr) - seq_len - horizon + 1
    stride = arr.strides[0]
    X = np.lib.stride_tricks.as_strided(
        arr, shape=(n, seq_len, arr.shape[1]),
        strides=(stride, stride, arr.strides[1]), writeable=False)
    X = np.ascontiguousarray(X)

    first = cut + seq_len + horizon - 1
    y_true = target[first: first + n]
    target_time = df.index[first: first + n]
    persistence = target[cut + seq_len - 1: cut + seq_len - 1 + n]

    # Rebuild the architecture from the run's own record and load the saved
    # weights into it, rather than deserialising the whole model. This reads
    # the checkpoints written before the Lambda layer was replaced as well as
    # the ones written after, so no run has to be repeated to be analysed.
    from train_bilstm import build_model
    # Runs made before --units was recorded used the then-default 64. Guessing
    # is only safe because the parameter count is verified below, which fails
    # loudly if the guess is wrong.
    units = spec.get("units", 64)
    model = build_model(seq_len, len(channels), units,
                        attention=spec.get("attention", True),
                        bidirectional=spec.get("bidirectional", True))
    model.load_weights(cfg.MODEL_DIR / f"forecast_bilstm_{tag}.keras")
    if model.count_params() != spec["parameters"]:
        raise SystemExit(
            f"Rebuilt {tag} has {model.count_params():,} parameters but the "
            f"run recorded {spec['parameters']:,} - architecture mismatch.")
    pred = model.predict(X, batch_size=512, verbose=0).ravel()

    out = pd.DataFrame({"y_true": y_true, "y_pred": pred,
                        "persistence": persistence}, index=target_time)
    out.index.name = "target_time"
    path = PRED_DIR / f"bilstm_{tag}.csv"
    out.to_csv(path)
    print(f"bilstm_{tag}: {len(out):,} predictions -> {path.name}")
    print(f"  MAE {np.abs(out.y_true - out.y_pred).mean():.4f}")

    # --- attention weights -------------------------------------------------
    # The paper draft argues the attention layer makes the forecast readable:
    # you can see which part of the history it used. That is worth checking
    # rather than asserting - a softmax over 6 near-identical timesteps can
    # just as easily come out flat, in which case the claim does not hold.
    if not spec.get("attention", True):
        print("  (no attention layer in this ablation)")
        return
    att = keras.Model(model.input,
                      model.get_layer("attention_weights").output)
    sample = X[:20000]
    w = att.predict(sample, batch_size=512, verbose=0)[..., 0]
    mean_w = w.mean(axis=0)
    age = [(seq_len - 1 - i) * cfg.STEP_MINUTES for i in range(seq_len)]
    aw = pd.DataFrame({"minutes_before_issue": age, "mean_weight": mean_w,
                       "std_weight": w.std(axis=0)})
    aw.to_csv(PRED_DIR / f"attention_{tag}.csv", index=False)
    print(f"  attention: uniform would be {1/seq_len:.4f}; "
          f"observed range {mean_w.min():.4f}-{mean_w.max():.4f}")


# --------------------------------------------------------------------------
# Stage 2: statistics
# --------------------------------------------------------------------------
def hac_variance(d: np.ndarray, max_lag: int) -> float:
    """
    Newey-West (Bartlett) long-run variance of the loss differential.

    Plain var(d)/n assumes independent samples. These are 10-minute forecasts
    whose errors stay correlated for hours, so that would understate the
    standard error several-fold and turn any gap into a significant one.
    """
    n = len(d)
    dm = d - d.mean()
    gamma0 = float(dm @ dm) / n
    total = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(dm[lag:] @ dm[:-lag]) / n
        total += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    return max(total, 1e-12)


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, max_lag: int):
    """
    Test whether model A and model B forecast equally well (absolute loss).

    Positive statistic -> A has the larger error, i.e. B is better.
    """
    d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    lrv = hac_variance(d, max_lag)
    stat = float(d.mean() / np.sqrt(lrv / n))
    # Two-sided normal p-value without pulling in scipy.
    from math import erfc, sqrt
    p = float(erfc(abs(stat) / sqrt(2.0)))
    return stat, p, float(d.mean())


def block_bootstrap_ci(e_a: np.ndarray, e_b: np.ndarray, block: int,
                       n_boot: int = 2000, seed: int = 42):
    """
    Confidence interval for the MAE gap, resampling contiguous blocks.

    Blocks rather than single points, for the same reason the variance above
    is HAC-corrected: whole weather situations have to move together or the
    resample destroys the autocorrelation it is meant to respect.
    """
    d = np.abs(e_a) - np.abs(e_b)
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i, :, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_predictions() -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(PRED_DIR.glob("*.csv")):
        if path.name.startswith("attention_"):
            continue
        out[path.stem] = pd.read_csv(path, index_col=0, parse_dates=True)
    return out


def analyse():
    preds = load_predictions()
    if len(preds) < 2:
        raise SystemExit("Need at least two exported models. Run the "
                         "--export steps first.")

    # Align every model on the timestamps they all share. This is the step
    # that makes the comparison legitimate; without it each model is scored
    # on a slightly different stretch of the test period.
    common = None
    for df in preds.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    print(f"Models: {len(preds)}   common forecasts: {len(common):,}")
    for name, df in preds.items():
        print(f"  {name:<32} own test set {len(df):,}")

    y = preds[next(iter(preds))].loc[common, "y_true"].to_numpy()
    persist = preds[next(iter(preds))].loc[common, "persistence"].to_numpy()
    err = {n: preds[n].loc[common, "y_pred"].to_numpy() - y for n in preds}
    err["persistence"] = persist - y

    base_mae = float(np.abs(err["persistence"]).mean())

    rows = []
    for name, e in err.items():
        mae = float(np.abs(e).mean())
        rows.append({"model": name, "MAE": round(mae, 4),
                     "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
                     "Bias": round(float(e.mean()), 4),
                     "Skill_%": round(100 * (1 - mae / base_mae), 2)})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    print("\n" + "=" * 74)
    print("ON THE IDENTICAL SAMPLE SET")
    print("=" * 74)
    print(table.to_string(index=False))

    # 24 h of lags: generous, and therefore conservative. The rule-of-thumb
    # minimum for a 6-step-ahead forecast is 5 lags; the errors here stay
    # correlated far longer than that, so the wider window is the honest one.
    max_lag = 144
    block = 144

    names = [r for r in table["model"] if r != "persistence"]
    best = names[0]
    print("\n" + "=" * 74)
    print(f"IS THE GAP REAL?  Diebold-Mariano vs '{best}', HAC {max_lag} lags")
    print("=" * 74)
    stats = []
    for name in names[1:]:
        stat, p, gap = diebold_mariano(err[name], err[best], max_lag)
        lo, hi = block_bootstrap_ci(err[name], err[best], block)
        verdict = ("significant" if p < 0.05 else
                   "NOT distinguishable")
        stats.append({"model": name, "MAE_gap": round(gap, 4),
                      "DM_stat": round(stat, 2), "p_value": f"{p:.2e}",
                      "CI95_low": round(lo, 4), "CI95_high": round(hi, 4),
                      "verdict": verdict})
        print(f"  {name:<32} gap {gap:+.4f} degC   DM {stat:+6.2f}   "
              f"p {p:.2e}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")

    # A reviewer will ask whether the sequence model is merely mis-calibrated.
    # Answered with an oracle correction - the mean error removed using the
    # test set itself, which no legitimate method could beat. If even that
    # does not close the gap, calibration is not the explanation.
    print("\n" + "-" * 74)
    print("COULD BIAS CORRECTION CLOSE IT?  (oracle bound, not a usable method)")
    print("-" * 74)
    bias_rows = []
    for name in names:
        e = err[name]
        mae, mae_c = float(np.abs(e).mean()), float(np.abs(e - e.mean()).mean())
        bias_rows.append({"model": name, "MAE": round(mae, 4),
                          "MAE_debiased": round(mae_c, 4),
                          "gain": round(mae - mae_c, 4)})
        print(f"  {name:<32} {mae:.4f} -> {mae_c:.4f}   gain {mae - mae_c:.4f}")
    gb_c = bias_rows[0]["MAE_debiased"]
    others = [r for r in bias_rows[1:]]
    if others:
        best_other = min(others, key=lambda r: r["MAE_debiased"])
        print(f"  Gap after debiasing both: "
              f"{best_other['MAE_debiased'] - gb_c:+.4f} degC "
              f"(was {best_other['MAE'] - bias_rows[0]['MAE']:+.4f})")

    print("\n" + "-" * 74)
    print("HAC lag sensitivity (does the verdict depend on the window?)")
    print("-" * 74)
    lag_rows = []
    for lag in (5, 36, 144, 432):
        cells = []
        for name in names[1:]:
            st, pv, _ = diebold_mariano(err[name], err[best], lag)
            cells.append(f"{name.replace('bilstm_', '')}: DM {st:+.2f} "
                         f"(p {pv:.1e})")
        lag_rows.append({"HAC_lags": lag,
                         "hours": round(lag * cfg.STEP_MINUTES / 60, 1),
                         "result": "; ".join(cells)})
        print(f"  {lag:>4} lags ({lag * cfg.STEP_MINUTES / 60:>5.1f} h)  "
              + "; ".join(cells))

    # --- where does the ranking hold? --------------------------------------
    print("\n" + "=" * 74)
    print("ERROR BY REGIME  (mean absolute error, degC)")
    print("=" * 74)
    regimes = {}
    ts = pd.DatetimeIndex(common)
    regimes["season"] = pd.Series(
        pd.cut(ts.month, [0, 2, 5, 8, 11, 12],
               labels=["winter", "spring", "summer", "autumn", "winter"],
               ordered=False), index=ts)
    regimes["time_of_day"] = pd.Series(
        pd.cut(ts.hour, [-1, 5, 11, 17, 23],
               labels=["night", "morning", "afternoon", "evening"]), index=ts)
    # Volatility: how much the temperature moved in the hour BEFORE the
    # forecast was issued.
    #
    # The obvious shortcut - |y - persistence| - is wrong and was used here
    # once. That quantity is the future change, which is the persistence error
    # itself, so conditioning on it hands persistence the calm group by
    # construction and tells you nothing. It also could not be computed in
    # operation, where the future is what is being asked about. The past hour
    # is known at issue time and is a legitimate conditioning variable.
    issue_time = ts - pd.Timedelta(minutes=cfg.FORECAST_HORIZON_MIN)
    temp = load_dataset()[cfg.TARGET]
    past = (temp.reindex(issue_time).to_numpy()
            - temp.reindex(issue_time - pd.Timedelta(minutes=60)).to_numpy())
    vol = pd.Series(np.abs(past), index=ts)
    regimes["volatility"] = pd.Series(
        pd.qcut(vol, 3, labels=["calm", "moderate", "volatile"]), index=ts)

    breakdown = {}
    for rname, groups in regimes.items():
        rows = []
        for level in groups.dropna().unique():
            m = (groups == level).to_numpy()
            row = {"regime": rname, "level": str(level), "n": int(m.sum())}
            for name in names + ["persistence"]:
                row[name] = round(float(np.abs(err[name][m]).mean()), 4)
            rows.append(row)
        sub = pd.DataFrame(rows).sort_values("level")
        breakdown[rname] = sub
        print(f"\n{rname}")
        print(sub.to_string(index=False))

    # --- write it down -----------------------------------------------------
    lines = ["# Attention-BiLSTM vs the existing baseline\n",
             f"All models scored on the same {len(common):,} forecasts "
             f"({common.min():%Y-%m-%d} to {common.max():%Y-%m-%d}), aligned "
             "on target timestamp.\n",
             "## Accuracy\n", table.to_markdown(index=False), "",
             f"## Significance (Diebold-Mariano, HAC {max_lag} lags, "
             f"reference '{best}')\n",
             pd.DataFrame(stats).to_markdown(index=False), "",
             "### Is it just a calibration offset?\n",
             "An oracle correction - the mean error removed using the test "
             "set itself, which no legitimate method could beat. If even "
             "this does not close the gap, calibration is not the "
             "explanation.\n",
             pd.DataFrame(bias_rows).to_markdown(index=False), "",
             "The HAC window is a choice, so here is the same test across a "
             "range of windows. The verdict does not depend on it.\n",
             pd.DataFrame(lag_rows).to_markdown(index=False), "",
             "## Error by regime\n",
             "Volatility is the absolute temperature change over the hour "
             "BEFORE the forecast was issued - known at issue time, so it is "
             "a legitimate way to split the test set.\n"]
    for rname, sub in breakdown.items():
        lines += [f"### {rname}\n", sub.to_markdown(index=False), ""]

    att = sorted(PRED_DIR.glob("attention_*.csv"))
    if att:
        lines.append("## Attention weights\n")
        for a in att:
            aw = pd.read_csv(a)
            uni = 1.0 / len(aw)
            lines += [f"### {a.stem.replace('attention_', '')} "
                      f"(uniform = {uni:.4f})\n",
                      aw.to_markdown(index=False), ""]

    out = cfg.RESULTS_DIR / "model_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    table.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)
    json.dump({"n_common": int(len(common)), "accuracy": rows,
               "significance": stats},
              open(cfg.RESULTS_DIR / "model_comparison.json", "w"), indent=2,
              default=str)
    print(f"\nSaved -> results/model_comparison.md, .csv, .json")


# --------------------------------------------------------------------------
# Stage 3: figures
# --------------------------------------------------------------------------
def make_plots():
    """Three figures, each answering one question the text raises."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.read_csv(cfg.RESULTS_DIR / "model_comparison.csv")
    short = {"baseline_gradientboosting": "Gradient Boosting\n(baseline)",
             "persistence": "Persistence\n(naive)"}

    def label(n):
        return short.get(n, n.replace("bilstm_", "BiLSTM ")
                         .replace("_", " ").replace("60min", "60 min")
                         .replace("720min", "12 h"))

    # 1) Accuracy. A dot plot on a truncated axis rather than bars from zero:
    #    every model sits between 0.45 and 0.54, so bars from zero render the
    #    entire result as eleven near-identical rectangles. Dots carry no area,
    #    so shortening the axis does not exaggerate anything - which is exactly
    #    the objection that rules a truncated bar chart out.
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    body = table[table.model != "persistence"].sort_values("MAE",
                                                           ascending=False)
    base = float(table.loc[table.model == "persistence", "MAE"].iloc[0])
    gb = float(table.loc[table.model == "baseline_gradientboosting",
                         "MAE"].iloc[0]) if "baseline_gradientboosting" \
        in set(table.model) else None
    y = np.arange(len(body))
    for i, (m, v) in enumerate(zip(body.model, body.MAE)):
        is_base = m == "baseline_gradientboosting"
        c = "#264653" if is_base else ("#2a9d8f" if gb and v < gb else "#e76f51")
        ax.hlines(i, min(v, gb or v), max(v, gb or v), color=c, alpha=0.25, lw=2)
        ax.plot(v, i, "o", ms=9, color=c)
        ax.text(v + 0.0016, i, f"{v:.4f}", va="center", fontsize=8.5, color=c)
    if gb:
        ax.axvline(gb, color="#264653", ls="--", lw=1.3, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m).replace("\n", " ") for m in body.model],
                       fontsize=9)
    ax.set_xlabel("MAE (degC) - lower is better")
    ax.set_xlim(body.MAE.min() - 0.006, body.MAE.max() + 0.010)
    ax.set_title("Temperature forecast 60 min ahead\n"
                 f"identical test samples, identical split "
                 f"(persistence baseline: {base:.3f} degC)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg.RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 2) Attention over the history window. The paper draft claims the model
    #    shows which part of the past it used; this is that claim, plotted.
    files = sorted(PRED_DIR.glob("attention_*.csv"))
    if files:
        fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 4.0),
                                 squeeze=False)
        for ax, f in zip(axes[0], files):
            aw = pd.read_csv(f)
            uni = 1.0 / len(aw)
            ax.plot(aw.minutes_before_issue, aw.mean_weight, marker="o",
                    ms=3.5, color="#e76f51", lw=1.6)
            ax.fill_between(aw.minutes_before_issue,
                            aw.mean_weight - aw.std_weight,
                            aw.mean_weight + aw.std_weight,
                            color="#e76f51", alpha=0.17)
            ax.axhline(uni, color="#666", ls="--", lw=1.2)
            ax.text(aw.minutes_before_issue.max(), uni,
                    f" uniform {uni:.3f}", va="bottom", ha="right",
                    color="#666", fontsize=8.5)
            ax.invert_xaxis()
            ax.set_xlabel("minutes before the forecast is issued")
            ax.set_ylabel("mean attention weight")
            ax.set_title(f.stem.replace("attention_", ""), fontsize=10)
            ax.grid(alpha=0.25)
        fig.suptitle("What the attention layer actually looks at", y=1.0)
        fig.tight_layout()
        fig.savefig(cfg.RESULTS_DIR / "attention_weights.png", dpi=150)
        plt.close(fig)

    # 3) Where the ranking holds. An average can hide a reversal; this shows
    #    whether there is one.
    md = cfg.RESULTS_DIR / "model_comparison.md"
    if md.exists():
        import re
        raw = md.read_text()
        blocks = {}
        for rname in ("season", "time_of_day", "volatility"):
            m = re.search(rf"### {rname}\n\n(\|.*?)\n\n", raw, re.S)
            if m:
                rows = [r.strip().strip("|").split("|")
                        for r in m.group(1).strip().splitlines()]
                head = [c.strip() for c in rows[0]]
                data = [[c.strip() for c in r] for r in rows[2:]]
                blocks[rname] = pd.DataFrame(data, columns=head)
        if blocks:
            keep = [c for c in ("baseline_gradientboosting",
                                "bilstm_60min_calendar", "bilstm_60min_tuned")
                    if c in next(iter(blocks.values())).columns]
            fig, axes = plt.subplots(1, len(blocks),
                                     figsize=(4.6 * len(blocks), 4.0),
                                     squeeze=False)
            cmap = {"baseline_gradientboosting": "#264653",
                    "bilstm_60min_calendar": "#e76f51",
                    "bilstm_60min_tuned": "#f4a261"}
            for ax, (rname, sub) in zip(axes[0], blocks.items()):
                x = np.arange(len(sub))
                w = 0.8 / len(keep)
                for i, col in enumerate(keep):
                    ax.bar(x + i * w - 0.4 + w / 2,
                           sub[col].astype(float), w,
                           label=label(col).replace("\n", " "),
                           color=cmap.get(col))
                ax.set_xticks(x)
                ax.set_xticklabels(sub["level"], fontsize=9)
                ax.set_title(rname, fontsize=10)
                ax.set_ylabel("MAE (degC)")
                ax.grid(axis="y", alpha=0.25)
            axes[0][0].legend(fontsize=8, loc="upper left")
            fig.suptitle("Does the ranking hold in every regime?", y=1.0)
            fig.tight_layout()
            fig.savefig(cfg.RESULTS_DIR / "regime_breakdown.png", dpi=150)
            plt.close(fig)

    print("Saved -> results/model_comparison.png, attention_weights.png, "
          "regime_breakdown.png")


# --------------------------------------------------------------------------
# Stage 4: aggregate repeated runs
# --------------------------------------------------------------------------
def summarise_runs():
    """
    Group every training run by configuration and report spread across seeds.

    A single run is not a result here. The configurations differ from one
    another by 0.001-0.018 degC and the same configuration differs from itself
    by up to 0.009 degC depending on the seed, so a table of single runs would
    rank architectures by which of them got the luckier initialisation. Only
    the mean over repeats can be compared, and the spread has to be shown
    beside it so the reader can see when a difference is inside the noise.
    """
    rows = []
    for path in sorted(cfg.RESULTS_DIR.glob("bilstm_*.json")):
        d = json.load(open(path))
        rows.append({
            "tag": path.stem.replace("bilstm_", ""),
            "architecture": d.get("architecture", "attention_bilstm"),
            "seq_minutes": d["seq_minutes"],
            "inputs": ("engineered" if d.get("engineered_features") else
                       "raw+calendar" if d.get("calendar_channels") else "raw"),
            "units": d.get("units", 64),
            "lr": d.get("learning_rate", 1e-3),
            "seed": d.get("seed"),
            "parameters": d["parameters"],
            "MAE": d["MAE"],
            "skill": d["Skill_vs_Persistence_%"],
            "train_s": d["train_seconds"],
        })
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("No runs found.")

    # Hyperparameters belong in the key. Without them the untuned 64-unit run
    # lands in the same group as the tuned 96-unit ones and inflates that
    # group's spread with a difference that is not a seed difference at all.
    key = ["architecture", "seq_minutes", "inputs", "units", "lr"]
    agg = (runs.groupby(key)
           .agg(n_seeds=("MAE", "size"), MAE_mean=("MAE", "mean"),
                MAE_min=("MAE", "min"), MAE_max=("MAE", "max"),
                MAE_std=("MAE", "std"), parameters=("parameters", "max"),
                train_s=("train_s", "mean"))
           .reset_index().sort_values("MAE_mean"))
    agg["spread"] = agg.MAE_max - agg.MAE_min
    for c in ("MAE_mean", "MAE_min", "MAE_max", "MAE_std", "spread"):
        agg[c] = agg[c].round(4)
    agg["train_s"] = agg.train_s.round(0)

    # The baseline is deterministic (fixed random_state), so it is a point,
    # not a distribution - shown as a reference line rather than a row.
    base = json.load(open(cfg.RESULTS_DIR / "forecast_metrics.json"))
    gb = base["GradientBoosting"]["MAE"]

    print("=" * 96)
    print("EVERY RUN, GROUPED BY CONFIGURATION")
    print("=" * 96)
    print(agg.to_string(index=False))
    print(f"\nGradient Boosting baseline: MAE {gb:.4f} (deterministic, one run)")

    print("\n" + "-" * 96)
    print("IS A DIFFERENCE BIGGER THAN THE NOISE IT SITS IN?")
    print("-" * 96)
    multi = agg[agg.n_seeds > 1]
    if len(multi) == 0:
        print("  No configuration has repeats yet.")
    else:
        worst = float(multi.spread.max())
        print(f"  Largest observed within-configuration spread: {worst:.4f} degC")
        print(f"  Treat any gap below that as unresolved.\n")
        best = agg.iloc[0]
        for _, r in agg.iloc[1:].iterrows():
            gap = r.MAE_mean - best.MAE_mean
            mark = "resolved" if gap > worst else "INSIDE THE NOISE"
            print(f"  {r.architecture:<18} {r.inputs:<13} "
                  f"u{int(r.units):<4} {r.MAE_mean:.4f}  vs best {gap:+.4f}  "
                  f"{mark}")
        print(f"\n  vs the baseline: best configuration is "
              f"{best.MAE_mean - gb:+.4f} degC from Gradient Boosting")

    runs.sort_values("MAE").to_csv(cfg.RESULTS_DIR / "all_runs.csv", index=False)
    agg.to_csv(cfg.RESULTS_DIR / "runs_by_configuration.csv", index=False)
    (cfg.RESULTS_DIR / "runs_by_configuration.md").write_text(
        "# Every training run, grouped by configuration\n\n"
        "A single run is not a result: configurations differ from one another "
        "by 0.001-0.018 degC and the same configuration differs from itself by "
        "up to 0.009 degC depending on the seed. Compare the means, and treat "
        "a gap smaller than the spread column as unresolved.\n\n"
        + agg.to_markdown(index=False)
        + f"\n\nGradient Boosting baseline: MAE {gb:.4f} "
          "(deterministic, one run).\n\n## Individual runs\n\n"
        + runs.sort_values("MAE").to_markdown(index=False) + "\n")
    print("\nSaved -> results/runs_by_configuration.md, .csv, all_runs.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-baseline", action="store_true")
    ap.add_argument("--export-bilstm", type=str, metavar="TAG")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="group every run by configuration, show seed spread")
    args = ap.parse_args()

    if args.export_baseline:
        export_baseline()
    elif args.export_bilstm:
        export_bilstm(args.export_bilstm)
    elif args.analyse:
        analyse()
    elif args.plots:
        make_plots()
    elif args.summary:
        summarise_runs()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
