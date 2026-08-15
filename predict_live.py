"""
Live inference: ThingSpeak -> trained models -> forecast + alerts.

This is the only script that touches ThingSpeak, and it only ever READS from
it. Training never uses ThingSpeak data, because Wokwi values are set by hand
and are not real weather. Keeping that separation clean is the whole point of
the architecture.

    python src/predict_live.py                 # fetch from ThingSpeak
    python src/predict_live.py --csv data.csv  # offline, from a CSV
    python src/predict_live.py --watch 300     # rerun every 300 seconds

Configure channel ID, API key and field mapping in config.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data_loader import VALID_RANGES, clean
from features import build_features
from sensor_check import run_checks
from train_anomaly import ANOMALY_FEATURES


def fetch_thingspeak(channel_id: str | None = None, read_key: str | None = None,
                     results: int | None = None) -> pd.DataFrame:
    """Pull the most recent feed entries from a ThingSpeak channel."""
    try:
        import requests
    except ImportError:
        raise SystemExit("pip install requests")

    ts = cfg.THINGSPEAK
    channel_id = channel_id or ts["channel_id"]
    read_key = read_key if read_key is not None else ts["read_api_key"]
    results = results or ts["results"]

    if channel_id == "YOUR_CHANNEL_ID":
        raise SystemExit(
            "ThingSpeak is not configured yet. Put your channel ID and read "
            "API key into config.py, or run with --csv for an offline test.")

    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {"results": results}
    if read_key:
        params["api_key"] = read_key

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    feeds = r.json().get("feeds", [])
    if not feeds:
        raise SystemExit("ThingSpeak returned no data for this channel.")

    df = pd.DataFrame(feeds)
    # ThingSpeak reports UTC; the training data is in the dataset's local
    # time. Shift before dropping the timezone, so the hour-of-day features
    # mean the same thing at inference as they did during training.
    stamps = pd.to_datetime(df["created_at"], utc=True) + \
        pd.Timedelta(hours=cfg.DATASET_UTC_OFFSET_HOURS)
    df.index = stamps.dt.tz_localize(None)
    df = df.rename(columns=ts["field_map"])

    keep = [c for c in ts["field_map"].values() if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    print(f"Fetched {len(df)} points from channel {channel_id} "
          f"({df.index.min()} -> {df.index.max()})")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df.index = pd.to_datetime(df[tcol])
    return df.drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce")


def select_forecast_model(df: pd.DataFrame):
    """
    Pick the best model the available history can actually feed.

    The main model needs 60 minutes of uninterrupted data. Wokwi rarely
    delivers that, because the browser pauses the simulation whenever its tab
    loses focus. The fallback model gets by on a single 10-minute lag and
    costs only ~0.7 points of skill, so degrading to it beats refusing to
    predict at all.

    Returns (bundle, feature_row). Raises SystemExit if neither can be fed.
    """
    candidates = [cfg.MODEL_DIR / "forecast_model.joblib",
                  cfg.MODEL_DIR / "forecast_model_fast.joblib"]
    tried = []

    for path in candidates:
        if not path.exists():
            continue
        bundle = joblib.load(path)
        # Each bundle records the windows it was trained with, so the live
        # matrix is rebuilt exactly as the model expects. Falling back to the
        # config defaults here would silently feed the fast model features
        # built to the main model's geometry.
        X = build_features(df,
                           lags_min=bundle.get("lags_min"),
                           tendency_min=bundle.get("tendency_min"),
                           rolling_min=bundle.get("rolling_min"))
        need = bundle["feature_names"]
        if any(c not in X.columns for c in need):
            tried.append(f"{path.name}: feature mismatch, retrain it")
            continue
        rows = X[need].dropna()
        if rows.empty:
            tried.append(f"{path.name}: needs more history")
            continue
        return bundle, rows.iloc[[-1]]

    span = df.index.max() - df.index.min()
    detail = "\n  ".join(tried) if tried else "no trained model found"
    raise SystemExit(
        f"No usable forecast model.\n  {detail}\n\n"
        f"The feed spans {span} across {len(df)} rows on a {cfg.RESAMPLE} "
        f"grid. Let the Wokwi simulation run longer - and keep its tab in the "
        f"foreground, because the browser pauses it otherwise, which turns "
        f"wall-clock time into gaps.\n"
        f"If no model exists yet: python3 train_forecast.py && "
        f"python3 train_fallback.py")


def analyse(df: pd.DataFrame) -> dict:
    """Run forecast + both detection layers on the most recent sample."""
    an = joblib.load(cfg.MODEL_DIR / "anomaly_model.joblib")

    # Keep the unresampled feed for Layer 2. The forecast and Layer 1 need
    # the 10-minute grid their models were trained on, but Layer 2 is a plain
    # comparison of two sensors and is strictly better off without it:
    # averaging halves a step change and delays detection by up to an hour.
    raw = df.copy()

    df = clean(df, resample=cfg.RESAMPLE)
    df = df.ffill(limit=3).dropna(subset=["temperature"])

    # A quantity that survives cleaning as all-NaN was present in the feed but
    # implausible throughout - in the simulator that means a slider parked
    # outside the physical range (VALID_RANGES in data_loader.py). Without
    # this check the run fails later as "not enough history", which sends you
    # looking for the problem in entirely the wrong place.
    for col in cfg.BASE_COLUMNS:
        if col in df.columns and df[col].isna().all():
            lo, hi = VALID_RANGES[col]
            raise SystemExit(
                f"Every '{col}' reading was rejected as implausible "
                f"(valid range {lo} to {hi}). Check the {col} slider in Wokwi "
                f"- it is set outside that range.")

    # --- Forecast ----------------------------------------------------------
    fc, last = select_forecast_model(df)
    forecast = float(fc["model"].predict(last)[0])
    now_t = float(df["temperature"].iloc[-1])

    # A reading inside the sensor's operating range can still lie outside
    # anything the model was trained on. The BMP180 reads down to 300 hPa;
    # the training data spans 913-1015 hPa. Below that the model extrapolates
    # and returns a confident number with no basis - it has no notion of
    # having left familiar territory. Saying so is the difference between a
    # forecast that looks broken and one that explains itself.
    extrapolating = [
        f"{col} {float(df[col].iloc[-1]):.1f} outside training range "
        f"{lo:.0f}-{hi:.0f}"
        for col, (lo, hi) in cfg.TRAINING_RANGES.items()
        if col in df.columns and not lo <= float(df[col].iloc[-1]) <= hi
    ]

    # --- Layer 1 -----------------------------------------------------------
    # Built separately from the forecast matrix: the anomaly features are
    # pinned to the config windows the detector was fitted on, which are not
    # necessarily the ones the selected forecast model uses.
    X = build_features(df)
    if any(c not in X.columns for c in ANOMALY_FEATURES):
        Xa = X.iloc[0:0]
    else:
        Xa = X[ANOMALY_FEATURES].dropna()
    if len(Xa):
        a_last = Xa.iloc[[-1]]
        l1_flag = bool(an["model"].predict(a_last)[0] == -1)
        l1_score = float(-an["model"].decision_function(a_last)[0])
    else:
        l1_flag, l1_score = False, float("nan")

    # --- Layer 2 -----------------------------------------------------------
    l2_source = raw if cfg.CROSSCHECK_ON_RAW_FEED else df
    flags = run_checks(l2_source)
    l2 = flags.iloc[-1]
    l2_flag = bool(l2["sensor_fault"])
    l2_detail = [k for k, v in l2.items() if v and k != "sensor_fault"]
    l2_spacing = flags.attrs.get("spacing_min", cfg.STEP_MINUTES)
    l2_reaction = round(l2_spacing * flags.attrs.get("persistence_samples", 0), 1)

    # --- Interpretation ----------------------------------------------------
    if l2_flag:
        verdict = "HARDWARE FAULT - readings untrustworthy"
    elif l1_flag:
        verdict = "ENVIRONMENTAL ANOMALY - unusual weather, sensors agree"
    else:
        verdict = "NORMAL"

    return {
        "timestamp": df.index[-1].isoformat(),
        "temperature_now": round(now_t, 2),
        "humidity_now": round(float(df["humidity"].iloc[-1]), 1)
        if "humidity" in df else None,
        "pressure_now": round(float(df["pressure"].iloc[-1]), 2)
        if "pressure" in df else None,
        "forecast_temperature": round(forecast, 2),
        "forecast_horizon_min": fc["horizon_min"],
        "forecast_change": round(forecast - now_t, 2),
        "model": fc["model_name"],
        "anomaly_flag": int(l1_flag),
        "anomaly_score": round(l1_score, 4),
        "sensor_fault_flag": int(l2_flag),
        "sensor_fault_detail": l2_detail,
        "sensor_check_reaction_min": l2_reaction,
        "extrapolating": extrapolating,
        "verdict": verdict,
    }


def report(res: dict) -> None:
    arrow = "up" if res["forecast_change"] > 0.2 else \
        ("down" if res["forecast_change"] < -0.2 else "steady")
    print("\n" + "=" * 58)
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}   last reading {res['timestamp']}")
    print("=" * 58)
    print(f"  Now        {res['temperature_now']:>7.2f} degC   "
          f"{res['humidity_now']} %   {res['pressure_now']} hPa")
    print(f"  Forecast   {res['forecast_temperature']:>7.2f} degC  "
          f"in {res['forecast_horizon_min']} min "
          f"({res['forecast_change']:+.2f}, {arrow})")
    print(f"  Model      {res['model']}")
    print("-" * 58)
    print(f"  Layer 1 anomaly     {'YES' if res['anomaly_flag'] else 'no':<5} "
          f"(score {res['anomaly_score']})")
    print(f"  Layer 2 sensor fault {'YES' if res['sensor_fault_flag'] else 'no':<5} "
          f"{res['sensor_fault_detail'] if res['sensor_fault_detail'] else ''}"
          f"  (reacts after {res['sensor_check_reaction_min']} min)")
    if res.get("extrapolating"):
        print("-" * 58)
        for note in res["extrapolating"]:
            print(f"  [warn] {note}")
        print("  Forecast is an extrapolation and is not supported by the "
              "training data.")
    print(f"\n  >> {res['verdict']}")
    print("=" * 58)


def push_thingspeak(res: dict, dry_run: bool = False) -> None:
    """
    Write results to the prediction channel, so the mobile app can read them.

    Driven by THINGSPEAK["write_channel_fields"] rather than a fixed field
    list: the mapping is what the app is built against, and having it declared
    in one place means the config and the actual upload cannot drift apart.
    """
    ts = cfg.THINGSPEAK
    key = ts.get("write_api_key", "")
    if dry_run:
        print("  [dry-run] nothing written to ThingSpeak")
        return
    if not key:
        print("  [info] no write key configured - result not pushed")
        return

    import requests

    payload = {"api_key": key}
    skipped = []
    for field, result_key in ts["write_channel_fields"].items():
        value = res.get(result_key)
        # NaN has to be filtered out explicitly: anomaly_score is NaN whenever
        # Layer 1 had too little history, and ThingSpeak stores the literal
        # string "nan", which then breaks the app's number parsing.
        if value is None or (isinstance(value, float) and np.isnan(value)):
            skipped.append(result_key)
            continue
        payload[field] = value
    if skipped:
        print(f"  [warn] no value, field left empty: {skipped}")

    try:
        r = requests.post("https://api.thingspeak.com/update", data=payload,
                          timeout=15)
        entry = r.text.strip()
        # ThingSpeak answers with the new entry id, or a plain "0" when the
        # write was rejected - almost always the free tier's 15 s per-channel
        # rate limit. It returns HTTP 200 either way, so the body is the only
        # way to tell success from failure.
        if entry == "0":
            print("  [warn] ThingSpeak rejected the write (entry 0). Usually "
                  "the 15 s rate limit - increase --watch.")
        else:
            print(f"  Pushed to channel {ts.get('write_channel_id', '?')} "
                  f"(entry {entry})")
    except Exception as e:
        print(f"  [warn] push failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="read from CSV instead of ThingSpeak")
    ap.add_argument("--channel", type=str, default=None)
    ap.add_argument("--key", type=str, default=None)
    ap.add_argument("--watch", type=int, default=0,
                    help="repeat every N seconds")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    ap.add_argument("--no-push", action="store_true",
                    help="analyse only, write nothing to ThingSpeak")
    args = ap.parse_args()

    def once():
        # With --json, stdout has to carry the JSON document and nothing
        # else. The loading and cleaning steps print progress as they go, so
        # that chatter is redirected to stderr for the duration - otherwise
        # it lands in front of the document and every parser chokes on it.
        # stderr stays visible, so problems are still reported.
        stream = sys.stderr if args.json else sys.stdout
        with contextlib.redirect_stdout(stream):
            df = load_csv(args.csv) if args.csv else \
                fetch_thingspeak(args.channel, args.key)
            res = analyse(df)
            if not args.json:
                report(res)
            push_thingspeak(res, dry_run=args.no_push)

        if args.json:
            print(json.dumps(res, indent=2))
        return res

    if args.watch:
        print(f"Watching, interval {args.watch}s. Ctrl+C to stop.")
        while True:
            try:
                once()
            except Exception as e:
                print(f"[error] {e}")
            time.sleep(args.watch)
    else:
        once()


if __name__ == "__main__":
    main()
