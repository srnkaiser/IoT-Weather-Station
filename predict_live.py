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
from data_loader import clean
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
    df.index = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(None)
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


def analyse(df: pd.DataFrame) -> dict:
    """Run forecast + both detection layers on the most recent sample."""
    fc = joblib.load(cfg.MODEL_DIR / "forecast_model.joblib")
    an = joblib.load(cfg.MODEL_DIR / "anomaly_model.joblib")

    df = clean(df, resample=cfg.RESAMPLE)
    df = df.ffill(limit=3).dropna(subset=["temperature"])

    X = build_features(df)

    # --- Forecast ----------------------------------------------------------
    need = fc["feature_names"]
    missing = [c for c in need if c not in X.columns]
    if missing:
        raise SystemExit(
            f"Feature mismatch: the model expects {missing} but the live data "
            f"cannot produce them. Retrain with the same config.")

    Xf = X[need].dropna()
    if Xf.empty:
        raise SystemExit(
            f"Not enough history. The features need about "
            f"{max(cfg.ROLLING_MIN) // 60} h of continuous data; "
            f"ThingSpeak currently provides less.")

    last = Xf.iloc[[-1]]
    forecast = float(fc["model"].predict(last)[0])
    now_t = float(df["temperature"].iloc[-1])

    # --- Layer 1 -----------------------------------------------------------
    Xa = X[ANOMALY_FEATURES].dropna()
    if len(Xa):
        a_last = Xa.iloc[[-1]]
        l1_flag = bool(an["model"].predict(a_last)[0] == -1)
        l1_score = float(-an["model"].decision_function(a_last)[0])
    else:
        l1_flag, l1_score = False, float("nan")

    # --- Layer 2 -----------------------------------------------------------
    flags = run_checks(df)
    l2 = flags.iloc[-1]
    l2_flag = bool(l2["sensor_fault"])
    l2_detail = [k for k, v in l2.items() if v and k != "sensor_fault"]

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
          f"{res['sensor_fault_detail'] if res['sensor_fault_detail'] else ''}")
    print(f"\n  >> {res['verdict']}")
    print("=" * 58)


def push_thingspeak(res: dict) -> None:
    """Optional: write results back to a second ThingSpeak channel."""
    key = cfg.THINGSPEAK.get("write_api_key", "")
    if not key:
        return
    import requests
    payload = {"api_key": key,
               "field1": res["forecast_temperature"],
               "field2": res["anomaly_flag"],
               "field3": res["sensor_fault_flag"]}
    try:
        r = requests.post("https://api.thingspeak.com/update", data=payload, timeout=15)
        print(f"  Pushed to ThingSpeak (entry {r.text})")
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
    args = ap.parse_args()

    def once():
        df = load_csv(args.csv) if args.csv else \
            fetch_thingspeak(args.channel, args.key)
        res = analyse(df)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            report(res)
        push_thingspeak(res)
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
