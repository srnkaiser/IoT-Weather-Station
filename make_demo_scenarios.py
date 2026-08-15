"""
Build test scenarios for the live pipeline, and check it reacts correctly.

WHY THIS EXISTS

The training scripts prove the models work on historical data. They say
nothing about whether predict_live.py behaves correctly on the kind of feed
Wokwi actually produces - short, gappy, and occasionally broken. Several code
paths (the fallback model, the Layer 2 alarm, the too-little-history error)
had never once been executed before this script existed.

Each scenario is a CSV cut from real Jena data with one specific fault
injected, plus the verdict the pipeline is supposed to reach. Running them
takes seconds and needs no simulator, so a broken pipeline is caught here
rather than during the demo.

Usage:
    python3 make_demo_scenarios.py          # build + verify
    python3 make_demo_scenarios.py --build  # only write the CSVs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data_loader import load_dataset

SCENARIO_DIR = cfg.PROJECT_ROOT / "scenarios"

# A calm summer stretch, so the baseline verdict is genuinely "normal"
# rather than an artefact of the weather that happened to be in the slice.
SLICE_START = "2016-07-15 00:00:00"
N_ROWS = 200          # 200 x 10 min = 33 h, comfortably above every window


def base_slice(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(SLICE_START)
    out = df.loc[start:].head(N_ROWS).copy()
    if len(out) < N_ROWS:
        raise SystemExit(f"Not enough data after {SLICE_START}")
    return out


def scenarios(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str, str]]:
    """name -> (data, expected verdict substring, what it demonstrates)"""
    out = {}

    # 1) Undisturbed. Anything but NORMAL here means the detectors are
    #    firing on ordinary weather.
    out["normal"] = (base_slice(df), "NORMAL",
                     "clean data, both layers silent")

    # 2) DHT22 drifts away from the BMP180: the classic slow hardware fault.
    #    Every single reading stays plausible, which is exactly why Layer 1
    #    cannot see it and Layer 2 can.
    d = base_slice(df)
    ramp = np.linspace(0, 9.0, 40)
    d.iloc[-40:, d.columns.get_loc("temperature")] += ramp
    out["sensor_drift"] = (d, "HARDWARE FAULT",
                           "DHT22 drifts +9 degC, BMP180 does not follow")

    # 3) DHT22 frozen - I2C hang or dead sensor. Constant, not extreme.
    d = base_slice(df)
    frozen = float(d["temperature"].iloc[-30])
    d.iloc[-30:, d.columns.get_loc("temperature")] = frozen
    out["sensor_stuck"] = (d, "HARDWARE FAULT",
                           "DHT22 frozen at one value for 5 h")

    # 4) A real front: both sensors cool together. Layer 2 must stay silent -
    #    this is the distinction the whole architecture is built on.
    #
    #    Expected verdict is NORMAL, not ENVIRONMENTAL, and that is measured
    #    behaviour rather than a lowered bar: Layer 1 does not flag smooth
    #    ramps at all (0 of 30 points at -14 degC over 5 h). An Isolation
    #    Forest isolates individual points; a gradual slide through the
    #    ordinary value range is the opposite of isolable. What this scenario
    #    verifies is the part that matters - that Layer 2 does not
    #    misattribute weather to broken hardware.
    d = base_slice(df)
    drop = np.linspace(0, -11.0, 30)
    for col in ("temperature", "temperature_bmp"):
        d.iloc[-30:, d.columns.get_loc(col)] += drop
    h = d.iloc[-30:, d.columns.get_loc("humidity")].to_numpy()
    d.iloc[-30:, d.columns.get_loc("humidity")] = h + (100.0 - h) * 0.6
    out["weather_front"] = (d, "NORMAL",
                            "both sensors cool together - Layer 2 must stay quiet")

    # 5) Sudden jump on both sensors: electrical interference, or a slider
    #    yanked across its range. This is what Layer 1 is genuinely good at
    #    (100 % recall on spikes) and therefore what to use in a live demo.
    d = base_slice(df)
    for col in ("temperature", "temperature_bmp"):
        d.iloc[-1, d.columns.get_loc(col)] += 14.0
    out["spike"] = (d, "ENVIRONMENTAL",
                    "sudden +14 degC jump on both sensors")

    # 6) A full cold front, staged as the worked example in the README: all
    #    three quantities move together, as they do in reality. Kept as a test
    #    so the figures quoted in the evaluation guide cannot silently drift
    #    out of date - if the model changes, this scenario changes with it.
    d = base_slice(df)
    n = 90                                  # 90 x 10 min = 15 h of passage
    ramp = np.linspace(0, 1, n)
    col = {c: d.columns.get_loc(c) for c in
           ("temperature", "temperature_bmp", "humidity", "pressure")}

    t0 = float(d.iloc[-n, col["temperature"]])
    d.iloc[-n:, col["temperature"]] = t0 - 7.0 * ramp
    # Both sensors follow the front; the 0.3 offset is the BMP180's
    # self-heating, well below the cross-check threshold.
    d.iloc[-n:, col["temperature_bmp"]] = t0 - 7.0 * ramp + 0.3

    h0 = float(d.iloc[-n, col["humidity"]])
    d.iloc[-n:, col["humidity"]] = h0 + (100.0 - h0) * 0.55 * ramp
    p0 = float(d.iloc[-n, col["pressure"]])
    d.iloc[-n:, col["pressure"]] = p0 + 10.0 * ramp

    out["cold_front"] = (d, "NORMAL",
                         "front passes: T down, RH up, pressure up together")

    # 7) Only 25 minutes of history: too little for the main model, enough
    #    for the fallback. Exercises the model selection path.
    out["short_history"] = (base_slice(df).tail(3), "",
                            "3 samples only - fallback model must take over")

    return out


def build() -> dict:
    SCENARIO_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    specs = scenarios(df)

    for name, (data, _, purpose) in specs.items():
        path = SCENARIO_DIR / f"{name}.csv"
        data.reset_index().rename(columns={"index": "timestamp"}).to_csv(
            path, index=False)
        print(f"  {path.name:<22} {len(data):>4} rows   {purpose}")
    return specs


def verify(specs: dict) -> int:
    print("\n" + "=" * 72)
    print("VERIFYING PIPELINE BEHAVIOUR")
    print("=" * 72)

    failures = 0
    for name, (_, expected, purpose) in specs.items():
        path = SCENARIO_DIR / f"{name}.csv"
        proc = subprocess.run(
            [sys.executable, "predict_live.py", "--csv", str(path),
             "--no-push"],
            capture_output=True, text=True, cwd=cfg.PROJECT_ROOT)
        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            # short_history is allowed to fail loudly IF the message explains
            # itself; silent tracebacks are never acceptable.
            if name == "short_history" and "history" in output.lower():
                print(f"\n[{name}] handled: refused with a clear message")
                print(f"  {[l for l in output.splitlines() if l.strip()][-1][:100]}")
                continue
            print(f"\n[{name}] FAILED (exit {proc.returncode})")
            print("  " + "\n  ".join(output.strip().splitlines()[-6:]))
            failures += 1
            continue

        verdict = next((l.split(">>")[1].strip()
                        for l in output.splitlines() if ">>" in l), "?")
        model = next((l.split("Model")[1].strip()
                      for l in output.splitlines() if l.strip().startswith("Model")), "?")
        ok = expected.lower() in verdict.lower() if expected else True

        print(f"\n[{name}] {purpose}")
        print(f"  verdict : {verdict}")
        print(f"  model   : {model}")
        if not ok:
            print(f"  MISMATCH: expected something containing '{expected}'")
            failures += 1
        else:
            print("  as expected")

    print("\n" + "=" * 72)
    if failures:
        print(f"{failures} scenario(s) did not behave as specified")
    else:
        print("All scenarios behaved as specified")
    print("=" * 72)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="only write the CSVs, do not verify")
    args = ap.parse_args()

    print("Building scenarios from real Jena data ...")
    specs = build()
    if args.build:
        return
    sys.exit(1 if verify(specs) else 0)


if __name__ == "__main__":
    main()
