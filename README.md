# AI Weather Station — AI Module

**Team 7** · Bruno, Luka, Sören, Michael
Project 7: Weather Station · ESP32 + DHT22 + BMP180

External, locally trained AI model for weather forecasting and anomaly detection.
Meets Dr. Saha's requirement: own trained model, Python, local, own dataset.

---

## 1. Hardware reference

| ThingSpeak | Quantity | Sensor | Role |
|---|---|---|---|
| field1 | Temperature °C | DHT22 | Target variable + feature |
| field2 | Humidity % | DHT22 | Feature |
| field3 | Air pressure hPa | BMP180 | Feature (as a trend) |
| field4 | Temperature °C | BMP180 | Cross-comparison for sensor errors |

> **Naming note:** The sensor is called **BMP180**, not BME180.
> BMP = *Pressure*, BME = *Environment* (with humidity). Write this correctly in the paper.

**Central design rule:** The model may use only quantities that the DHT22 and
BMP180 can later provide. A model trained on wind speed is useless if the ESP32
cannot measure it.

---

## 2. Why training is not performed on ThingSpeak data

In Wokwi, sensor values are set using sliders or a scenario file. Therefore, what
is in your ThingSpeak channel consists of **set values, not real weather patterns**.
They are unusable as training data.

Therefore, the clean separation:

```
Real historical weather dataset  ──►  Training (offline, one-time)
                                              │
                                              ▼
Wokwi / ESP32 ──► ThingSpeak ────────────►  completed model ──► Forecast + alarm
```

ThingSpeak is **read only**, never used for training.

---

## 3. Installation

```bash
git clone https://github.com/srnkaiser/IoT-Weather-Station.git
cd IoT-Weather-Station
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

### Windows (PowerShell)

`run_all.sh` is a Bash script. Either use Git Bash/WSL or run the pipeline in
PowerShell with the following command:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe make_testdata.py; & .\.venv\Scripts\python.exe train_forecast.py; & .\.venv\Scripts\python.exe train_anomaly.py; & .\.venv\Scripts\python.exe sensor_check.py; & .\.venv\Scripts\python.exe evaluate_combined.py
```

Immediate test with synthetic data (the complete pipeline in one command):

```bash
./run_all.sh
```

The Python files are located directly in the project folder (there is no
`src/` subdirectory). The script automatically uses `.venv/bin/python` if the
virtual environment exists.

---

## 4. Obtain the real dataset

According to the lecturer, you should find a suitable dataset yourselves. Two
candidates that exactly match your sensor setup:

### Recommendation: Jena Climate (Max Planck Institute for Biogeochemistry)

Time series from 2009–2016, 10-minute resolution, containing `T (degC)`, `rh (%)`, `p (mbar)` —
an exact match for DHT22 + BMP180. More than 400,000 measurements.

```bash
# macOS / Linux: download and unpack directly at the expected location
mkdir -p data
curl -L https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip \
  -o /tmp/jena_climate.zip
unzip -p /tmp/jena_climate.zip jena_climate_2009_2016.csv \
  > data/jena_climate_2009_2016.csv
rm /tmp/jena_climate.zip
```

On Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip -OutFile jena_climate.zip
Expand-Archive jena_climate.zip -DestinationPath data -Force
Remove-Item jena_climate.zip
```

Then in `config.py`:

```python
DATASET = "jena"
```

Then run the pipeline again:

```bash
./run_all.sh
```

The models and metrics are now generated with real data. The file must be named
exactly `data/jena_climate_2009_2016.csv`.

### Alternative: DWD Open Data

Hourly station data from the German Weather Service, e.g. Stuttgart-Echterdingen.
Gives your paper a local connection. Temperature and humidity are in product `TU`,
and air pressure in `P0` — you must merge both files using `MESS_DATUM`.
The format is semicolon-separated, and missing values are `-999`.

The official downloads are in the DWD Open Data directory for
[temperature/humidity (TU)](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/)
and [air pressure (P0)](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/pressure/historical/).
This alternative requires the described merge into `data/dwd_stuttgart.csv`; for a
directly executable approach, please use Jena.

Then set `DATASET = "dwd"` and, if necessary, `RESAMPLE = "1h"`, `STEP_MINUTES = 60`.

> After switching to real data, **run all scripts again**. The figures from the
> synthetic dataset must not be included in the paper.

---

## 5. Project structure

```
IoT-Weather-Station/
├── config.py                  All parameters centralized
├── run_all.sh                 Complete pipeline
├── requirements.txt
├── data/                      Datasets
├── models/                    Trained models (.joblib)
├── results/                   Metrics, tables, plots → for the paper
├── data_loader.py             Loading, cleaning, uniform schema
├── make_testdata.py           Synthetic test data
├── features.py                Feature engineering
├── train_forecast.py          Forecasting model + model comparison
├── train_anomaly.py           Isolation Forest + error injection
├── sensor_check.py            Layer 2: Sensor cross-comparison
├── evaluate_combined.py       Overall evaluation of both layers
├── predict_live.py            ThingSpeak → forecast + alarm
└── train_lstm.py              Optional: LSTM for the comparison table
```

---

## 6. The three design decisions you must justify in the paper

### 6.1 Persistence baseline

Weather is strongly autocorrelated. A 1-hour temperature forecast always looks
good in absolute figures — an R² above 0.97 comes for free. Only the difference
from the naïve forecast “the temperature stays the same” is meaningful. Therefore,
the persistence baseline is included throughout the evaluation, and the **Skill
Score** (the share of baseline error removed by the model) is the actual metric.

### 6.2 Chronological split, never random

With a random split, the model sees the future during training and is evaluated
on the past. This makes every metric look better. We split strictly by time:
the first 80% for training, the last 20% for testing.

### 6.3 Altitude-invariant air pressure

Absolute air pressure depends on the station altitude. Jena is located at about
155 m, whereas your location is not. A model trained on absolute hPa values
carries this offset as a learned bias. Therefore: remove absolute pressure values
and instead use the deviation from its own 24-hour mean plus the **pressure trend**
over 3 and 6 hours. This is altitude-invariant — and falling pressure is, in any
case, the classic synoptic predictor of an approaching front.

Can be disabled via `ALTITUDE_INVARIANT_PRESSURE = False` in `config.py`.

---

## 7. The two-layer anomaly architecture

The most important conceptual part of the project — and what distinguishes you
from other groups.

```
                    Measurement
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ Layer 1           │      │ Layer 2              │
│ Isolation Forest  │      │ Sensor-Kreuzvergleich│
│ statistical       │      │ deterministic        │
└─────────┬─────────┘      └──────────┬───────────┘
          │                           │
          └───────────┬───────────────┘
                      ▼
       Layer 2 silent + Layer 1 triggers → real weather event
       Layer 2 triggers                  → hardware fault, data unusable
```

**Why two layers?** The Isolation Forest is structurally blind to two types of
errors:

- **stuck** (frozen value): a constant value is not statistically extreme; it is
  only constant.
- **drift** (slowly increasing offset): every individual value remains within a
  plausible range.

Layer 2 catches both through hardware redundancy: DHT22 and BMP180 measure the
same temperature at the same location. **Weather affects both sensors. A fault affects one.**

The threshold is formed using *median + k · MAD*, not mean + k · σ. MAD is robust
— a drifting sensor inflates the standard deviation and would raise a σ-based
threshold above the very error it is intended to detect.

**Evaluation without labels:** Anomaly detection is unsupervised, so there is
nothing to measure. We therefore inject errors of known types at known positions
into the test section and measure precision/recall against them. This yields a
real quantitative table instead of “the model found a few anomalies.”

---

## 8. Live operation

Fill in `config.py`:

```python
THINGSPEAK = dict(
    channel_id="1234567",
    read_api_key="XXXXXXXX",   # leave empty if the channel is public
    ...
)
```

Before starting, check that the four `field_map` entries match your ThingSpeak
channel: `field1` DHT22 temperature, `field2` humidity, `field3` BMP180 air
pressure, and `field4` BMP180 temperature. For a public channel, leave
`read_api_key` empty; for a private channel, the **Read API Key** is required.
Both models must have been trained beforehand with `./run_all.sh`.

Then:

```bash
python3 predict_live.py              # einmalig
python3 predict_live.py --watch 300  # every 5 minutes
python3 predict_live.py --json       # maschinenlesbar
```

Optionally write back to a second ThingSpeak channel (forecast, anomaly flag,
sensor-error flag) via `write_api_key`.

**Warm-up time warning:** The features need about 24 hours of uninterrupted
history (the longest rolling window). Before that, `predict_live.py` returns an
error. For the demo: run Wokwi with an accelerated scenario or set
`ROLLING_MIN = [60, 360]`.

---

## 9. Mapping to the required paper structure

| Section in the paper | Source |
|---|---|
| Proposed AI-enabled Architecture | Sections 2 + 7 of this README |
| Hardware Design | Table in Section 1 |
| Cloud Integration (ThingSpeak) | `predict_live.py` |
| AI Model | `features.py`, `train_forecast.py`, Section 6 |
| Experimental Results | `results/forecast_metrics.md`, `results/combined_results.md` |
| | Plots: `results/*.png` |
| Conclusion / Future Scope | Section 10 |

---

## 10. Limitations — state them honestly

A reviewer looks for exactly these. Naming them yourself is stronger than having
them discovered.

1. **Humidity depends on a single sensor.** The BMP180 cannot measure humidity.
   If the DHT22 fails, this quantity is completely lost — there is no redundancy
   and thus no cross-comparison for it.
2. **One location, one period.** Trained on data from one station. Transfer to a
   different climate has not been validated.
3. **Layer 1 barely detects drift and stuck** (recall below 10%). This is not a
   weakness of the implementation, but a fundamental limitation of pointwise
   outlier detection — and precisely the justification for Layer 2.
4. **Contamination is a manually set parameter.** It directly determines the
   false-alarm rate and cannot be properly optimized without labels.
5. **Only one forecast horizon.** 60 minutes. Longer horizons are possible via
   `FORECAST_HORIZON_MIN`, but have not been evaluated.
6. **No precipitation, no wind.** Without suitable sensors, real weather
   forecasting in the meteorological sense is not possible — this is about
   short-term extrapolation of local measurements.

---

## 11. Next steps

- [ ] Download the real dataset, switch `DATASET`, retrain everything
- [ ] Enter ThingSpeak credentials in `config.py`
- [ ] Check field1–field4 assignments against your channel
- [ ] Optionally use `train_lstm.py` for the comparison table
- [ ] Include plots from `results/` in the paper
