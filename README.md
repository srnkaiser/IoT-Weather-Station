# IoT Weather Station

This project trains local models for temperature forecasting and sensor-fault detection. The weather station uses an ESP32 with DHT22 and BMP180 sensors and can evaluate measurements from ThingSpeak.

## Prerequisites

- Python 3.10 or newer
- Git
- Optional for live operation: a ThingSpeak channel containing the weather-station measurements

## Installation

```bash
git clone https://github.com/srnkaiser/IoT-Weather-Station.git
cd IoT-Weather-Station
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Training data

The default configuration uses the Jena dataset. If `data/jena_climate_2009_2016.csv` is not available, download it as follows:

```bash
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

Alternatively, select a different data format through `DATASET` in `config.py`:

- `synthetic`: generated test data
- `jena`: Jena climate data
- `dwd`: DWD data file at `data/dwd_stuttgart.csv`
- `generic`: custom CSV file at `data/weather.csv`

Configure the column mapping for custom files in `DATASET_SPECS` in `config.py`.

## Train the models

Run the full pipeline:

```bash
./run_all.sh
```

The script generates test data when needed, trains the forecasting and anomaly-detection models, and saves models in `models/` and evaluation results in `results/`.

On Windows, run the same pipeline with:

```powershell
& .\.venv\Scripts\python.exe make_testdata.py
& .\.venv\Scripts\python.exe train_forecast.py
& .\.venv\Scripts\python.exe train_anomaly.py
& .\.venv\Scripts\python.exe sensor_check.py
& .\.venv\Scripts\python.exe evaluate_combined.py
```

## Configure ThingSpeak for live operation

Add the channel and, when necessary, the Read API Key in `config.py`:

```python
THINGSPEAK = dict(
    channel_id="3451792",
    read_api_key="",  # leave empty for public channels
)
```

The field mapping must match the ThingSpeak channel:

| ThingSpeak field | Value |
|---|---|
| `field1` | DHT22 temperature |
| `field2` | DHT22 humidity |
| `field3` | BMP180 temperature |
| `field5` | BMP180 air pressure |

You can then start the forecast:

```bash
python3 predict_live.py              # one-time run
python3 predict_live.py --watch 120  # every 2 minutes
python3 predict_live.py --json       # JSON output
python3 predict_live.py --no-push    # analyse without writing to ThingSpeak
```

Feature calculation requires 60 minutes of uninterrupted measurements. If less
history is available, `predict_live.py` automatically falls back to a
short-memory model that needs about 20 minutes and costs 0.7 percentage points
of forecast skill.

Keep the Wokwi browser tab in the foreground while collecting data. Browsers
suspend timers in background tabs, which pauses the simulation and leaves gaps
in the history.

## Verifying the system

```bash
python3 make_demo_scenarios.py   # seven end-to-end tests, no simulator needed
python3 explain_model.py         # how the forecast responds to each input
```

`make_demo_scenarios.py` builds test cases from real weather data with specific
faults injected and checks that the pipeline reaches the expected verdict for
each. It runs in seconds and requires neither Wokwi nor ThingSpeak.

## Demonstrating it live

Three things to know before moving a slider in Wokwi, otherwise the system
looks unresponsive when it is working correctly:

- **The prediction channel only updates while `predict_live.py` is running.**
  The measurement channel updates by itself; the prediction channel does not.
  Run `python3 predict_live.py --watch 60 --no-push` to see results locally.
- **A slider change takes up to 10 minutes to take full effect**, because
  readings are averaged onto a 10-minute grid. Layer 2 is the exception and
  reacts in 3 minutes.
- **Move sliders slowly and keep the Wokwi tab in the foreground.** Rapid
  changes produce rates that do not occur in real weather, so the model
  correctly reports an anomaly instead of a useful forecast. Background tabs
  are suspended by the browser, which pauses the simulation.

| Step | Action | Expected |
|---|---|---|
| 0 | 20 °C / 60 % / 1013 hPa, both temperature sliders equal, wait 10 min | `NORMAL` |
| 1 | raise pressure slowly to 1019 hPa | forecast drops ~1.4 °C |
| 2 | jump temperature by 14 °C at once, both sensors | `ENVIRONMENTAL ANOMALY` |
| 3 | move only the DHT22 slider, wait 3 min | `HARDWARE FAULT` |

Step 3 is the core of the architecture: weather affects both sensors, a defect
affects one. The forecast keeps being computed during a fault — the flag, not
the absence of a number, carries the information.

## Project structure

```text
config.py               Central configuration
run_all.sh              Full training pipeline
data/                   Training data (downloaded, not in the repository)
models/                 Trained models
results/                Metrics and plots
data_loader.py          Loading, cleaning, unified schema
features.py             Feature engineering
train_forecast.py       Temperature-forecast training
train_fallback.py       Short-memory model for limited history
train_anomaly.py        Anomaly-detection training (Layer 1)
sensor_check.py         Temperature-sensor comparison (Layer 2)
evaluate_combined.py    Combined evaluation of both layers
tune_contamination.py   Sensitivity analysis of the anomaly threshold
explain_model.py        Model response analysis
make_demo_scenarios.py  End-to-end verification
predict_live.py         Fetch and evaluate ThingSpeak data
Weather_Station_App.aia MIT App Inventor project
```
