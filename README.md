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
python3 predict_live.py --watch 300  # every 5 minutes
python3 predict_live.py --json       # JSON output
```

Feature calculation requires approximately 24 hours of uninterrupted measurements. For a shorter test period, reduce `ROLLING_MIN` in `config.py`.

## Project structure

```text
config.py              Central configuration
run_all.sh              Full training pipeline
data/                   Training data
models/                 Trained models
results/                Metrics and plots
train_forecast.py       Temperature-forecast training
train_anomaly.py        Anomaly-detection training
sensor_check.py         Temperature-sensor comparison
predict_live.py         Fetch and evaluate ThingSpeak data
```
