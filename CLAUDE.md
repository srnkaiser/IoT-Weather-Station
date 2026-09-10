# CLAUDE.md — project context

Read this before doing anything in this repository.

## What this project is now

This started as a graded university project (Project 7, Weather Station,
DHBW Heidenheim) and was submitted on 15 August 2026. It went well enough
that Prof. Dr. Himadri Nath Saha proposed extending it into a conference
paper for **BRAINCON** (https://braincon.brainallianz.org).

The paper is now a joint effort between three parties:

| Who | Role |
|---|---|
| Prof. Dr. Saha | supervision |
| A doctoral researcher | drafted the paper, owns the federated learning part |
| Team 7 (Bruno, Luka, Sören, Michael) | owns the existing system and the forecasting experiments |

**Submission deadline: end of September 2026.**

## The division of work — do not cross these lines

The doctoral researcher assigned the tasks explicitly. Stay inside our part.

**Ours:**
- Attention-BiLSTM forecasting model
- Comparison against the existing baseline on identical data

**Theirs — do not build these:**
- FedAvg / FedProx experiments
- non-IID client partitioning
- Integration of the anomaly-detection component into the federated pipeline

If something in their area looks wrong, raise it with them. Do not implement
around it.

## Rules for the experiments

These come from the doctoral researcher's brief and from the original paper.
Breaking any of them makes the results incomparable and therefore useless.

1. **Same dataset.** Jena Climate 2009-2016, as loaded by `data_loader.py`.
   420,551 raw observations, 420,204 after cleaning.
2. **Same chronological split.** First 80 % train, last 20 % test. Never
   shuffle. A random split lets the model see the future during training.
3. **Same forecast horizon.** 60 minutes.
4. **Compare against the existing baseline**, reproduced below.
5. **Report skill over persistence, not R².** Hourly temperature is strongly
   autocorrelated: the naive "no change" forecast alone reaches R² = 0.984,
   so absolute goodness-of-fit says almost nothing about what a model adds.

## The baseline to beat

Measured on the test portion (2015-2016), 84,041 samples never seen during
training. Reproduce with `./run_all.sh`.

| Model | MAE (°C) | RMSE | R² | Skill vs. persistence |
|---|---|---|---|---|
| Persistence | 0.720 | 1.014 | 0.9843 | — |
| Ridge | 0.509 | 0.728 | 0.9919 | +29.4 % |
| Random Forest | 0.467 | 0.685 | 0.9929 | +35.1 % |
| **Gradient Boosting** | **0.463** | **0.674** | **0.9931** | **+35.7 %** |

## What we already know, so nobody re-derives it

- **A sequence model may well lose to gradient boosting here.** At a
  60-minute horizon on a single station, tree ensembles are very strong on
  engineered lag features. Measured evidence: removing 50 minutes of history
  from the feature set cost only 0.7 percentage points of skill
  (`results/fallback_comparison.md`), which suggests most usable signal sits
  in the current state and the short-term tendency. If the BiLSTM does not
  beat 0.463 °C, that is a reportable result, not a bug — but it must be
  explained rather than tuned away.
- **The two-layer anomaly architecture works and is measured.** Layer 2
  reaches precision 1.000 with zero false alarms across 82,447 clean
  samples; detected events are classified correctly as weather in 100 % of
  cases and as hardware faults in 94.1 %. See `results/combined_results.md`.
- **Layer 1's weakness is structural, not a tuning failure.** Precision
  stays between 0.06 and 0.09 across a tenfold range of the contamination
  parameter (`results/contamination_sensitivity.md`).
- **Absolute pressure must stay out of the features.** Jena averages 989 hPa,
  the deployed BMP180 reports around 1013 hPa. Training on absolute pressure
  teaches the model that 989 hPa is normal.

## Repository layout

```
config.py               all parameters, single source of truth
data_loader.py          loading, cleaning, unified schema
features.py             feature engineering
train_forecast.py       the baseline model comparison
train_lstm.py           starting point for the BiLSTM work
train_fallback.py       short-memory model
train_anomaly.py        Layer 1
sensor_check.py         Layer 2
evaluate_combined.py    both layers together
tune_contamination.py   sensitivity analysis
explain_model.py        model response analysis
make_demo_scenarios.py  end-to-end verification
predict_live.py         live inference against ThingSpeak
```

Training data is not committed (41 MB); the download command is in the
README. Trained models are committed so `predict_live.py` runs immediately.

## Environment

MacBook Air M2, 8 GB RAM. Memory is the binding constraint. Sequence tensors
for this dataset are small (~0.36 GB in float32), so training fits, but do
not assume headroom for large batch sizes or wide layers.

## Verification

Before claiming anything works:

```bash
python3 make_demo_scenarios.py   # seven end-to-end scenarios
./run_all.sh                     # full retrain, reproduces every number above
```

Numbers quoted in the paper must come from `results/`, never from memory.
