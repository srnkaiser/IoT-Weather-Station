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
  **Settled, September 2026:** it does not beat it, the gap is 0.026 °C once
  the inputs are matched, and the explanation is measured rather than
  asserted — see the results section above. `train_lstm.py` said the same
  thing in the original project, before any of it was run.
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

## Measured results of the BiLSTM work (September 2026)

Everything below is reproduced by `compare_models.py`; numbers live in
`results/model_comparison.md`. Do not re-derive them.

### How to run the comparison

The baseline was fitted under sklearn 1.4 and the sequence models need
TensorFlow, whose environment carries sklearn 1.9, so the two cannot be
loaded in one process. Predictions are exported separately, then analysed:

```bash
python3 compare_models.py --export-baseline
.venv-tf/bin/python compare_models.py --export-bilstm 60min_calendar
python3 compare_models.py --analyse
python3 compare_models.py --plots
```

### The headline

On the **same 83,959 forecasts**, aligned on target timestamp:

| Model | MAE (°C) | Skill | vs. baseline |
|---|---|---|---|
| **Gradient Boosting** | **0.4632** | +35.71 % | — |
| Attention-BiLSTM, 60 min, with calendar features | 0.4896 | +32.05 % | +0.0264 |
| Attention-BiLSTM, 60 min, tuned | 0.5378 | +25.36 % | +0.0746 |
| Attention-BiLSTM, 12 h | 0.5419 | +24.79 % | +0.0787 |
| Persistence | 0.7205 | — | — |

(The BiLSTM row is one seed; see point 3 below — the seed-to-seed spread of
that configuration is large enough that a mean over five runs is the number
to quote. Updated once those runs finish.)

The remaining gap of 0.026 °C is statistically significant
(Diebold-Mariano +9.21, p ≈ 3e-20, 95 % CI [+0.021, +0.033]) and the verdict
does not depend on the HAC window. It is also small: 5.7 % relative.

### Four things that were learned the hard way

1. **The comparison was not like-for-like.** The baseline gets four calendar
   features (`hour_sin/cos`, `doy_sin/cos`); the BiLSTM was given three raw
   channels. Adding them via `--calendar` is worth **0.048 °C** — more than
   every hyperparameter change combined. Always pass `--calendar`. A BiLSTM
   result without it understates the architecture by roughly 7 percentage
   points of skill.

2. **Hyperparameter tuning is not where the gap is.** Lowering the learning
   rate to 3e-4, widening to 96 units, patience 12 and ReduceLROnPlateau
   together moved MAE from 0.5385 to 0.5381. Four ten-thousandths. Do not
   spend time here.

3. **Run-to-run spread depends on the configuration, and two seeds are not
   enough to know it.** Without calendar features, seeds 42 and 7 give 0.5381
   and 0.5372 — a spread of 0.001 °C. *With* them the same pair gives 0.4897
   and 0.4812, a spread of **0.0085 °C**, nine times larger and about a third
   of the gap being reported. The stronger configuration is the less stable
   one. Quote a mean over five seeds for the calendar configuration, never a
   single run. Individual runs are exactly reproducible via `--seed`.

4. **The attention layer says the long history is unused.** Given 72
   timesteps (12 h), the model puts **73 % of its attention on the last
   60 minutes** and 59 % on the last 10. It discards eleven of the twelve
   hours it was handed. This is the mechanistic explanation for why the 12-hour
   variant does not beat the 60-minute one, and it is a result in its own
   right — see `results/attention_weights.png`.

### The ranking holds everywhere

Broken down by season (4), time of day (4) and pre-forecast volatility (3),
Gradient Boosting wins all eleven regimes. There is no regime where the
sequence model takes the lead, so the average is not hiding a reversal. The
gap is narrowest in volatile conditions (+4.1 % relative) and widest in calm
ones (+9.3 %).

### A trap in the regime analysis

Volatility must be measured over the hour **before** the forecast is issued.
Using `|y - persistence|` looks equivalent and is not: that is the future
change, i.e. the persistence error itself, so conditioning on it hands
persistence the calm group by construction. It is also unavailable in
operation. This was written wrongly once and corrected.

### Saved models must not use a Lambda layer

Every checkpoint produced before c32f862 was unloadable: `keras.layers.Lambda`
serialises as a pickled function with no declared output shape and
`load_model` fails on it. `train_bilstm.py` now uses a registered
`AttentionPool` layer. Old checkpoints still load through `load_weights` into
a rebuilt architecture, which is what `compare_models.py` does.

## Repository layout

```
config.py               all parameters, single source of truth
data_loader.py          loading, cleaning, unified schema
features.py             feature engineering
train_forecast.py       the baseline model comparison
train_lstm.py           starting point for the BiLSTM work
train_bilstm.py         Attention-BiLSTM, the conference-paper experiment
compare_models.py       aligned comparison, significance, regimes, attention
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
