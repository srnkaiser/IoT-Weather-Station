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
  **Settled, September 2026, and the expectation was wrong.** On raw
  channels the sequence model loses badly, as predicted. On the *same
  engineered features* it wins: 0.4511 against 0.4633. The signal the trees
  exploit is in the features, and a sequence model given those features
  extracts more of it, not less — see the results section above.
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

22 training runs. Full narrative in `results/bilstm_comparison.md`, numbers in
`results/model_comparison.md` and `results/runs_by_configuration.md`.
Reproduce everything with `./run_bilstm.sh`. Do not re-derive these.

### The result, in one line

A sequence model beats the baseline, but only on the engineered features.

| Model | Inputs | MAE | Skill | Runs |
|---|---|---|---|---|
| **Attention-BiLSTM** | **engineered features** | **0.4511** | +37.4 % | 3 |
| Gradient Boosting | engineered features | 0.4633 | +35.7 % | 1 |
| BiLSTM, no attention | raw + calendar | 0.4718 | +34.5 % | 4 |
| Plain LSTM | raw + calendar | 0.4727 | +34.4 % | 4 |
| Attention-BiLSTM | raw + calendar | 0.4853 | +32.6 % | 5 |
| Attention-BiLSTM | raw only | 0.5376 | +25.4 % | 2 |

State it as "all three runs beat the baseline, by 0.007 to 0.016 degC", not as
the mean alone: the seed spread of that configuration (0.0094) is close to the
gap (0.0122). Significance uses the **median** seed, never the best:
DM +5.94, p = 2.9e-9, CI [+0.0090, +0.0176], HAC-corrected, stable across HAC
windows and across all eleven regimes tested.

### How to run the comparison

The baseline was fitted under sklearn 1.4 and the sequence models need
TensorFlow, whose environment carries sklearn 1.9, so the two cannot be loaded
in one process. Predictions are exported separately, then analysed:

```bash
python3 compare_models.py --export-baseline
.venv-tf/bin/python compare_models.py --export-bilstm 60min_features
python3 compare_models.py --analyse     # aligned comparison + significance
python3 compare_models.py --summary     # every run, grouped, with seed spread
python3 compare_models.py --plots
```

### Six things learned the hard way

1. **Match the inputs before comparing.** The baseline gets four calendar
   features the raw sequence does not. Withholding them cost 0.048 degC and
   produced a first conclusion ("the architecture does not work") that was
   simply wrong. Supplying the full engineered set via `--features` reversed
   the result outright.

2. **Hyperparameter tuning is not where anything is.** Learning rate 3e-4,
   96 units, patience 12, ReduceLROnPlateau: 0.5385 to 0.5381. Do not spend
   time here.

3. **Two seeds are not enough to know the spread.** Measured twice in this
   project, wrongly both times. Without calendar features the spread is 0.001;
   with them 0.0094; the stronger configuration is the less stable one. Run
   four or five, quote the range, and let `--summary` mark gaps smaller than
   the spread as unresolved.

4. **Neither half of "Attention-BiLSTM" earns its place.** Attention costs
   0.0135 degC. Bidirectionality doubles parameters for a difference inside
   the noise. The plain LSTM matches everything with 43,073 parameters against
   92,290 - and is more stable across seeds.

5. **The model itself says long history is useless.** Given 12 hours it puts
   73 % of its attention on the last 60 minutes. Accuracy agrees: 12 h vs
   60 min differs by 0.0007. A 2-step window (70 min lookback) matches a
   6-step one (0.4515 vs 0.4511, p = 0.26).

6. **Condition on the past, never on the future.** The volatility breakdown
   first used `|y - persistence|`, which is the persistence error itself and
   hands persistence the calm group by construction. It must be the
   temperature change over the hour *before* the forecast is issued.

### Reproducibility, precisely

Runs are seeded (`--seed`) and reproduce to about 1e-4, not bit-exactly:
`enable_op_determinism()` is deliberately not set. That is two orders of
magnitude below the differences measured. **Training durations are not
comparable** - the runs were executed several at a time and the timings
reflect contention. Quote the ordering, never the seconds.

### Saved models must not use a Lambda layer

Every checkpoint produced before c32f862 was unloadable: `keras.layers.Lambda`
serialises as a pickled function with no declared output shape and
`load_model` fails on it. `train_bilstm.py` now uses a registered
`AttentionPool` layer. Old checkpoints still load through `load_weights` into
a rebuilt architecture, which is what `compare_models.py` does.

### For the federated section (theirs, not ours)

The recommended model has 43,073 parameters: 168 KB per client per round in
float32, against 361 KB for the full Attention-BiLSTM. Gradient Boosting
cannot be averaged at all, which is the actual argument for a neural model
here - independent of the 0.012 degC.

## Repository layout

```
config.py               all parameters, single source of truth
data_loader.py          loading, cleaning, unified schema
features.py             feature engineering
train_forecast.py       the baseline model comparison
train_lstm.py           starting point for the BiLSTM work
train_bilstm.py         Attention-BiLSTM, the conference-paper experiment
compare_models.py       aligned comparison, significance, regimes, attention
run_bilstm.sh           reproduces the whole conference-paper grid
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
