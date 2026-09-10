# Attention-BiLSTM vs. the existing baseline

Jena Climate, chronological 80/20 split, 60-minute forecast horizon.
All models evaluated on the same test period (2015-2016).

| Model | MAE (°C) | RMSE | R² | Skill vs. persistence |
|---|---|---|---|---|
| **Gradient Boosting** | 0.4633 | 0.6735 | 0.9931 | +35.69 % |
| Random Forest | 0.4674 | 0.6845 | 0.9929 | +35.12 % |
| Ridge | 0.5087 | 0.7284 | 0.9919 | +29.38 % |
| Attention-BiLSTM (60 min) | 0.5385 | 0.7920 | 0.9904 | +25.28 % |
| Attention-BiLSTM (12 h) | 0.5421 | 0.7775 | 0.9908 | +24.78 % |
| Persistence (baseline) | 0.7204 | 1.0141 | 0.9843 | — |

## Reading

The Attention-BiLSTM does not beat the engineered-feature models. At 60 minutes
of history - identical information to the baseline - it reaches 0.5385 °C
against 0.4633 °C for Gradient Boosting, and it also
trails Ridge regression.

Extending the history to 12 hours does not help: 0.5421 °C, marginally
worse than the 60-minute variant (0.5385 °C). Twelve times more input
produced no improvement.

That second result is the informative one. It rules out the obvious objection
that the sequence model simply had too little sequence to work with, and it is
consistent with an earlier measurement in this project: removing 50 minutes of
history from the feature-based model cost only 0.7 percentage points of skill
(`fallback_comparison.md`). At a 60-minute horizon on a single station, almost
all usable signal sits in the current state and its short-term tendency - which
is precisely what engineered lag features capture directly and what a recurrent
model has to rediscover from raw sequences.

## Method notes

- Both variants: bidirectional LSTM (64 units) with additive attention over the
  time axis, 43,138 parameters, Adam at 1e-3, batch 256, early stopping on
  validation MAE (patience 5). 60-min variant stopped at epoch 22 (best 17),
  12-h variant at epoch 8 (best 3).
- Scaling statistics were computed on the training portion only.
- The persistence baseline was recomputed for each sequence length rather than
  reused, because sequence construction drops a different number of leading
  rows and the sample sets must match exactly.
- Hyperparameters were not tuned extensively. The gap of ~0.08 °C is large
  enough that tuning is unlikely to reverse the ordering, but this has not been
  exhaustively tested and should be stated as such.
