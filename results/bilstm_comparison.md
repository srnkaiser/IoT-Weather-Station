# Sequence models vs the engineered-feature baseline

Jena Climate, chronological 80/20 split, 60-minute forecast horizon, 22
training runs. Numbers come from `model_comparison.md` (aligned comparison and
significance) and `runs_by_configuration.md` (every run, grouped by
configuration). Reproduce with `../run_bilstm.sh`.

## The result

A sequence model does beat the baseline - but only when it is given the same
engineered features. On raw sensor channels it loses badly.

| Model | Inputs | MAE (degC) | Skill | Runs |
|---|---|---|---|---|
| **Attention-BiLSTM** | **engineered features** | **0.4511** | **+37.4 %** | 3 |
| Gradient Boosting (baseline) | engineered features | 0.4633 | +35.7 % | 1 |
| BiLSTM, no attention | raw + calendar | 0.4718 | +34.5 % | 4 |
| Plain LSTM | raw + calendar | 0.4727 | +34.4 % | 4 |
| Attention-BiLSTM | raw + calendar | 0.4853 | +32.6 % | 5 |
| Attention-BiLSTM | raw channels only | 0.5376 | +25.4 % | 2 |
| Persistence | - | 0.7205 | - | - |

Means over seeds. All three runs of the winning configuration beat the
baseline, by 0.007 to 0.016 degC - that is the honest way to state it, because
the seed spread of that configuration (0.0094) is close to the gap (0.0122).
Quoting the mean alone would imply a precision the data does not carry.

Significance, using the **median** of the three seeds rather than the best:
Diebold-Mariano +5.94 against Gradient Boosting, p = 2.9e-9, 95 % CI
[+0.0090, +0.0176]. The variance is HAC-corrected over 24 h because forecast
errors here stay correlated for hours; the verdict is unchanged across HAC
windows from 0.8 h to 72 h. The ranking also holds in all eleven regimes
tested - four seasons, four parts of the day, three volatility terciles.

## Three findings about the architecture

**The attention layer costs accuracy.** Removing it improves MAE from 0.4853
to 0.4718. Bidirectionality doubles the parameters for nothing: a plain
forward LSTM reaches 0.4727 with 43,073 parameters against 92,290, and the
difference to the best ablation is 0.0009 - far inside the noise. Neither half
of the name "Attention-BiLSTM" earns its place on this task.

**The attention weights say the long history is unused.** Given 12 hours, the
model places 73 % of its attention on the last 60 minutes and 59 % on the last
10. Measured accuracy agrees: 12 h and 60 min differ by 0.0007 degC. The
interpretability claim holds - the weights are readable - and what they say is
that the model discards eleven of the twelve hours it is handed. See
`attention_weights.png`.

**The simpler architectures are also more stable.** Across seeds the
no-attention BiLSTM spreads 0.0028 degC and the plain LSTM 0.0073, against
0.0096 for the full architecture. For clients that train independently this
may matter beyond accuracy.

## The objection this had to survive

The engineered feature set already contains lags back to 60 minutes, so a
6-step window over it reaches roughly 120 minutes into the past while the
baseline sees 60. The advantage could therefore have been extra history rather
than a better model. It is not: with a 2-step window - about 70 minutes of
lookback, essentially the baseline's - the same model reaches 0.4515 against
0.4511. Diebold-Mariano between the two: +1.12, p = 0.26, confidence interval
spanning zero. The lookback is not where the advantage comes from.

## What this cost to find out

The first version of this comparison reported the BiLSTM at 0.5385 against
0.4633 and concluded the architecture did not work. Two thirds of that gap was
an error in the comparison, not a property of the model: the baseline received
four calendar features (hour and day-of-year, sine and cosine encoded) that the
sequence model did not. Matching the inputs closed most of it; supplying the
full engineered set reversed the result.

Two further corrections are recorded in the git history: run-to-run spread was
reported from two runs and turned out to be nine times larger on the stronger
configuration, and the volatility breakdown initially conditioned on
`|y - persistence|`, which is the persistence error itself and hands
persistence the calm group by construction.

## Scope

Federated learning, non-IID partitioning and the integration of the anomaly
detection are the doctoral researcher's part of the paper and are deliberately
not touched here. The one number from this work that feeds into them: the
recommended model has 43,073 parameters, which is 168 KB per client per
round in float32, against 361 KB for the full Attention-BiLSTM.
