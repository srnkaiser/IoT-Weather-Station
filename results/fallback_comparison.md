# Forecast with reduced history

The fallback model uses a single 10-minute lag, so it can run after about 20 minutes of simulation instead of 60. The gap between the two rows is what the longer history is worth.

|                             |    MAE |   RMSE |     R2 |   Skill_vs_Persistence_% |
|:----------------------------|-------:|-------:|-------:|-------------------------:|
| Main model (60 min history) | 0.4633 | 0.6735 | 0.9931 |                    35.69 |
| Fallback (10 min history)   | 0.4684 | 0.6793 | 0.993  |                    34.98 |
