# Forecast results (60 min horizon, dataset: jena)

| Model            |    MAE |   RMSE |     R2 |   Skill_vs_Persistence_% |   train_seconds |
|:-----------------|-------:|-------:|-------:|-------------------------:|----------------:|
| Persistence      | 0.7204 | 1.0141 | 0.9843 |                     0    |             0   |
| Ridge            | 0.5087 | 0.7284 | 0.9919 |                    29.38 |             0.2 |
| RandomForest     | 0.4674 | 0.6845 | 0.9929 |                    35.12 |            51.1 |
| GradientBoosting | 0.4633 | 0.6735 | 0.9931 |                    35.69 |             8.4 |
