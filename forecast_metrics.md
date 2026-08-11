# Forecast results (60 min horizon, dataset: synthetic)

| Model            |    MAE |   RMSE |     R2 |   Skill_vs_Persistence_% |   train_seconds |
|:-----------------|-------:|-------:|-------:|-------------------------:|----------------:|
| Persistence      | 0.8011 | 1.0095 | 0.9789 |                     0    |             0   |
| Ridge            | 0.687  | 0.8622 | 0.9846 |                    14.24 |             0.5 |
| RandomForest     | 0.6926 | 0.8679 | 0.9844 |                    13.54 |            86.7 |
| GradientBoosting | 0.6797 | 0.8512 | 0.985  |                    15.15 |             8.7 |
