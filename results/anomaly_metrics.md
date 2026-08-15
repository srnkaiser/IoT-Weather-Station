# Anomaly detection results

## Overall

|                  |      Value |
|:-----------------|-----------:|
| Precision        |     0.0675 |
| Recall           |     0.1425 |
| F1               |     0.0917 |
| TP               |   179      |
| FP               |  2471      |
| FN               |  1077      |
| TN               | 79976      |
| FalseAlarmRate_% |     2.997  |

## By injected fault type

| FaultType   |   n_samples |   Recall |   MeanScore |
|:------------|------------:|---------:|------------:|
| spike       |          12 |   1      |      0.0511 |
| stuck       |         167 |   0.012  |     -0.1458 |
| drift       |         637 |   0.011  |     -0.1464 |
| dropout     |         213 |   0.3286 |     -0.0098 |
| frontal     |         227 |   0.3877 |     -0.0228 |
| clean       |       82447 |   0.03   |     -0.1443 |
