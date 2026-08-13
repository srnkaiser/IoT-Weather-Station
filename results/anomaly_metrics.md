# Anomaly detection results

## Overall

|                  |      Value |
|:-----------------|-----------:|
| Precision        |     0.0875 |
| Recall           |     0.1855 |
| F1               |     0.1189 |
| TP               |   233      |
| FP               |  2430      |
| FN               |  1023      |
| TN               | 79993      |
| FalseAlarmRate_% |     2.948  |

## By injected fault type

| FaultType   |   n_samples |   Recall |   MeanScore |
|:------------|------------:|---------:|------------:|
| spike       |          12 |   1      |      0.0543 |
| stuck       |         167 |   0.012  |     -0.1443 |
| drift       |         637 |   0.0126 |     -0.1428 |
| dropout     |         213 |   0.3239 |     -0.0092 |
| frontal     |         227 |   0.6256 |      0.0042 |
| clean       |       82423 |   0.0295 |     -0.1418 |
