# Anomaly detection results

## Overall

|                  |      Value |
|:-----------------|-----------:|
| Precision        |     0.3046 |
| Recall           |     0.269  |
| F1               |     0.2857 |
| TP               |   332      |
| FP               |   758      |
| FN               |   902      |
| TN               | 19029      |
| FalseAlarmRate_% |     3.831  |

## By injected fault type

| FaultType   |   n_samples |   Recall |   MeanScore |
|:------------|------------:|---------:|------------:|
| spike       |          12 |   1      |      0.0561 |
| stuck       |         146 |   0.0274 |     -0.064  |
| drift       |         636 |   0.0582 |     -0.0935 |
| dropout     |         213 |   0.6432 |      0.0251 |
| frontal     |         227 |   0.6256 |      0.011  |
| clean       |       19787 |   0.0383 |     -0.0998 |
