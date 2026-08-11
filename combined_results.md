# Two-layer anomaly detection results

Dataset: `synthetic`  |  Test samples: 21,021  |  Injected anomalies: 1,234

## Overall

| Detector                |   Precision |   Recall |     F1 |   FalseAlarmRate_% |
|:------------------------|------------:|---------:|-------:|-------------------:|
| Layer1_IsolationForest  |      0.3046 |   0.269  | 0.2857 |              3.831 |
| Layer2_SensorCrossCheck |      0.9985 |   0.5438 | 0.7041 |              0.005 |
| Fused_OR                |      0.5259 |   0.6823 | 0.594  |              3.836 |

## Recall by fault type

| FaultType   | Class         |     n |   Layer1 |   Layer2 |   Fused |
|:------------|:--------------|------:|---------:|---------:|--------:|
| spike       | sensor fault  |    12 |   1      |   0      |  1      |
| stuck       | sensor fault  |   146 |   0.027  |   0.397  |  0.418  |
| drift       | sensor fault  |   636 |   0.058  |   0.591  |  0.61   |
| dropout     | sensor fault  |   213 |   0.643  |   0.826  |  0.981  |
| frontal     | environmental |   227 |   0.626  |   0.269  |  0.758  |
| clean       | -             | 19787 |   0.0383 |   0.0001 |  0.0384 |
