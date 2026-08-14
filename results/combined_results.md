# Two-layer anomaly detection results

Dataset: `jena`  |  Test samples: 83,679  |  Injected anomalies: 1,256

## Overall

| Detector                |   Precision |   Recall |     F1 |   FalseAlarmRate_% |
|:------------------------|------------:|---------:|-------:|-------------------:|
| Layer1_IsolationForest  |      0.0875 |   0.1855 | 0.1189 |              2.948 |
| Layer2_SensorCrossCheck |      1      |   0.5462 | 0.7065 |              0     |
| Fused_OR                |      0.2548 |   0.6616 | 0.3679 |              2.948 |

## Recall by fault type

| FaultType   | Class         |     n |   Layer1 |   Layer2 |   Fused |
|:------------|:--------------|------:|---------:|---------:|--------:|
| spike       | sensor fault  |    12 |   1      |    0     |  1      |
| stuck       | sensor fault  |   167 |   0.012  |    0.491 |  0.491  |
| drift       | sensor fault  |   637 |   0.013  |    0.57  |  0.582  |
| dropout     | sensor fault  |   213 |   0.324  |    0.826 |  0.925  |
| frontal     | environmental |   227 |   0.626  |    0.286 |  0.744  |
| clean       | -             | 82423 |   0.0295 |    0     |  0.0295 |
