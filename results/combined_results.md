# Two-layer anomaly detection results

Dataset: `jena`  |  Test samples: 83,703  |  Injected anomalies: 1,256

## Overall

| Detector                |   Precision |   Recall |     F1 |   FalseAlarmRate_% |
|:------------------------|------------:|---------:|-------:|-------------------:|
| Layer1_IsolationForest  |      0.0675 |   0.1425 | 0.0917 |              2.997 |
| Layer2_SensorCrossCheck |      1      |   0.5048 | 0.6709 |              0     |
| Fused_OR                |      0.2357 |   0.6067 | 0.3395 |              2.997 |

## Recall by fault type

| FaultType   | Class         |     n |   Layer1 |   Layer2 |   Fused |
|:------------|:--------------|------:|---------:|---------:|--------:|
| spike       | sensor fault  |    12 |    1     |    0     |   1     |
| stuck       | sensor fault  |   167 |    0.012 |    0.491 |   0.491 |
| drift       | sensor fault  |   637 |    0.011 |    0.59  |   0.601 |
| dropout     | sensor fault  |   213 |    0.329 |    0.826 |   0.925 |
| frontal     | environmental |   227 |    0.388 |    0     |   0.388 |
| clean       | -             | 82447 |    0.03  |    0     |   0.03  |

## Layer 2 scored against hardware faults only

The overall table scores every detector against 'was anything injected here'. That penalises Layer 2 for correct behaviour: a frontal passage is weather, both sensors see it, and Layer 2 is meant to stay silent. Scored against the faults it actually targets:

Precision 1.0 | Recall 0.6161 | F1 0.7625 | FalseAlarmRate_% 0.0

## Classification of detected events

- Weather events detected: 88, correctly identified as weather: 88
- Hardware faults detected: 674, correctly identified as faults: 634
