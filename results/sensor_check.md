# Layer 2: deterministic sensor fault detection

Cross-check threshold: 2.45 degC

Precision 1.0000 | Recall 0.5462 | F1 0.7065

| FaultType   |   n_samples |   Recall |
|:------------|------------:|---------:|
| spike       |          12 |   0      |
| stuck       |         167 |   0.491  |
| drift       |         637 |   0.5699 |
| dropout     |         213 |   0.8263 |
| frontal     |         227 |   0.2863 |
| clean       |       82898 |   0      |
