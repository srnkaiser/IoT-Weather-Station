# Layer 2: deterministic sensor fault detection

Cross-check threshold: 2.18 degC

Precision 0.9985 | Recall 0.5438 | F1 0.7041

| FaultType   |   n_samples |   Recall |
|:------------|------------:|---------:|
| spike       |          12 |   0      |
| stuck       |         146 |   0.3973 |
| drift       |         636 |   0.5912 |
| dropout     |         213 |   0.8263 |
| frontal     |         227 |   0.2687 |
| clean       |       19805 |   0.0001 |
