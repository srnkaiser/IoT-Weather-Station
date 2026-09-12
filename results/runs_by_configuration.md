# Every training run, grouped by configuration

A single run is not a result: configurations differ from one another by 0.001-0.018 degC and the same configuration differs from itself by up to 0.009 degC depending on the seed. Compare the means, and treat a gap smaller than the spread column as unresolved.

| architecture     |   seq_minutes | inputs       |   units |     lr |   n_seeds |   MAE_mean |   MAE_min |   MAE_max |   MAE_std |   parameters |   train_s |   spread |
|:-----------------|--------------:|:-------------|--------:|-------:|----------:|-----------:|----------:|----------:|----------:|-------------:|----------:|---------:|
| attention_bilstm |            60 | engineered   |      96 | 0.0003 |         3 |     0.4511 |    0.4469 |    0.4563 |    0.0048 |       119170 |      1952 |   0.0094 |
| attention_bilstm |            20 | engineered   |      96 | 0.0003 |         1 |     0.4515 |    0.4515 |    0.4515 |  nan      |       119170 |       742 |   0      |
| bilstm           |            60 | raw+calendar |      96 | 0.0003 |         4 |     0.4718 |    0.4705 |    0.4733 |    0.0012 |        86081 |      2194 |   0.0028 |
| lstm             |            60 | raw+calendar |      96 | 0.0003 |         4 |     0.4727 |    0.4688 |    0.4761 |    0.003  |        43073 |      1432 |   0.0073 |
| attention_lstm   |            60 | raw+calendar |      96 | 0.0003 |         1 |     0.4796 |    0.4796 |    0.4796 |  nan      |        46210 |       705 |   0      |
| attention_bilstm |           720 | raw+calendar |      96 | 0.0003 |         1 |     0.4848 |    0.4848 |    0.4848 |  nan      |        92290 |      7414 |   0      |
| attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |         5 |     0.4853 |    0.4801 |    0.4897 |    0.0044 |        92290 |      1924 |   0.0096 |
| attention_bilstm |            60 | raw          |      96 | 0.0003 |         2 |     0.5376 |    0.5372 |    0.5381 |    0.0006 |        89218 |       833 |   0.0009 |
| attention_bilstm |            60 | raw          |      64 | 0.001  |         1 |     0.5385 |    0.5385 |    0.5385 |  nan      |        43138 |       201 |   0      |
| attention_bilstm |           720 | raw          |      64 | 0.001  |         1 |     0.5421 |    0.5421 |    0.5421 |  nan      |        43138 |       754 |   0      |

Gradient Boosting baseline: MAE 0.4633 (deterministic, one run).

## Individual runs

| tag                | architecture     |   seq_minutes | inputs       |   units |     lr |   seed |   parameters |    MAE |   skill |   train_s |
|:-------------------|:-----------------|--------------:|:-------------|--------:|-------:|-------:|-------------:|-------:|--------:|----------:|
| 60min_features_s7  | attention_bilstm |            60 | engineered   |      96 | 0.0003 |      7 |       119170 | 0.4469 |   37.99 |    1455.4 |
| 60min_features     | attention_bilstm |            60 | engineered   |      96 | 0.0003 |     42 |       119170 | 0.4502 |   37.53 |    1481.3 |
| 20min_features     | attention_bilstm |            20 | engineered   |      96 | 0.0003 |     42 |       119170 | 0.4515 |   37.35 |     742.1 |
| 60min_features_s1  | attention_bilstm |            60 | engineered   |      96 | 0.0003 |      1 |       119170 | 0.4563 |   36.68 |    2919.1 |
| 60min_plainlstm_s1 | lstm             |            60 | raw+calendar |      96 | 0.0003 |      1 |        43073 | 0.4688 |   34.94 |    1576.7 |
| 60min_noatt_s2     | bilstm           |            60 | raw+calendar |      96 | 0.0003 |      2 |        86081 | 0.4705 |   34.71 |    2060.9 |
| 60min_noatt_s7     | bilstm           |            60 | raw+calendar |      96 | 0.0003 |      7 |        86081 | 0.4711 |   34.63 |    2339   |
| 60min_noatt        | bilstm           |            60 | raw+calendar |      96 | 0.0003 |     42 |        86081 | 0.4721 |   34.49 |    1328.2 |
| 60min_plainlstm_s7 | lstm             |            60 | raw+calendar |      96 | 0.0003 |      7 |        43073 | 0.4724 |   34.45 |    1241.2 |
| 60min_noatt_s1     | bilstm           |            60 | raw+calendar |      96 | 0.0003 |      1 |        86081 | 0.4733 |   34.32 |    3046.7 |
| 60min_plainlstm    | lstm             |            60 | raw+calendar |      96 | 0.0003 |     42 |        43073 | 0.4735 |   34.3  |     616.9 |
| 60min_plainlstm_s2 | lstm             |            60 | raw+calendar |      96 | 0.0003 |      2 |        43073 | 0.4761 |   33.93 |    2294.7 |
| 60min_unidir       | attention_lstm   |            60 | raw+calendar |      96 | 0.0003 |     42 |        46210 | 0.4796 |   33.45 |     704.8 |
| 60min_calendar_s1  | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |      1 |        92290 | 0.4801 |   33.37 |    1857   |
| 60min_calendar_s7  | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |      7 |        92290 | 0.4812 |   33.23 |     974.4 |
| 720min_calendar    | attention_bilstm |           720 | raw+calendar |      96 | 0.0003 |     42 |        92290 | 0.4848 |   32.73 |    7414   |
| 60min_calendar_s2  | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |      2 |        92290 | 0.4868 |   32.44 |    3525.5 |
| 60min_calendar_s3  | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |      3 |        92290 | 0.4886 |   32.2  |    2364.7 |
| 60min_calendar     | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |     42 |        92290 | 0.4897 |   32.05 |     897.1 |
| 60min_tuned_s7     | attention_bilstm |            60 | raw          |      96 | 0.0003 |      7 |        89218 | 0.5372 |   25.45 |     810.8 |
| 60min_tuned        | attention_bilstm |            60 | raw          |      96 | 0.0003 |    nan |        89218 | 0.5381 |   25.33 |     854.3 |
| 60min              | attention_bilstm |            60 | raw          |      64 | 0.001  |    nan |        43138 | 0.5385 |   25.28 |     201.3 |
| 720min             | attention_bilstm |           720 | raw          |      64 | 0.001  |    nan |        43138 | 0.5421 |   24.78 |     753.7 |
