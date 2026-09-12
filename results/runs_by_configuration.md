# Every training run, grouped by configuration

A single run is not a result: configurations differ from one another by 0.001-0.018 degC and the same configuration differs from itself by up to 0.009 degC depending on the seed. Compare the means, and treat a gap smaller than the spread column as unresolved.

| architecture     |   seq_minutes | inputs       |   units |     lr |   n_seeds |   MAE_mean |   MAE_min |   MAE_max |   MAE_std |   parameters |   train_s |   spread |
|:-----------------|--------------:|:-------------|--------:|-------:|----------:|-----------:|----------:|----------:|----------:|-------------:|----------:|---------:|
| bilstm           |            60 | raw+calendar |      96 | 0.0003 |         1 |     0.4721 |    0.4721 |    0.4721 |  nan      |        86081 |      1328 |   0      |
| lstm             |            60 | raw+calendar |      96 | 0.0003 |         1 |     0.4735 |    0.4735 |    0.4735 |  nan      |        43073 |       617 |   0      |
| attention_lstm   |            60 | raw+calendar |      96 | 0.0003 |         1 |     0.4796 |    0.4796 |    0.4796 |  nan      |        46210 |       705 |   0      |
| attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |         2 |     0.4855 |    0.4812 |    0.4897 |    0.006  |        92290 |       936 |   0.0085 |
| attention_bilstm |            60 | raw          |      96 | 0.0003 |         2 |     0.5376 |    0.5372 |    0.5381 |    0.0006 |        89218 |       833 |   0.0009 |
| attention_bilstm |            60 | raw          |      64 | 0.001  |         1 |     0.5385 |    0.5385 |    0.5385 |  nan      |        43138 |       201 |   0      |
| attention_bilstm |           720 | raw          |      64 | 0.001  |         1 |     0.5421 |    0.5421 |    0.5421 |  nan      |        43138 |       754 |   0      |

Gradient Boosting baseline: MAE 0.4633 (deterministic, one run).

## Individual runs

| tag               | architecture     |   seq_minutes | inputs       |   units |     lr |   seed |   parameters |    MAE |   skill |   train_s |
|:------------------|:-----------------|--------------:|:-------------|--------:|-------:|-------:|-------------:|-------:|--------:|----------:|
| 60min_noatt       | bilstm           |            60 | raw+calendar |      96 | 0.0003 |     42 |        86081 | 0.4721 |   34.49 |    1328.2 |
| 60min_plainlstm   | lstm             |            60 | raw+calendar |      96 | 0.0003 |     42 |        43073 | 0.4735 |   34.3  |     616.9 |
| 60min_unidir      | attention_lstm   |            60 | raw+calendar |      96 | 0.0003 |     42 |        46210 | 0.4796 |   33.45 |     704.8 |
| 60min_calendar_s7 | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |      7 |        92290 | 0.4812 |   33.23 |     974.4 |
| 60min_calendar    | attention_bilstm |            60 | raw+calendar |      96 | 0.0003 |     42 |        92290 | 0.4897 |   32.05 |     897.1 |
| 60min_tuned_s7    | attention_bilstm |            60 | raw          |      96 | 0.0003 |      7 |        89218 | 0.5372 |   25.45 |     810.8 |
| 60min_tuned       | attention_bilstm |            60 | raw          |      96 | 0.0003 |    nan |        89218 | 0.5381 |   25.33 |     854.3 |
| 60min             | attention_bilstm |            60 | raw          |      64 | 0.001  |    nan |        43138 | 0.5385 |   25.28 |     201.3 |
| 720min            | attention_bilstm |           720 | raw          |      64 | 0.001  |    nan |        43138 | 0.5421 |   24.78 |     753.7 |
