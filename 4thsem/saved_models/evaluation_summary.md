| Target   | Model            |       R2 |   Adj_R2 |      RMSE |       MAE |         MSE | Target_Description            |    CV_R2 |   Fit_Time_Sec |
|:---------|:-----------------|---------:|---------:|----------:|----------:|------------:|:------------------------------|---------:|---------------:|
| Cs       | Ridge Regression | 0.902686 | 0.902095 | 3.04084   | 2.42072   |  9.24673    | Compressive Strength (Cs_Mpa) | 0.898903 |           7.19 |
| Cs       | Naive Bayes      | 0.78193  | 0.780605 | 4.55203   | 3.13075   | 20.721      | Compressive Strength (Cs_Mpa) | 0.796477 |           0.1  |
| Cs       | XGBoost          | 0.995444 | 0.995417 | 0.657925  | 0.418766  |  0.432865   | Compressive Strength (Cs_Mpa) | 0.994538 |           2.07 |
| Cs       | LightGBM         | 0.995526 | 0.995499 | 0.651999  | 0.427852  |  0.425103   | Compressive Strength (Cs_Mpa) | 0.99417  |          11.61 |
| Cs       | ANN (PyTorch)    | 0.992122 | 0.992074 | 0.86518   | 0.604295  |  0.748537   | Compressive Strength (Cs_Mpa) | 0.989932 |         262.34 |
| Ts       | Ridge Regression | 0.921321 | 0.920709 | 0.252397  | 0.189055  |  0.0637043  | Tensile Strength (Ts_Mpa)     | 0.907517 |           8.53 |
| Ts       | Naive Bayes      | 0.570435 | 0.567096 | 0.589752  | 0.386285  |  0.347807   | Tensile Strength (Ts_Mpa)     | 0.432301 |           0.09 |
| Ts       | XGBoost          | 0.997268 | 0.997247 | 0.0470313 | 0.0287144 |  0.00221194 | Tensile Strength (Ts_Mpa)     | 0.996754 |           2.09 |
| Ts       | LightGBM         | 0.997272 | 0.997251 | 0.0469948 | 0.0290714 |  0.00220851 | Tensile Strength (Ts_Mpa)     | 0.99686  |          11.57 |
| Ts       | ANN (PyTorch)    | 0.994983 | 0.994944 | 0.0637366 | 0.0470309 |  0.00406236 | Tensile Strength (Ts_Mpa)     | 0.991877 |         203.99 |
| Fs       | Ridge Regression | 0.98652  | 0.986312 | 0.275142  | 0.199171  |  0.0757031  | Flexural Strength (Fs_Mpa)    | 0.986396 |           9.83 |
| Fs       | Naive Bayes      | 0.967426 | 0.966924 | 0.4277    | 0.13112   |  0.182927   | Flexural Strength (Fs_Mpa)    | 0.972714 |           0.05 |
| Fs       | XGBoost          | 0.99963  | 0.999624 | 0.0455771 | 0.0237746 |  0.00207727 | Flexural Strength (Fs_Mpa)    | 0.999534 |           1.55 |
| Fs       | LightGBM         | 0.999577 | 0.999571 | 0.0487132 | 0.0259741 |  0.00237298 | Flexural Strength (Fs_Mpa)    | 0.999496 |           9.21 |
| Fs       | ANN (PyTorch)    | 0.999143 | 0.99913  | 0.0693699 | 0.0550549 |  0.00481218 | Flexural Strength (Fs_Mpa)    | 0.99899  |          77.76 |