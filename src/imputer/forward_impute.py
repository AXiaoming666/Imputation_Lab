import numpy as np


def forward_impute(data_with_missing):
    data_imputed = data_with_missing.copy()
    n_samples, n_features = data_imputed.shape

    for i in range(n_features):
        impute_value = data_imputed[0, i]
        for j in range(1, n_samples):
            if np.isnan(data_imputed[j, i]):
                if np.isnan(impute_value):
                    continue
                else:
                    data_imputed[j, i] = impute_value
            else:
                if np.isnan(impute_value):
                    for k in range(0, j):
                        data_imputed[k, i] = data_imputed[j, i]
                impute_value = data_imputed[j, i]
    
    return data_imputed