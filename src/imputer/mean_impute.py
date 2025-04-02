import numpy as np


def mean_impute(data_with_missing):
    data_imputed = data_with_missing.copy()
    n_samples, n_features = data_imputed.shape

    for i in range(n_features):
        impute_value = np.nanmean(data_imputed[:, i])
        data_imputed[:, i][np.isnan(data_imputed[:, i])] = impute_value
    
    return data_imputed