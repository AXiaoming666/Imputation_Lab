import numpy as np
import pandas as pd


def forward_impute(missing_set: pd.DataFrame) -> pd.DataFrame:
    imputed_set = missing_set.copy(deep=True)
    n_samples, n_features = imputed_set.shape

    for i in range(n_features):
        impute_value = imputed_set.iloc[0, i]
        for j in range(1, n_samples):
            if np.isnan(imputed_set.iloc[j, i]):
                if np.isnan(impute_value):
                    continue
                else:
                    imputed_set.iloc[j, i] = impute_value
            else:
                if np.isnan(impute_value):
                    for k in range(0, j):
                        imputed_set.iloc[k, i] = imputed_set.iloc[j, i]
                impute_value = imputed_set.iloc[j, i]
    
    return imputed_set