import json
import pandas as pd
import os

dataframes = []

for foldername, subfolders, filenames in os.walk('./results'):
    for filename in filenames:
        with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
            df = pd.json_normalize(data)
            dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df = combined_df.rename(columns={'config.seed': 'seed',
                                          'config.dataset_name': 'dataset',
                                          'config.missing_rate' : 'missing_rate',
                                          'config.missing_type' : 'missing_type',
                                          'config.completeness_rate' : 'completeness_rate',
                                          'config.imputation_method' : 'imputation_method',
                                          'config.forecast_model' : 'forecast_model',
                                          'imputed_metrics.RMSE' : 'imputed_rmse',
                                          'imputed_metrics.MAE' : 'imputed_mae',
                                          'imputed_metrics.R2' : 'imputed_r2',
                                          'imputed_metrics.KL divergence' : 'imputed_kl', 
                                          'imputed_metrics.KS statistic' : 'imputed_ks',
                                          'imputed_metrics.W2 distance' : 'imputed_w2',
                                          'imputed_metrics.Sliced Wasserstein distance' : 'imputed_sliced_w2',
                                          'forecast_metrics.mse' : 'forecast_mse',
                                          'forecast_metrics.mae' : 'forecast_mae'})

combined_df.to_csv('results.csv', index=False)