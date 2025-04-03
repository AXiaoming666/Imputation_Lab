import json
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

dataframes = []

for foldername, subfolders, filenames in os.walk('./results'):
    for filename in filenames:
        with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)
            df = pd.json_normalize(data)
            dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

df_renamed = combined_df.copy()
column_mapping = {}
for col in df.columns:
    new_col = col.replace('.', '_')
    new_col = new_col.replace(' ', '_')
    column_mapping[col] = new_col
    df_renamed[new_col] = combined_df[col]

for col in df.columns:
    if col in df_renamed.columns:
        df_renamed.drop(columns=[col], inplace=True, errors='ignore')

mse_base = 0.10529112070798874
mae_base = 0.23513168096542358

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

model_formula = 'forecast_metrics_mse ~ config_missing_rate + C(config_missing_type) + config_completeness_rate + C(config_imputation_method)'
model = ols(model_formula, data=df_renamed).fit()
anova_results = sm.stats.anova_lm(model, typ=3)
print(anova_results)

model_formula = 'forecast_metrics_mae ~ config_missing_rate + C(config_missing_type) + config_completeness_rate + C(config_imputation_method)'
model = ols(model_formula, data=df_renamed).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

model_formula = 'forecast_metrics_mse ~ imputed_metrics_RMSE + imputed_metrics_MAE + imputed_metrics_R2 + imputed_metrics_KL_divergence' + \
                '+ imputed_metrics_KS_statistic + imputed_metrics_W2_distance + imputed_metrics_Sliced_Wasserstein_distance'
model = ols(model_formula, data=df_renamed).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

model_formula = 'forecast_metrics_mae ~ imputed_metrics_RMSE + imputed_metrics_MAE + imputed_metrics_R2 + imputed_metrics_KL_divergence' + \
                '+ imputed_metrics_KS_statistic + imputed_metrics_W2_distance + imputed_metrics_Sliced_Wasserstein_distance'
model = ols(model_formula, data=df_renamed).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)