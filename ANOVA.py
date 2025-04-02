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
anova_results = sm.stats.anova_lm(model, typ=2)
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


""" def new_func(
    config_param='missing_rate',
    metrics_param='mse',
    aborted_methods=None,
    point_size=1,
):
    mse_means = {}
    mae_means = {}
    mse_stds = {}
    mae_stds = {}
    for imputation_method in df_renamed['config_imputation_method'].unique():
        for config in df_renamed[f'config_{config_param}'].unique():
            subset = df_renamed[(df_renamed['config_imputation_method'] == imputation_method) & 
                                (df_renamed[f'config_{config_param}'] == config)]
            if subset.empty:
                continue
            mse_means[(imputation_method, config)] = subset['forecast_metrics_mse'].mean()
            mse_stds[(imputation_method, config)] = subset['forecast_metrics_mse'].std()
            mae_means[(imputation_method, config)] = subset['forecast_metrics_mae'].mean()
            mae_stds[(imputation_method, config)] = subset['forecast_metrics_mae'].std()
            
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = sorted(df_renamed[f'config_{config_param}'].unique())
    imputation_methods = sorted(df_renamed['config_imputation_method'].unique())
    if not aborted_methods is None:
        imputation_methods = [method for method in imputation_methods if method not in aborted_methods]
    
    colors = plt.cm.tab10(range(len(imputation_methods)))
    color_dict = dict(zip(imputation_methods, colors))

    y_positions = range(len(configs))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(configs)

    if metrics_param == 'mse':
        metrics_means = mse_means
        metrics_stds = mse_stds
        metrics_base = mse_base
    elif metrics_param == 'mae':
        metrics_means = mae_means
        metrics_stds = mae_stds
        metrics_base = mae_base
    
    for i, config in enumerate(configs):
        for imputation_method in imputation_methods:
            if (imputation_method, config) in metrics_means:
                ax.scatter(
                metrics_means[(imputation_method, config)], 
                i,
                s=metrics_stds[(imputation_method, config)] * 1000 * point_size,
                color=color_dict[imputation_method],
                label=f'{imputation_method}' if i == 0 else "",
                alpha=0.7
                )

    ax.axvline(x=metrics_base, color='r', linestyle='--', label=f'Baseline {metrics_param.upper()}: {metrics_base:.4f}')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    legend = ax.legend(by_label.values(), by_label.keys(), loc='best')

    for handle in legend.legend_handles:
        if isinstance(handle, matplotlib.collections.PathCollection):
            handle.set_sizes([100])

    ax.set_xlabel(metrics_param.upper())
    ax.set_title(f'{metrics_param.upper()} Means by Imputation Method and {config_param.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f'./visualization/{metrics_param.upper()}_{config_param}_without_{aborted_methods}.png')

for config_param in ['missing_rate', 'completeness_rate', 'missing_type']:
    for metrics_param in ['mse', 'mae']:
        for aborted_methods, point_size in [(None, 1), (['mean'], 10), (['knn', 'mean', 'xgboost', 'IIM'], 100)]:
            new_func(config_param=config_param, metrics_param=metrics_param, aborted_methods=aborted_methods, point_size=point_size) """