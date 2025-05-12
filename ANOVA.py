import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def explained_deviation(anova_table):
    explained_deviations = {}
    total_ss = anova_table['sum_sq'].sum()
    for factor in anova_table.index[:-1]:
        ss_factor = anova_table.loc[factor, 'sum_sq']
        deviation = ss_factor / total_ss
        explained_deviations[factor] = deviation
    return explained_deviations

# 新增函数，用于绘制偏 eta 平方的柱状图
def plot_explained_deviation(partial_etas, title):
    factors = list(partial_etas.keys())
    eta_values = list(partial_etas.values())

    plt.figure(figsize=(10, 6))
    plt.bar(factors, eta_values, color='blue')
    plt.xlabel('Factors')
    plt.ylabel('Deviance')
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

results = pd.read_csv('./results.csv')
results = results[results['missing_rate'] != 0]

result = results[results["dataset"] == "exchange_rate"]
model = ols('forecast_mse ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MSE by factors in Exchange Rate')

model = ols('forecast_mae ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MAE by factors in Exchange Rate')

model = ols('forecast_mse ~ impute_rmse + impute_mae + impute_r2 + impute_kl_divergence + impute_ks_statistic + impute_w2_distance + impute_sliced_kl_divergence + impute_sliced_ks_statistic + impute_sliced_w2_distance', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MSE by imputation metrics in Exchange Rate')

model = ols('forecast_mae ~ impute_rmse + impute_mae + impute_r2 + impute_kl_divergence + impute_ks_statistic + impute_w2_distance + impute_sliced_kl_divergence + impute_sliced_ks_statistic + impute_sliced_w2_distance', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MAE by imputation metrics in Exchange Rate')



result = results[results["dataset"] == "illness"]
model = ols('forecast_mse ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MSE by factors in Illness')

model = ols('forecast_mae ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MAE by factors in Illness')

model = ols('forecast_mse ~ impute_rmse + impute_mae + impute_r2 + impute_kl_divergence + impute_ks_statistic + impute_w2_distance + impute_sliced_kl_divergence + impute_sliced_ks_statistic + impute_sliced_w2_distance', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MSE by imputation metrics in Illness')

model = ols('forecast_mae ~ impute_rmse + impute_mae + impute_r2 + impute_kl_divergence + impute_ks_statistic + impute_w2_distance + impute_sliced_kl_divergence + impute_sliced_ks_statistic + impute_sliced_w2_distance', data=result).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
explained_deviations = explained_deviation(anova_table)
plot_explained_deviation(explained_deviations, 'ANOVA for MAE by imputation metrics in Illness')