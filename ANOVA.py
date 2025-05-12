import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns  # 新增导入seaborn库
from scipy.stats import pearsonr  # 新增导入pearsonr函数
matplotlib.use('Agg')

def explained_deviation(anova_table):
    explained_deviations = {}
    total_ss = anova_table['sum_sq'].sum()
    for factor in anova_table.index[:-1]:
        ss_factor = anova_table.loc[factor, 'sum_sq']
        deviation = ss_factor / total_ss
        explained_deviations[factor] = deviation
    return explained_deviations

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

# 新增函数：计算皮尔逊相关系数
def calculate_pearson_corr(data):
    impute_metrics = ['impute_rmse', 'impute_mae', 'impute_r2', 'impute_kl_divergence', 'impute_ks_statistic', 'impute_w2_distance', 'impute_sliced_kl_divergence', 'impute_sliced_ks_statistic', 'impute_sliced_w2_distance']
    forecast_metrics = ['forecast_mse', 'forecast_mae']
    corr_matrix = pd.DataFrame(index=impute_metrics, columns=forecast_metrics)
    p_value_matrix = pd.DataFrame(index=impute_metrics, columns=forecast_metrics)

    for impute_metric in impute_metrics:
        for forecast_metric in forecast_metrics:
            corr, p_value = pearsonr(data[impute_metric], data[forecast_metric])
            corr_matrix.loc[impute_metric, forecast_metric] = corr
            p_value_matrix.loc[impute_metric, forecast_metric] = p_value

    return corr_matrix, p_value_matrix

# 新增函数：绘制相关系数热力图
def plot_corr_heatmap(corr_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_corr_heatmap.png')
    plt.close()

results = pd.read_csv('./results.csv')
results = results[results['missing_rate'] != 0]

# 对每个数据集进行分析
datasets = ["exchange_rate", "illness"]
for dataset in datasets:
    result = results[results["dataset"] == dataset]

    # 计算相关系数
    corr_matrix, p_value_matrix = calculate_pearson_corr(result)

    # 绘制相关系数热力图
    plot_corr_heatmap(corr_matrix, f'Pearson Correlation between Imputation and Forecast Metrics in {dataset}')

    model = ols('forecast_mse ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    explained_deviations = explained_deviation(anova_table)
    plot_explained_deviation(explained_deviations, f'ANOVA for MSE by factors in {dataset}')

    model = ols('forecast_mae ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    explained_deviations = explained_deviation(anova_table)
    plot_explained_deviation(explained_deviations, 'ANOVA for MAE by factors in Illness')