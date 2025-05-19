import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr
import os

save_path = './visualization/ANOVA/'
os.makedirs(save_path, exist_ok=True)

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
    
    combined = list(zip(factors, eta_values))
    combined.sort(key=lambda x: x[1], reverse=True)
    factors, eta_values = zip(*combined)

    plt.figure(figsize=(10, 6))
    plt.barh(factors, eta_values, color='blue')
    plt.ylabel('Factors')
    plt.xlabel('Deviance')
    plt.tight_layout()
    plt.savefig(f'{save_path}{title.replace(" ", "_")}.png')
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
    plt.savefig(f'{save_path}{title.replace(" ", "_")}_corr_heatmap.png')
    plt.close()

results = pd.read_csv('./results.csv')
results = results[results['missing_rate'] != 0]

# 对每个数据集进行分析
datasets = ["exchange_rate", "illness"]
for dataset in datasets:
    result = results[results["dataset"] == dataset]

    # 计算相关系数
    corr_matrix, p_value_matrix = calculate_pearson_corr(result)
    print(f"Pearson Correlation Matrix for {dataset}:")
    print(corr_matrix)
    print(f"P-Value Matrix for {dataset}:")
    print(p_value_matrix)

    # 绘制相关系数热力图
    plot_corr_heatmap(corr_matrix, f'Pearson Correlation between Imputation and Forecast Metrics in {dataset}')

    model = ols('forecast_mse ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    explained_deviations = explained_deviation(anova_table)
    plot_explained_deviation(explained_deviations, f'ANOVA for MSE by factors in {dataset}')

    model = ols('forecast_mae ~ missing_rate * C(missing_type) * complete_rate * C(imputer) * C(forecast_model)', data=result).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    explained_deviations = explained_deviation(anova_table)
    plot_explained_deviation(explained_deviations, f'ANOVA for MAE by factors in {dataset}')