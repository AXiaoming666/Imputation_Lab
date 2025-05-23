import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr
import os
import re

save_path = './visualization/ANOVA/'

def explained_deviation(anova_table):
    explained_deviations = {}
    total_ss = anova_table['sum_sq'].sum()
    for factor in anova_table.index[:-1]:
        ss_factor = anova_table.loc[factor, 'sum_sq']
        deviation = ss_factor / total_ss
        explained_deviations[factor] = deviation
    return explained_deviations

def plot_explained_deviation(partial_etas, title):
    # 过滤出 deviance 大于 0.002 的数据
    filtered_etas = {factor: value for factor, value in partial_etas.items() if value > 0.002}
    factors = list(filtered_etas.keys())
    eta_values = list(filtered_etas.values())
    
    factors = [re.sub(r'C\((.*?)\)', r'\1', factor) for factor in factors]
    factors = [factor.replace(':', ':\n') for factor in factors]
    
    combined = list(zip(factors, eta_values))
    combined.sort(key=lambda x: x[1], reverse=True)
    factors, eta_values = zip(*combined)

    plt.figure(figsize=(16, 6))
    
    # 根据柱子索引归一化，生成从深到浅的颜色值
    num_bars = len(factors)
    normalized_indices = [(num_bars - i - 1) / (num_bars - 1) for i in range(num_bars)]
    
    # 生成同一色系（蓝色系）的颜色列表，从左到右颜色逐渐变浅
    colors = [(0, 0, value) for value in normalized_indices]
    
    # 设置 align 为 edge，width 为负数，让柱子向左绘制，使用同一色系颜色
    bars = plt.bar(factors, eta_values, color=colors, align='center', width=0.4)
    plt.xlabel('Factors')
    plt.ylabel('Deviance')

    # 移除边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 添加浅色网格线
    plt.grid(True, axis='y', color='lightgray', linestyle='--', alpha=0.7)

    # 调整标签位置以适应垂直柱状图
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', 
                 ha='center', va='bottom', color='black', fontsize=10)

    # 调整 x 轴标签位置，使其尾部对准柱体右侧
    plt.xticks(rotation=45, ha='center')
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

if __name__ == '__main__':
    
    os.makedirs(save_path, exist_ok=True)
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
        plot_explained_deviation(explained_deviations, f'ANOVA for MAE by factors in {dataset}')