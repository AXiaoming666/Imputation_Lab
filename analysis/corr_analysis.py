import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr
import os

save_path = './visualization/corr_analysis/'

def calculate_pearson_corr(data):
    impute_metrics = ['impute_rmse', 'impute_mae', 'impute_r2', 'impute_kl_divergence', 'impute_ks_statistic', 'impute_w2_distance', 'impute_sliced_kl_divergence', 'impute_sliced_ks_statistic', 'impute_sliced_w2_distance']
    forecast_metrics = ['forecast_mse', 'forecast_mae']
    corr_matrix = pd.DataFrame(index=impute_metrics, columns=forecast_metrics)

    for impute_metric in impute_metrics:
        for forecast_metric in forecast_metrics:
            corr, p_value = pearsonr(data[impute_metric], data[forecast_metric])
            corr_matrix.loc[impute_metric, forecast_metric] = corr

    return corr_matrix

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
        corr_matrix = calculate_pearson_corr(result)
        # 绘制相关系数热力图
        plot_corr_heatmap(corr_matrix, f'Pearson Correlation between Imputation and Forecast Metrics in {dataset}')
    
    
    
    impute_metrics = ['impute_rmse', 'impute_mae', 'impute_r2', 'impute_kl_divergence', 'impute_ks_statistic', 'impute_w2_distance', 'impute_sliced_kl_divergence', 'impute_sliced_ks_statistic', 'impute_sliced_w2_distance']
    
    corr_matrix = pd.DataFrame(index=impute_metrics, columns=impute_metrics)

    for impute_metric1 in impute_metrics:
        for impute_metric2 in impute_metrics:
            corr, p_value = pearsonr(results[impute_metric1], results[impute_metric2])
            corr_matrix.loc[impute_metric1, impute_metric2] = corr
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation between Imputation Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}imputation_metrics_corr_heatmap.png')
    plt.close()