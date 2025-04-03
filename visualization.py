import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json


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

def scatter_plot(
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
    plt.close(fig)


def box_plot():
    # Create a figure with subplots for MSE and MAE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Get unique imputation methods
    imputation_methods = sorted(df_renamed['config_imputation_method'].unique())
    
    # Group data by imputation method for boxplots
    rmse_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_RMSE'] 
                for method in imputation_methods]
    mae_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_MAE'] 
                for method in imputation_methods]
    
    # Create boxplots with showfliers=False to hide outliers
    ax1.boxplot(rmse_data, tick_labels=imputation_methods, showfliers=False)
    ax1.set_title('RMSE by Imputation Method')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.boxplot(mae_data, tick_labels=imputation_methods, showfliers=False)
    ax2.set_title('MAE by Imputation Method')
    ax2.set_ylabel('MAE')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.set_xticklabels(imputation_methods, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('./visualization/boxplot_I.png')
    plt.close(fig)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    KL_divergence_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_KL_divergence'] 
                          for method in imputation_methods]
    KS_statistic_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_KS_statistic'] 
                         for method in imputation_methods]
    W2_distance_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_W2_distance'] 
                        for method in imputation_methods]
    ax1.boxplot(KL_divergence_data, tick_labels=imputation_methods, showfliers=False)
    ax1.set_title('KL Divergence by Imputation Method')
    ax1.set_ylabel('KL Divergence')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.boxplot(KS_statistic_data, tick_labels=imputation_methods, showfliers=False)
    ax2.set_title('KS Statistic by Imputation Method')
    ax2.set_ylabel('KS Statistic')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax3.boxplot(W2_distance_data, tick_labels=imputation_methods, showfliers=False)
    ax3.set_title('W2 Distance by Imputation Method')
    ax3.set_ylabel('W2 Distance')
    ax3.grid(True, linestyle='--', alpha=0.7)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels(imputation_methods, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./visualization/boxplot_II.png')
    plt.close(fig)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    Sliced_Wasserstein_distance_data = [df_renamed[df_renamed['config_imputation_method'] == method]['imputed_metrics_Sliced_Wasserstein_distance']
                                        for method in imputation_methods]
    ax1.boxplot(Sliced_Wasserstein_distance_data, tick_labels=imputation_methods, showfliers=False)
    ax1.set_title('Sliced Wasserstein Distance by Imputation Method')
    ax1.set_ylabel('Sliced Wasserstein Distance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    for ax in [ax1]:
        ax.set_xticklabels(imputation_methods, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./visualization/boxplot_III.png')
    plt.close(fig)
    

if __name__ == "__main__":
    os.makedirs('./visualization', exist_ok=True)
    """ for config_param in ['missing_rate', 'completeness_rate', 'missing_type']:
        for metrics_param in ['mse', 'mae']:
            for aborted_methods, point_size in [(None, 1), (['mean'], 10), (['knn', 'mean', 'xgboost', 'IIM'], 100)]:
                scatter_plot(config_param=config_param, metrics_param=metrics_param, aborted_methods=aborted_methods, point_size=point_size) """
    box_plot()