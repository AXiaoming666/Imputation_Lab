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
    aborted_methods=None
):
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = sorted(df_renamed[f'config_{config_param}'].unique())
    imputation_methods = sorted(df_renamed['config_imputation_method'].unique())
    if aborted_methods is not None:
        imputation_methods = [method for method in imputation_methods if method not in aborted_methods]
    
    colors = plt.cm.tab10(range(len(imputation_methods)))
    color_dict = dict(zip(imputation_methods, colors))

    # Set up the plot
    ax.set_ylabel(metrics_param.upper())
    ax.set_xlabel(config_param.replace("_", " ").title())
    ax.set_title(f'{metrics_param.upper()} by Imputation Method and {config_param.replace("_", " ").title()}')
    
    # Group data for box plot
    grouped_data = []
    labels = []
    positions = []
    
    # Calculate positions for proper alignment
    method_count = len(imputation_methods)
    box_width = 0.8  # width of each box
    group_width = box_width * method_count
    
    for i, config in enumerate(configs):
        config_center = i  # Center of each config group
        
        for j, imputation_method in enumerate(imputation_methods):
            # Position each boxplot within its config group
            position = config_center + (j - method_count/2 + 0.5) * box_width / method_count
            
            subset = df_renamed[(df_renamed['config_imputation_method'] == imputation_method) & 
                                (df_renamed[f'config_{config_param}'] == config)]
            if not subset.empty:
                if metrics_param == 'mse':
                    data = subset['forecast_metrics_mse']
                elif metrics_param == 'mae':
                    data = subset['forecast_metrics_mae']
                
                grouped_data.append(data)
                labels.append(f"{imputation_method}_{config}")
                positions.append(position)
    
    # Create box plot
    boxplot = ax.boxplot(grouped_data, positions=positions, patch_artist=True, 
                         showfliers=False, widths=box_width/method_count*0.9)
    
    # Color boxes by imputation method
    method_boxes = {}
    for i, (box, label) in enumerate(zip(boxplot['boxes'], labels)):
        method = label.split('_')[0]
        box.set(facecolor=color_dict[method])
        if method not in method_boxes:
            method_boxes[method] = box
    
    # Add baseline line
    metrics_base = mse_base if metrics_param == 'mse' else mae_base
    ax.axhline(y=metrics_base, color='r', linestyle='--', 
               label=f'Baseline {metrics_param.upper()}: {metrics_base:.4f}')
    
    # Set x-ticks at the center of each config group
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    
    # Create custom legend for imputation methods
    handles = [method_boxes[method] for method in imputation_methods]
    ax.legend(handles, imputation_methods, loc='best')
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Create filename with proper handling of None
    aborted_str = "" if aborted_methods is None else f"_without_{'_'.join(aborted_methods)}"
    plt.savefig(f'./visualization/{metrics_param.upper()}_{config_param}{aborted_str}.png')
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
    for config_param in ['missing_rate', 'completeness_rate', 'missing_type']:
        for metrics_param in ['mse', 'mae']:
            for aborted_methods in [None, ['mean'], ['knn', 'mean', 'xgboost', 'IIM']]:
                scatter_plot(config_param=config_param, metrics_param=metrics_param, aborted_methods=aborted_methods)
    box_plot()