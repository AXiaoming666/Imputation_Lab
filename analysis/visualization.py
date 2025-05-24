import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
import os
import numpy as np  # 新增导入 numpy

save_path = 'visualization/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + 'by_forecast_model/', exist_ok=True)
os.makedirs(save_path + 'by_imputer/', exist_ok=True)

if __name__ == '__main__':
    results = pd.read_csv('results.csv')
    
    for dataset in results['dataset'].unique():
        for forecast_metric in ['forecast_mse', 'forecast_mae']:
            result = results[(results['dataset'] == dataset)]

            benchmark = result[result["missing_rate"] == 0][["forecast_model", forecast_metric]]

            result = result[result["missing_rate"]!= 0]
            
            # 计算每个数据集、forecast_model 和 missing_rate 组合下的 forecast_metric_value 均值和标准差
            grouped_data = result.groupby(['forecast_model','missing_rate'])[forecast_metric].agg(['mean','std']).reset_index()
            
            # 设置图片清晰度
            plt.rcParams['figure.dpi'] = 300
            
            # 创建画布，设置 margin_titles 为 True，避免图例被覆盖，同时调整宽高比
            g = sns.FacetGrid(grouped_data, margin_titles=True, height=5, aspect=1.7)
            
            # 获取调色板
            palette = sns.color_palette('bright', n_colors=len(grouped_data['forecast_model'].unique()))
            model_color_map = dict(zip(grouped_data['forecast_model'].unique(), palette))
            
            # 在每个子图上绘制散点图，点的大小与标准差成正比
            g.map_dataframe(sns.scatterplot, x='missing_rate', y='mean', size='std', hue='forecast_model', palette='bright', sizes=(20, 200))

            # 添加浅色网格线
            for ax in g.axes.flat:
                ax.grid(True, color='lightgray', linestyle='--', alpha=0.7)
                
            # 获取图例句柄和标签
            handles, labels = g.axes[0, 0].get_legend_handles_labels()
            labels[labels.index('std')] ='standard_deviation'
            
            # 找到标准差图例的起始和结束索引
            std_start_idx = labels.index('standard_deviation')
            std_end_idx = next((i for i, label in enumerate(labels[std_start_idx:], start=std_start_idx) if label == labels[0]), len(labels))
            
            # 筛选出 3 个不同大小的标准差图例
            std_handles = handles[std_start_idx:std_end_idx]
            std_labels = labels[std_start_idx:std_end_idx]
            num_std_legend = 3
            step = len(std_handles) // num_std_legend if len(std_handles) > num_std_legend - 1 else 1
            selected_std_handles = [std_handles[0]] + [std_handles[i * step] for i in range(1, num_std_legend)] + [std_handles[-1]]
            selected_std_labels = [std_labels[0]] + [std_labels[i * step] for i in range(1, num_std_legend)] + [std_labels[-1]]
            
            # 移除原有的标准差图例
            handles = handles[:std_start_idx] + handles[std_end_idx:]
            labels = labels[:std_start_idx] + labels[std_end_idx:]
            
            # 添加筛选后的标准差图例
            handles = handles[:std_start_idx] + selected_std_handles + handles[std_start_idx + 1:]
            labels = labels[:std_start_idx] + selected_std_labels + labels[std_start_idx + 1:]
            
            # 创建一个从模型名称到图例句柄的映射
            model_to_handle = dict(zip(labels, handles))
            
            # 新增黑色基准误差线的图例句柄和标签，仅用于图例
            # 修改 linestyle 为 '-.'，与图中基准线样式保持一致
            benchmark_line = Line2D([0], [0], color='black', linestyle='-.', alpha=0.7)
            benchmark_handles = [benchmark_line]
            benchmark_labels = ['benchmark']
            
            # 为每个预测模型绘制基准误差线，保持原有颜色
            for model in benchmark['forecast_model'].unique():
                model_benchmark = benchmark[benchmark['forecast_model'] == model][forecast_metric].values[0]
                if model in model_to_handle:
                    color = model_color_map[model]  # 使用正确的颜色
                    g.axes[0, 0].axhline(y=model_benchmark, color=color, linestyle='-.', alpha=0.7)

            # 合并图例句柄和标签
            handles.extend(benchmark_handles)
            labels.extend(benchmark_labels)

            # 手动添加图例，调整位置和布局
            # 减少列数以避免图例过于拥挤
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.45, 1.1), ncol=len(handles), frameon=False, fontsize='x-small', columnspacing=0.5)

            # 设置轴标签
            g.set_axis_labels('missing_rate', forecast_metric)

            g.figure.suptitle(f'{forecast_metric} vs forecast_model in {dataset}', y=0.95)

            # 调整子图布局，为标题和图例腾出更多空间
            plt.subplots_adjust(top=0.83)

            # 显示图形
            plt.savefig(f'{save_path}by_forecast_model/{dataset}_{forecast_metric}_vs_forecast_model_with_benchmark.png')
            plt.close()  # 关闭当前图形，避免内存泄漏
            
            
            
            
            
            result = results[(results['dataset'] == dataset)]

            benchmark = result[result["missing_rate"] == 0][["forecast_model", forecast_metric]]

            result = result[result["missing_rate"]!= 0]
            
            # 计算每个数据集、forecast_model 和 missing_rate 组合下的 forecast_metric_value 均值和标准差
            grouped_data = result.groupby(['forecast_model','missing_rate'])[forecast_metric].agg(['mean','std']).reset_index()
            
            # 设置图片清晰度
            plt.rcParams['figure.dpi'] = 300
            
            # 创建画布，设置 margin_titles 为 True，避免图例被覆盖，同时调整宽高比
            g = sns.FacetGrid(grouped_data, margin_titles=True, height=5, aspect=1.7)
            
            # 获取调色板
            palette = sns.color_palette('bright', n_colors=len(grouped_data['forecast_model'].unique()))
            model_color_map = dict(zip(grouped_data['forecast_model'].unique(), palette))
            
            # 在每个子图上绘制散点图，点的大小与标准差成正比
            g.map_dataframe(sns.scatterplot, x='missing_rate', y='mean', size='std', hue='forecast_model', palette='bright', sizes=(20, 200))
            
            # 为每个预测模型绘制拟合直线并标记斜率
            for model in grouped_data['forecast_model'].unique():
                model_data = grouped_data[grouped_data['forecast_model'] == model]
                x = model_data['missing_rate']
                y = model_data['mean']
                slope, intercept = np.polyfit(x, y, 1)  # 计算斜率和截距
                color = model_color_map[model]
                line = g.axes[0, 0].plot(x, slope * x + intercept, linestyle='--', color=color)  # 绘制虚线拟合直线

            # 添加浅色网格线
            for ax in g.axes.flat:
                ax.grid(True, color='lightgray', linestyle='--', alpha=0.7)
                
            # 获取图例句柄和标签
            handles, labels = g.axes[0, 0].get_legend_handles_labels()
            labels[labels.index('std')] ='standard_deviation'
            
            # 找到标准差图例的起始和结束索引
            std_start_idx = labels.index('standard_deviation')
            std_end_idx = next((i for i, label in enumerate(labels[std_start_idx:], start=std_start_idx) if label == labels[0]), len(labels))
            
            # 筛选出 3 个不同大小的标准差图例
            std_handles = handles[std_start_idx:std_end_idx]
            std_labels = labels[std_start_idx:std_end_idx]
            num_std_legend = 3
            step = len(std_handles) // num_std_legend if len(std_handles) > num_std_legend - 1 else 1
            selected_std_handles = [std_handles[0]] + [std_handles[i * step] for i in range(1, num_std_legend)] + [std_handles[-1]]
            selected_std_labels = [std_labels[0]] + [std_labels[i * step] for i in range(1, num_std_legend)] + [std_labels[-1]]
            
            # 移除原有的标准差图例
            handles = handles[:std_start_idx] + handles[std_end_idx:]
            labels = labels[:std_start_idx] + labels[std_end_idx:]
            
            # 添加筛选后的标准差图例
            handles = handles[:std_start_idx] + selected_std_handles + handles[std_start_idx + 1:]
            labels = labels[:std_start_idx] + selected_std_labels + labels[std_start_idx + 1:]

            # 手动添加图例，调整位置和布局
            # 减少列数以避免图例过于拥挤
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.45, 1.1), ncol=len(handles), frameon=False, fontsize='x-small', columnspacing=0.5)

            # 设置轴标签
            g.set_axis_labels('missing_rate', forecast_metric)

            g.figure.suptitle(f'{forecast_metric} vs forecast_model in {dataset}', y=0.95)

            # 调整子图布局，为标题和图例腾出更多空间
            plt.subplots_adjust(top=0.83)

            # 显示图形
            plt.savefig(f'{save_path}by_forecast_model/{dataset}_{forecast_metric}_vs_forecast_model_with_asymptote.png')
            plt.close()  # 关闭当前图形，避免内存泄漏
    
    
    for dataset in results['dataset'].unique():
        for forecast_metric in ['forecast_mse', 'forecast_mae']:
            result = results[results['dataset'] == dataset]
            
            # 计算每个数据集、imputer 和 missing_rate 组合下的 forecast_metric_value 均值和标准差
            grouped_data = result.groupby(['imputer','missing_rate'])[forecast_metric].agg(['mean','std']).reset_index()
            
            # 设置图片清晰度
            plt.rcParams['figure.dpi'] = 300
            
            # 创建画布，设置 margin_titles 为 True，避免图例被覆盖，同时调整宽高比
            g = sns.FacetGrid(grouped_data, margin_titles=True, height=5, aspect=1.7)
            
            # 获取调色板
            palette = sns.color_palette('bright', n_colors=len(grouped_data['imputer'].unique()))
            imputer_color_map = dict(zip(grouped_data['imputer'].unique(), palette))
            
            # 在每个子图上绘制散点图，点的大小与标准差成正比
            g.map_dataframe(sns.scatterplot, x='missing_rate', y='mean', size='std', hue='imputer', palette='bright', sizes=(20, 200))

            # 添加浅色网格线
            for ax in g.axes.flat:
                ax.grid(True, color='lightgray', linestyle='--', alpha=0.7)
                
            # 获取图例句柄和标签
            handles, labels = g.axes[0, 0].get_legend_handles_labels()
            labels[labels.index('std')] ='standard_deviation'
            
            # 找到标准差图例的起始和结束索引
            std_start_idx = labels.index('standard_deviation')
            std_end_idx = next((i for i, label in enumerate(labels[std_start_idx:], start=std_start_idx) if label == labels[0]), len(labels))
            
            # 筛选出 3 个不同大小的标准差图例
            std_handles = handles[std_start_idx:std_end_idx]
            std_labels = labels[std_start_idx:std_end_idx]
            num_std_legend = 3
            step = len(std_handles) // num_std_legend if len(std_handles) > num_std_legend - 1 else 1
            selected_std_handles = [std_handles[0]] + [std_handles[i * step] for i in range(1, num_std_legend)] + [std_handles[-1]]
            selected_std_labels = [std_labels[0]] + [std_labels[i * step] for i in range(1, num_std_legend)] + [std_labels[-1]]
            
            # 移除原有的标准差图例
            handles = handles[:std_start_idx] + handles[std_end_idx:]
            labels = labels[:std_start_idx] + labels[std_end_idx:]
            
            # 添加筛选后的标准差图例
            handles = handles[:std_start_idx] + selected_std_handles + handles[std_start_idx + 1:]
            labels = labels[:std_start_idx] + selected_std_labels + labels[std_start_idx + 1:]

            # 手动添加图例，调整位置和布局
            # 减少列数以避免图例过于拥挤
            plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.45, 1.1), ncol=len(handles), frameon=False, fontsize='x-small', columnspacing=0.5)

            # 设置轴标签
            g.set_axis_labels('missing_rate', forecast_metric)

            g.figure.suptitle(f'{forecast_metric} vs imputer in {dataset}', y=0.95)

            # 调整子图布局，为标题和图例腾出更多空间
            plt.subplots_adjust(top=0.83)

            # 显示图形
            plt.savefig(f'{save_path}by_imputer/{dataset}_{forecast_metric}_vs_imputer.png')
            plt.close()  # 关闭当前图形，避免内存泄漏