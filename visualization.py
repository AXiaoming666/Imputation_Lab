import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

result_list = []
for foldername, subfolders, filenames in os.walk('./results'):
    for filename in filenames:
        with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
            result = json.load(file)
            result_list.append(result)

for metrics in ["RMSE", "MAE", "R2", "KL divergence", "KS statistic", "W2 distance", "Sliced Wasserstein distance", "mse", "mae"]:
    for completeness_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for missing_type in ['MCAR', 'MAR', 'F-MNAR', 'D-MNAR']:
            x_list = []
            y_list = []
            label_list = []
            for result in result_list:
                if result['config']['missing_type'] == missing_type \
                    and result['config']['completeness_rate'] == completeness_rate:
                        x = result['config']['missing_rate']
                        if metrics == "mse" or metrics == "mae":
                            y = result["forecast_metrics"][metrics]
                        else:
                            y = result["imputed_metrics"][metrics]
                        label = result['config']['imputation_method']
                        
                        if label not in label_list:
                            label_list.append(label)
                            x_list.append([])
                            y_list.append([])
                        index = label_list.index(label)
                        x_list[index].append(x)
                        y_list[index].append(y)

            plt.figure(figsize=(10, 6))
            for label_index in range(len(label_list)):
                points = sorted(zip(x_list[label_index], y_list[label_index]))
                x, y = zip(*points)
                plt.plot(x, y, label=label_list[label_index], marker='o')
            plt.xlabel('Missing rate')
            plt.ylabel(metrics)
            plt.title(f"{missing_type} with {completeness_rate*100}% complete attributes")
            plt.legend()
            plt.savefig(f"./visualization/{missing_type}_{completeness_rate}_{metrics}.png")
            plt.close()