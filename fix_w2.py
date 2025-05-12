import os
import numpy as np
import pandas as pd

from src.evaluation import Evaluate


results_dir = './results'
for root, dirs, files in os.walk(results_dir):
    for dir in dirs:
        if dir == "exchange_rate":
            original_data = pd.read_csv("./Time-Series-Library/dataset/exchange_rate/exchange_rate.csv")
            true_pred = np.load("./results/exchange_rate/pred_true.npy")
            n_sample_res = true_pred.shape[0]
            n_sample_dev = original_data.shape[0] - n_sample_res
        else:
            original_data = pd.read_csv("./Time-Series-Library/dataset/illness/national_illness.csv")
            true_pred = np.load("./results/illness/pred_true.npy")
            n_sample_res = true_pred.shape[0]
            n_sample_dev = original_data.shape[0] - n_sample_res
        dev_set = original_data.iloc[:n_sample_dev, 1:].copy(deep=True)
        mean = np.mean(dev_set.values, axis=0, keepdims=True)
        std = np.std(dev_set.values, axis=0, keepdims=True)
        dev_set = (dev_set - mean) / std
        for root_, dirs_, files_ in os.walk(os.path.join(root, dir)):
            for dir_ in dirs_:
                imputed_set = np.load(os.path.join(root_, dir_, 'imputed_set.npy'))
                imputed_set = (imputed_set - mean) / std
                imputed_set = pd.DataFrame(imputed_set, columns=dev_set.columns)
                imputed_metrics = np.load(os.path.join(root_, dir_, 'imputed_metrics.npy'), allow_pickle=True).item()
                fixed_metrics = Evaluate(dev_set, imputed_set)
                assert imputed_metrics['RMSE'] == fixed_metrics['RMSE']
                print(imputed_metrics["W2_distance"], fixed_metrics["W2_distance"])
                imputed_metrics["W2_distance"] = fixed_metrics["W2_distance"]
                imputed_metrics["sliced_W2_distance"] = fixed_metrics["sliced_W2_distance"]
                np.save(os.path.join(root_, dir_, 'imputed_metrics.npy'), imputed_metrics)