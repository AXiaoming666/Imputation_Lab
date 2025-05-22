import numpy as np
import pandas as pd
import os

results_dir = "./results"

results = {
    "dataset": [],
    "missing_rate": [],
    "missing_type": [],
    "complete_rate": [],
    "imputer": [],
    "forecast_model": [],
    "impute_rmse": [],
    "impute_mae": [],
    "impute_r2": [],
    "impute_kl_divergence": [],
    "impute_ks_statistic": [],
    "impute_w2_distance": [],
    "impute_sliced_kl_divergence": [],
    "impute_sliced_ks_statistic": [],
    "impute_sliced_w2_distance": [],
    "forecast_mse": [],
    "forecast_mae": [],
}

for dataset in os.listdir(results_dir):
    if os.path.isfile(os.path.join(results_dir, dataset)):
        continue
    pred_true = np.load(os.path.join(results_dir, dataset, "pred_true.npy"))
    feature_num = pred_true.shape[1]
    for impute_parameters in os.listdir(os.path.join(results_dir, dataset)):
        if os.path.isfile(os.path.join(results_dir, dataset, impute_parameters)):
            continue
        if impute_parameters == "ground":
            impute_parameter = {"missing_rate": 0, "missing_type": None, "complete_rate": None, "imputer": None}
            imputed_metrics = {"RMSE": None, 
                               "MAE": None,
                               "R2": None,
                               "KL_divergence": None,
                               "KS_statistic": None,
                               "W2_distance": None,
                               "sliced_KL_divergence": None,
                               "sliced_KS_statistic": None,
                               "sliced_W2_distance": None}
        else:
            impute_parameter = {
                "missing_rate": float(impute_parameters.split("_")[0]),
                "missing_type": impute_parameters.split("_")[1],
                "complete_rate": float(impute_parameters.split("_")[2]) / feature_num,
                "imputer": impute_parameters.split("_")[3]
            }
            imputed_metrics = np.load(os.path.join(results_dir, dataset, impute_parameters, "imputed_metrics.npy"), allow_pickle=True).item()
        for forecast_model in os.listdir(os.path.join(results_dir, dataset, impute_parameters)):
            if os.path.isfile(os.path.join(results_dir, dataset, impute_parameters, forecast_model)):
                continue
            pred_metrics = np.load(os.path.join(results_dir, dataset, impute_parameters, forecast_model, "pred_metrics.npy"), allow_pickle=True).item()
            results["dataset"].append(dataset)
            results["missing_rate"].append(impute_parameter["missing_rate"])
            results["missing_type"].append(impute_parameter["missing_type"])
            results["complete_rate"].append(impute_parameter["complete_rate"])
            results["imputer"].append(impute_parameter["imputer"])
            results["forecast_model"].append(forecast_model)
            results["impute_rmse"].append(imputed_metrics["RMSE"])
            results["impute_mae"].append(imputed_metrics["MAE"])
            results["impute_r2"].append(imputed_metrics["R2"])
            results["impute_kl_divergence"].append(imputed_metrics["KL_divergence"])
            results["impute_ks_statistic"].append(imputed_metrics["KS_statistic"])
            results["impute_w2_distance"].append(imputed_metrics["W2_distance"])
            results["impute_sliced_kl_divergence"].append(imputed_metrics["sliced_KL_divergence"])
            results["impute_sliced_ks_statistic"].append(imputed_metrics["sliced_KS_statistic"])
            results["impute_sliced_w2_distance"].append(imputed_metrics["sliced_W2_distance"])
            results["forecast_mse"].append(pred_metrics["mse"])
            results["forecast_mae"].append(pred_metrics["mae"])
            
results = pd.DataFrame(results)
results.to_csv("./results.csv", index=False)