import os
import shap
import pycatch22
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

results_dir = "./results"

def extract_results(results_dir):
    train_X = []
    train_y = []
    
    for dataset in os.listdir(results_dir):
        ground_true = pd.read_csv("./Time-Series-Library/datasets/" + dataset + ".csv").iloc[:, 1:].values
        pred_true = np.load(os.path.join(results_dir, dataset, "pred_true.npy"))
        
        t_ground_true = ground_true.T
        t_pred_true = pred_true.T
        
        ground_true_features = []
        
        for i in range(t_ground_true.shape[0]):
            ground_true_features.append(pycatch22.catch22_all(t_ground_true[i]))
        
        ground_true_features = np.array(ground_true_features)
        
        
        impute_params = os.listdir(os.path.join(results_dir, dataset))
        for impute_parameters in tqdm(impute_params, desc=f"Processing impute parameters for {dataset}"):
            if os.path.isfile(os.path.join(results_dir, dataset, impute_parameters)):
                continue
            if impute_parameters == "ground":
                for feature in ground_true_features:
                    train_X.append(feature)
            else:
                imputed_set = np.load(os.path.join(results_dir, dataset, impute_parameters, "imputed_set.npy"))
                
                t_imputed_set = imputed_set.T

                imputed_features = []
                
                for i in range(t_imputed_set.shape[0]):
                    imputed_features.append(pycatch22.catch22_all(t_imputed_set[i]))
                    
                imputed_features = np.array(imputed_features)
                
                for feature in imputed_features:
                    train_X.append(feature)

            mae_mean = []
            
            for forecast_model in os.listdir(os.path.join(results_dir, dataset, impute_parameters)):
                if os.path.isfile(os.path.join(results_dir, dataset, impute_parameters, forecast_model)):
                    continue
                pred_set = np.load(os.path.join(results_dir, dataset, impute_parameters, forecast_model, "pred.npy"))
                t_pred_set = pred_set.T
                
                mae = []
                
                for i in range(t_pred_set.shape[0]):
                    mae.append(np.mean(np.abs(t_pred_set[i] - t_pred_true[i])))
                
                mae = np.array(mae)

                mae_mean.append(mae)
            
            mae_mean = np.array(mae_mean)
            mae_mean = np.mean(mae_mean, axis=0)

            for mae in mae_mean:
                train_y.append(mae)
            
            print(train_X[-1])
            print(train_y[-1])

    return np.array(train_X), np.array(train_y)

if __name__ == "__main__":
    train_X, train_y = extract_results(results_dir)
    
    print(train_X.shape)
    print(train_y.shape)