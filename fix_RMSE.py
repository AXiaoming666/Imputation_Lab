import numpy as np
import random
from src.data_loading import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import *
import os
import json


def main():
    for foldername, subfolders, filenames in os.walk('./results'):
        for filename in filenames:
            with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
                results = json.load(file)
                if results['imputed_metrics']['RMSE'] != results['imputed_metrics']['MAE']:
                    continue
            
            config = results['config']
            
            np.random.seed(config['seed'])
            random.seed(config['seed'])
            
            data = DataLoader(config['dataset_name'])
            MissingSimulation(data, config['missing_rate'], config['missing_type'], config['completeness_rate'])
            
            imputation_methods = {
                'forward': forward_impute,
                'mean': mean_impute,
                'knn': KNN_impute,
                'xgboost': XGBoost_impute,
                'IIM': IIM
            }
            
            impute_func = imputation_methods.get(config['imputation_method'])
            if impute_func is None:
                raise ValueError('Invalid imputation method')
            
            data_imputed = impute_func(data.get_incomplete_data())
            
            RMSE = np.sqrt(((data.get_y_train_complete() - data.separate_time_features(data_imputed)) ** 2).mean())
            
            results['imputed_metrics']['RMSE'] = RMSE
            
            with open(os.path.join(foldername, filename), 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=4)

if __name__ == '__main__':
    main()