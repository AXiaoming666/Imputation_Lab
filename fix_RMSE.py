import argparse
import numpy as np
import random
from src.data_loading import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import *
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='exchange_rate')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--missing_type', type=str, default='MCAR')
    parser.add_argument('--completeness_rate', type=float, default=0.8)
    parser.add_argument('--imputation_method', type=str, default='XGBoost')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = {
        "seed": args.seed,
        "dataset_name": args.dataset_name,
        "missing_rate": args.missing_rate,
        "missing_type": args.missing_type,
        "completeness_rate": args.completeness_rate,
        "imputation_method": args.imputation_method,
    }
    
    name = "{}_{}_{}_{}_{}_TimesNet.json".format(
        config['dataset_name'],
        config['missing_rate'],
        config['imputation_method'],
        config['missing_type'],
        config['completeness_rate']
    )
    
    for foldername, subfolders, filenames in os.walk('./results'):
        for filename in filenames:
            if filename == name:
                with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data['imputed_metrics']['RMSE'] != data['imputed_metrics']['MAE']:
                        return


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
    
    for foldername, subfolders, filenames in os.walk('./results'):
        for filename in filenames:
            if filename == name:
                with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    data['imputed_metrics']['RMSE'] = RMSE
                with open(os.path.join(foldername, filename), 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=4)

if __name__ == '__main__':
    main()