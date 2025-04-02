import numpy as np
import random
from src.data_loading import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import *
from src.evaluation_metrics import Evaluate


def main(config):
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
    data.save_imputed_data(data.separate_time_features(data_imputed))
    
    metrics = Evaluate(data.get_y_train_complete(), data.separate_time_features(data_imputed))
    
    np.save("./config.npy", config)
    np.save("./metrics.npy", metrics)