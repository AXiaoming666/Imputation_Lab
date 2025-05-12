import os
import numpy as np
import pandas as pd
import torch
import geomloss
from concurrent.futures import ProcessPoolExecutor

def data2distribution(data):
    data = data - data.min() + 1e-10
    data /= data.sum()
    return data

def Evaluate(dev_set: pd.DataFrame, imputed_set: pd.DataFrame) -> dict:
    imputed_set = imputed_set[dev_set.columns]
    
    assert dev_set.shape == imputed_set.shape, "data and imputed_set must have the same shape"
    assert not np.any(np.isnan(imputed_set)), "imputed_set must not contain any NaN values"
    
    w2_distance = calculate_2D_W2_distance(dev_set.values, imputed_set.values)
    sliced_W2_distance = calculate_sliced_metrics(dev_set.values, imputed_set.values)

    metrics = {
        "W2_distance": w2_distance,
        "sliced_W2_distance": sliced_W2_distance
    }
    return metrics

def calculate_2D_W2_distance(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(calculate_W2_distance, 
                                    [dev_set[:, i] for i in range(dev_set.shape[1])], 
                                    [imputed_set[:, i] for i in range(imputed_set.shape[1])]))
    
    w2_distance = sum(results) / dev_set.shape[1]
    return w2_distance

def calculate_W2_distance(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    dev_set_distribution = data2distribution(dev_set).reshape(-1, 1)
    imputed_set_distribution = data2distribution(imputed_set).reshape(-1, 1)
    
    OTLoss = geomloss.SamplesLoss(
        loss="sinkhorn", p=2,
        cost=geomloss.utils.squared_distances,
        blur=0.1**(1/2), backend="tensorized"
    )
    pW = OTLoss(torch.from_numpy(dev_set_distribution), torch.from_numpy(imputed_set_distribution))
    return pW.item()

def calculate_sliced_metrics(dev_set: np.ndarray, imputed_set: np.ndarray, n_slices: int = 100) -> tuple:
    projections = np.random.randn(n_slices, dev_set.shape[1])
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    dev_set = (dev_set - dev_set.mean(axis=0)) / dev_set.std(axis=0)
    imputed_set = (imputed_set - imputed_set.mean(axis=0)) / imputed_set.std(axis=0)

    projection_pairs = [(dev_set @ proj, imputed_set @ proj) for proj in projections]
    args = list(zip(*projection_pairs))

    with ProcessPoolExecutor(max_workers=8) as executor:
        W2_distances = list(executor.map(calculate_W2_distance, *args))
    
    sliced_W2_distance = sum(W2_distances) / n_slices

    return sliced_W2_distance


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
                print(imputed_metrics["W2_distance"], fixed_metrics["W2_distance"])
                imputed_metrics["W2_distance"] = fixed_metrics["W2_distance"]
                imputed_metrics["sliced_W2_distance"] = fixed_metrics["sliced_W2_distance"]
                np.save(os.path.join(root_, dir_, 'imputed_metrics.npy'), imputed_metrics)