import numpy as np
import pandas as pd
import torch
import geomloss
from scipy.stats import ks_2samp
from concurrent.futures import ProcessPoolExecutor


def data2distribution(data):
    data = data - data.min() + 1e-10
    data /= data.sum()
    return data


def Evaluate(dev_set: pd.DataFrame, imputed_set: pd.DataFrame) -> dict:
    imputed_set = imputed_set[dev_set.columns]
    
    assert dev_set.shape == imputed_set.shape, "data and imputed_set must have the same shape"
    assert not np.any(np.isnan(imputed_set)), "imputed_set must not contain any NaN values"
    
    rmse = calculate_RMSE(dev_set.values, imputed_set.values)
    mae = calculate_MAE(dev_set.values, imputed_set.values)
    r2 = calculate_R2(dev_set.values, imputed_set.values)
    kl_divergence = calculate_2D_KL_divergence(dev_set.values, imputed_set.values)
    ks_statistic = calculate_2D_KS_statistic(dev_set.values, imputed_set.values)
    w2_distance = calculate_2D_W2_distance(dev_set.values, imputed_set.values)
    sliced_KL_divergence, sliced_KS_statistic, sliced_W2_distance = calculate_sliced_metrics(dev_set.values, imputed_set.values)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "KL_divergence": kl_divergence,
        "KS_statistic": ks_statistic,
        "W2_distance": w2_distance,
        "sliced_KL_divergence": sliced_KL_divergence,
        "sliced_KS_statistic": sliced_KS_statistic,
        "sliced_W2_distance": sliced_W2_distance
    }
    return metrics


def calculate_RMSE(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    return np.sqrt(((dev_set - imputed_set) ** 2).mean())


def calculate_MAE(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    return np.abs(dev_set - imputed_set).mean()


def calculate_R2(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    return 1 - (((dev_set - imputed_set) ** 2).sum() / ((dev_set - dev_set.mean()) ** 2).sum())


def calculate_KL_divergence(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    dev_distribution = data2distribution(dev_set)
    imputed_distribution = data2distribution(imputed_set)
    return (dev_distribution * np.log(dev_distribution / imputed_distribution)).sum()


def calculate_2D_KL_divergence(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    KL_divergence = 0
    for i in range(dev_set.shape[1]):
        KL_divergence += calculate_KL_divergence(dev_set[:, i], imputed_set[:, i])
    return KL_divergence / dev_set.shape[1]


def calculate_KS_statistic(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    return ks_2samp(dev_set, imputed_set, method='asymp')[0]


def calculate_2D_KS_statistic(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    statistic = 0
    for i in range(dev_set.shape[1]):
        statistic += calculate_KS_statistic(dev_set[:, i], imputed_set[:, i])
    return statistic / dev_set.shape[1]


def calculate_W2_distance(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    dev_set_distribution = data2distribution(dev_set).reshape(-1, 1)
    imputed_set_distribution = data2distribution(imputed_set).reshape(-1, 1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OTLoss = geomloss.SamplesLoss(
        loss="sinkhorn", p=2,
        cost=geomloss.utils.squared_distances,
        blur=0.01**(1/2), backend="tensorized"
    )
    OTLoss.to(device)
    dev_set_tensor = torch.from_numpy(dev_set_distribution).to(device)
    imputed_set_tensor = torch.from_numpy(imputed_set_distribution).to(device)
    
    pW = OTLoss(dev_set_tensor, imputed_set_tensor)
    return pW.cpu().item()


def calculate_2D_W2_distance(dev_set: np.ndarray, imputed_set: np.ndarray) -> float:
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(calculate_W2_distance, 
                                    [dev_set[:, i] for i in range(dev_set.shape[1])], 
                                    [imputed_set[:, i] for i in range(imputed_set.shape[1])]))
    
    w2_distance = sum(results) / dev_set.shape[1]
    return w2_distance


def calculate_sliced_metrics(dev_set: np.ndarray, imputed_set: np.ndarray, n_slices: int = 100) -> tuple:
    projections = np.random.randn(n_slices, dev_set.shape[1])
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    dev_set = (dev_set - dev_set.mean(axis=0)) / dev_set.std(axis=0)
    imputed_set = (imputed_set - imputed_set.mean(axis=0)) / imputed_set.std(axis=0)

    projection_pairs = [(dev_set @ proj, imputed_set @ proj) for proj in projections]
    args = list(zip(*projection_pairs))

    with ProcessPoolExecutor(max_workers=8) as executor:
        KL_divergences = list(executor.map(calculate_KL_divergence, *args))
        KS_statistics = list(executor.map(calculate_KS_statistic, *args))
        W2_distances = list(executor.map(calculate_W2_distance, *args))
    
    sliced_KL_divergence = sum(KL_divergences) / n_slices
    sliced_KS_statistic = sum(KS_statistics) / n_slices
    sliced_W2_distance = sum(W2_distances) / n_slices

    return sliced_KL_divergence, sliced_KS_statistic, sliced_W2_distance