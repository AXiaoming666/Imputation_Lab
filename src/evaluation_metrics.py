import numpy as np
import ot
from scipy.stats import ks_2samp
from concurrent.futures import ProcessPoolExecutor


def data2distribution(data):
    data = data - data.min() + 1e-10
    data /= data.sum()
    return data


def Evaluate(data_original, data_imputed):
    assert data_original.shape == data_imputed.shape, "data and data_imputed must have the same shape"
    assert np.any(np.isnan(data_imputed)) == False, "data_imputed must not contain any NaN values"

    metrics = {
        "RMSE": calculate_RMSE(data_original, data_imputed),
        "MAE": calculate_MAE(data_original, data_imputed),
        "R2": calculate_R2(data_original, data_imputed),
        "KL divergence": calculate_KL_divergence(data_original, data_imputed),
        "KS statistic": calculate_KS_statistic(data_original, data_imputed),
        "W2 distance": calculate_2D_W2_distance(data_original, data_imputed),
        "Sliced Wasserstein distance": calculate_sliced_wasserstein_distance(data_original, data_imputed)
    }
    return metrics


def calculate_RMSE(data_original, data_imputed):
    return np.sqrt(((data_original - data_imputed) ** 2).mean())


def calculate_MAE(data_original, data_imputed):
    return np.abs(data_original - data_imputed).mean()


def calculate_R2(data_original, data_imputed):
    return 1 - (((data_original - data_imputed) ** 2).sum() / ((data_original - data_original.mean()) ** 2).sum())


def calculate_KL_divergence(data_original, data_imputed):
    KL_divergence = 0
    for i in range(data_original.shape[1]):
        feature = data2distribution(data_original[:, i])
        imputed_feature = data2distribution(data_imputed[:, i])
        KL_divergence += (feature * np.log(feature / imputed_feature)).sum()
    return KL_divergence / data_original.shape[1]


def calculate_KS_statistic(data, data_imputed):
    statistic = 0
    for i in range(data.shape[1]):
        statistic += ks_2samp(data[:, i], data_imputed[:, i])[0]
    return statistic / data.shape[1]


def calculate_W2_distance(data_original, data_imputed):
    distribution_original = data2distribution(data_original)
    distribution_imputed = data2distribution(data_imputed)
    M = ot.dist(distribution_original[:, None], distribution_imputed[:, None])
    w2_distance = ot.emd2(distribution_original, distribution_imputed, M, numItermax=1000000000)
    return w2_distance


def calculate_2D_W2_distance(data_original, data_imputed):
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(calculate_W2_distance, 
                                    [data_original[:, i] for i in range(data_original.shape[1])], 
                                    [data_imputed[:, i] for i in range(data_imputed.shape[1])]))
    
    w2_distance = sum(results) / data_original.shape[1]
    return w2_distance


def calculate_sliced_wasserstein_distance(data_original, data_imputed, n_slices=100):
    projections = np.random.randn(n_slices, data_original.shape[1])
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    data_original = (data_original - data_original.mean(axis=0)) / data_original.std(axis=0)
    data_imputed = (data_imputed - data_imputed.mean(axis=0)) / data_imputed.std(axis=0)

    projection_pairs = [(data_original @ proj, data_imputed @ proj) for proj in projections]
    args = list(zip(*projection_pairs))

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(calculate_W2_distance, *args))
    
    sliced_wasserstein_distance = sum(results) / n_slices

    return sliced_wasserstein_distance