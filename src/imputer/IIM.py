import numpy as np
import optuna
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import pandas as pd

max_search_l = 20
max_search_k = 20

def IIM(missing_set: pd.DataFrame, alpha: float, k: int) -> pd.DataFrame:
    imputed_set = missing_set.copy(deep=True)
    complete_sample_idx = np.where(np.all(~missing_set.isna(), axis=1))[0]
    incomplete_sample_idx = np.where(np.any(missing_set.isna(), axis=1))[0]
    complete_feature_idx = np.where(np.all(~missing_set.isna(), axis=0))[0]
    incomplete_feature_idx = np.where(np.any(missing_set.isna(), axis=0))[0]
    
    
    def Learning(l: int, alpha: float) -> np.ndarray:
        phi = list()
        for sample_idx in complete_sample_idx:
            nearest_neighbours_idx = Nearest_Neighbours(sample_idx, l)
            phi.append(Linear_Regression(nearest_neighbours_idx, alpha))
        return np.array(phi)


    def Nearest_Neighbours(sample_idx: int, k: int) -> np.ndarray:
        distance = cdist(missing_set.iloc[complete_sample_idx, complete_feature_idx].values, missing_set.iloc[[sample_idx], complete_feature_idx].values, metric='euclidean').flatten()
        nearest_neighbours_idx = complete_sample_idx[np.argsort(distance)[:k]]
        return nearest_neighbours_idx
            

    def Linear_Regression(sample_idx: int, alpha: float) -> np.ndarray:
        X = missing_set.iloc[sample_idx, complete_feature_idx].values
        X = np.atleast_2d(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = missing_set.iloc[sample_idx, incomplete_feature_idx].values
        Y = np.atleast_2d(Y)
        phi = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
        return phi


    def Imputation(sample_idx: int, k: int, phi: np.ndarray) -> np.ndarray:
        nearest_neighbours_idx = Nearest_Neighbours(sample_idx, k)
        candidate = list()
        for neighbour_idx in nearest_neighbours_idx:
            candidate.append(Candidate(sample_idx, neighbour_idx, phi))
        candidate = np.array(candidate)
        data_imputed = Combine(candidate)
        return data_imputed


    def Candidate(sample_idx: int, neighbour_idx: int, phi: np.ndarray) -> np.ndarray:
        X = missing_set.iloc[[sample_idx], complete_feature_idx].values
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = X @ phi[np.where(complete_sample_idx == neighbour_idx)[0], :]
        return np.atleast_1d(Y.squeeze())


    def Combine(candidate: np.ndarray) -> np.ndarray:
        distance = np.sum(np.sqrt(np.sum((candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        weight = 1 / (distance + 1e-10)
        weight = weight / np.sum(weight, axis=0)
        data_imputed = weight @ candidate
        return data_imputed


    def Adaptive(alpha: float, k: int) -> np.ndarray:
        search_l = min(max_search_l, complete_sample_idx.shape[0])
        phi_l = list()
        for l in range(1, search_l+1):
            phi_l.append(Learning(l, alpha))
        phi_l = np.array(phi_l)
        cost = np.zeros((complete_sample_idx.shape[0], search_l))
        for sample_idx in complete_sample_idx:
            nearest_neighbours = Nearest_Neighbours(sample_idx, k)
            for neighbour_idx in nearest_neighbours:
                for l in range(1, search_l+1):
                    data_imputed = Candidate(sample_idx, neighbour_idx, phi_l[l-1])
                    cost[np.where(complete_sample_idx == neighbour_idx)[0], l-1] += np.sum((missing_set.iloc[sample_idx, incomplete_feature_idx].values - data_imputed) ** 2)
        phi = list()
        for sample_idx in complete_sample_idx:
            l_min = np.argmin(cost[np.where(complete_sample_idx == sample_idx)[0], :]) + 1
            phi.append(phi_l[l_min-1, np.where(complete_sample_idx == sample_idx)[0]].squeeze(axis=0))
        phi = np.array(phi)
        return phi
    

    phi = Adaptive(alpha, k)

    for sample_idx in incomplete_sample_idx:
        data_imputed = Imputation(sample_idx, k, phi)
        imputed_set.iloc[sample_idx, incomplete_feature_idx] = data_imputed
    return imputed_set


def IIM_adaptive(missing_set: pd.DataFrame) -> pd.DataFrame:
    complete_sample_idx = np.where(np.all(~missing_set.isna(), axis=1))[0]
    incomplete_feature_idx = np.where(np.any(missing_set.isna(), axis=0))[0]
    missing_rate = np.sum(missing_set.iloc[:, incomplete_feature_idx].isna().values) / (missing_set.shape[0] * incomplete_feature_idx.shape[0])
    data_train = missing_set.iloc[complete_sample_idx, :].copy(deep=True)
    missing_mask = np.random.choice([True, False], data_train.iloc[:, incomplete_feature_idx].shape, p=[missing_rate, 1 - missing_rate])
    
    n_complete_samples_train = np.sum(np.all(~missing_mask, axis=1))
    if n_complete_samples_train < 10:
        missing_counts = np.sum(missing_mask, axis=1)
        non_zero_indices = np.where(missing_counts > 0)[0]
        if len(non_zero_indices) > 0:
            sorted_indices = non_zero_indices[np.argsort(missing_counts[non_zero_indices])]
            num_samples_to_fix = min(10 - n_complete_samples_train, len(sorted_indices))
            for i in range(num_samples_to_fix):
                min_idx = sorted_indices[i]
                feature_idx = np.where(missing_mask[min_idx])[0]
                missing_mask[min_idx, feature_idx] = False
    
    data_train.iloc[:, incomplete_feature_idx] = np.where(missing_mask, np.nan, data_train.iloc[:, incomplete_feature_idx])
    n_complete_samples_train = np.sum(np.all(~np.isnan(data_train.values), axis=1))

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1e5, log=True)
        k = trial.suggest_int('k', 1, min(n_complete_samples_train, max_search_k))
        data_imputed = IIM(data_train, alpha, k)
        return np.sqrt(np.mean((missing_set.iloc[complete_sample_idx, :].values - data_imputed.values) ** 2))

    def optimize(n_trials):
        study = optuna.load_study(study_name='mystudy', storage='sqlite:///example.db')
        study.optimize(objective, n_trials=n_trials)

    study = optuna.create_study(study_name='mystudy', direction='minimize',
                                sampler=optuna.samplers.TPESampler(multivariate=True),
                                storage='sqlite:///example.db',
                                load_if_exists=True
                                )
    r = Parallel(n_jobs=-1)([delayed(optimize)(10) for _ in range(10)])
    alpha = study.best_params['alpha']
    k = study.best_params['k']
    optuna.delete_study(study_name="mystudy", storage="sqlite:///example.db")
    imputed_set = IIM(missing_set, alpha, k)
    return imputed_set