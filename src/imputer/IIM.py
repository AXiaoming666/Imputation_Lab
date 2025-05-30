import numpy as np
import optuna
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import pandas as pd

max_search_l = 20
max_search_k = 20

def IIM(missing_set: np.array, alpha: float, k: int) -> np.array:
    imputed_set = missing_set.copy()
    complete_sample_idx = np.where(np.all(~np.isnan(imputed_set), axis=1))[0]
    incomplete_sample_idx = np.where(np.any(np.isnan(imputed_set), axis=1))[0]
    complete_feature_idx = np.where(np.all(~np.isnan(imputed_set), axis=0))[0]
    incomplete_feature_idx = np.where(np.any(np.isnan(imputed_set), axis=0))[0]
    
    
    def Learning(l: int, alpha: float) -> np.array:
        phi = list()
        for sample_idx in complete_sample_idx:
            nearest_neighbours_idx = Nearest_Neighbours(sample_idx, l)
            phi.append(Linear_Regression(nearest_neighbours_idx, alpha))
        return np.array(phi)


    def Nearest_Neighbours(sample_idx: int, k: int) -> np.array:
        distance = cdist(imputed_set[complete_sample_idx, :][:, complete_feature_idx], imputed_set[sample_idx, complete_feature_idx].reshape(1, -1), metric='euclidean').flatten()
        nearest_neighbours_idx = complete_sample_idx[np.argsort(distance)[:k]]
        return nearest_neighbours_idx
            

    def Linear_Regression(sample_idx: int, alpha: float) -> np.array:
        X = np.atleast_2d(imputed_set[sample_idx, :][:, complete_feature_idx])
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = imputed_set[sample_idx, :][:, incomplete_feature_idx]
        phi = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
        return phi


    def Imputation(sample_idx: int, k: int, phi: np.array) -> np.array:
        nearest_neighbours_idx = Nearest_Neighbours(sample_idx, k)
        candidate = list()
        for neighbour_idx in nearest_neighbours_idx:
            candidate.append(Candidate(sample_idx, neighbour_idx, phi))
        candidate = np.array(candidate)
        data_imputed = Combine(candidate)
        return data_imputed


    def Candidate(sample_idx: int, neighbour_idx: int, phi: np.array) -> np.array:
        X = np.atleast_2d(imputed_set[sample_idx, complete_feature_idx])
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y = X @ phi[np.where(complete_sample_idx == neighbour_idx)[0], :]
        return np.atleast_1d(Y.squeeze())


    def Combine(candidate: np.array) -> np.array:
        distance = np.sum(np.sqrt(np.sum((candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
        weight = 1 / (distance + 1e-10)
        weight = weight / np.sum(weight, axis=0)
        data_imputed = weight @ candidate
        return data_imputed


    def Adaptive(alpha: float, k: int) -> np.array:
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
                    cost[np.where(complete_sample_idx == neighbour_idx)[0], l-1] += np.sum((imputed_set[sample_idx, incomplete_feature_idx] - data_imputed) ** 2)
        phi = list()
        for sample_idx in complete_sample_idx:
            l_min = np.argmin(cost[np.where(complete_sample_idx == sample_idx)[0], :]) + 1
            phi.append(phi_l[l_min-1, np.where(complete_sample_idx == sample_idx)[0]].squeeze(axis=0))
        phi = np.array(phi)
        return phi
    

    phi = Adaptive(alpha, k)

    for sample_idx in incomplete_sample_idx:
        data_imputed = Imputation(sample_idx, k, phi)
        for feature_idx in incomplete_feature_idx:
            if np.isnan(imputed_set[sample_idx, feature_idx]):
                imputed_set[sample_idx, feature_idx] = data_imputed[np.where(incomplete_feature_idx == feature_idx)[0]].squeeze()
    return imputed_set


def IIM_adaptive(missing_set: pd.DataFrame) -> pd.DataFrame:
    imputed_set = missing_set.to_numpy(copy=True)
    complete_sample_idx = np.where(np.all(~np.isnan(imputed_set), axis=1))[0]
    incomplete_feature_idx = np.where(np.any(np.isnan(imputed_set), axis=0))[0]
    missing_rate = np.sum(np.isnan(imputed_set[:, incomplete_feature_idx])) / (imputed_set.shape[0] * incomplete_feature_idx.shape[0])
    data_train = imputed_set[complete_sample_idx, :].copy()
    missing_mask = np.random.choice([True, False], data_train[:, incomplete_feature_idx].shape, p=[missing_rate, 1 - missing_rate])
    
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
    
    data_train[:, incomplete_feature_idx] = np.where(missing_mask, np.nan, data_train[:, incomplete_feature_idx])
    n_complete_samples_train = np.sum(np.all(~missing_mask, axis=1))

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1e5, log=True)
        k = trial.suggest_int('k', 1, min(n_complete_samples_train, max_search_k))
        data_imputed = IIM(data_train, alpha, k)
        return np.sqrt(np.mean((imputed_set[complete_sample_idx, :] - data_imputed) ** 2))

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
    imputed_set = IIM(imputed_set, alpha, k)
    imputed_set = pd.DataFrame(imputed_set, columns=missing_set.columns)
    return imputed_set