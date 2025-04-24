import numpy as np
import scipy.optimize as optimize
import argparse
from src.data_loader import DataLoader


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def MissingSimulation(data: DataLoader, args: argparse.Namespace) -> None:
    features = data.get_dev_set()
    time_features = data.get_time_features()
    n_samples, n_features = features.shape
    n_time_features = time_features.shape[1]
    
    if args.dataset == "traffic":
        n_complete_features = int(n_features * args.completeness_rate)
    else:
        n_complete_features = args.complete_num
        assert n_complete_features <= n_features, f"Number of complete features ({n_complete_features}) must be less than or equal to total number of features ({n_features})."
    
    n_missing_features = n_features - n_complete_features
    complete_feature_idx = np.random.choice(n_features, n_complete_features, replace=False)
    missing_feature_idx = np.setdiff1d(np.arange(n_features), complete_feature_idx)

    mask = features.copy(deep=True).astype(bool)
    mask.iloc[:, complete_feature_idx] = False

    missing_type = args.missing_type
    missing_rate = args.missing_rate

    if missing_type == 'MCAR':
        mask.iloc[:, missing_feature_idx] = np.random.choice([True, False], (n_samples, n_missing_features), p=[missing_rate, 1-missing_rate])
    else:
        if missing_type == 'MAR':
            missing_related_feature_idx = complete_feature_idx
        elif missing_type == 'F-MNAR':
            missing_related_feature_idx = missing_feature_idx
        else:
            missing_related_feature_idx = range(n_features)

        n_missing_related_feature = len(missing_related_feature_idx) + n_time_features
        coeffs = np.random.randn(n_missing_related_feature, n_missing_features)
        Wx = np.dot(np.concatenate([features.iloc[:, missing_related_feature_idx], time_features], axis=1), coeffs)
        coeffs /= np.std(Wx, axis=0, keepdims=True)

        intercepts = np.zeros(n_missing_features)
        for i in range(n_missing_features):
            def f(x):
                return sigmoid(np.dot(np.concatenate([features.iloc[:, missing_related_feature_idx], time_features], axis=1), coeffs[:, i]) + x).mean() - missing_rate
            intercepts[i] = optimize.bisect(f, -50, 50)
    
        ps = sigmoid(np.dot(np.concatenate([features.iloc[:, missing_related_feature_idx], time_features], axis=1), coeffs) + intercepts)
        ber = np.random.rand(n_samples, n_missing_features)
        mask.iloc[:, missing_feature_idx] = ber < ps
        
    n_complete_samples = np.sum((~mask).all(axis=1))
    if n_complete_samples < 20:
        missing_counts = np.sum(mask, axis=1)
        non_zero_indices = np.where(missing_counts > 0)[0]
        if len(non_zero_indices) > 0:
            sorted_indices = non_zero_indices[np.argsort(missing_counts[non_zero_indices])]
            num_samples_to_fix = min(20 - n_complete_samples, len(sorted_indices))
            for i in range(num_samples_to_fix):
                min_idx = sorted_indices[i]
                feature_idx = np.where(mask.iloc[min_idx])[0]
                mask.iloc[min_idx, feature_idx] = False

    data.set_mask(mask)