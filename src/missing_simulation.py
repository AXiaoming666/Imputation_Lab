import numpy as np
import scipy.optimize as optimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def MissingSimulation(data, missing_rate, missing_type, completeness_rate):
    features = data.get_y_train_complete()
    time_features = data.get_X_train_complete()
    n_samples, n_features = features.shape
    n_time_features = time_features.shape[1]

    n_complete_features = max(1, int(n_features * completeness_rate))
    n_missing_features = n_features - n_complete_features
    complete_feature_idx = np.random.choice(n_features, n_complete_features, replace=False)
    missing_feature_idx = np.setdiff1d(np.arange(n_features), complete_feature_idx)

    missing_mask = np.zeros((n_samples, n_features), dtype=bool)
    missing_mask[:, complete_feature_idx] = False

    if missing_type == 'MCAR':
        missing_mask[:, missing_feature_idx] = np.random.choice([True, False], (n_samples, n_missing_features), p=[missing_rate, 1-missing_rate])
    else:
        if missing_type == 'MAR':
            missing_related_feature_idx = complete_feature_idx
        elif missing_type == 'F-MNAR':
            missing_related_feature_idx = missing_feature_idx
        else:
            missing_related_feature_idx = range(n_features)

        n_missing_related_feature = len(missing_related_feature_idx) + n_time_features
        coeffs = np.random.randn(n_missing_related_feature, n_missing_features)
        Wx = np.dot(np.concatenate([features[:, missing_related_feature_idx], time_features], axis=1), coeffs)
        coeffs /= np.std(Wx, axis=0, keepdims=True)


        intercepts = np.zeros(n_missing_features)
        for i in range(n_missing_features):
            def f(x):
                return sigmoid(np.dot(np.concatenate([features[:, missing_related_feature_idx], time_features], axis=1), coeffs[:, i]) + x).mean() - missing_rate
            intercepts[i] = optimize.bisect(f, -50, 50)
    
        ps = sigmoid(np.dot(np.concatenate([features[:, missing_related_feature_idx], time_features], axis=1), coeffs) + intercepts)
        ber = np.random.rand(n_samples, n_missing_features)
        missing_mask[:, missing_feature_idx] = ber < ps
        
    n_complete_samples = np.sum(np.all(~missing_mask, axis=1))
    if n_complete_samples < 20:
        missing_counts = np.sum(missing_mask, axis=1)
        non_zero_indices = np.where(missing_counts > 0)[0]
        if len(non_zero_indices) > 0:
            sorted_indices = non_zero_indices[np.argsort(missing_counts[non_zero_indices])]
            num_samples_to_fix = min(20 - n_complete_samples, len(sorted_indices))
            for i in range(num_samples_to_fix):
                min_idx = sorted_indices[i]
                feature_idx = np.where(missing_mask[min_idx])[0]
                missing_mask[min_idx, feature_idx] = False

    data.set_missing_mask(missing_mask)