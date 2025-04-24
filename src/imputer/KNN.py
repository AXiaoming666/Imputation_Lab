import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


def KNN_impute(missing_set: pd.DataFrame) -> pd.DataFrame:
    imputed_set = missing_set.copy(deep=True)
    complete_features = np.where(np.all(~imputed_set.isna(), axis=0))[0]
    incomplete_features = np.where(np.any(imputed_set.isna(), axis=0))[0]
    complete_samples = np.where(np.all(~imputed_set.isna(), axis=1))[0]
    incomplete_samples = np.where(np.any(imputed_set.isna(), axis=1))[0]

    X_train = imputed_set.iloc[complete_samples, complete_features]
    y_train = imputed_set.iloc[complete_samples, incomplete_features]
    X_pred = imputed_set.iloc[incomplete_samples, complete_features]

    best_params = adaptive_param_learning(X_train, y_train)
    knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'], p=best_params['p'])
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_pred)
    for i in range(len(incomplete_samples)):
        for j in range(len(incomplete_features)):
            if pd.isna(imputed_set.iloc[incomplete_samples[i], incomplete_features[j]]):
                imputed_set.iloc[incomplete_samples[i], incomplete_features[j]] = y_pred[i, j]

    return imputed_set


def adaptive_param_learning(X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    param_grid = {
        "n_neighbors" : range(1, min(X_train.shape[0], 11)),
        "weights" : ['uniform', 'distance'],
        "p" : [1, 2]
    }
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_