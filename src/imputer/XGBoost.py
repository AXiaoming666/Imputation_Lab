import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
import pandas as pd

def XGBoost_impute(missing_set: pd.DataFrame) -> pd.DataFrame:
    imputed_set = missing_set.copy(deep=True)
    n_samples, n_features = imputed_set.shape

    for feature in range(n_features):
        if imputed_set.iloc[:, feature].isnull().any():
            incomplete_samples = imputed_set[imputed_set.iloc[:, feature].isnull()]
            complete_samples = imputed_set[imputed_set.iloc[:, feature].notnull()]

            X_train = complete_samples.drop(complete_samples.columns[feature], axis=1)
            y_train = complete_samples.iloc[:, feature]
            best_param = adaptive_param_learning(X_train, y_train)

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            best_model = xgb.train(best_param, dtrain, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)

            X_pred = incomplete_samples.drop(incomplete_samples.columns[feature], axis=1)
            dpred = xgb.DMatrix(X_pred)
            y_pred = best_model.predict(dpred)
            imputed_set.loc[imputed_set.iloc[:, feature].isnull(), imputed_set.columns[feature]] = y_pred
        
    return imputed_set


def adaptive_param_learning(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e-1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1e-1, log=True),
            'objective' : 'reg:squarederror',
            'eval_metric' : 'rmse',
            'device' : 'cuda',
            'tree_method' : 'hist'
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)

        cv_results = xgb.cv(
            param,
            dtrain,
            nfold=5,
            early_stopping_rounds=50,
            metrics=['rmse']
        )
        
        return cv_results['test-rmse-mean'].iloc[-1]
    

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )
    study.optimize(objective, n_trials=100, n_jobs=-1)

    best_param = study.best_params
    best_param['objective'] = 'reg:squarederror'
    best_param['eval_metric'] = 'rmse'
    best_param['device'] = 'cuda'
    best_param['tree_method'] = 'hist'

    return best_param