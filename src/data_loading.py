import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, dataset_name):
        if dataset_name == 'illness':
            self.df_data = pd.read_csv(f"./Time-Series-Library/dataset/illness/national_illness.csv")
            self.n_sample_dev = 773
        elif dataset_name == 'exchange_rate':
            self.df_data = pd.read_csv(f"./Time-Series-Library/dataset/exchange_rate/exchange_rate.csv")
            self.n_sample_dev = 6071
        elif dataset_name == 'traffic':
            self.df_data = pd.read_csv(f"./Time-Series-Library/dataset/traffic/traffic.csv")
            self.n_sample_dev = 14036
        else:
            raise ValueError("Invalid dataset name")
        self.mean = np.mean(self.df_data.iloc[:, 1:].values, axis=0, keepdims=True)
        self.std = np.std(self.df_data.iloc[:, 1:].values, axis=0, keepdims=True)
        self.df_data.iloc[:, 1:] = (self.df_data.iloc[:, 1:] - self.mean) / self.std
        
        self.n_sample, self.n_feature = self.df_data.shape
        self.n_sample_res = self.n_sample - self.n_sample_dev

        self.time_features = self.extract_time_features(self.df_data.iloc[:, 0])
        self.time_features_mean = np.mean(self.time_features.values, axis=0, keepdims=True)
        self.time_features_std = np.std(self.time_features.values, axis=0, keepdims=True)
        self.time_features = (self.time_features - self.time_features_mean) / self.time_features_std

        self.n_time_features = self.time_features.shape[1]

        self.X_train_complete = self.time_features.iloc[:self.n_sample_dev, :].values
        self.y_train_complete = self.df_data.iloc[:self.n_sample_dev, 1:].values
        self.X_test = self.time_features.iloc[self.n_sample_dev:, :].values
        self.y_test = self.df_data.iloc[self.n_sample_dev:, 1:].values


    def extract_time_features(self, time_stamps):
        time_features = pd.DataFrame()

        month = pd.to_datetime(time_stamps).dt.month.astype(float)
        day = pd.to_datetime(time_stamps).dt.day.astype(float)
        hour = pd.to_datetime(time_stamps).dt.hour.astype(float)
        minute = pd.to_datetime(time_stamps).dt.minute.astype(float)
        second = pd.to_datetime(time_stamps).dt.second.astype(float)

        time_features['year'] = pd.to_datetime(time_stamps).dt.year.astype(float)
        time_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        time_features['month_cos'] = np.cos(2 * np.pi * month / 12)
        time_features['day_sin'] = np.sin(2 * np.pi * day / 31)
        time_features['day_cos'] = np.cos(2 * np.pi * day / 31)
        time_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        time_features['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        time_features['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        time_features['second_sin'] = np.sin(2 * np.pi * second / 60)
        time_features['second_cos'] = np.cos(2 * np.pi * second / 60)

        for feature in time_features.columns:
            if np.std(time_features[feature]) == 0:
                time_features = time_features.drop(feature, axis=1)

        return time_features
    

    def get_y_train_complete(self):
        return self.y_train_complete
    

    def get_X_train_complete(self):
        return self.X_train_complete
    

    def set_missing_mask(self, missing_mask):
        self.missing_mask = missing_mask
        self.y_train_incomplete = self.y_train_complete.copy()
        self.y_train_incomplete[self.missing_mask] = np.nan
        self.incomplete_data = np.concatenate([self.X_train_complete, self.y_train_incomplete], axis=1)


    def get_incomplete_data(self):
        return self.incomplete_data
    

    def separate_time_features(self, data):
        features = data[:, self.n_time_features:]
        return features
    
    def save_imputed_data(self, imputed_train_data):
        imputed_data = self.df_data.copy()
        imputed_data.iloc[:self.n_sample_dev, 1:] = imputed_train_data
        imputed_data.iloc[:, 1:] = (imputed_data.iloc[:, 1:] * self.std) + self.mean
        imputed_data.to_csv(f"./imputed_data.csv", index=False)