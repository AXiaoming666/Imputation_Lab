import numpy as np
import pandas as pd
import argparse
import os


class DataLoader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.__load_dataset__(args.dataset)
        self.__normalize_data__()
        self.__extract_time_features__()
        

    def __load_dataset__(self, dataset: str) -> None:
        if dataset == 'illness':
            self.original_data = pd.read_csv("./Time-Series-Library/dataset/illness/national_illness.csv")
            true_pred = np.load("./results/illness/pred_true.npy")
            n_sample_res = true_pred.shape[0]
            self.n_sample_dev = self.original_data.shape[0] - n_sample_res
        elif dataset == 'exchange_rate':
            self.original_data = pd.read_csv("./Time-Series-Library/dataset/exchange_rate/exchange_rate.csv")
            true_pred = np.load("./results/exchange_rate/pred_true.npy")
            n_sample_res = true_pred.shape[0]
            self.n_sample_dev = self.original_data.shape[0] - n_sample_res
        
        self.dev_set = self.original_data.iloc[:self.n_sample_dev, 1:].copy(deep=True)

    def __normalize_data__(self) -> None:
        self.mean = np.mean(self.dev_set.values, axis=0, keepdims=True)
        self.std = np.std(self.dev_set.values, axis=0, keepdims=True)
        self.dev_set = (self.dev_set - self.mean) / self.std
    
    def __extract_time_features__(self) -> None:
        time_stamps = self.original_data.iloc[:self.n_sample_dev, 0]
        
        self.time_features = pd.DataFrame()
        
        self.time_features['year'] = pd.to_datetime(time_stamps).dt.year.astype(float)
        
        month = pd.to_datetime(time_stamps).dt.month.astype(float)
        day = pd.to_datetime(time_stamps).dt.day.astype(float)
        hour = pd.to_datetime(time_stamps).dt.hour.astype(float)
        minute = pd.to_datetime(time_stamps).dt.minute.astype(float)
        second = pd.to_datetime(time_stamps).dt.second.astype(float)

        self.time_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        self.time_features['month_cos'] = np.cos(2 * np.pi * month / 12)
        self.time_features['day_sin'] = np.sin(2 * np.pi * day / 31)
        self.time_features['day_cos'] = np.cos(2 * np.pi * day / 31)
        self.time_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.time_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        self.time_features['minute_sin'] = np.sin(2 * np.pi * minute / 60)
        self.time_features['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        self.time_features['second_sin'] = np.sin(2 * np.pi * second / 60)
        self.time_features['second_cos'] = np.cos(2 * np.pi * second / 60)

        for feature in self.time_features.columns:
            if np.std(self.time_features[feature]) == 0:
                self.time_features.drop(feature, axis=1, inplace=True)
            
        self.time_features = (self.time_features - np.mean(self.time_features.values, axis=0, keepdims=True)) / np.std(self.time_features.values, axis=0, keepdims=True)
        
    
    def get_dev_set(self) -> pd.DataFrame:
        return self.dev_set
    
    def get_time_features(self) -> pd.DataFrame:
        return self.time_features
    
    def set_mask(self, mask: pd.DataFrame) -> None:
        self.mask = mask
        self.missing_set = self.dev_set.copy(deep=True)
        self.missing_set[self.mask] = np.nan
    
    def get_missing_set(self) -> pd.DataFrame:
        return pd.concat([self.time_features, self.missing_set], axis=1)
    
    def fix_missing(self, imputed_data: pd.DataFrame) -> None:
        self.imputed_set = imputed_data[self.dev_set.columns].copy(deep=True)
    
    def get_imputed_set(self) -> pd.DataFrame:
        return self.imputed_set * self.std + self.mean
        
    def save_processed_data(self) -> None:
        processed_data = self.original_data.copy(deep=True)
        processed_data.iloc[:self.n_sample_dev, 1:] = self.imputed_set.copy(deep=True) * self.std + self.mean
        os.makedirs("./temp/", exist_ok=True)
        processed_data.to_csv(f"./temp/processed_data.csv", index=False)
    
    def load_imputed_data(self) -> None:
        result_path = "./results/{}/{}_{}_{}_{}/".format(
            self.args.dataset,
            self.args.missing_rate,
            self.args.missing_type,
            self.args.complete_num,
            self.args.imputer
        )
        imputed_set = np.load(result_path + "imputed_set.npy")
        imputed_set = pd.DataFrame(imputed_set, columns=self.dev_set.columns)
        imputed_set = (imputed_set - self.mean) / self.std
        self.imputed_set = imputed_set.copy(deep=True)