import argparse
import random
import numpy as np
import subprocess
import os
import pandas as pd
import shutil

from src.data_loader import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import impute
from src.evaluation import Evaluate


def TSLib(args: argparse.Namespace) -> subprocess.CompletedProcess:
    command = ["/home/cxz/anaconda3/envs/TSLib-env/bin/python", "-u",
               "run.py",
               "--task_name", "long_term_forecast",
               "--is_training", "1",
               "--model_id", "0",
               "--model", args.forecast_model,
               "--data", "custom",
               "--features", "M", 
               "--factor", "3",
               "--des", "\'Exp\'",
               "--itr", "1",
               "--inverse",
               "--d_model", "64",
               "--d_ff", "64"]
    
    if args.missing_rate == 0:
        if args.dataset == "exchange_rate":
            command += ["--root_path", "./dataset/exchange_rate",
                        "--data_path", "exchange_rate.csv"]
        elif args.dataset == "illness":
            command += ["--root_path", "./dataset/illness",
                        "--data_path", "national_illness.csv"]
        elif args.dataset == "traffic":
            command += ["--root_path", "./dataset/traffic",
                        "--data_path", "traffic.csv"]
    else:
        command += ["--root_path", "../temp/",
                    "--data_path", "processed_data.csv"]
    
    if args.dataset == "exchange_rate":
        command += ["--seq_len", "96",
                    "--label_len", "48",
                    "--pred_len", "96",
                    "--enc_in", "8",
                    "--dec_in", "8",
                    "--c_out", "8"]
    elif args.dataset == "illness":
        command += ["--seq_len", "36",
                    "--label_len", "18",
                    "--pred_len", "24",
                    "--enc_in", "7",
                    "--dec_in", "7",
                    "--c_out", "7"]
    elif args.dataset == "traffic":
        command += ["--seq_len", "96",
                    "--label_len", "48",
                    "--pred_len", "96",
                    "--enc_in", "862",
                    "--dec_in", "862",
                    "--c_out", "862"]
    
    if args.forecast_model == "TimesNet":
        command += ["--e_layers", "2",
                    "--d_layers", "1",
                    "--top_k", "5"]
    elif args.forecast_model == "iTransformer":
        command += ["--e_layers", "4",
                    "--d_layers", "1",
                    "--batch_size", "16",
                    "--learning_rate", "0.001"]
    elif args.forecast_model == "TimeXer":
        command += ["--e_layers", "3",
                    "--batch_size", "16",
                    "--learning_rate", "0.001"]
    return subprocess.run(command, check=True, cwd="./Time-Series-Library")

def TSLib_output2TS(TSLib_output: np.ndarray) -> np.ndarray:
    TS = []
    for i in range(TSLib_output.shape[0]):
        TS.append(TSLib_output[i, 0, :])
    for i in range(TSLib_output.shape[1] - 1):
        TS.append(TSLib_output[-1, i + 1, :])
    return np.array(TS)

def get_TSLib_output() -> tuple[np.ndarray, np.ndarray, dict]:
    base_dir = "./Time-Series-Library/results/"
    entries = os.listdir(base_dir)
    subdirs = [entry for entry in entries if os.path.isdir(os.path.join(base_dir, entry))]
    subdir = os.path.join(base_dir, subdirs[0])
    metrics = np.load(subdir + "/metrics.npy")
    pred = np.load(subdir + "/pred.npy")
    true = np.load(subdir + "/true.npy")
    metrics = {"pred": {"mae": metrics[0],
               "mse": metrics[1],
               "rmse": metrics[2],
               "mape": metrics[3],
               "mspe": metrics[4]}}
    pred = TSLib_output2TS(pred)
    true = TSLib_output2TS(true)
    return pred, true, metrics

def save_results(args: argparse.Namespace, imputed_metrics: dict|None, imputed_set: pd.DataFrame|None) -> None:
    pred, pred_true, pred_metrics = get_TSLib_output()
    
    result_path = f"./results/{args.dataset}/"
    os.makedirs(result_path, exist_ok=True)
    
    if args.missing_rate == 0:
        np.save(result_path + "pred_true.npy", pred_true)
        result_path += f"ground/{args.forecast_model}/"
        os.makedirs(result_path, exist_ok=True)
        np.save(result_path + "pred.npy", pred)
        np.save(result_path + "metrics.npy", pred_metrics)
    else:
        result_path = "./results/{}/{}_{}_{}_{}/".format(
            args.dataset,
            args.missing_rate,
            args.missing_type,
            args.complete_num if args.dataset != "traffic" else args.complete_rate,
            args.imputer
        )
        os.makedirs(result_path, exist_ok=True)
        imputed_set.to_csv(result_path + "imputed_set.csv", index=False)
        metrics = {"impute": imputed_metrics} | pred_metrics
        np.save(result_path + "metrics.npy", metrics)
        np.save(result_path + "pred.npy", pred)
    
def delete_temp() -> None:
    temp_dir = "./temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    else:
        print("Temporary directory does not exist.")
    
    results_dir = "./Time-Series-Library/checkpoints/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    else:
        print("Checkpoints directory does not exist.")
    
    results_dir = "./Time-Series-Library/results/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    else:
        print("Results directory does not exist.")
        
    results_dir = "./Time-Series-Library/test_results/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    else:
        print("Test results directory does not exist.")
        
    if os.path.exists("./Time-Series-Library/result_long_term_forecast.txt"):
        os.remove("./Time-Series-Library/result_long_term_forecast.txt")
    else:
        print("result_long_term_forecast.txt does not exist.")

if __name__ == "__main__":
    fix_seed = 42
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="exchange_rate", choices=["exchange_rate", "illness", "traffic"], help="Name of dataset")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Missing rate in correpted features")
    parser.add_argument("--missing_type", type=str, default="MCAR", choices=["MCAR", "MAR", "F-MNAR", "D-MNAR"], help="Missing type")
    parser.add_argument("--complete_num", type=int, default=1, help="Number of features not to be corrupted")
    parser.add_argument("--complete_rate", type=float, default=0.9, help="Complete rate of features")
    parser.add_argument("--imputer", type=str, default="Forward", choices=["Mean", "Forward", "KNN", "XGBoost", "IIM"], help="Imputation method")
    parser.add_argument("--forecast_model", type=str, default="TimesNet", choices=["TimesNet", "iTransformer", "TimeXer"], help="Forecasting model")
    
    args = parser.parse_args()
    
    if args.missing_rate == 0:
        TSLib(args)
        
        save_results(args, None, None)
        
        delete_temp()
    else:
        data = DataLoader(args)
        
        MissingSimulation(data, args)
        
        missing_set = data.get_missing_set()
        
        imputed_set = impute(missing_set, method=args.imputer)
        
        data.fix_missing(imputed_set)
        
        imputed_metrics = Evaluate(data.get_dev_set(), imputed_set)
        
        data.save_processed_data()
        
        TSLib(args)
        
        save_results(args, imputed_metrics, data.get_imputed_set())
        
        delete_temp()