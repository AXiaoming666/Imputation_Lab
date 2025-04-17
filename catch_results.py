import numpy as np
import re
import json
import sys

def load_config(file_path):
    return np.load(file_path, allow_pickle=True).item()

def extract_metrics(line):
    mse = re.search(r'mse:(\d+\.\d+)', line)
    mae = re.search(r'mae:(\d+\.\d+)', line)
    return {'mse': float(mse.group(1)), 'mae': float(mae.group(1))}

def read_second_line(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 1:
                return line.strip()
    return ""

def save_results(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    forecast_model = sys.argv[1]

    config = load_config('./config.npy')
    config['forecast_model'] = forecast_model
    imputed_metrics = load_config('./metrics.npy')

    second_line = read_second_line('./Time-Series-Library/result_long_term_forecast.txt')
    forecast_metrics = extract_metrics(second_line)

    result_dict = {
        'config': config,
        'imputed_metrics': imputed_metrics,
        'forecast_metrics': forecast_metrics
    }

    name = "{}_{}_{}_{}_{}_{}.json".format(
        config['dataset_name'],
        config['missing_rate'],
        config['imputation_method'],
        config['missing_type'],
        config['completeness_rate'],
        config['forecast_model']
    )

    save_results(f"./results/{name}", result_dict)

if __name__ == "__main__":
    main()