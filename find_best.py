import os
import json
from pathlib import Path

def read_json_files(directory='./results/'):
    """
    Read all JSON files from the specified directory.
    
    Args:
        directory (str): Path to the directory containing JSON files
    
    Returns:
        dict: Dictionary with filenames as keys and JSON content as values
    """
    results = {}
    results_dir = Path(directory)
    
    # Check if directory exists
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Directory {directory} does not exist or is not a directory")
        return results
    
    # Iterate through all files in the directory
    for file_path in results_dir.glob('*.json'):
        try:
            with open(file_path, 'r') as file:
                results[file_path.name] = json.load(file)
                print(f"Successfully loaded {file_path.name}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path.name}")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return results

if __name__ == "__main__":
    json_data = read_json_files()
    print(f"Read {len(json_data)} JSON files from ./results/")
    
    mse_min = float('inf')
    mae_min = float('inf')
    
    for filename, data in json_data.items():
        if data["config"]["imputation_method"] != "IIM":
            if data["forecast_metrics"]["mse"] < mse_min:
                mse_min = data["forecast_metrics"]["mse"]
                mse_best = filename
            if data["forecast_metrics"]["mae"] < mae_min:
                mae_min = data["forecast_metrics"]["mae"]
                mae_best = filename
    print(f"Best MSE: {mse_best} with value {mse_min}")
    print(f"Best MAE: {mae_best} with value {mae_min}")