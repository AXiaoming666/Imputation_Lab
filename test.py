import os
import json

undo = 0
for foldername, subfolders, filenames in os.walk('./results'):
    for filename in filenames:
        with open(os.path.join(foldername, filename), 'r', encoding='utf-8') as file:
            results = json.load(file)
            if results['imputed_metrics']['RMSE'] == results['imputed_metrics']['MAE']:
                undo += 1

print(undo)