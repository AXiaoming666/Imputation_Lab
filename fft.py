import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('Agg')

from src.data_loading import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import *
from src.evaluation_metrics import Evaluate


np.random.seed(42)
random.seed(42)

data = DataLoader("exchange_rate")
MissingSimulation(data, 0.9, "MCAR", 0.1)

data_imputed = forward_impute(data.get_incomplete_data())

complete_data = data.get_y_train_complete()
imputed_data = data.separate_time_features(data_imputed)

fft_complete = np.fft.fft(complete_data)
fft_imputed = np.fft.fft(imputed_data)

fft_complete = np.mean(np.abs(fft_complete), axis=1)
fft_imputed = np.mean(np.abs(fft_imputed), axis=1)

n = len(complete_data)

fft_complete = fft_complete[:n // 2]
fft_imputed = fft_imputed[:n // 2]

plt.figure(figsize=(12, 6))

# Original plots
plt.subplot(2, 1, 1)
plt.plot(range(0, n // 2), fft_complete, label='Complete Data', alpha=0.7)
plt.plot(range(0, n // 2), fft_imputed, label='Imputed Data', alpha=0.7, linestyle='--')

# Find top 5 amplitudes for complete data
top5_complete_idx = np.argsort(fft_complete)[-5:]
top5_complete_val = fft_complete[top5_complete_idx]

# Find top 5 amplitudes for imputed data
top5_imputed_idx = np.argsort(fft_imputed)[-5:]
top5_imputed_val = fft_imputed[top5_imputed_idx]

print("Top 5 Amplitudes for Complete Data:")
for i in range(5):
    print(f"Index: {top5_complete_idx[i]}, Value: {top5_complete_val[i]}")
print("Top 5 Amplitudes for Imputed Data:")
for i in range(5):
    print(f"Index: {top5_imputed_idx[i]}, Value: {top5_imputed_val[i]}")

plt.title('FFT Comparison: Complete vs Imputed Data')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Difference plot
plt.subplot(2, 1, 2)
plt.plot(range(0, n // 2), fft_complete - fft_imputed, color='red', label='Difference (Complete - Imputed)')
plt.title('Difference between Complete and Imputed FFT')
plt.xlabel('Period by days')
plt.ylabel('Magnitude Difference')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fft_comparison.png')
plt.close()