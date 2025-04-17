import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('Agg')

from src.data_loading import DataLoader
from src.missing_simulation import MissingSimulation
from src.imputer import *


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

# Find top 5 amplitudes for complete data
top5_complete_idx = np.argsort(fft_complete)[-5:]
top5_complete_idx.sort()  # Sort to get the correct order for zoom
top5_imputed_idx = np.argsort(fft_imputed)[-5:]
top5_imputed_idx.sort()  # Sort to get the correct order for zoom

# Define the zoom range to include the top 5 amplitudes
zoom_start = min(top5_complete_idx[0], top5_imputed_idx[0]) - 5
zoom_end = max(top5_complete_idx[-1], top5_imputed_idx[-1]) + 5

plt.figure(figsize=(12, 12))

# Original plots
plt.subplot(2, 1, 1)
plt.plot(range(0, n // 2), fft_complete, label='Complete Data', alpha=0.7)
plt.plot(range(0, n // 2), fft_imputed, label='Imputed Data', alpha=0.7, linestyle='--')
plt.title('FFT Comparison: Complete vs Imputed Data')
plt.axvspan(zoom_start, zoom_end, color='yellow', alpha=0.2, label='Zoom Region')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

shared_peaks = set(top5_complete_idx).intersection(set(top5_imputed_idx))
shared_peaks = list(shared_peaks)
# Find overlap in top 5 peaks and remove from either list to create unique sets
top5_complete_idx = [idx for idx in top5_complete_idx if idx not in shared_peaks]
top5_imputed_idx = [idx for idx in top5_imputed_idx if idx not in shared_peaks]

# Zoomed-in plot
plt.subplot(2, 1, 2)
plt.plot(range(0, n // 2), fft_complete, label='Complete Data', alpha=0.7)
plt.plot(range(0, n // 2), fft_imputed, label='Imputed Data', alpha=0.7, linestyle='--')

for idx in shared_peaks:
    plt.plot(idx, fft_complete[idx], 'bo', markersize=3, label='Shared Peaks' if idx == shared_peaks[0] else "")
    plt.plot([idx, idx], [0, fft_complete[idx]], 'b--', linewidth=0.8)

# Mark top 5 peaks for complete data
for idx in top5_complete_idx:
    plt.plot(idx, fft_complete[idx], 'ro', markersize=3, label='Complete Peaks' if idx == top5_complete_idx[0] else "")
    plt.plot([idx, idx], [0, fft_complete[idx]], 'r--', linewidth=0.8)

# Mark top 5 peaks for imputed data
for idx in top5_imputed_idx:
    plt.plot(idx, fft_imputed[idx], 'go', markersize=3, label='Imputed Peaks' if idx == top5_imputed_idx[0] else "")
    plt.plot([idx, idx], [0, fft_imputed[idx]], 'g--', linewidth=0.8)

plt.xlim(zoom_start, zoom_end)
plt.ylim(3.6, 4.1)
plt.title('Zoomed-in FFT Comparison (Top 5 Periods Marked)')
plt.xlabel('Period by days')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fft_comparison.png')
plt.close()