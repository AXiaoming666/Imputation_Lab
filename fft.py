import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    
    n = x.shape[1]
    
    values, indices = torch.topk(frequency_list, k)
    
    plt.figure(figsize=(10, 6))
    freq = np.linspace(0, n//2+1, len(frequency_list))
    
    plt.plot(freq, frequency_list.detach().cpu().numpy())
    
    # Calculate x-axis range based on peak frequencies
    min_freq = max(0, min(freq[indices]) - 10)
    max_freq = min(n//2, max(freq[indices]) + 10)
    plt.xlim(min_freq, max_freq)
    
    # 标注峰值
    for i, (idx, val) in enumerate(zip(indices, values)):
        plt.plot(freq[idx], val.item(), 'ro')  # 红点标记峰值
        plt.annotate(f'Peak {i+1}\n(f={freq[idx]:.1f})',  # 添加标注
                    xy=(freq[idx], val.item()),
                    xytext=(10, 10),
                    textcoords='offset points')
    
    plt.title('Frequency Domain')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig('frequency_domain.png')
    plt.close()
    
    period = x.shape[1] // indices
    return period, abs(xf).mean(-1)[:, indices]

data = pd.read_csv('./Time-Series-Library/dataset/weather/weather.csv')
data = data.values[:, 1:].astype(np.float64)
data = data.reshape(1, data.shape[0], data.shape[1])
data = torch.tensor(data, dtype=torch.float64)
print(FFT_for_Period(data, 5))