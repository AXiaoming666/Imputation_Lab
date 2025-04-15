import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use('Agg')

ts = pd.read_csv("./Time-Series-Library/dataset/exchange_rate/exchange_rate.csv", encoding='utf-8')
ts = ts.set_index('date')

ts.iloc[:, 0] = ts.iloc[:, :].mean(axis=1)
ts = ts.iloc[:, 0]

# 应用STL分解
stl = STL(ts, seasonal=7, trend=30, period=7, robust=True)
result = stl.fit()

# 提取分解后的成分
seasonal = result.seasonal
trend = result.trend
resid = result.resid

# 绘制原始数据及其分解成分
plt.figure(figsize=(14, 8))

plt.subplot(4, 1, 1)
plt.plot(ts, label='Original')
plt.legend(loc='upper left')

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='upper left')

plt.subplot(4, 1, 4)
plt.plot(resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('stl_decomposition.png')
