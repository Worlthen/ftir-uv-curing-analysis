import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 收集所有CSV文件
csv_files = [f for f in os.listdir() if f.endswith('.csv') and not f.endswith('_diff.csv')]

# 创建主DataFrame存储所有数据
data = []

for file in csv_files:
    # 从文件名提取曝光时间
    match = re.search(r'(\d+)s', file)
    if not match:
        continue
    exposure_time = int(match.group(1))
    
    # 读取数据
    df = pd.read_csv(file)
    df['ExposureTime'] = exposure_time
    df['Filename'] = file
    
    # 添加到主数据集
    data.append(df)

# 合并所有数据
all_data = pd.concat(data)

# 按曝光时间分组绘制光谱
plt.figure(figsize=(10, 6))
for time, group in all_data.groupby('ExposureTime'):
    plt.plot(group['Wavenumber'], group['Absorbance'], label=f'{time}s')

plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Absorbance')
plt.title('FTIR Absorption Spectra at Different Exposure Times')
plt.legend()
plt.show()

# 计算差异光谱 (最长曝光时间 - 最短曝光时间)
times = sorted(all_data['ExposureTime'].unique())
if len(times) >= 2:
    longest = all_data[all_data['ExposureTime'] == times[-1]]
    shortest = all_data[all_data['ExposureTime'] == times[0]]
    
    diff = longest.set_index('Wavenumber')['Absorbance'] - shortest.set_index('Wavenumber')['Absorbance']
    
    plt.figure(figsize=(10, 6))
    plt.plot(diff.index, diff.values)
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Absorbance Difference')
    plt.title(f'Difference Spectrum ({times[-1]}s - {times[0]}s)')
    plt.show()
    
    # 保存差异光谱
    diff.to_csv('difference_spectrum.csv', header=['AbsorbanceDifference'])

# 保存整合数据
all_data.to_csv('all_spectra.csv', index=False)