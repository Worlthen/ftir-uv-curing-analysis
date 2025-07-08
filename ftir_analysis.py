import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import correlate

# 1. 基线校正函数
def baseline_als(y, lam=1000, p=0.05, niter=10):
    """Asymmetric Least Squares 基线校正"""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# 2. 归一化函数
def normalize_spectrum(y, method='max'):
    """归一化方法"""
    if method == 'max':
        return y / np.max(y)
    elif method == 'area':
        return y / np.trapz(y)
    else:
        return y

# 3. 光谱对齐函数
def align_spectra(reference, target):
    """使用交叉相关对齐光谱"""
    correlation = correlate(reference, target, mode='same')
    shift = np.argmax(correlation) - len(reference)//2
    return np.roll(target, shift)

# 主分析函数
def analyze_ftir_data():
    # 读取所有CSV文件，排除生成的文件
    csv_files = [f for f in os.listdir() if f.endswith('.csv') and \
                 not f.endswith('_diff.csv') and \
                 f != 'all_spectra.csv' and \
                 f != 'difference_spectrum.csv']
    
    # 存储处理后的数据
    processed_data = []
    
    # 按曝光时间排序
    def get_exposure_time(f):
        match = re.search(r'(\d+)s', f)
        return int(match.group(1)) if match else 0
    
    csv_files.sort(key=get_exposure_time)
    
    # 处理每个文件
    reference = None
    for file in csv_files:
        # 提取曝光时间
        exposure_time = get_exposure_time(file)
        
        # 读取数据
        df = pd.read_csv(file) # Removed header=None and names parameters
        x = df['Wavenumber'].values
        y = df['Absorbance'].values.astype(float)
        
        # 基线校正
        baseline = baseline_als(y)
        y_corrected = y - baseline
        
        # 归一化
        y_normalized = normalize_spectrum(y_corrected)
        
        # 光谱对齐(以第一个光谱为参考)
        if reference is None:
            reference = y_normalized
        else:
            y_normalized = align_spectra(reference, y_normalized)
        
        # 保存处理后的数据
        processed_data.append({
            'ExposureTime': exposure_time,
            'Wavenumber': x,
            'Absorbance': y_normalized,
            'Filename': file
        })
    
    # 绘制处理后的光谱
    plt.figure(figsize=(12, 8))
    for data in processed_data:
        plt.plot(data['Wavenumber'], data['Absorbance'], 
                label=f"{data['ExposureTime']}s")
    
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Absorbance')
    plt.title('Processed FTIR Spectra')
    plt.legend()
    plt.show()
    
    # 计算并绘制差异光谱
    if len(processed_data) >= 2:
        initial = processed_data[0]['Absorbance']
        final = processed_data[-1]['Absorbance']
        difference = final - initial
        
        plt.figure(figsize=(12, 6))
        plt.plot(processed_data[0]['Wavenumber'], difference)
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Absorbance Difference')
        plt.title(f'Difference Spectrum ({processed_data[-1]["ExposureTime"]}s - {processed_data[0]["ExposureTime"]}s)')
        plt.show()
        
        # 保存差异光谱
        diff_df = pd.DataFrame({
            'Wavenumber': processed_data[0]['Wavenumber'],
            'AbsorbanceDifference': difference
        })
        diff_df.to_csv('ftir_difference_spectrum.csv', index=False)
    
    return processed_data

# 执行分析
if __name__ == '__main__':
    analyze_ftir_data()