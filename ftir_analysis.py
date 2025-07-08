import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import correlate

# 1. Baseline correction function
def baseline_als(y, lam=1000, p=0.05, niter=10):
    """Asymmetric Least Squares baseline correction"""
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# 2. Normalization function
def normalize_spectrum(y, method='max'):
    """Normalization methods"""
    if method == 'max':
        return y / np.max(y)
    elif method == 'area':
        return y / np.trapz(y)
    else:
        return y

# 3. Spectral alignment function
def align_spectra(reference, target):
    """Align spectra using cross-correlation"""
    correlation = correlate(reference, target, mode='same')
    shift = np.argmax(correlation) - len(reference)//2
    return np.roll(target, shift)

# Main analysis function
def analyze_ftir_data():
    # Read all CSV files, excluding generated files
    csv_files = [f for f in os.listdir() if f.endswith('.csv') and \
                 not f.endswith('_diff.csv') and \
                 f != 'all_spectra.csv' and \
                 f != 'difference_spectrum.csv']
    
    # Store processed data
    processed_data = []

    # Sort by exposure time
    def get_exposure_time(f):
        match = re.search(r'(\d+)s', f)
        return int(match.group(1)) if match else 0
    
    csv_files.sort(key=get_exposure_time)
    
    # Process each file
    reference = None
    for file in csv_files:
        # Extract exposure time
        exposure_time = get_exposure_time(file)

        # Read data
        df = pd.read_csv(file) # Removed header=None and names parameters
        x = df['Wavenumber'].values
        y = df['Absorbance'].values.astype(float)
        
        # Baseline correction
        baseline = baseline_als(y)
        y_corrected = y - baseline

        # Normalization
        y_normalized = normalize_spectrum(y_corrected)

        # Spectral alignment (using first spectrum as reference)
        if reference is None:
            reference = y_normalized
        else:
            y_normalized = align_spectra(reference, y_normalized)
        
        # Save processed data
        processed_data.append({
            'ExposureTime': exposure_time,
            'Wavenumber': x,
            'Absorbance': y_normalized,
            'Filename': file
        })
    
    # Plot processed spectra
    plt.figure(figsize=(12, 8))
    for data in processed_data:
        plt.plot(data['Wavenumber'], data['Absorbance'], 
                label=f"{data['ExposureTime']}s")
    
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Normalized Absorbance')
    plt.title('Processed FTIR Spectra')
    plt.legend()
    plt.show()
    
    # Calculate and plot difference spectra
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
        
        # Save difference spectrum
        diff_df = pd.DataFrame({
            'Wavenumber': processed_data[0]['Wavenumber'],
            'AbsorbanceDifference': difference
        })
        diff_df.to_csv('ftir_difference_spectrum.csv', index=False)
    
    return processed_data

# Execute analysis
if __name__ == '__main__':
    analyze_ftir_data()