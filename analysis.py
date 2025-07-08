import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Collect all CSV files
csv_files = [f for f in os.listdir() if f.endswith('.csv') and not f.endswith('_diff.csv')]

# Create main DataFrame to store all data
data = []

for file in csv_files:
    # Extract exposure time from filename
    match = re.search(r'(\d+)s', file)
    if not match:
        continue
    exposure_time = int(match.group(1))
    
    # Read data
    df = pd.read_csv(file)
    df['ExposureTime'] = exposure_time
    df['Filename'] = file
    
    # Add to main dataset
    data.append(df)

# Merge all data
all_data = pd.concat(data)

# Plot spectra grouped by exposure time
plt.figure(figsize=(10, 6))
for time, group in all_data.groupby('ExposureTime'):
    plt.plot(group['Wavenumber'], group['Absorbance'], label=f'{time}s')

plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Absorbance')
plt.title('FTIR Absorption Spectra at Different Exposure Times')
plt.legend()
plt.show()

# Calculate difference spectrum (longest exposure time - shortest exposure time)
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
    
    # Save difference spectrum
    diff.to_csv('difference_spectrum.csv', header=['AbsorbanceDifference'])

# Save integrated data
all_data.to_csv('all_spectra.csv', index=False)