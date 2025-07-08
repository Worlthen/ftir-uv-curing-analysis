import os
import numpy as np
import pandas as pd
from brukeropusreader import read_file

# Get all files in current directory with OPUS extensions
files = [f for f in os.listdir() if f.endswith(('.0', '.1', '.2', '.3'))]

for file_path in files:
    try:
        # Read OPUS file
        opus_data = read_file(file_path)

        # Extract data
        x = np.arange(len(opus_data["AB"]))  # Wavenumber or index
        y = opus_data["AB"]  # Absorbance data

        # Create DataFrame and save as CSV
        df = pd.DataFrame({'Wavenumber': x, 'Absorbance': y})

        # Modify filename generation logic, preserve original extension
        base_name = os.path.splitext(file_path)[0]
        ext = os.path.splitext(file_path)[1]
        csv_path = f"{base_name}{ext}.csv"
        
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted {file_path} to {csv_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
import matplotlib.pyplot as plt


file_path = '0s_10%_365nm.1'
opus_data = read_file(file_path)
print(f"AB array shape: {opus_data['AB'].shape}")


x = np.arange(len(opus_data["AB"]))  
y = opus_data["AB"]


plt.plot(x, y)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("FTIR Spectrum from .0 file")
plt.grid()
plt.show()