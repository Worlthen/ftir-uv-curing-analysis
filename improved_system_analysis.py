"""
Photoresist UV Exposure FTIR Spectral Analysis System - Improved Version
Complete implementation with fixes for data ordering and English labels

Features include:
1. Photochemical reaction kinetics analysis
2. Molecular structure change characterization
3. Reaction kinetics modeling
4. Multivariate statistical analysis
5. GUI data range selection

Key Improvements:
- Fixed data ordering to prevent head-tail connection
- All Chinese characters replaced with English
- Proper wavenumber sorting (left to right)
- Enhanced data validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal, stats
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use English fonts and avoid character display issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FTIRAnalyzer:
    """FTIR Spectral Analysis Main Class"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.analysis_results = {}
        self.exposure_times = []
        self.wavenumbers = []

    def load_data(self, filepath='all_spectra.csv'):
        """Load spectral data with support for both integrated and individual files"""
        try:
            # Determine file type and load accordingly
            if self.is_integrated_file(filepath):
                return self.load_integrated_data(filepath)
            else:
                return self.load_individual_file(filepath)
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

    def is_integrated_file(self, filepath):
        """Check if file is an integrated data file with multiple time points"""
        try:
            # Read first few rows to check structure
            sample_data = pd.read_csv(filepath, nrows=10)

            # Check if it has the integrated format columns
            required_columns = ['ExposureTime', 'Wavenumber', 'Absorbance']
            has_integrated_format = all(col in sample_data.columns for col in required_columns)

            if has_integrated_format:
                # Check if there are multiple exposure times
                full_data = pd.read_csv(filepath)
                unique_times = full_data['ExposureTime'].nunique()
                return unique_times > 1

            return False
        except:
            return False

    def load_integrated_data(self, filepath):
        """Load integrated data file (multiple time points in one file)"""
        print("Loading integrated data file...")
        self.data = pd.read_csv(filepath)

        # Remove duplicates and average repeated measurements
        self.data = self.average_duplicate_measurements(self.data)

        print(f"Integrated data loaded: {len(self.data)} data points")

        # Extract time points and wavenumbers with proper sorting
        self.exposure_times = sorted(self.data['ExposureTime'].unique())
        self.wavenumbers = sorted(self.data['Wavenumber'].unique())

        # Validate data integrity
        self.validate_data_integrity()
        return True

    def load_individual_file(self, filepath):
        """Load individual spectrum file and prompt for exposure time"""
        print("Loading individual spectrum file...")

        # Try different possible column names for individual files
        possible_formats = [
            ['Wavenumber', 'Absorbance'],
            ['wavenumber', 'absorbance'],
            ['Wavenumbers', 'Absorbances'],
            ['Wave Number', 'Absorbance'],
            ['cm-1', 'Abs'],
            ['X', 'Y']
        ]

        data = pd.read_csv(filepath)
        print(f"File columns: {list(data.columns)}")

        # Find the correct column mapping
        wn_col, abs_col = None, None
        for wn_name, abs_name in possible_formats:
            if wn_name in data.columns and abs_name in data.columns:
                wn_col, abs_col = wn_name, abs_name
                break

        # If standard names not found, try to auto-detect
        if wn_col is None:
            # Assume first column is wavenumber, second is absorbance
            if len(data.columns) >= 2:
                wn_col = data.columns[0]
                abs_col = data.columns[1]
                print(f"Auto-detected columns: {wn_col} (wavenumber), {abs_col} (absorbance)")
            else:
                raise ValueError("Could not identify wavenumber and absorbance columns")

        # Get exposure time from user or filename
        exposure_time = self.extract_exposure_time_from_filename(filepath)
        if exposure_time is None:
            # Default to 0 if cannot extract from filename
            exposure_time = 0
            print(f"Using default exposure time: {exposure_time}s")
        else:
            print(f"Extracted exposure time from filename: {exposure_time}s")

        # Create integrated format
        integrated_data = []
        for _, row in data.iterrows():
            integrated_data.append({
                'ExposureTime': exposure_time,
                'Wavenumber': row[wn_col],
                'Absorbance': row[abs_col]
            })

        self.data = pd.DataFrame(integrated_data)

        # Remove duplicates and average repeated measurements
        self.data = self.average_duplicate_measurements(self.data)

        # Extract time points and wavenumbers
        self.exposure_times = sorted(self.data['ExposureTime'].unique())
        self.wavenumbers = sorted(self.data['Wavenumber'].unique())

        print(f"Individual file loaded: {len(self.data)} data points")
        self.validate_data_integrity()
        return True

    def extract_exposure_time_from_filename(self, filepath):
        """Extract exposure time from filename if possible"""
        import re
        import os

        filename = os.path.basename(filepath)

        # Common patterns for exposure time in filenames
        patterns = [
            r'(\d+)s',           # e.g., "5s.csv", "spectrum_10s.csv"
            r'(\d+)sec',         # e.g., "5sec.csv"
            r't(\d+)',           # e.g., "t5.csv", "spectrum_t10.csv"
            r'time(\d+)',        # e.g., "time5.csv"
            r'exp(\d+)',         # e.g., "exp5.csv"
            r'(\d+)_',           # e.g., "5_spectrum.csv"
            r'_(\d+)\.',         # e.g., "spectrum_5.csv"
        ]

        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return int(match.group(1))

        return None

    def average_duplicate_measurements(self, data):
        """Average measurements with same exposure time and wavenumber"""
        print("Averaging duplicate measurements...")

        # Group by ExposureTime and Wavenumber, then average Absorbance
        averaged_data = data.groupby(['ExposureTime', 'Wavenumber'], as_index=False).agg({
            'Absorbance': 'mean'
        })

        original_count = len(data)
        averaged_count = len(averaged_data)

        if original_count > averaged_count:
            print(f"Averaged {original_count - averaged_count} duplicate measurements")
            print(f"Final data points: {averaged_count}")

        return averaged_data

    def validate_data_integrity(self):
        """Validate loaded data integrity"""
        if len(self.wavenumbers) < 10:
            print("Warning: Very few wavenumber points detected")

        # Ensure wavenumber range is reasonable for FTIR
        wn_min, wn_max = min(self.wavenumbers), max(self.wavenumbers)
        if wn_min < 400 or wn_max > 4000:
            print(f"Warning: Unusual wavenumber range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")

        print(f"Exposure time points: {self.exposure_times}")
        print(f"Wavenumber range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
        print(f"Total wavenumber points: {len(self.wavenumbers)}")

    def preprocess_spectrum(self, spectrum_data, baseline_method='als', 
                          normalize_method='max', smooth=True):
        """
        Spectrum preprocessing pipeline with proper data ordering
        
        Parameters:
        spectrum_data: Single time point spectrum data
        baseline_method: Baseline correction method ('als', 'polynomial')
        normalize_method: Normalization method ('max', 'area', 'vector')
        smooth: Whether to apply smoothing
        """
        # Sort data by wavenumber to ensure proper order (CRITICAL FIX)
        spectrum_data_sorted = spectrum_data.sort_values('Wavenumber').reset_index(drop=True)
        wavenumbers = spectrum_data_sorted['Wavenumber'].values
        absorbances = spectrum_data_sorted['Absorbance'].values
        
        # Ensure data is monotonic and properly ordered
        if len(wavenumbers) > 1:
            # Check if data is in descending order and reverse if needed
            if wavenumbers[0] > wavenumbers[-1]:
                wavenumbers = wavenumbers[::-1]
                absorbances = absorbances[::-1]
                print("Data reversed to ensure ascending wavenumber order")
        
        # Remove any duplicate wavenumbers
        unique_indices = np.unique(wavenumbers, return_index=True)[1]
        wavenumbers = wavenumbers[unique_indices]
        absorbances = absorbances[unique_indices]
        
        # 1. Baseline correction
        if baseline_method == 'als':
            baseline = self.baseline_correction_als(absorbances)
            corrected = absorbances - baseline
        else:
            # Simple polynomial baseline
            p = np.polyfit(wavenumbers, absorbances, 2)
            baseline = np.polyval(p, wavenumbers)
            corrected = absorbances - baseline

        # 2. Smoothing
        if smooth and len(corrected) > 5:
            window_length = min(5, len(corrected) if len(corrected) % 2 == 1 else len(corrected) - 1)
            if window_length >= 3:
                corrected = signal.savgol_filter(corrected, window_length, 2)

        # 3. Normalization
        if normalize_method == 'max':
            max_abs = np.max(np.abs(corrected))
            if max_abs > 0:
                corrected = corrected / max_abs
        elif normalize_method == 'area':
            area = trapz(np.abs(corrected), wavenumbers)
            if area > 0:
                corrected = corrected / area
        elif normalize_method == 'vector':
            norm = np.linalg.norm(corrected)
            if norm > 0:
                corrected = corrected / norm

        return pd.DataFrame({
            'Wavenumber': wavenumbers,
            'Absorbance': corrected
        })

    def baseline_correction_als(self, y, lam=1000, p=0.05, niter=10):
        """
        Asymmetric Least Squares (ALS) baseline correction algorithm

        Parameters:
        y: Spectral data
        lam: Smoothing parameter
        p: Asymmetry parameter
        niter: Number of iterations
        """
        L = len(y)
        if L < 3:
            return np.zeros_like(y)
            
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)

        for i in range(niter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)

        return z

    def process_all_spectra(self):
        """Process all spectra with consistent ordering"""
        if self.data is None:
            print("No data loaded")
            return False

        processed_data = {}
        
        for time in self.exposure_times:
            time_data = self.data[self.data['ExposureTime'] == time]
            if len(time_data) > 0:
                processed_spectrum = self.preprocess_spectrum(time_data)
                processed_data[time] = processed_spectrum
                print(f"Processed spectrum for {time}s exposure")

        self.processed_data = processed_data
        print(f"Successfully processed {len(processed_data)} spectra")
        return True

    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        if not self.process_all_spectra():
            return False

        print("Running complete analysis...")

        # 1. Calculate difference spectra
        diff_spectra = self.calculate_difference_spectra(self.processed_data)
        self.analysis_results['difference_spectra'] = diff_spectra

        # 2. Analyze key wavenumbers
        key_analysis = self.analyze_key_wavenumbers()
        self.analysis_results['key_analysis'] = key_analysis

        # 3. PCA analysis
        pca_results = self.perform_pca_analysis()
        self.analysis_results['pca'] = pca_results

        print("Analysis completed successfully!")
        return True

    def run_selective_analysis(self, filtered_data, selected_times):
        """Run analysis on selected data subset"""
        if filtered_data is None or len(filtered_data) == 0:
            print("No data to analyze")
            return False

        print(f"Running selective analysis on {len(selected_times)} time points...")

        # Temporarily store original data
        original_data = self.data
        original_times = self.exposure_times
        original_wavenumbers = self.wavenumbers

        try:
            # Set filtered data as current data
            self.data = filtered_data
            self.exposure_times = selected_times
            self.wavenumbers = sorted(filtered_data['Wavenumber'].unique())

            # Process selected spectra
            if not self.process_all_spectra():
                return False

            # 1. Calculate difference spectra
            diff_spectra = self.calculate_difference_spectra(self.processed_data)
            self.analysis_results['difference_spectra'] = diff_spectra

            # 2. Analyze key wavenumbers
            key_analysis = self.analyze_key_wavenumbers()
            self.analysis_results['key_analysis'] = key_analysis

            # 3. PCA analysis
            pca_results = self.perform_pca_analysis()
            self.analysis_results['pca'] = pca_results

            print("Selective analysis completed successfully!")
            return True

        except Exception as e:
            print(f"Selective analysis failed: {e}")
            return False
        finally:
            # Restore original data
            self.data = original_data
            self.exposure_times = original_times
            self.wavenumbers = original_wavenumbers

    def calculate_difference_spectra(self, processed_data):
        """Calculate difference spectra with proper data handling"""
        reference_time = min(processed_data.keys())
        reference_spectrum = processed_data[reference_time]
        
        # Ensure reference spectrum is properly sorted
        reference_spectrum = reference_spectrum.sort_values('Wavenumber').reset_index(drop=True)
        reference_wavenumbers = reference_spectrum['Wavenumber'].values
        reference_absorbance = reference_spectrum['Absorbance'].values

        difference_spectra = {}

        for time in processed_data.keys():
            if time != reference_time:
                current_spectrum = processed_data[time]
                current_spectrum = current_spectrum.sort_values('Wavenumber').reset_index(drop=True)
                current_wavenumbers = current_spectrum['Wavenumber'].values
                current_absorbance = current_spectrum['Absorbance'].values

                # Interpolate to common wavenumber grid if needed
                if not np.array_equal(reference_wavenumbers, current_wavenumbers):
                    current_absorbance = np.interp(reference_wavenumbers, 
                                                 current_wavenumbers, 
                                                 current_absorbance)

                difference = current_absorbance - reference_absorbance
                
                difference_spectra[time] = pd.DataFrame({
                    'Wavenumber': reference_wavenumbers,
                    'Difference': difference,
                    'Time': time
                })

        return difference_spectra

    def analyze_key_wavenumbers(self, wavenumber_ranges=None):
        """Analyze kinetics of key wavenumber regions"""
        if self.data is None:
            print("Data not loaded")
            return {}

        if wavenumber_ranges is None:
            # Key wavenumber regions for photoresist analysis (all English names)
            wavenumber_ranges = {
                'C=C Unsaturated Bonds': (1600, 1700),
                'C=O Carbonyl Groups': (1700, 1800),
                'C-H Aliphatic Stretch': (2800, 3000),
                'O-H Hydroxyl Groups': (3200, 3600),
                'Aromatic C=C Bonds': (1450, 1600)
            }

        key_analysis = {}
        
        for region_name, (wn_min, wn_max) in wavenumber_ranges.items():
            region_data = []
            
            for time in self.exposure_times:
                time_data = self.data[
                    (self.data['ExposureTime'] == time) & 
                    (self.data['Wavenumber'] >= wn_min) & 
                    (self.data['Wavenumber'] <= wn_max)
                ]
                
                if len(time_data) > 0:
                    avg_absorbance = time_data['Absorbance'].mean()
                    region_data.append({
                        'time': time,
                        'absorbance': avg_absorbance
                    })
            
            if len(region_data) >= 3:
                region_df = pd.DataFrame(region_data)
                
                # Fit kinetic model
                kinetic_fit = self.fit_kinetic_model(region_df)
                
                key_analysis[region_name] = {
                    'wavenumber_range': (wn_min, wn_max),
                    'time_series': region_df,
                    'kinetic_fit': kinetic_fit
                }

        return key_analysis

    def fit_kinetic_model(self, time_data, model='first_order'):
        """Fit kinetic model to time series data"""
        times = time_data['time'].values
        absorbances = time_data['absorbance'].values

        # Filter valid data
        valid_mask = (absorbances > 0) & np.isfinite(absorbances)
        times = times[valid_mask]
        absorbances = absorbances[valid_mask]

        if len(times) < 3:
            return None

        try:
            if model == 'first_order':
                # ln(A/A0) = -kt
                A0 = absorbances[0]
                if A0 <= 0:
                    return None
                    
                ln_ratios = np.log(absorbances / A0)
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, ln_ratios)
                k = -slope
                r2 = r_value**2

                return {
                    'k': k,
                    'A0': A0,
                    'r2': r2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'model': 'first_order',
                    'half_life': np.log(2) / k if k > 0 else np.inf
                }
        except Exception as e:
            print(f"Kinetic fitting failed: {e}")
            return None

    def perform_pca_analysis(self):
        """Perform PCA analysis on processed spectra"""
        if not self.processed_data:
            return None

        # Prepare data matrix with consistent wavenumber grid
        spectra_matrix = []
        times = []
        
        # Use the first spectrum as reference for wavenumber grid
        reference_time = min(self.processed_data.keys())
        reference_wavenumbers = self.processed_data[reference_time]['Wavenumber'].values
        
        for time in sorted(self.processed_data.keys()):
            spectrum = self.processed_data[time]
            # Interpolate to reference grid if needed
            if not np.array_equal(spectrum['Wavenumber'].values, reference_wavenumbers):
                absorbance = np.interp(reference_wavenumbers, 
                                     spectrum['Wavenumber'].values,
                                     spectrum['Absorbance'].values)
            else:
                absorbance = spectrum['Absorbance'].values
                
            spectra_matrix.append(absorbance)
            times.append(time)

        spectra_matrix = np.array(spectra_matrix)
        
        # Standardize data
        scaler = StandardScaler()
        spectra_scaled = scaler.fit_transform(spectra_matrix)
        
        # Perform PCA
        pca = PCA()
        scores = pca.fit_transform(spectra_scaled)
        loadings = pca.components_
        
        return {
            'scores': scores,
            'loadings': loadings,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'times': times,
            'wavenumbers': reference_wavenumbers,
            'scaler': scaler,
            'pca_model': pca
        }


class FTIRAnalysisGUI:
    """FTIR Analysis GUI Interface with English labels"""

    def __init__(self):
        self.analyzer = FTIRAnalyzer()
        self.root = tk.Tk()
        self.root.title("FTIR Spectral Analysis System")
        self.root.geometry("1200x800")

        # Create main frame
        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets with enhanced data selection"""
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)

        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(side='left', padx=5, pady=5)

        ttk.Button(file_frame, text="Load Data File",
                  command=self.load_data_file).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Run Analysis",
                  command=self.run_analysis).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Save Results",
                  command=self.save_results).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Save Plots",
                  command=self.save_plots).pack(side='left', padx=2)
        ttk.Button(file_frame, text="Kinetic Plots",
                  command=self.plot_kinetics_detailed).pack(side='left', padx=2)

        # Data selection frame
        selection_frame = ttk.LabelFrame(control_frame, text="Data Selection")
        selection_frame.pack(side='left', padx=5, pady=5, fill='x', expand=True)

        # Time point selection
        ttk.Label(selection_frame, text="Time Points:").grid(row=0, column=0, padx=5, sticky='w')
        self.time_selection_frame = ttk.Frame(selection_frame)
        self.time_selection_frame.grid(row=0, column=1, padx=5, sticky='ew')

        # Wavenumber range selection
        ttk.Label(selection_frame, text="Wavenumber Range:").grid(row=1, column=0, padx=5, sticky='w')
        range_frame = ttk.Frame(selection_frame)
        range_frame.grid(row=1, column=1, padx=5, sticky='ew')

        ttk.Label(range_frame, text="Min:").pack(side='left')
        self.wn_min_var = tk.StringVar(value="400")
        ttk.Entry(range_frame, textvariable=self.wn_min_var, width=8).pack(side='left', padx=2)

        ttk.Label(range_frame, text="Max:").pack(side='left')
        self.wn_max_var = tk.StringVar(value="4000")
        ttk.Entry(range_frame, textvariable=self.wn_max_var, width=8).pack(side='left', padx=2)

        # Analysis options
        options_frame = ttk.LabelFrame(control_frame, text="Analysis Options")
        options_frame.pack(side='right', padx=5, pady=5)

        # Baseline correction method
        ttk.Label(options_frame, text="Baseline:").grid(row=0, column=0, padx=5, sticky='w')
        self.baseline_var = tk.StringVar(value='als')
        baseline_combo = ttk.Combobox(options_frame, textvariable=self.baseline_var,
                                    values=['als', 'polynomial'], width=10)
        baseline_combo.grid(row=0, column=1, padx=5)

        # Normalization method
        ttk.Label(options_frame, text="Normalization:").grid(row=1, column=0, padx=5, sticky='w')
        self.norm_var = tk.StringVar(value='max')
        norm_combo = ttk.Combobox(options_frame, textvariable=self.norm_var,
                                values=['max', 'area', 'vector'], width=10)
        norm_combo.grid(row=1, column=1, padx=5)

        # Plot area
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please load a data file")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='bottom', fill='x')

        # Initialize time checkboxes (will be populated when data is loaded)
        self.time_checkboxes = {}

    def load_data_file(self):
        """Load data file with enhanced selection options"""
        filename = filedialog.askopenfilename(
            title="Select FTIR Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                if self.analyzer.load_data(filename):
                    self.status_var.set(f"Data loaded: {filename.split('/')[-1]}")
                    self.create_time_selection_checkboxes()
                    self.update_wavenumber_range()
                    messagebox.showinfo("Success",
                                      f"Data loaded successfully!\n"
                                      f"Time points: {len(self.analyzer.exposure_times)}\n"
                                      f"Wavenumber points: {len(self.analyzer.wavenumbers)}")
                else:
                    messagebox.showerror("Error", "Failed to load data!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def create_time_selection_checkboxes(self):
        """Create checkboxes for time point selection"""
        # Clear existing checkboxes
        for widget in self.time_selection_frame.winfo_children():
            widget.destroy()
        self.time_checkboxes.clear()

        # Create checkboxes for each time point
        for i, time in enumerate(self.analyzer.exposure_times):
            var = tk.BooleanVar(value=True)  # Default: all selected
            self.time_checkboxes[time] = var

            cb = ttk.Checkbutton(self.time_selection_frame,
                               text=f"{time}s",
                               variable=var)
            cb.grid(row=0, column=i, padx=2)

        # Add "Select All" and "Clear All" buttons
        ttk.Button(self.time_selection_frame, text="All",
                  command=self.select_all_times).grid(row=1, column=0, padx=2, sticky='w')
        ttk.Button(self.time_selection_frame, text="None",
                  command=self.clear_all_times).grid(row=1, column=1, padx=2, sticky='w')

    def select_all_times(self):
        """Select all time points"""
        for var in self.time_checkboxes.values():
            var.set(True)

    def clear_all_times(self):
        """Clear all time point selections"""
        for var in self.time_checkboxes.values():
            var.set(False)

    def update_wavenumber_range(self):
        """Update wavenumber range based on loaded data"""
        if self.analyzer.wavenumbers:
            min_wn = min(self.analyzer.wavenumbers)
            max_wn = max(self.analyzer.wavenumbers)
            self.wn_min_var.set(f"{min_wn:.0f}")
            self.wn_max_var.set(f"{max_wn:.0f}")

    def get_selected_times(self):
        """Get list of selected time points"""
        selected_times = []
        for time, var in self.time_checkboxes.items():
            if var.get():
                selected_times.append(time)
        return selected_times

    def get_wavenumber_range(self):
        """Get selected wavenumber range"""
        try:
            min_wn = float(self.wn_min_var.get())
            max_wn = float(self.wn_max_var.get())
            return min_wn, max_wn
        except ValueError:
            messagebox.showerror("Error", "Invalid wavenumber range values!")
            return None, None

    def run_analysis(self):
        """Run complete analysis with selected data"""
        if self.analyzer.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        # Get selected parameters
        selected_times = self.get_selected_times()
        if not selected_times:
            messagebox.showerror("Error", "Please select at least one time point!")
            return

        min_wn, max_wn = self.get_wavenumber_range()
        if min_wn is None or max_wn is None:
            return

        if min_wn >= max_wn:
            messagebox.showerror("Error", "Minimum wavenumber must be less than maximum!")
            return

        self.status_var.set("Running analysis on selected data...")
        self.root.update()

        try:
            # Filter data based on selections
            filtered_data = self.filter_data(selected_times, min_wn, max_wn)

            # Run analysis on filtered data
            if self.analyzer.run_selective_analysis(filtered_data, selected_times):
                self.status_var.set("Analysis completed")
                self.plot_results()
                messagebox.showinfo("Success",
                                  f"Analysis completed successfully!\n"
                                  f"Time points analyzed: {len(selected_times)}\n"
                                  f"Wavenumber range: {min_wn:.0f} - {max_wn:.0f} cm⁻¹")
            else:
                messagebox.showerror("Error", "Analysis failed!")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis error: {str(e)}")

    def filter_data(self, selected_times, min_wn, max_wn):
        """Filter data based on selected time points and wavenumber range"""
        if self.analyzer.data is None:
            return None

        # Filter by time points and wavenumber range
        filtered_data = self.analyzer.data[
            (self.analyzer.data['ExposureTime'].isin(selected_times)) &
            (self.analyzer.data['Wavenumber'] >= min_wn) &
            (self.analyzer.data['Wavenumber'] <= max_wn)
        ].copy()

        return filtered_data

    def plot_results(self):
        """Plot analysis results with proper data ordering and English labels"""
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if not self.analyzer.analysis_results:
            return

        # Create figure with subplots
        fig = Figure(figsize=(14, 10))

        # Store figure for saving
        self.current_figure = fig

        # 1. Spectral evolution
        ax1 = fig.add_subplot(221)
        processed_data = self.analyzer.processed_data

        if processed_data:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, (time, spectrum) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                # Ensure data is properly sorted before plotting
                spectrum_sorted = spectrum.sort_values('Wavenumber')
                ax1.plot(spectrum_sorted['Wavenumber'], spectrum_sorted['Absorbance'],
                        color=color, label=f'{time}s', linewidth=1.5)

        ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ax1.set_ylabel('Normalized Absorbance', fontsize=10)
        ax1.set_title('Spectral Evolution', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)

        # 2. Key wavenumber time series with kinetic fits
        ax2 = fig.add_subplot(222)
        if 'key_analysis' in self.analyzer.analysis_results:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (region, data) in enumerate(self.analyzer.analysis_results['key_analysis'].items()):
                if 'time_series' in data:
                    ts = data['time_series']
                    color = colors[i % len(colors)]

                    # Plot experimental data points
                    ax2.scatter(ts['time'], ts['absorbance'],
                              color=color, s=50, alpha=0.8,
                              label=f'{region} (Data)', zorder=3)

                    # Plot kinetic fit if available
                    if 'kinetic_fit' in data and data['kinetic_fit']:
                        fit = data['kinetic_fit']
                        t_fit = np.linspace(ts['time'].min(), ts['time'].max(), 100)

                        if fit['model'] == 'first_order':
                            y_fit = fit['A0'] * np.exp(-fit['k'] * t_fit)
                            ax2.plot(t_fit, y_fit, '--', color=color, linewidth=2,
                                   label=f'{region} (Fit: k={fit["k"]:.2e} s⁻¹)', alpha=0.7)

        ax2.set_xlabel('Exposure Time (s)', fontsize=10)
        ax2.set_ylabel('Average Absorbance', fontsize=10)
        ax2.set_title('Kinetic Analysis of Key Regions', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)

        # 3. Difference spectra
        ax3 = fig.add_subplot(223)
        if 'difference_spectra' in self.analyzer.analysis_results:
            diff_spectra = self.analyzer.analysis_results['difference_spectra']
            for time, diff_data in diff_spectra.items():
                # Ensure proper sorting
                diff_sorted = diff_data.sort_values('Wavenumber')
                ax3.plot(diff_sorted['Wavenumber'], diff_sorted['Difference'],
                        label=f'{time}s', linewidth=1.5)

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ax3.set_ylabel('Absorbance Difference', fontsize=10)
        ax3.set_title('Difference Spectra Analysis', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)

        # 4. PCA scores
        ax4 = fig.add_subplot(224)
        if 'pca' in self.analyzer.analysis_results:
            pca = self.analyzer.analysis_results['pca']
            scores = pca['scores']
            times = pca['times']

            scatter = ax4.scatter(scores[:, 0], scores[:, 1], c=times,
                                cmap='viridis', s=100, alpha=0.7)

            # Add time labels
            for i, time in enumerate(times):
                ax4.annotate(f'{time}s', (scores[i, 0], scores[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax4.set_xlabel(f'PC1 ({pca["explained_variance_ratio"][0]*100:.1f}%)', fontsize=10)
            ax4.set_ylabel(f'PC2 ({pca["explained_variance_ratio"][1]*100:.1f}%)', fontsize=10)
            ax4.set_title('PCA Score Plot', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=8)

            # Add color bar
            try:
                cbar = fig.colorbar(scatter, ax=ax4)
                cbar.set_label('Exposure Time (s)', fontsize=10)
                cbar.ax.tick_params(labelsize=8)
            except:
                pass

        # Add overall title
        fig.suptitle('FTIR Spectral Analysis Results', fontsize=16, fontweight='bold')
        fig.tight_layout()

        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar for plot interaction
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()

    def save_results(self):
        """Save analysis results"""
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "No analysis results to save!")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                if filename.endswith('.csv'):
                    self.save_results_csv(filename)
                elif filename.endswith('.xlsx'):
                    self.save_results_excel(filename)
                else:
                    self.save_results_text(filename)

                self.status_var.set(f"Results saved: {filename.split('/')[-1]}")
                messagebox.showinfo("Success", "Results saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def save_results_text(self, filename):
        """Save results as text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("FTIR Spectral Analysis Results\n")
            f.write("=" * 40 + "\n\n")

            # Write analysis parameters
            selected_times = self.get_selected_times()
            min_wn, max_wn = self.get_wavenumber_range()
            f.write("Analysis Parameters:\n")
            f.write(f"  Selected time points: {selected_times}\n")
            f.write(f"  Wavenumber range: {min_wn:.0f} - {max_wn:.0f} cm⁻¹\n")
            f.write(f"  Baseline method: {self.baseline_var.get()}\n")
            f.write(f"  Normalization method: {self.norm_var.get()}\n\n")

            # Write key analysis results
            if 'key_analysis' in self.analyzer.analysis_results:
                f.write("Key Wavenumber Analysis:\n")
                for region, data in self.analyzer.analysis_results['key_analysis'].items():
                    f.write(f"\n{region}:\n")
                    f.write(f"  Range: {data['wavenumber_range']} cm⁻¹\n")
                    if data['kinetic_fit']:
                        fit = data['kinetic_fit']
                        f.write(f"  Rate constant: {fit['k']:.2e} s⁻¹\n")
                        f.write(f"  R²: {fit['r2']:.3f}\n")
                        f.write(f"  Half-life: {fit['half_life']:.1f} s\n")

            # Write PCA results
            if 'pca' in self.analyzer.analysis_results:
                pca = self.analyzer.analysis_results['pca']
                f.write(f"\nPCA Analysis:\n")
                f.write(f"  PC1 variance explained: {pca['explained_variance_ratio'][0]*100:.1f}%\n")
                f.write(f"  PC2 variance explained: {pca['explained_variance_ratio'][1]*100:.1f}%\n")
                f.write(f"  Total variance (PC1+PC2): {sum(pca['explained_variance_ratio'][:2])*100:.1f}%\n")

    def save_results_csv(self, filename):
        """Save results as CSV file"""
        # Save key analysis time series data
        if 'key_analysis' in self.analyzer.analysis_results:
            all_data = []
            for region, data in self.analyzer.analysis_results['key_analysis'].items():
                if 'time_series' in data:
                    ts = data['time_series']
                    for _, row in ts.iterrows():
                        all_data.append({
                            'Region': region,
                            'Time': row['time'],
                            'Absorbance': row['absorbance']
                        })

            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(filename, index=False)

    def save_results_excel(self, filename):
        """Save results as Excel file"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Save key analysis data
            if 'key_analysis' in self.analyzer.analysis_results:
                for region, data in self.analyzer.analysis_results['key_analysis'].items():
                    if 'time_series' in data:
                        ts = data['time_series']
                        sheet_name = region.replace('/', '_')[:31]  # Excel sheet name limit
                        ts.to_excel(writer, sheet_name=sheet_name, index=False)

    def save_plots(self):
        """Save current plots"""
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            messagebox.showerror("Error", "No plots to save! Please run analysis first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Plots",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                # Get DPI based on file format
                dpi = 300 if filename.lower().endswith(('.png', '.jpg', '.jpeg')) else None

                self.current_figure.savefig(filename, dpi=dpi, bbox_inches='tight',
                                          facecolor='white', edgecolor='none')
                self.status_var.set(f"Plots saved: {filename.split('/')[-1]}")
                messagebox.showinfo("Success", "Plots saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plots: {str(e)}")

    def plot_kinetics_detailed(self):
        """Create detailed kinetic analysis plots in a separate window"""
        if not self.analyzer.analysis_results or 'key_analysis' not in self.analyzer.analysis_results:
            messagebox.showerror("Error", "No kinetic analysis results available! Please run analysis first.")
            return

        # Create new window for kinetic plots
        kinetic_window = tk.Toplevel(self.root)
        kinetic_window.title("Detailed Kinetic Analysis")
        kinetic_window.geometry("1200x800")

        # Create figure with subplots for each region
        key_analysis = self.analyzer.analysis_results['key_analysis']
        n_regions = len(key_analysis)

        if n_regions == 0:
            messagebox.showinfo("Info", "No kinetic data available for plotting.")
            kinetic_window.destroy()
            return

        # Calculate subplot layout
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols

        fig = Figure(figsize=(15, 5 * n_rows))

        # Plot each region
        for i, (region, data) in enumerate(key_analysis.items()):
            if 'time_series' not in data:
                continue

            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ts = data['time_series']

            # Plot experimental data
            ax.scatter(ts['time'], ts['absorbance'],
                      color='red', s=80, alpha=0.8,
                      label='Experimental Data', zorder=3, edgecolors='darkred')

            # Plot kinetic fit if available
            if 'kinetic_fit' in data and data['kinetic_fit']:
                fit = data['kinetic_fit']
                t_fit = np.linspace(ts['time'].min(), ts['time'].max(), 200)

                if fit['model'] == 'first_order':
                    y_fit = fit['A0'] * np.exp(-fit['k'] * t_fit)
                    ax.plot(t_fit, y_fit, 'b-', linewidth=3, alpha=0.8,
                           label=f'First Order Fit\nk = {fit["k"]:.2e} s⁻¹\nR² = {fit["r2"]:.3f}')

                    # Add half-life annotation
                    if fit['half_life'] < np.inf:
                        ax.axvline(x=fit['half_life'], color='green', linestyle='--', alpha=0.7,
                                 label=f'Half-life = {fit["half_life"]:.1f} s')

            # Formatting
            ax.set_xlabel('Exposure Time (s)', fontsize=12)
            ax.set_ylabel('Average Absorbance', fontsize=12)
            ax.set_title(f'{region}\n({data["wavenumber_range"][0]:.0f}-{data["wavenumber_range"][1]:.0f} cm⁻¹)',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

            # Set reasonable y-axis limits
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        fig.suptitle('Detailed Kinetic Analysis - All Regions', fontsize=16, fontweight='bold')
        fig.tight_layout()

        # Embed in new window
        canvas = FigureCanvasTkAgg(fig, kinetic_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, kinetic_window)
        toolbar.update()

        # Add save button for kinetic plots
        button_frame = ttk.Frame(kinetic_window)
        button_frame.pack(side='bottom', fill='x', padx=10, pady=5)

        def save_kinetic_plots():
            filename = filedialog.asksaveasfilename(
                title="Save Kinetic Plots",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                try:
                    dpi = 300 if filename.lower().endswith(('.png', '.jpg', '.jpeg')) else None
                    fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                              facecolor='white', edgecolor='none')
                    messagebox.showinfo("Success", f"Kinetic plots saved: {filename.split('/')[-1]}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {str(e)}")

        ttk.Button(button_frame, text="Save Kinetic Plots",
                  command=save_kinetic_plots).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close",
                  command=kinetic_window.destroy).pack(side='right', padx=5)

    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    """Main function to run the application"""
    print("Starting FTIR Spectral Analysis System...")
    print("Key improvements:")
    print("- Fixed data ordering (no head-tail connection)")
    print("- All English labels and interface")
    print("- Enhanced data validation")
    print("- Proper wavenumber sorting")

    app = FTIRAnalysisGUI()
    app.run()


if __name__ == "__main__":
    main()
