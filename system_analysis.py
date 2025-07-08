"""
Photoresist UV Exposure FTIR Spectral Analysis System
Complete implementation based on analysis tool architecture
Features include:
1. Photochemical reaction kinetics analysis
2. Molecular structure change characterization
3. Reaction kinetics modeling
4. Multivariate statistical analysis
5. GUI data range selection
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

# Set matplotlib to use English fonts and avoid Chinese character issues
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class FTIRAnalyzer:
    """Main class for FTIR spectral analysis"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.analysis_results = {}
        self.exposure_times = []
        self.wavenumbers = []

    def load_data(self, filepath='all_spectra.csv'):
        """Load spectral data"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {len(self.data)} data points")
            # Extract time points and wavenumbers
            self.exposure_times = sorted(self.data['ExposureTime'].unique())
            self.wavenumbers = sorted(self.data['Wavenumber'].unique())
            print(f"Exposure time points: {self.exposure_times}")
            print(f"Wavenumber range: {min(self.wavenumbers):.1f} - {max(self.wavenumbers):.1f} cm⁻¹")
            print(f"Total wavenumber points: {len(self.wavenumbers)}")
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

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
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        for i in range(niter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def preprocess_spectrum(self, spectrum_data, baseline_method='als',
                            normalize_method='max', smooth=True):
        """
        Spectrum preprocessing pipeline
        Parameters:
        spectrum_data: Single time point spectrum data
        baseline_method: Baseline correction method ('als', 'polynomial')
        normalize_method: Normalization method ('max', 'area', 'vector')
        smooth: Whether to apply smoothing
        """
        # Sort data by wavenumber to ensure proper order
        spectrum_data_sorted = spectrum_data.sort_values('Wavenumber')
        wavenumbers = spectrum_data_sorted['Wavenumber'].values
        absorbances = spectrum_data_sorted['Absorbance'].values

        # Ensure data is monotonic (left to right)
        if len(wavenumbers) > 1 and wavenumbers[0] > wavenumbers[-1]:
            wavenumbers = wavenumbers[::-1]
            absorbances = absorbances[::-1]

        # 1. Baseline Correction
        if baseline_method == 'als':
            baseline = self.baseline_correction_als(absorbances)
            corrected = absorbances - baseline
        elif baseline_method == 'polynomial':
            coeffs = np.polyfit(wavenumbers, absorbances, 3)
            baseline = np.polyval(coeffs, wavenumbers)
            corrected = absorbances - baseline
        else:
            corrected = absorbances

        # 2. Smoothing
        if smooth:
            corrected = signal.savgol_filter(corrected, 5, 2)

        # 3. Normalization
        if normalize_method == 'max':
            corrected = corrected / np.max(np.abs(corrected))
        elif normalize_method == 'area':
            corrected = corrected / trapz(np.abs(corrected), wavenumbers)
        elif normalize_method == 'vector':
            corrected = corrected / np.linalg.norm(corrected)

        return pd.DataFrame({
            'Wavenumber': wavenumbers,
            'Absorbance': corrected
        })

    def fit_kinetic_model(self, time_data, model='first_order'):
        """
        Kinetic model fitting
        Parameters:
        time_data: DataFrame containing time and absorbance columns
        model: Kinetic model type ('first_order', 'zero_order', 'second_order')
        """
        times = time_data['time'].values
        absorbances = time_data['absorbance'].values

        # Filter invalid data
        valid_mask = (absorbances > 0) & np.isfinite(absorbances)
        times = times[valid_mask]
        absorbances = absorbances[valid_mask]
        if len(times) < 3:
            return None

        try:
            if model == 'first_order':
                A0 = absorbances[0]
                ln_ratios = np.log(absorbances / A0)
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, ln_ratios)
                k = -slope
                r2 = r_value ** 2
                return {
                    'k': k,
                    'A0': A0,
                    'r2': r2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'model': 'first_order',
                    'half_life': np.log(2) / k if k > 0 else np.inf
                }
            elif model == 'zero_order':
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, absorbances)
                k = -slope
                A0 = intercept
                r2 = r_value ** 2
                return {
                    'k': k,
                    'A0': A0,
                    'r2': r2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'model': 'zero_order',
                    'half_life': A0 / (2 * k) if k > 0 else np.inf
                }
            elif model == 'second_order':
                A0 = absorbances[0]
                inv_diff = 1 / absorbances - 1 / A0
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, inv_diff)
                k = slope
                r2 = r_value ** 2
                return {
                    'k': k,
                    'A0': A0,
                    'r2': r2,
                    'p_value': p_value,
                    'std_err': std_err,
                    'model': 'second_order',
                    'half_life': 1 / (k * A0) if k > 0 else np.inf
                }
        except Exception as e:
            print(f"Kinetic fitting failed: {e}")
            return None

    def perform_pca(self, spectra_matrix, n_components=3):
        """
        Principal Component Analysis (PCA)
        Parameters:
        spectra_matrix: Spectral matrix (n_samples, n_features)
        n_components: Number of principal components
        """
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(spectra_matrix)

        # PCA analysis
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(scaled_data)

        return {
            'scores': scores,
            'loadings': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'scaler': scaler,
            'pca_model': pca
        }

    def calculate_difference_spectra(self, processed_data):
        """
        Calculate difference spectra
        Parameters:
        processed_data: Preprocessed spectral data dictionary
        """
        reference_time = min(processed_data.keys())
        reference_spectrum = processed_data[reference_time]
        reference_wavenumbers = reference_spectrum['Wavenumber'].values
        reference_absorbance = reference_spectrum['Absorbance'].values

        difference_spectra = {}

        for time in processed_data.keys():
            if time != reference_time:
                current_spectrum = processed_data[time]
                current_wavenumbers = current_spectrum['Wavenumber'].values
                current_absorbance = current_spectrum['Absorbance'].values

                # Interpolate to align with reference wavenumbers
                if len(current_wavenumbers) != len(reference_wavenumbers) or \
                        not np.allclose(current_wavenumbers, reference_wavenumbers):
                    current_absorbance_interp = np.interp(reference_wavenumbers,
                                                         current_wavenumbers,
                                                         current_absorbance)
                    diff = current_absorbance_interp - reference_absorbance
                else:
                    diff = current_absorbance - reference_absorbance

                difference_spectra[time] = pd.DataFrame({
                    'Wavenumber': reference_wavenumbers,
                    'Difference': diff,
                    'Time': time
                })

        return difference_spectra

    def analyze_key_wavenumbers(self, wavenumber_ranges=None):
        """
        Analyze kinetics of key wavenumber regions
        Parameters:
        wavenumber_ranges: Key wavenumber range dictionary
        """
        if self.data is None:
            print("Data not loaded")
            return {}

        if wavenumber_ranges is None:
            wavenumber_ranges = {
                'C=C unsaturated bonds': (1600, 1700),
                'C-H bending vibration': (1400, 1500),
                'Aromatic ring vibration': (1500, 1600),
                'C-O stretching': (1000, 1300)
            }

        key_analysis = {}

        for region_name, (wn_min, wn_max) in wavenumber_ranges.items():
            mask = (self.data['Wavenumber'] >= wn_min) & (self.data['Wavenumber'] <= wn_max)
            region_data = self.data[mask]

            if len(region_data) > 0:
                time_series = []
                for time in self.exposure_times:
                    time_data = region_data[region_data['ExposureTime'] == time]
                    if len(time_data) > 0:
                        avg_abs = time_data['Absorbance'].mean()
                        time_series.append({'time': time, 'absorbance': avg_abs})

                if len(time_series) > 2:
                    time_df = pd.DataFrame(time_series)
                    models = ['first_order', 'zero_order', 'second_order']
                    best_fit = None
                    best_r2 = -1

                    for model in models:
                        fit_result = self.fit_kinetic_model(time_df, model)
                        if fit_result and fit_result['r2'] > best_r2:
                            best_r2 = fit_result['r2']
                            best_fit = fit_result

                    key_analysis[region_name] = {
                        'wavenumber_range': (wn_min, wn_max),
                        'time_series': time_df,
                        'best_fit': best_fit,
                        'peak_position': region_data.groupby('ExposureTime')['Absorbance'].idxmax(),
                        'peak_intensity': region_data.groupby('ExposureTime')['Absorbance'].max()
                    }

        return key_analysis

    def perform_comprehensive_analysis(self, baseline_method='als',
                                     normalize_method='max',
                                     selected_times=None,
                                     selected_wavenumbers=None):
        """
        Perform comprehensive analysis
        Parameters:
        baseline_method: Baseline correction method
        normalize_method: Normalization method
        selected_times: Selected time points
        selected_wavenumbers: Selected wavenumber range
        """
        if self.data is None:
            print("Please load data first")
            return

        print("Starting comprehensive analysis...")

        analysis_data = self.data.copy()

        if selected_times:
            analysis_data = analysis_data[analysis_data['ExposureTime'].isin(selected_times)]
        if selected_wavenumbers:
            wn_min, wn_max = selected_wavenumbers
            analysis_data = analysis_data[
                (analysis_data['Wavenumber'] >= wn_min) &
                (analysis_data['Wavenumber'] <= wn_max)
            ]

        # 1. Data preprocessing
        print("1. Data preprocessing...")
        processed_data = {}
        for time in sorted(analysis_data['ExposureTime'].unique()):
            time_data = analysis_data[analysis_data['ExposureTime'] == time]
            processed_spectrum = self.preprocess_spectrum(
                time_data, baseline_method, normalize_method
            )
            processed_data[time] = processed_spectrum
        self.processed_data = processed_data

        # 2. Difference spectra analysis
        print("2. Difference spectra analysis...")
        difference_spectra = self.calculate_difference_spectra(processed_data)

        # 3. Key wavenumber kinetic analysis
        print("3. Key wavenumber kinetic analysis...")
        key_analysis = self.analyze_key_wavenumbers()

        # 4. PCA analysis
        print("4. Principal Component Analysis...")
        times = sorted(processed_data.keys())
        min_length = min(len(processed_data[t]['Absorbance']) for t in times)
        spectra_matrix = []
        for t in times:
            spectrum = processed_data[t]['Absorbance'].values[:min_length]
            spectra_matrix.append(spectrum)
        spectra_matrix = np.array(spectra_matrix)

        pca_results = self.perform_pca(spectra_matrix)

        # Save analysis results
        self.analysis_results = {
            'processed_data': processed_data,
            'difference_spectra': difference_spectra,
            'key_analysis': key_analysis,
            'pca_results': pca_results,
            'analysis_times': times,
            'parameters': {
                'baseline_method': baseline_method,
                'normalize_method': normalize_method,
                'selected_times': selected_times,
                'selected_wavenumbers': selected_wavenumbers
            }
        }

        print("Comprehensive analysis completed!")
        return self.analysis_results

    def generate_report(self):
        """Generate analysis report"""
        if not self.analysis_results:
            print("Please run analysis first")
            return

        report = []
        report.append("=" * 60)
        report.append("FTIR Spectral Analysis Report")
        report.append("=" * 60)

        # Basic information
        report.append("\nData Overview:")
        if self.data is not None:
            report.append(f"- Total data points: {len(self.data)}")
        report.append(f"- Exposure time points: {len(self.exposure_times)}")
        if self.exposure_times:
            report.append(f"- Time range: {min(self.exposure_times)} - {max(self.exposure_times)} sec")
        if self.wavenumbers:
            report.append(f"- Wavenumber range: {min(self.wavenumbers):.1f} - {max(self.wavenumbers):.1f} cm⁻¹")

        # Key wavenumber analysis
        if 'key_analysis' in self.analysis_results:
            report.append("\nKey Wavenumber Region Analysis:")
            for region, data in self.analysis_results['key_analysis'].items():
                if data['best_fit']:
                    fit = data['best_fit']
                    report.append(f"\n{region} ({data['wavenumber_range'][0]}-{data['wavenumber_range'][1]} cm⁻¹):")
                    report.append(f"  - Best model: {fit['model']}")
                    report.append(f"  - Rate constant: {fit['k']:.2e} s⁻¹")
                    report.append(f"  - R² value: {fit['r2']:.4f}")
                    report.append(f"  - Half-life: {fit['half_life']:.2f} sec")

        # PCA Results
        if 'pca_results' in self.analysis_results:
            pca = self.analysis_results['pca_results']
            report.append("\nPrincipal Component Analysis:")
            for i, var in enumerate(pca['explained_variance_ratio']):
                report.append(f"  - PC{i+1}: {var*100:.1f}% variance explained")
            report.append(f"  - Cumulative variance: {pca['cumulative_variance'][-1]*100:.1f}%")

        report_text = "\n".join(report)
        print(report_text)

        # Save report
        with open('ftir_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\nReport saved to: ftir_analysis_report.txt")

        return report_text


class FTIRAnalysisGUI:
    """FTIR Analysis GUI Interface"""

    def __init__(self):
        self.analyzer = FTIRAnalyzer()
        self.root = tk.Tk()
        self.root.title("FTIR Spectral Analysis System")
        self.root.geometry("1200x800")
        self.create_widgets()

    def create_widgets(self):
        """Create GUI components"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.data_frame = ttk.Frame(notebook)
        notebook.add(self.data_frame, text="Data Loading and Selection")
        self.create_data_selection_frame()

        self.analysis_frame = ttk.Frame(notebook)
        notebook.add(self.analysis_frame, text="Analysis Results")
        self.create_analysis_frame()

        self.plot_frame = ttk.Frame(notebook)
        notebook.add(self.plot_frame, text="Data Visualization")
        self.create_plot_frame()

    def create_data_selection_frame(self):
        file_frame = ttk.LabelFrame(self.data_frame, text="Data File", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(file_frame, text="Load Data File",
                   command=self.load_data_file).pack(side='left', padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=10)

        selection_frame = ttk.LabelFrame(self.data_frame, text="Data Range Selection", padding=10)
        selection_frame.pack(fill='both', expand=True, padx=10, pady=5)

        time_frame = ttk.Frame(selection_frame)
        time_frame.pack(fill='x', pady=5)
        ttk.Label(time_frame, text="Select Exposure Times:").pack(side='left')
        self.time_listbox = tk.Listbox(time_frame, selectmode='multiple', height=6)
        self.time_listbox.pack(side='left', padx=10, fill='x', expand=True)
        time_scroll = ttk.Scrollbar(time_frame, orient='vertical', command=self.time_listbox.yview)
        time_scroll.pack(side='right', fill='y')
        self.time_listbox.config(yscrollcommand=time_scroll.set)

        wavenumber_frame = ttk.Frame(selection_frame)
        wavenumber_frame.pack(fill='x', pady=5)
        ttk.Label(wavenumber_frame, text="Wavenumber Range:").pack(side='left')
        ttk.Label(wavenumber_frame, text="Min").pack(side='left', padx=(20, 5))
        self.wn_min_var = tk.StringVar()
        ttk.Entry(wavenumber_frame, textvariable=self.wn_min_var, width=10).pack(side='left', padx=5)
        ttk.Label(wavenumber_frame, text="Max").pack(side='left', padx=(20, 5))
        self.wn_max_var = tk.StringVar()
        ttk.Entry(wavenumber_frame, textvariable=self.wn_max_var, width=10).pack(side='left', padx=5)

        param_frame = ttk.LabelFrame(selection_frame, text="Analysis Parameters", padding=10)
        param_frame.pack(fill='x', pady=5)

        ttk.Label(param_frame, text="Baseline Correction:").grid(row=0, column=0, sticky='w', padx=5)
        self.baseline_var = tk.StringVar(value='als')
        baseline_combo = ttk.Combobox(param_frame, textvariable=self.baseline_var,
                                      values=['als', 'polynomial'], state='readonly')
        baseline_combo.grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(param_frame, text="Normalization:").grid(row=0, column=2, sticky='w', padx=5)
        self.normalize_var = tk.StringVar(value='max')
        normalize_combo = ttk.Combobox(param_frame, textvariable=self.normalize_var,
                                       values=['max', 'area', 'vector'], state='readonly')
        normalize_combo.grid(row=0, column=3, padx=5, sticky='w')

        ttk.Button(selection_frame, text="Run Analysis",
                   command=self.run_analysis).pack(pady=10)

    def create_analysis_frame(self):
        result_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Results", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.result_text = tk.Text(result_frame, wrap='word', height=20)
        self.result_text.pack(side='left', fill='both', expand=True)
        result_scroll = ttk.Scrollbar(result_frame, orient='vertical', command=self.result_text.yview)
        result_scroll.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=result_scroll.set)

        button_frame = ttk.Frame(self.analysis_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(button_frame, text="Generate Report",
                   command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Results",
                   command=self.save_results).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Results",
                   command=self.clear_results).pack(side='left', padx=5)

    def create_plot_frame(self):
        control_frame = ttk.LabelFrame(self.plot_frame, text="Plot Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(control_frame, text="Spectral Evolution",
                   command=self.plot_spectral_evolution).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Kinetics Plot",
                   command=self.plot_kinetics).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Difference Spectra",
                   command=self.plot_difference_spectra).pack(side='left', padx=5)
        ttk.Button(control_frame, text="PCA Analysis",
                   command=self.plot_pca).pack(side='left', padx=5)

        self.plot_frame_canvas = ttk.Frame(self.plot_frame)
        self.plot_frame_canvas.pack(fill='both', expand=True, padx=10, pady=5)

    def load_data_file(self):
        filename = filedialog.askopenfilename(
            title="Select FTIR Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            if self.analyzer.load_data(filename):
                self.file_label.config(text=f"Loaded: {filename.split('/')[-1]}")
                self.update_time_listbox()
                self.update_wavenumber_range()
                messagebox.showinfo("Success", "Data loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load data!")

    def update_time_listbox(self):
        self.time_listbox.delete(0, tk.END)
        for time in self.analyzer.exposure_times:
            self.time_listbox.insert(tk.END, f"{time}s")
        for i in range(len(self.analyzer.exposure_times)):
            self.time_listbox.selection_set(i)

    def update_wavenumber_range(self):
        if self.analyzer.wavenumbers:
            self.wn_min_var.set(str(min(self.analyzer.wavenumbers)))
            self.wn_max_var.set(str(max(self.analyzer.wavenumbers)))

    def run_analysis(self):
        if self.analyzer.data is None:
            messagebox.showerror("Error", "Please load a data file first!")
            return

        try:
            selected_indices = self.time_listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "Please select at least one time point!")
                return

            selected_times = [self.analyzer.exposure_times[i] for i in selected_indices]

            try:
                wn_min = float(self.wn_min_var.get()) if self.wn_min_var.get() else None
                wn_max = float(self.wn_max_var.get()) if self.wn_max_var.get() else None
                selected_wavenumbers = (wn_min, wn_max) if wn_min and wn_max else None
            except ValueError:
                selected_wavenumbers = None

            self.analyzer.perform_comprehensive_analysis(
                baseline_method=self.baseline_var.get(),
                normalize_method=self.normalize_var.get(),
                selected_times=selected_times,
                selected_wavenumbers=selected_wavenumbers
            )

            self.display_analysis_results()
            messagebox.showinfo("Success", "Analysis completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def display_analysis_results(self):
        if not self.analyzer.analysis_results:
            return

        results = self.analyzer.analysis_results
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "FTIR Spectral Analysis Results\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n")

        params = results['parameters']
        self.result_text.insert(tk.END, "Analysis Parameters:\n")
        self.result_text.insert(tk.END, f"- Baseline Method: {params['baseline_method']}\n")
        self.result_text.insert(tk.END, f"- Normalization Method: {params['normalize_method']}\n")
        self.result_text.insert(tk.END, f"- Analysis Time Points: {len(results['analysis_times'])} points\n")
        if params['selected_wavenumbers']:
            wn_min, wn_max = params['selected_wavenumbers']
            self.result_text.insert(tk.END, f"- Wavenumber Range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹\n")

        if 'key_analysis' in results:
            self.result_text.insert(tk.END, "\nKey Wavenumber Region Analysis:\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            for region, data in results['key_analysis'].items():
                if data['best_fit']:
                    fit = data['best_fit']
                    self.result_text.insert(tk.END, f"\n{region}:\n")
                    self.result_text.insert(tk.END, f"  Wavenumber Range: {data['wavenumber_range'][0]}-{data['wavenumber_range'][1]} cm⁻¹\n")
                    self.result_text.insert(tk.END, f"  Best Model: {fit['model']}\n")
                    self.result_text.insert(tk.END, f"  Rate Constant: {fit['k']:.2e} s⁻¹\n")
                    self.result_text.insert(tk.END, f"  R² Value: {fit['r2']:.4f}\n")
                    self.result_text.insert(tk.END, f"  Half-life: {fit['half_life']:.2f} sec\n")
                    self.result_text.insert(tk.END, f"  p-value: {fit['p_value']:.2e}\n")

        if 'pca_results' in results:
            pca = results['pca_results']
            self.result_text.insert(tk.END, "\nPrincipal Component Analysis Results:\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            for i, var in enumerate(pca['explained_variance_ratio']):
                self.result_text.insert(tk.END, f"PC{i+1}: {var*100:.1f}% variance explained\n")
            self.result_text.insert(tk.END, f"Cumulative Variance: {pca['cumulative_variance'][-1]*100:.1f}%\n")

    def generate_report(self):
        if self.analyzer.analysis_results:
            self.analyzer.generate_report()
            messagebox.showinfo("Success", "Report generated and saved to file!")
        else:
            messagebox.showerror("Error", "Please run analysis first!")

    def save_results(self):
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "No analysis results to save!")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                content = self.result_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Results saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)

    def plot_spectral_evolution(self):
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "Please run analysis first!")
            return

        for widget in self.plot_frame_canvas.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 8))
        ax1 = fig.add_subplot(221)
        processed_data = self.analyzer.analysis_results['processed_data']

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (time, spectrum) in enumerate(processed_data.items()):
            color = colors[i % len(colors)]
            ax1.plot(spectrum['Wavenumber'], spectrum['Absorbance'],
                     color=color, label=f'{time}s', linewidth=1.5)

        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Normalized Absorbance')
        ax1.set_title('Spectral Evolution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(222)
        if 'key_analysis' in self.analyzer.analysis_results:
            for region, data in self.analyzer.analysis_results['key_analysis'].items():
                if 'time_series' in data:
                    ts = data['time_series']
                    ax2.plot(ts['time'], ts['absorbance'], 'o-', label=region, linewidth=2)

        ax2.set_xlabel('Exposure Time (s)')
        ax2.set_ylabel('Average Absorbance')
        ax2.set_title('Key Wavenumber Region Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(223)
        if 'difference_spectra' in self.analyzer.analysis_results:
            diff_spectra = self.analyzer.analysis_results['difference_spectra']
            for time, diff_data in diff_spectra.items():
                ax3.plot(diff_data['Wavenumber'], diff_data['Difference'],
                         label=f'{time}s', linewidth=1.5)

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Wavenumber (cm⁻¹)')
        ax3.set_ylabel('Absorbance Difference')
        ax3.set_title('Difference Spectra Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(224)
        if 'pca_results' in self.analyzer.analysis_results:
            pca = self.analyzer.analysis_results['pca_results']
            scores = pca['scores']
            times = self.analyzer.analysis_results['analysis_times']
            scatter = ax4.scatter(scores[:, 0], scores[:, 1],
                                  c=times, cmap='viridis', s=100)
            for i, time in enumerate(times):
                ax4.annotate(f'{time}s', (scores[i, 0], scores[i, 1]),
                             xytext=(5, 5), textcoords='offset points')
            ax4.set_xlabel(f'PC1 ({pca["explained_variance_ratio"][0]*100:.1f}%)')
            ax4.set_ylabel(f'PC2 ({pca["explained_variance_ratio"][1]*100:.1f}%)')
            ax4.set_title('PCA Score Plot')
            ax4.grid(True, alpha=0.3)
            cbar = fig.colorbar(scatter, ax=ax4)
            cbar.set_label('Exposure Time (s)')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_kinetics(self):
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "Please run analysis first!")
            return

        for widget in self.plot_frame_canvas.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 8))
        if 'key_analysis' in self.analyzer.analysis_results:
            key_analysis = self.analyzer.analysis_results['key_analysis']
            n_regions = len(key_analysis)
            if n_regions > 0:
                rows = int(np.ceil(n_regions / 2))
                cols = min(2, n_regions)
                for i, (region, data) in enumerate(key_analysis.items()):
                    ax = fig.add_subplot(rows, cols, i + 1)
                    if 'time_series' in data and data['best_fit']:
                        ts = data['time_series']
                        fit = data['best_fit']
                        ax.scatter(ts['time'], ts['absorbance'],
                                   color='red', s=50, label='Experimental Data', zorder=3)
                        t_fit = np.linspace(ts['time'].min(), ts['time'].max(), 100)
                        y_fit = None
                        if fit['model'] == 'first_order':
                            y_fit = fit['A0'] * np.exp(-fit['k'] * t_fit)
                        elif fit['model'] == 'zero_order':
                            y_fit = fit['A0'] - fit['k'] * t_fit
                        elif fit['model'] == 'second_order':
                            y_fit = fit['A0'] / (1 + fit['k'] * fit['A0'] * t_fit)
                        if y_fit is not None:
                            ax.plot(t_fit, y_fit, 'b-', linewidth=2,
                                    label=f'{fit["model"]} (R²={fit["r2"]:.3f})')
                        ax.set_xlabel('Exposure Time (s)')
                        ax.set_ylabel('Absorbance')
                        ax.set_title(f'{region}\nk = {fit["k"]:.2e} s⁻¹')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_difference_spectra(self):
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "Please run analysis first!")
            return

        for widget in self.plot_frame_canvas.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        if 'difference_spectra' in self.analyzer.analysis_results:
            diff_spectra = self.analyzer.analysis_results['difference_spectra']
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, (time, diff_data) in enumerate(diff_spectra.items()):
                color = colors[i % len(colors)]
                ax.plot(diff_data['Wavenumber'], diff_data['Difference'],
                        color=color, label=f'{time}s', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Absorbance Difference')
            ax.set_title('Difference Spectra Analysis (Relative to Initial)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.95, 'Positive peaks: Product formation', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax.text(0.02, 0.05, 'Negative peaks: Reactant consumption', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_pca(self):
        if not self.analyzer.analysis_results:
            messagebox.showerror("Error", "Please run analysis first!")
            return

        for widget in self.plot_frame_canvas.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(12, 8))
        if 'pca_results' in self.analyzer.analysis_results:
            pca = self.analyzer.analysis_results['pca_results']
            scores = pca['scores']
            loadings = pca['loadings']
            times = self.analyzer.analysis_results['analysis_times']

            ax1 = fig.add_subplot(221)
            scatter = ax1.scatter(scores[:, 0], scores[:, 1],
                                  c=times, cmap='viridis', s=100, edgecolors='black')
            for i, time in enumerate(times):
                ax1.annotate(f'{time}s', (scores[i, 0], scores[i, 1]),
                             xytext=(5, 5), textcoords='offset points')
            ax1.plot(scores[:, 0], scores[:, 1], 'k--', alpha=0.5, linewidth=1)
            ax1.set_xlabel(f'PC1 ({pca["explained_variance_ratio"][0]*100:.1f}%)')
            ax1.set_ylabel(f'PC2 ({pca["explained_variance_ratio"][1]*100:.1f}%)')
            ax1.set_title('PCA Score Plot')
            ax1.grid(True, alpha=0.3)
            try:
                cbar = fig.colorbar(scatter, ax=ax1)
                cbar.set_label('Exposure Time (s)')
            except:
                pass

            ax2 = fig.add_subplot(222)
            pc_numbers = range(1, len(pca['explained_variance_ratio']) + 1)
            ax2.bar(pc_numbers, pca['explained_variance_ratio'] * 100, alpha=0.7)
            ax2.plot(pc_numbers, pca['cumulative_variance'] * 100, 'ro-', linewidth=2)
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Variance Explained (%)')
            ax2.set_title('Variance Explained Plot')
            ax2.grid(True, alpha=0.3)

            ax3 = fig.add_subplot(223)
            wavenumbers = self.analyzer.wavenumbers
            if len(wavenumbers) == len(loadings[0]):
                ax3.plot(wavenumbers, loadings[0], 'b-', linewidth=2)
                ax3.set_xlabel('Wavenumber (cm⁻¹)')
                ax3.set_ylabel('PC1 Loadings')
                ax3.set_title('PC1 Loadings Plot')
                ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(224)
            if len(wavenumbers) == len(loadings[1]):
                ax4.plot(wavenumbers, loadings[1], 'r-', linewidth=2)
                ax4.set_xlabel('Wavenumber (cm⁻¹)')
                ax4.set_ylabel('PC2 Loadings')
                ax4.set_title('PC2 Loadings Plot')
                ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def run(self):
        self.root.mainloop()


def main():
    print("Launching FTIR Spectral Analysis System...")
    app = FTIRAnalysisGUI()
    if app.analyzer.load_data():
        print("Data loaded successfully, updating GUI info")
        app.file_label.config(text="Loaded: all_spectra.csv")
        app.update_time_listbox()
        app.update_wavenumber_range()
    else:
        print("Automatic data loading failed, please load manually in GUI")
    app.run()


if __name__ == "__main__":
    main()