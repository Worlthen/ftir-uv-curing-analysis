"""
FTIR UV Curing Analysis Engine
Specialized analysis for photopolymerization and UV curing processes
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
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib to use English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FTIRUVCuringAnalyzer:
    """
    Main FTIR analysis engine for UV curing processes
    
    Features:
    - UV curing specific analysis
    - Kinetic modeling
    - Chemical interpretation
    - Statistical analysis
    - Automated reporting
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.analysis_results = {}
        self.exposure_times = []
        self.wavenumbers = []
        
        # UV curing specific wavenumber regions
        self.uv_curing_regions = {
            'acrylate_cc': (1620, 1640),      # C=C double bonds in acrylates
            'aromatic_cc': (1580, 1620),      # Aromatic C=C
            'carbonyl': (1700, 1750),         # C=O stretch
            'ch_aliphatic': (2800, 3000),     # Aliphatic C-H
            'ch_aromatic': (3000, 3100),      # Aromatic C-H
            'oh_groups': (3200, 3600),        # O-H stretch
            'ether_co': (1000, 1300),         # C-O ether linkages
            'aromatic_ring': (1500, 1600),    # Aromatic ring vibrations
            'ch_bending': (1400, 1500),       # C-H bending
        }
        
    def load_data(self, filepath: str = 'integrated_spectra.csv') -> bool:
        """
        Load integrated spectral data
        
        Args:
            filepath: Path to the integrated CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data = pd.read_csv(filepath)
            
            # Validate required columns
            required_columns = ['Wavenumber', 'Absorbance', 'ExposureTime']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Extract unique exposure times and wavenumbers
            self.exposure_times = sorted(self.data['ExposureTime'].unique())
            self.wavenumbers = sorted(self.data['Wavenumber'].unique())
            
            logger.info(f"Loaded data: {len(self.exposure_times)} time points, {len(self.wavenumbers)} wavenumbers")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def baseline_correction(self, spectrum: np.ndarray, method: str = 'als', **kwargs) -> np.ndarray:
        """
        Apply baseline correction to spectrum
        
        Args:
            spectrum: Input spectrum
            method: Baseline correction method ('als', 'polynomial')
            **kwargs: Additional parameters for baseline correction
            
        Returns:
            Baseline-corrected spectrum
        """
        if method == 'als':
            return self._als_baseline(spectrum, **kwargs)
        elif method == 'polynomial':
            return self._polynomial_baseline(spectrum, **kwargs)
        else:
            logger.warning(f"Unknown baseline method: {method}")
            return spectrum
    
    def _als_baseline(self, y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction
        """
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
            
        return y - z
    
    def _polynomial_baseline(self, y: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        Polynomial baseline correction
        """
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return y - baseline
    
    def normalize_spectrum(self, spectrum: np.ndarray, method: str = 'max') -> np.ndarray:
        """
        Normalize spectrum
        
        Args:
            spectrum: Input spectrum
            method: Normalization method ('max', 'area', 'snv')
            
        Returns:
            Normalized spectrum
        """
        if method == 'max':
            return spectrum / np.max(np.abs(spectrum))
        elif method == 'area':
            return spectrum / np.trapz(np.abs(spectrum))
        elif method == 'snv':  # Standard Normal Variate
            return (spectrum - np.mean(spectrum)) / np.std(spectrum)
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return spectrum
    
    def preprocess_data(self, baseline_method: str = 'als', norm_method: str = 'max') -> pd.DataFrame:
        """
        Preprocess all spectral data
        
        Args:
            baseline_method: Baseline correction method
            norm_method: Normalization method
            
        Returns:
            Preprocessed data DataFrame
        """
        if self.data is None:
            logger.error("No data loaded")
            return None
        
        processed_data = []
        
        for time in self.exposure_times:
            time_data = self.data[self.data['ExposureTime'] == time].copy()
            time_data = time_data.sort_values('Wavenumber')
            
            # Apply baseline correction
            corrected_spectrum = self.baseline_correction(
                time_data['Absorbance'].values, 
                method=baseline_method
            )
            
            # Apply normalization
            normalized_spectrum = self.normalize_spectrum(
                corrected_spectrum, 
                method=norm_method
            )
            
            time_data['ProcessedAbsorbance'] = normalized_spectrum
            processed_data.append(time_data)
        
        self.processed_data = pd.concat(processed_data, ignore_index=True)
        logger.info("Data preprocessing completed")
        
        return self.processed_data
    
    def analyze_cc_consumption(self, wavenumber_range: Tuple[float, float] = (1620, 1640)) -> Dict:
        """
        Analyze C=C double bond consumption during UV curing
        
        Args:
            wavenumber_range: Wavenumber range for C=C analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if self.processed_data is None:
            logger.error("No processed data available")
            return {}
        
        # Extract C=C region data
        mask = (self.processed_data['Wavenumber'] >= wavenumber_range[0]) & \
               (self.processed_data['Wavenumber'] <= wavenumber_range[1])
        cc_data = self.processed_data[mask]
        
        # Calculate integrated peak area for each time point
        cc_areas = []
        for time in self.exposure_times:
            time_mask = cc_data['ExposureTime'] == time
            time_spectrum = cc_data[time_mask].sort_values('Wavenumber')
            
            if len(time_spectrum) > 0:
                area = trapz(time_spectrum['ProcessedAbsorbance'], 
                           time_spectrum['Wavenumber'])
                cc_areas.append(area)
            else:
                cc_areas.append(0)
        
        # Calculate conversion
        initial_area = cc_areas[0] if cc_areas else 0
        conversion = [(initial_area - area) / initial_area * 100 if initial_area > 0 else 0 
                     for area in cc_areas]
        
        # Fit kinetic models
        kinetic_results = self.fit_kinetic_models(self.exposure_times, conversion)
        
        results = {
            'wavenumber_range': wavenumber_range,
            'exposure_times': self.exposure_times,
            'cc_areas': cc_areas,
            'conversion_percent': conversion,
            'kinetic_models': kinetic_results
        }
        
        return results
    
    def fit_kinetic_models(self, times: List[float], conversion: List[float]) -> Dict:
        """
        Fit various kinetic models to conversion data
        
        Args:
            times: Exposure times
            conversion: Conversion percentages
            
        Returns:
            Dictionary with fitted model parameters
        """
        models = {}
        
        # Convert to numpy arrays
        t = np.array(times)
        conv = np.array(conversion)
        
        # Zero-order kinetics: C = k*t
        try:
            popt_zero, pcov_zero = optimize.curve_fit(
                lambda t, k: k * t, t, conv, maxfev=1000
            )
            r2_zero = self.calculate_r_squared(conv, popt_zero[0] * t)
            models['zero_order'] = {
                'rate_constant': popt_zero[0],
                'r_squared': r2_zero,
                'equation': 'C = k*t'
            }
        except:
            models['zero_order'] = {'error': 'Fitting failed'}
        
        # First-order kinetics: C = C_max * (1 - exp(-k*t))
        try:
            popt_first, pcov_first = optimize.curve_fit(
                lambda t, c_max, k: c_max * (1 - np.exp(-k * t)), 
                t, conv, maxfev=1000, bounds=([0, 0], [200, 10])
            )
            fitted_first = popt_first[0] * (1 - np.exp(-popt_first[1] * t))
            r2_first = self.calculate_r_squared(conv, fitted_first)
            models['first_order'] = {
                'c_max': popt_first[0],
                'rate_constant': popt_first[1],
                'r_squared': r2_first,
                'equation': 'C = C_max * (1 - exp(-k*t))'
            }
        except:
            models['first_order'] = {'error': 'Fitting failed'}
        
        # Autocatalytic model: C = C_max * t^n / (t50^n + t^n)
        try:
            popt_auto, pcov_auto = optimize.curve_fit(
                lambda t, c_max, t50, n: c_max * (t**n) / (t50**n + t**n), 
                t[1:], conv[1:], maxfev=1000, bounds=([0, 0, 0], [200, 100, 5])
            )
            fitted_auto = popt_auto[0] * (t**popt_auto[2]) / (popt_auto[1]**popt_auto[2] + t**popt_auto[2])
            r2_auto = self.calculate_r_squared(conv, fitted_auto)
            models['autocatalytic'] = {
                'c_max': popt_auto[0],
                't50': popt_auto[1],
                'n': popt_auto[2],
                'r_squared': r2_auto,
                'equation': 'C = C_max * t^n / (t50^n + t^n)'
            }
        except:
            models['autocatalytic'] = {'error': 'Fitting failed'}
        
        return models
    
    def calculate_r_squared(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        Calculate R-squared value
        """
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def analyze_multiple_regions(self, regions: Dict[str, Tuple[float, float]] = None) -> Dict:
        """
        Analyze multiple wavenumber regions
        
        Args:
            regions: Dictionary of region names and wavenumber ranges
            
        Returns:
            Dictionary containing analysis results for all regions
        """
        if regions is None:
            regions = self.uv_curing_regions
        
        results = {}
        
        for region_name, wavenumber_range in regions.items():
            logger.info(f"Analyzing region: {region_name} ({wavenumber_range[0]}-{wavenumber_range[1]} cm⁻¹)")
            
            # Analyze region
            region_results = self.analyze_cc_consumption(wavenumber_range)
            results[region_name] = region_results
        
        return results
    
    def perform_pca_analysis(self) -> Dict:
        """
        Perform Principal Component Analysis on spectral data
        
        Returns:
            Dictionary containing PCA results
        """
        if self.processed_data is None:
            logger.error("No processed data available")
            return {}
        
        # Prepare data matrix
        pivot_data = self.processed_data.pivot(
            index='ExposureTime', 
            columns='Wavenumber', 
            values='ProcessedAbsorbance'
        )
        
        # Handle missing values
        pivot_data = pivot_data.fillna(0)
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data)
        
        # Perform PCA
        pca = PCA()
        pca_scores = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        results = {
            'scores': pca_scores,
            'loadings': pca.components_,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'exposure_times': pivot_data.index.values,
            'wavenumbers': pivot_data.columns.values
        }
        
        logger.info(f"PCA completed: {len(explained_variance)} components")
        logger.info(f"First 3 PCs explain {cumulative_variance[2]*100:.1f}% of variance")

        return results

    def calculate_difference_spectra(self, reference_time: float = None) -> pd.DataFrame:
        """
        Calculate difference spectra relative to reference time

        Args:
            reference_time: Reference time point (default: first time point)

        Returns:
            DataFrame containing difference spectra
        """
        if self.processed_data is None:
            logger.error("No processed data available")
            return pd.DataFrame()

        if reference_time is None:
            reference_time = min(self.exposure_times)

        # Get reference spectrum
        ref_data = self.processed_data[
            self.processed_data['ExposureTime'] == reference_time
        ].sort_values('Wavenumber')

        difference_data = []

        for time in self.exposure_times:
            if time == reference_time:
                continue

            time_data = self.processed_data[
                self.processed_data['ExposureTime'] == time
            ].sort_values('Wavenumber')

            # Calculate difference
            merged = pd.merge(
                ref_data[['Wavenumber', 'ProcessedAbsorbance']],
                time_data[['Wavenumber', 'ProcessedAbsorbance']],
                on='Wavenumber',
                suffixes=('_ref', '_current')
            )

            merged['DifferenceAbsorbance'] = merged['ProcessedAbsorbance_current'] - merged['ProcessedAbsorbance_ref']
            merged['ExposureTime'] = time

            difference_data.append(merged[['Wavenumber', 'DifferenceAbsorbance', 'ExposureTime']])

        if difference_data:
            return pd.concat(difference_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def identify_significant_peaks(self, difference_data: pd.DataFrame, threshold: float = 0.01) -> Dict:
        """
        Identify significant peaks in difference spectra

        Args:
            difference_data: Difference spectra DataFrame
            threshold: Minimum peak height threshold

        Returns:
            Dictionary containing peak information
        """
        significant_peaks = {}

        for time in difference_data['ExposureTime'].unique():
            time_data = difference_data[
                difference_data['ExposureTime'] == time
            ].sort_values('Wavenumber')

            spectrum = time_data['DifferenceAbsorbance'].values
            wavenumbers = time_data['Wavenumber'].values

            # Find positive peaks (product formation)
            pos_peaks, _ = signal.find_peaks(spectrum, height=threshold)

            # Find negative peaks (reactant consumption)
            neg_peaks, _ = signal.find_peaks(-spectrum, height=threshold)

            significant_peaks[time] = {
                'positive_peaks': {
                    'wavenumbers': wavenumbers[pos_peaks],
                    'intensities': spectrum[pos_peaks]
                },
                'negative_peaks': {
                    'wavenumbers': wavenumbers[neg_peaks],
                    'intensities': spectrum[neg_peaks]
                }
            }

        return significant_peaks

    def run_automated_analysis(self, baseline_method: str = 'als', norm_method: str = 'max') -> Dict:
        """
        Run complete automated analysis pipeline

        Args:
            baseline_method: Baseline correction method
            norm_method: Normalization method

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting automated FTIR UV curing analysis")

        results = {
            'metadata': {
                'baseline_method': baseline_method,
                'normalization_method': norm_method,
                'analysis_regions': self.uv_curing_regions,
                'exposure_times': self.exposure_times,
                'wavenumber_range': [min(self.wavenumbers), max(self.wavenumbers)]
            }
        }

        # 1. Preprocess data
        logger.info("Step 1: Preprocessing data")
        self.preprocess_data(baseline_method, norm_method)

        # 2. Analyze multiple regions
        logger.info("Step 2: Analyzing multiple wavenumber regions")
        region_results = self.analyze_multiple_regions()
        results['region_analysis'] = region_results

        # 3. PCA analysis
        logger.info("Step 3: Performing PCA analysis")
        pca_results = self.perform_pca_analysis()
        results['pca_analysis'] = pca_results

        # 4. Difference spectra analysis
        logger.info("Step 4: Calculating difference spectra")
        diff_spectra = self.calculate_difference_spectra()
        results['difference_spectra'] = diff_spectra

        # 5. Peak identification
        logger.info("Step 5: Identifying significant peaks")
        significant_peaks = self.identify_significant_peaks(diff_spectra)
        results['significant_peaks'] = significant_peaks

        # 6. Chemical interpretation
        logger.info("Step 6: Chemical interpretation")
        chemical_interpretation = self.interpret_chemical_changes(region_results, significant_peaks)
        results['chemical_interpretation'] = chemical_interpretation

        logger.info("Automated analysis completed")
        self.analysis_results = results

        return results

    def interpret_chemical_changes(self, region_results: Dict, peak_data: Dict) -> Dict:
        """
        Interpret chemical changes based on spectral analysis

        Args:
            region_results: Results from multiple region analysis
            peak_data: Significant peak data

        Returns:
            Dictionary containing chemical interpretation
        """
        interpretation = {
            'reaction_mechanisms': [],
            'conversion_summary': {},
            'reaction_rates': {},
            'chemical_pathways': []
        }

        # Analyze C=C consumption (primary reaction)
        if 'acrylate_cc' in region_results:
            cc_data = region_results['acrylate_cc']
            if 'kinetic_models' in cc_data:
                best_model = self.find_best_kinetic_model(cc_data['kinetic_models'])
                interpretation['conversion_summary']['acrylate_cc'] = {
                    'final_conversion': max(cc_data['conversion_percent']),
                    'best_model': best_model,
                    'reaction_type': 'Photopolymerization'
                }

        # Analyze aromatic changes
        if 'aromatic_ring' in region_results:
            aromatic_data = region_results['aromatic_ring']
            interpretation['conversion_summary']['aromatic_ring'] = {
                'final_conversion': max(aromatic_data['conversion_percent']),
                'reaction_type': 'Aromatic modification'
            }

        # Identify reaction pathways
        interpretation['chemical_pathways'] = [
            {
                'pathway': 'Free radical polymerization',
                'evidence': 'C=C consumption in acrylate region',
                'wavenumber_range': '1620-1640 cm⁻¹'
            },
            {
                'pathway': 'Cross-linking formation',
                'evidence': 'Ether linkage formation',
                'wavenumber_range': '1000-1300 cm⁻¹'
            }
        ]

        return interpretation

    def find_best_kinetic_model(self, models: Dict) -> str:
        """
        Find the best kinetic model based on R-squared values

        Args:
            models: Dictionary of kinetic models

        Returns:
            Name of the best model
        """
        best_model = None
        best_r2 = -1

        for model_name, model_data in models.items():
            if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                best_r2 = model_data['r_squared']
                best_model = model_name

        return best_model if best_model else 'unknown'
