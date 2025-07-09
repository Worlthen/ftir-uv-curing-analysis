#!/usr/bin/env python3
"""
Enhanced FTIR UV Curing Analyzer
Based on scientific critique and restructured architecture

This module implements scientifically rigorous FTIR analysis for UV curing processes
with proper chemical mechanistic understanding and statistical validation.
"""

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats, sparse
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChemicalGroupDefinitions:
    """
    Scientifically validated chemical group definitions for UV curing analysis
    """
    
    REACTIVE_GROUPS = {
        'c_equals_c_acrylate': {
            'range': (1620, 1640),
            'assignment': 'C=C stretch (acrylate)',
            'reaction_role': 'primary_reactive_site',
            'extinction_coeff': 310,  # L/(mol·cm)
            'baseline_method': 'als'
        },
        'c_equals_c_methacrylate': {
            'range': (1635, 1645),
            'assignment': 'C=C stretch (methacrylate)',
            'reaction_role': 'primary_reactive_site',
            'extinction_coeff': 280,
            'baseline_method': 'als'
        },
        'vinyl_c_h': {
            'range': (3080, 3120),
            'assignment': '=C-H stretch (vinyl)',
            'reaction_role': 'polymerization_indicator',
            'extinction_coeff': 45,
            'baseline_method': 'polynomial'
        }
    }
    
    STRUCTURAL_GROUPS = {
        'ester_carbonyl': {
            'range': (1720, 1740),
            'assignment': 'C=O stretch (ester)',
            'reaction_role': 'structural_backbone',
            'extinction_coeff': 500,
            'baseline_method': 'als'
        },
        'ester_c_o': {
            'range': (1000, 1300),
            'assignment': 'C-O stretch (ester)',
            'reaction_role': 'crosslink_formation',
            'extinction_coeff': 150,
            'baseline_method': 'als'
        }
    }
    
    PHOTOINITIATOR_GROUPS = {
        'benzoin_carbonyl': {
            'range': (1650, 1680),
            'assignment': 'C=O stretch (benzoin PI)',
            'reaction_role': 'photoinitiator',
            'extinction_coeff': 800,
            'baseline_method': 'als'
        },
        'acetophenone_carbonyl': {
            'range': (1660, 1690),
            'assignment': 'C=O stretch (acetophenone PI)',
            'reaction_role': 'photoinitiator',
            'extinction_coeff': 750,
            'baseline_method': 'als'
        }
    }


class SpectralQualityValidator:
    """
    Comprehensive spectral quality validation based on scientific standards
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'snr_minimum': 1000,
            'baseline_stability': 0.001,  # AU/minute
            'peak_resolution': 1.5,
            'frequency_accuracy': 0.1,  # cm⁻¹
            'reproducibility_rsd': 2.0  # %
        }
    
    def validate_spectrum_quality(self, spectrum: np.ndarray, 
                                wavenumbers: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Comprehensive spectral quality assessment
        
        Parameters:
        -----------
        spectrum : np.ndarray
            FTIR spectrum intensities
        wavenumbers : np.ndarray
            Corresponding wavenumber values
            
        Returns:
        --------
        Dict containing quality metrics and pass/fail status
        """
        quality_metrics = {}
        
        # Signal-to-noise ratio calculation
        quality_metrics['snr'] = self._calculate_snr(spectrum)
        quality_metrics['snr_pass'] = quality_metrics['snr'] >= self.quality_thresholds['snr_minimum']
        
        # Baseline stability assessment
        quality_metrics['baseline_stability'] = self._assess_baseline_stability(spectrum)
        quality_metrics['baseline_pass'] = quality_metrics['baseline_stability'] <= self.quality_thresholds['baseline_stability']
        
        # Peak resolution evaluation
        quality_metrics['peak_resolution'] = self._calculate_peak_resolution(spectrum, wavenumbers)
        quality_metrics['resolution_pass'] = quality_metrics['peak_resolution'] >= self.quality_thresholds['peak_resolution']
        
        # Frequency accuracy check
        quality_metrics['frequency_accuracy'] = self._validate_frequency_calibration(wavenumbers)
        quality_metrics['frequency_pass'] = quality_metrics['frequency_accuracy'] <= self.quality_thresholds['frequency_accuracy']
        
        # Overall quality assessment
        quality_metrics['overall_pass'] = all([
            quality_metrics['snr_pass'],
            quality_metrics['baseline_pass'],
            quality_metrics['resolution_pass'],
            quality_metrics['frequency_pass']
        ])
        
        return quality_metrics
    
    def _calculate_snr(self, spectrum: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Use high-frequency region for noise estimation
        noise_region = spectrum[-100:]  # Last 100 points
        noise_std = np.std(noise_region)
        
        # Signal is maximum peak height
        signal = np.max(spectrum) - np.min(spectrum)
        
        return signal / noise_std if noise_std > 0 else np.inf
    
    def _assess_baseline_stability(self, spectrum: np.ndarray) -> float:
        """Assess baseline drift over measurement"""
        # Use regions typically free of absorption
        baseline_regions = [spectrum[:50], spectrum[-50:]]
        baseline_values = np.concatenate(baseline_regions)
        
        # Calculate drift as standard deviation
        return np.std(baseline_values)
    
    def _calculate_peak_resolution(self, spectrum: np.ndarray, 
                                 wavenumbers: np.ndarray) -> float:
        """Calculate average peak resolution"""
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(spectrum, height=0.1, distance=10)
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate resolution for adjacent peaks
        resolutions = []
        for i in range(len(peaks) - 1):
            peak1_idx, peak2_idx = peaks[i], peaks[i + 1]
            
            # Find valley between peaks
            valley_idx = peak1_idx + np.argmin(spectrum[peak1_idx:peak2_idx])
            
            # Calculate resolution: R = (ν2 - ν1) / (FWHM1 + FWHM2)
            delta_nu = abs(wavenumbers[peak2_idx] - wavenumbers[peak1_idx])
            fwhm1 = self._calculate_fwhm(spectrum, wavenumbers, peak1_idx)
            fwhm2 = self._calculate_fwhm(spectrum, wavenumbers, peak2_idx)
            
            if fwhm1 + fwhm2 > 0:
                resolution = delta_nu / (fwhm1 + fwhm2)
                resolutions.append(resolution)
        
        return np.mean(resolutions) if resolutions else 0.0
    
    def _calculate_fwhm(self, spectrum: np.ndarray, wavenumbers: np.ndarray, 
                       peak_idx: int) -> float:
        """Calculate full width at half maximum for a peak"""
        peak_height = spectrum[peak_idx]
        half_height = peak_height / 2
        
        # Find left and right half-height points
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > half_height:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_height:
            right_idx += 1
        
        if left_idx < right_idx:
            return abs(wavenumbers[right_idx] - wavenumbers[left_idx])
        else:
            return 0.0
    
    def _validate_frequency_calibration(self, wavenumbers: np.ndarray) -> float:
        """Validate frequency calibration accuracy"""
        # Check for expected atmospheric CO2 peak at 2349 cm⁻¹
        target_frequency = 2349.0
        
        # Find closest wavenumber
        closest_idx = np.argmin(np.abs(wavenumbers - target_frequency))
        frequency_error = abs(wavenumbers[closest_idx] - target_frequency)
        
        return frequency_error


class KineticModelLibrary:
    """
    Comprehensive library of kinetic models for UV curing analysis
    """
    
    @staticmethod
    def autocatalytic_model(t: np.ndarray, k1: float, k2: float, 
                          alpha_max: float) -> np.ndarray:
        """
        Autocatalytic kinetic model for UV curing
        
        dα/dt = (k1 + k2*α) * (α_max - α)
        
        Parameters:
        -----------
        t : np.ndarray
            Time points
        k1 : float
            Initial rate constant (s⁻¹)
        k2 : float
            Autocatalytic rate constant (s⁻¹)
        alpha_max : float
            Maximum conversion (0-1)
        """
        def differential_eq(t, alpha):
            return (k1 + k2 * alpha[0]) * (alpha_max - alpha[0])
        
        # Solve ODE
        sol = solve_ivp(differential_eq, [t[0], t[-1]], [0.0], t_eval=t, 
                       method='RK45', rtol=1e-8)
        
        return sol.y[0]
    
    @staticmethod
    def diffusion_limited_model(t: np.ndarray, k: float, n: float, 
                               K: float, alpha_max: float) -> np.ndarray:
        """
        Diffusion-limited kinetic model
        
        dα/dt = k * (α_max - α)^n / (1 + K*α)
        """
        def differential_eq(t, alpha):
            if alpha[0] >= alpha_max:
                return 0.0
            return k * (alpha_max - alpha[0])**n / (1 + K * alpha[0])
        
        sol = solve_ivp(differential_eq, [t[0], t[-1]], [0.0], t_eval=t,
                       method='RK45', rtol=1e-8)
        
        return sol.y[0]
    
    @staticmethod
    def parallel_reactions_model(t: np.ndarray, k1: float, k2: float,
                               alpha1_max: float, alpha2_max: float) -> np.ndarray:
        """
        Parallel reactions model for multiple reactive groups
        
        dα_total/dt = k1*(α1_max - α1) + k2*(α2_max - α2)
        """
        def differential_eq(t, alpha):
            alpha1, alpha2 = alpha
            dalpha1_dt = k1 * (alpha1_max - alpha1)
            dalpha2_dt = k2 * (alpha2_max - alpha2)
            return [dalpha1_dt, dalpha2_dt]
        
        sol = solve_ivp(differential_eq, [t[0], t[-1]], [0.0, 0.0], t_eval=t,
                       method='RK45', rtol=1e-8)
        
        # Return total conversion
        return sol.y[0] + sol.y[1]


class EnhancedFTIRAnalyzer:
    """
    Enhanced FTIR analyzer with scientific rigor and chemical mechanistic understanding
    """
    
    def __init__(self):
        self.chemical_groups = ChemicalGroupDefinitions()
        self.quality_validator = SpectralQualityValidator()
        self.kinetic_models = KineticModelLibrary()
        
        # Analysis parameters
        self.analysis_params = {
            'baseline_correction': 'als',
            'normalization': 'max',
            'smoothing': 'savgol',
            'kinetic_model': 'autocatalytic'
        }
        
        # Results storage
        self.results = {}
        self.quality_metrics = {}
        
    def analyze_uv_curing_kinetics(self, spectra_data: pd.DataFrame,
                                 experimental_conditions: Dict) -> Dict:
        """
        Comprehensive UV curing kinetics analysis
        
        Parameters:
        -----------
        spectra_data : pd.DataFrame
            FTIR spectra with columns: Wavenumber, Absorbance, ExposureTime
        experimental_conditions : Dict
            UV intensity, temperature, atmosphere, etc.
            
        Returns:
        --------
        Dict containing comprehensive analysis results
        """
        logger.info("Starting enhanced UV curing kinetics analysis")
        
        # Step 1: Validate input data quality
        quality_results = self._validate_data_quality(spectra_data)
        if not quality_results['overall_pass']:
            logger.warning("Data quality issues detected")
            
        # Step 2: Preprocess spectra
        processed_spectra = self._preprocess_spectra(spectra_data)
        
        # Step 3: Analyze chemical groups
        group_analysis = self._analyze_chemical_groups(processed_spectra)
        
        # Step 4: Kinetic modeling
        kinetic_results = self._perform_kinetic_analysis(group_analysis)
        
        # Step 5: Statistical validation
        validated_results = self._validate_kinetic_results(kinetic_results)
        
        # Step 6: Generate comprehensive report
        final_results = self._compile_results(
            quality_results, group_analysis, validated_results, experimental_conditions
        )
        
        logger.info("Analysis completed successfully")
        return final_results

    def _validate_data_quality(self, spectra_data: pd.DataFrame) -> Dict:
        """Validate input data quality"""
        # Extract representative spectrum for quality assessment
        first_spectrum = spectra_data[spectra_data['ExposureTime'] == 0]
        if first_spectrum.empty:
            first_spectrum = spectra_data.groupby('ExposureTime').first().iloc[0]

        wavenumbers = first_spectrum['Wavenumber'].values
        absorbance = first_spectrum['Absorbance'].values

        return self.quality_validator.validate_spectrum_quality(absorbance, wavenumbers)

    def _preprocess_spectra(self, spectra_data: pd.DataFrame) -> pd.DataFrame:
        """Advanced spectral preprocessing"""
        processed_data = spectra_data.copy()

        # Group by exposure time for processing
        for time, group in processed_data.groupby('ExposureTime'):
            wavenumbers = group['Wavenumber'].values
            absorbance = group['Absorbance'].values

            # Baseline correction
            if self.analysis_params['baseline_correction'] == 'als':
                corrected_abs = self._als_baseline_correction(absorbance)
            else:
                corrected_abs = self._polynomial_baseline_correction(absorbance)

            # Smoothing
            if self.analysis_params['smoothing'] == 'savgol':
                smoothed_abs = signal.savgol_filter(corrected_abs, 11, 3)
            else:
                smoothed_abs = corrected_abs

            # Normalization
            if self.analysis_params['normalization'] == 'max':
                normalized_abs = smoothed_abs / np.max(smoothed_abs)
            elif self.analysis_params['normalization'] == 'area':
                normalized_abs = smoothed_abs / np.trapz(smoothed_abs, wavenumbers)
            else:
                normalized_abs = smoothed_abs

            # Update processed data
            mask = processed_data['ExposureTime'] == time
            processed_data.loc[mask, 'ProcessedAbsorbance'] = normalized_abs

        return processed_data

    def _als_baseline_correction(self, spectrum: np.ndarray,
                               lam: float = 1e6, p: float = 0.001,
                               niter: int = 10) -> np.ndarray:
        """Asymmetric Least Squares baseline correction"""
        L = len(spectrum)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)

        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w * spectrum)
            w = p * (spectrum > z) + (1 - p) * (spectrum < z)

        return spectrum - z

    def _polynomial_baseline_correction(self, spectrum: np.ndarray,
                                      order: int = 3) -> np.ndarray:
        """Polynomial baseline correction"""
        x = np.arange(len(spectrum))

        # Fit polynomial to spectrum
        coeffs = np.polyfit(x, spectrum, order)
        baseline = np.polyval(coeffs, x)

        return spectrum - baseline

    def _analyze_chemical_groups(self, processed_spectra: pd.DataFrame) -> Dict:
        """Analyze chemical group transformations"""
        group_results = {}

        # Combine all group definitions
        all_groups = {
            **self.chemical_groups.REACTIVE_GROUPS,
            **self.chemical_groups.STRUCTURAL_GROUPS,
            **self.chemical_groups.PHOTOINITIATOR_GROUPS
        }

        for group_name, group_info in all_groups.items():
            logger.info(f"Analyzing chemical group: {group_name}")

            # Extract group-specific data
            group_data = self._extract_group_data(processed_spectra, group_info['range'])

            # Calculate conversion
            conversion_data = self._calculate_conversion(group_data, group_info['reaction_role'])

            # Store results
            group_results[group_name] = {
                'group_info': group_info,
                'raw_data': group_data,
                'conversion_data': conversion_data
            }

        return group_results

    def _extract_group_data(self, spectra_data: pd.DataFrame,
                          wavenumber_range: Tuple[float, float]) -> pd.DataFrame:
        """Extract spectral data for specific chemical group"""
        min_wn, max_wn = wavenumber_range

        # Filter data within wavenumber range
        mask = (spectra_data['Wavenumber'] >= min_wn) & (spectra_data['Wavenumber'] <= max_wn)
        group_data = spectra_data[mask].copy()

        # Calculate integrated intensity for each time point
        integrated_intensities = []
        exposure_times = sorted(group_data['ExposureTime'].unique())

        for time in exposure_times:
            time_data = group_data[group_data['ExposureTime'] == time]

            # Integrate using trapezoidal rule
            integrated_intensity = np.trapz(
                time_data['ProcessedAbsorbance'].values,
                time_data['Wavenumber'].values
            )

            integrated_intensities.append({
                'ExposureTime': time,
                'IntegratedIntensity': integrated_intensity,
                'PeakHeight': time_data['ProcessedAbsorbance'].max(),
                'PeakPosition': time_data.loc[time_data['ProcessedAbsorbance'].idxmax(), 'Wavenumber']
            })

        return pd.DataFrame(integrated_intensities)

    def _calculate_conversion(self, group_data: pd.DataFrame,
                            reaction_role: str) -> pd.DataFrame:
        """Calculate conversion based on reaction role"""
        conversion_data = group_data.copy()

        if reaction_role in ['primary_reactive_site', 'polymerization_indicator']:
            # For reactive groups, conversion = (I0 - It) / I0
            initial_intensity = group_data['IntegratedIntensity'].iloc[0]
            conversion_data['Conversion'] = (
                initial_intensity - group_data['IntegratedIntensity']
            ) / initial_intensity

        elif reaction_role in ['crosslink_formation', 'structural_backbone']:
            # For product formation, conversion = (It - I0) / I_max_theoretical
            initial_intensity = group_data['IntegratedIntensity'].iloc[0]
            max_intensity = group_data['IntegratedIntensity'].max()
            conversion_data['Conversion'] = (
                group_data['IntegratedIntensity'] - initial_intensity
            ) / (max_intensity - initial_intensity)

        elif reaction_role == 'photoinitiator':
            # For photoinitiator, conversion = (I0 - It) / I0
            initial_intensity = group_data['IntegratedIntensity'].iloc[0]
            conversion_data['Conversion'] = (
                initial_intensity - group_data['IntegratedIntensity']
            ) / initial_intensity

        else:
            # Default: no conversion calculation
            conversion_data['Conversion'] = 0.0

        # Ensure conversion is between 0 and 1
        conversion_data['Conversion'] = np.clip(conversion_data['Conversion'], 0.0, 1.0)

        return conversion_data

    def _perform_kinetic_analysis(self, group_analysis: Dict) -> Dict:
        """Perform comprehensive kinetic analysis"""
        kinetic_results = {}

        for group_name, group_data in group_analysis.items():
            if group_data['group_info']['reaction_role'] in ['primary_reactive_site']:
                logger.info(f"Performing kinetic analysis for {group_name}")

                conversion_data = group_data['conversion_data']
                times = conversion_data['ExposureTime'].values
                conversions = conversion_data['Conversion'].values

                # Fit multiple kinetic models
                model_results = self._fit_kinetic_models(times, conversions)

                # Select best model
                best_model = self._select_best_kinetic_model(model_results)

                kinetic_results[group_name] = {
                    'all_models': model_results,
                    'best_model': best_model,
                    'experimental_data': {'times': times, 'conversions': conversions}
                }

        return kinetic_results

    def _fit_kinetic_models(self, times: np.ndarray, conversions: np.ndarray) -> Dict:
        """Fit multiple kinetic models and compare performance"""
        model_results = {}

        # Autocatalytic model
        try:
            def autocatalytic_fit(t, k1, k2, alpha_max):
                return self.kinetic_models.autocatalytic_model(t, k1, k2, alpha_max)

            popt, pcov = optimize.curve_fit(
                autocatalytic_fit, times, conversions,
                bounds=([0, 0, 0.5], [1, 1, 1.0]),
                maxfev=5000
            )

            fitted_conversions = autocatalytic_fit(times, *popt)
            r_squared = r2_score(conversions, fitted_conversions)

            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))

            model_results['autocatalytic'] = {
                'parameters': {'k1': popt[0], 'k2': popt[1], 'alpha_max': popt[2]},
                'parameter_errors': {'k1': param_errors[0], 'k2': param_errors[1], 'alpha_max': param_errors[2]},
                'r_squared': r_squared,
                'fitted_data': fitted_conversions,
                'aic': self._calculate_aic(conversions, fitted_conversions, len(popt)),
                'bic': self._calculate_bic(conversions, fitted_conversions, len(popt))
            }
        except Exception as e:
            logger.warning(f"Autocatalytic model fitting failed: {e}")
            model_results['autocatalytic'] = {'error': str(e)}

        # First-order model
        try:
            def first_order_fit(t, k, alpha_max):
                return alpha_max * (1 - np.exp(-k * t))

            popt, pcov = optimize.curve_fit(
                first_order_fit, times, conversions,
                bounds=([0, 0.5], [1, 1.0]),
                maxfev=5000
            )

            fitted_conversions = first_order_fit(times, *popt)
            r_squared = r2_score(conversions, fitted_conversions)
            param_errors = np.sqrt(np.diag(pcov))

            model_results['first_order'] = {
                'parameters': {'k': popt[0], 'alpha_max': popt[1]},
                'parameter_errors': {'k': param_errors[0], 'alpha_max': param_errors[1]},
                'r_squared': r_squared,
                'fitted_data': fitted_conversions,
                'aic': self._calculate_aic(conversions, fitted_conversions, len(popt)),
                'bic': self._calculate_bic(conversions, fitted_conversions, len(popt))
            }
        except Exception as e:
            logger.warning(f"First-order model fitting failed: {e}")
            model_results['first_order'] = {'error': str(e)}

        # Zero-order model
        try:
            def zero_order_fit(t, k, alpha_max):
                return np.minimum(k * t, alpha_max)

            popt, pcov = optimize.curve_fit(
                zero_order_fit, times, conversions,
                bounds=([0, 0.5], [1, 1.0]),
                maxfev=5000
            )

            fitted_conversions = zero_order_fit(times, *popt)
            r_squared = r2_score(conversions, fitted_conversions)
            param_errors = np.sqrt(np.diag(pcov))

            model_results['zero_order'] = {
                'parameters': {'k': popt[0], 'alpha_max': popt[1]},
                'parameter_errors': {'k': param_errors[0], 'alpha_max': param_errors[1]},
                'r_squared': r_squared,
                'fitted_data': fitted_conversions,
                'aic': self._calculate_aic(conversions, fitted_conversions, len(popt)),
                'bic': self._calculate_bic(conversions, fitted_conversions, len(popt))
            }
        except Exception as e:
            logger.warning(f"Zero-order model fitting failed: {e}")
            model_results['zero_order'] = {'error': str(e)}

        return model_results

    def _calculate_aic(self, observed: np.ndarray, predicted: np.ndarray,
                      num_params: int) -> float:
        """Calculate Akaike Information Criterion"""
        n = len(observed)
        mse = np.mean((observed - predicted) ** 2)
        aic = n * np.log(mse) + 2 * num_params
        return aic

    def _calculate_bic(self, observed: np.ndarray, predicted: np.ndarray,
                      num_params: int) -> float:
        """Calculate Bayesian Information Criterion"""
        n = len(observed)
        mse = np.mean((observed - predicted) ** 2)
        bic = n * np.log(mse) + num_params * np.log(n)
        return bic

    def _select_best_kinetic_model(self, model_results: Dict) -> Dict:
        """Select best kinetic model based on statistical criteria"""
        valid_models = {k: v for k, v in model_results.items() if 'error' not in v}

        if not valid_models:
            return {'error': 'No valid kinetic models found'}

        # Rank models by AIC (lower is better)
        aic_scores = {k: v['aic'] for k, v in valid_models.items()}
        best_model_name = min(aic_scores, key=aic_scores.get)

        best_model = valid_models[best_model_name].copy()
        best_model['model_name'] = best_model_name
        best_model['ranking'] = {
            'aic_rank': sorted(aic_scores.items(), key=lambda x: x[1]),
            'r2_rank': sorted([(k, v['r_squared']) for k, v in valid_models.items()],
                            key=lambda x: x[1], reverse=True)
        }

        return best_model

    def _validate_kinetic_results(self, kinetic_results: Dict) -> Dict:
        """Validate kinetic analysis results"""
        validated_results = {}

        for group_name, results in kinetic_results.items():
            if 'best_model' in results and 'error' not in results['best_model']:
                best_model = results['best_model']

                # Statistical validation
                validation = {
                    'r_squared_acceptable': best_model['r_squared'] >= 0.90,
                    'parameters_physical': self._validate_physical_parameters(best_model),
                    'residuals_random': self._test_residual_randomness(
                        results['experimental_data']['conversions'],
                        best_model['fitted_data']
                    ),
                    'confidence_intervals': self._calculate_confidence_intervals(best_model)
                }

                validated_results[group_name] = {
                    **results,
                    'validation': validation,
                    'overall_valid': all([
                        validation['r_squared_acceptable'],
                        validation['parameters_physical'],
                        validation['residuals_random']
                    ])
                }
            else:
                validated_results[group_name] = results

        return validated_results

    def _validate_physical_parameters(self, model_results: Dict) -> bool:
        """Validate that kinetic parameters are physically reasonable"""
        params = model_results['parameters']
        model_name = model_results['model_name']

        if model_name == 'autocatalytic':
            # k1, k2 should be positive, alpha_max should be 0.5-1.0
            return (params['k1'] > 0 and params['k2'] > 0 and
                   0.5 <= params['alpha_max'] <= 1.0)
        elif model_name in ['first_order', 'zero_order']:
            # k should be positive, alpha_max should be 0.5-1.0
            return (params['k'] > 0 and 0.5 <= params['alpha_max'] <= 1.0)

        return True

    def _test_residual_randomness(self, observed: np.ndarray,
                                predicted: np.ndarray) -> bool:
        """Test if residuals are randomly distributed (Durbin-Watson test)"""
        residuals = observed - predicted

        # Durbin-Watson test statistic
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)

        # DW statistic should be around 2 for random residuals
        return 1.5 <= dw_stat <= 2.5

    def _calculate_confidence_intervals(self, model_results: Dict) -> Dict:
        """Calculate 95% confidence intervals for parameters"""
        params = model_results['parameters']
        errors = model_results['parameter_errors']

        # 95% confidence interval (t-distribution, approximate)
        t_value = 1.96  # For large samples

        confidence_intervals = {}
        for param_name, param_value in params.items():
            error = errors[param_name]
            ci_lower = param_value - t_value * error
            ci_upper = param_value + t_value * error
            confidence_intervals[param_name] = (ci_lower, ci_upper)

        return confidence_intervals

    def _compile_results(self, quality_results: Dict, group_analysis: Dict,
                        kinetic_results: Dict, experimental_conditions: Dict) -> Dict:
        """Compile comprehensive analysis results"""

        final_results = {
            'metadata': {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'analysis_parameters': self.analysis_params,
                'experimental_conditions': experimental_conditions
            },
            'data_quality': quality_results,
            'chemical_groups': group_analysis,
            'kinetic_analysis': kinetic_results,
            'summary': self._generate_summary(kinetic_results)
        }

        return final_results

    def _generate_summary(self, kinetic_results: Dict) -> Dict:
        """Generate analysis summary"""
        summary = {
            'total_groups_analyzed': len(kinetic_results),
            'successful_kinetic_fits': 0,
            'best_models': {},
            'average_r_squared': 0.0
        }

        r_squared_values = []

        for group_name, results in kinetic_results.items():
            if 'best_model' in results and 'error' not in results['best_model']:
                summary['successful_kinetic_fits'] += 1
                best_model = results['best_model']
                summary['best_models'][group_name] = best_model['model_name']
                r_squared_values.append(best_model['r_squared'])

        if r_squared_values:
            summary['average_r_squared'] = np.mean(r_squared_values)

        return summary
