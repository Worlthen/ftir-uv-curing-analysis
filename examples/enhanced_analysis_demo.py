#!/usr/bin/env python3
"""
Enhanced FTIR UV Curing Analysis - Demonstration Script

This script demonstrates the use of the enhanced FTIR analyzer with
scientific rigor and proper chemical mechanistic understanding.

Based on the critical analysis and restructured architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.enhanced_ftir_analyzer import EnhancedFTIRAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sample_data():
    """
    Load sample FTIR data for UV curing analysis
    
    This function creates realistic sample data that mimics actual
    UV curing FTIR measurements with proper chemical group signatures.
    """
    logger.info("Loading sample FTIR data for UV curing analysis")
    
    # Define experimental parameters
    wavenumbers = np.arange(4000, 400, -2)  # 4000-400 cm⁻¹, 2 cm⁻¹ resolution
    exposure_times = [0, 2, 5, 10, 20, 30, 60, 120]  # seconds
    
    # Create realistic FTIR spectra with chemical group signatures
    spectra_data = []
    
    for time in exposure_times:
        # Calculate conversion based on realistic UV curing kinetics
        conversion = 1 - np.exp(-0.05 * time)  # First-order approximation
        conversion = min(conversion, 0.95)  # Maximum 95% conversion
        
        # Generate spectrum with chemical group signatures
        spectrum = generate_realistic_spectrum(wavenumbers, conversion)
        
        # Add to dataset
        for i, wn in enumerate(wavenumbers):
            spectra_data.append({
                'Wavenumber': wn,
                'Absorbance': spectrum[i],
                'ExposureTime': time,
                'Filename': f'{time}s_uv_curing.csv'
            })
    
    return pd.DataFrame(spectra_data)


def generate_realistic_spectrum(wavenumbers, conversion):
    """
    Generate realistic FTIR spectrum for UV curing with proper chemical signatures

    Parameters:
    -----------
    wavenumbers : np.ndarray
        Wavenumber array
    conversion : float
        Degree of conversion (0-1)

    Returns:
    --------
    np.ndarray : Generated spectrum
    """
    spectrum = np.zeros_like(wavenumbers, dtype=np.float64)

    # Add baseline
    baseline = 0.1 + 0.05 * np.random.random(len(wavenumbers))
    spectrum = spectrum + baseline
    
    # C=C stretch (acrylate) - decreases with conversion
    c_equals_c_intensity = 0.8 * (1 - conversion) * gaussian_peak(wavenumbers, 1635, 15)
    spectrum = spectrum + c_equals_c_intensity

    # C=O stretch (ester) - relatively stable
    c_equals_o_intensity = 1.2 * gaussian_peak(wavenumbers, 1730, 20)
    spectrum = spectrum + c_equals_o_intensity

    # C-H stretch (alkyl) - increases with polymerization
    c_h_alkyl_intensity = 0.6 * (0.3 + 0.7 * conversion) * gaussian_peak(wavenumbers, 2920, 30)
    spectrum = spectrum + c_h_alkyl_intensity

    # =C-H stretch (vinyl) - decreases with conversion
    vinyl_c_h_intensity = 0.4 * (1 - conversion) * gaussian_peak(wavenumbers, 3100, 10)
    spectrum = spectrum + vinyl_c_h_intensity

    # C-O stretch (ester/ether) - increases with crosslinking
    c_o_intensity = 0.5 * (0.2 + 0.8 * conversion) * gaussian_peak(wavenumbers, 1200, 50)
    spectrum = spectrum + c_o_intensity

    # Photoinitiator C=O - decreases with photolysis
    pi_intensity = 0.3 * (1 - 0.9 * conversion) * gaussian_peak(wavenumbers, 1670, 12)
    spectrum = spectrum + pi_intensity

    # Add realistic noise
    noise = 0.01 * np.random.random(len(wavenumbers))
    spectrum = spectrum + noise
    
    return spectrum


def gaussian_peak(x, center, width):
    """Generate Gaussian peak"""
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def demonstrate_enhanced_analysis():
    """
    Demonstrate the enhanced FTIR analysis capabilities
    """
    logger.info("Starting enhanced FTIR UV curing analysis demonstration")
    
    # Load sample data
    spectra_data = load_sample_data()
    logger.info(f"Loaded {len(spectra_data)} spectral data points")
    
    # Define experimental conditions
    experimental_conditions = {
        'uv_wavelength': 365,  # nm
        'uv_intensity': 50,    # mW/cm²
        'temperature': 25,     # °C
        'atmosphere': 'air',
        'sample_thickness': 25,  # μm
        'photoinitiator': 'Irgacure 819',
        'photoinitiator_concentration': 2.0,  # wt%
        'monomer_system': 'TMPTA (trimethylolpropane triacrylate)'
    }
    
    # Initialize enhanced analyzer
    analyzer = EnhancedFTIRAnalyzer()
    
    # Configure analysis parameters
    analyzer.analysis_params.update({
        'baseline_correction': 'als',
        'normalization': 'max',
        'smoothing': 'savgol',
        'kinetic_model': 'autocatalytic'
    })
    
    # Perform comprehensive analysis
    logger.info("Performing comprehensive UV curing kinetics analysis")
    results = analyzer.analyze_uv_curing_kinetics(spectra_data, experimental_conditions)
    
    # Display results
    display_analysis_results(results)
    
    # Generate visualizations
    create_analysis_visualizations(results, spectra_data)
    
    logger.info("Enhanced analysis demonstration completed successfully")
    
    return results


def display_analysis_results(results):
    """
    Display comprehensive analysis results
    """
    print("\n" + "="*80)
    print("ENHANCED FTIR UV CURING ANALYSIS RESULTS")
    print("="*80)
    
    # Data quality assessment
    print("\n1. DATA QUALITY ASSESSMENT:")
    print("-" * 40)
    quality = results['data_quality']
    print(f"Signal-to-Noise Ratio: {quality['snr']:.1f} (Pass: {quality['snr_pass']})")
    print(f"Baseline Stability: {quality['baseline_stability']:.4f} AU (Pass: {quality['baseline_pass']})")
    print(f"Peak Resolution: {quality['peak_resolution']:.2f} (Pass: {quality['resolution_pass']})")
    print(f"Overall Quality: {'PASS' if quality['overall_pass'] else 'FAIL'}")
    
    # Chemical group analysis
    print("\n2. CHEMICAL GROUP ANALYSIS:")
    print("-" * 40)
    for group_name, group_data in results['chemical_groups'].items():
        group_info = group_data['group_info']
        conversion_data = group_data['conversion_data']
        final_conversion = conversion_data['Conversion'].iloc[-1]
        
        print(f"\n{group_name.upper().replace('_', ' ')}:")
        print(f"  Assignment: {group_info['assignment']}")
        print(f"  Wavenumber Range: {group_info['range'][0]}-{group_info['range'][1]} cm⁻¹")
        print(f"  Reaction Role: {group_info['reaction_role']}")
        print(f"  Final Conversion: {final_conversion:.2%}")
    
    # Kinetic analysis results
    print("\n3. KINETIC ANALYSIS RESULTS:")
    print("-" * 40)
    kinetic_results = results['kinetic_analysis']
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'best_model' in kinetic_data and 'error' not in kinetic_data['best_model']:
            best_model = kinetic_data['best_model']
            validation = kinetic_data.get('validation', {})
            
            print(f"\n{group_name.upper().replace('_', ' ')}:")
            print(f"  Best Model: {best_model['model_name']}")
            print(f"  R² Value: {best_model['r_squared']:.4f}")
            print(f"  Parameters:")
            for param, value in best_model['parameters'].items():
                error = best_model['parameter_errors'][param]
                print(f"    {param}: {value:.4e} ± {error:.4e}")
            
            if 'confidence_intervals' in validation:
                print(f"  95% Confidence Intervals:")
                for param, (lower, upper) in validation['confidence_intervals'].items():
                    print(f"    {param}: [{lower:.4e}, {upper:.4e}]")
            
            print(f"  Model Validation: {'PASS' if validation.get('overall_valid', False) else 'FAIL'}")
    
    # Summary statistics
    print("\n4. ANALYSIS SUMMARY:")
    print("-" * 40)
    summary = results['summary']
    print(f"Total Groups Analyzed: {summary['total_groups_analyzed']}")
    print(f"Successful Kinetic Fits: {summary['successful_kinetic_fits']}")
    print(f"Average R² Value: {summary['average_r_squared']:.4f}")
    
    print(f"\nBest Models Selected:")
    for group, model in summary['best_models'].items():
        print(f"  {group}: {model}")


def create_analysis_visualizations(results, spectra_data):
    """
    Create comprehensive visualizations of analysis results
    """
    logger.info("Creating analysis visualizations")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced FTIR UV Curing Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Spectral evolution
    ax1 = axes[0, 0]
    exposure_times = sorted(spectra_data['ExposureTime'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(exposure_times)))
    
    for i, time in enumerate(exposure_times[::2]):  # Plot every other time point
        time_data = spectra_data[spectra_data['ExposureTime'] == time]
        ax1.plot(time_data['Wavenumber'], time_data['Absorbance'], 
                color=colors[i*2], label=f'{time}s', linewidth=1.5)
    
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Absorbance')
    ax1.set_title('Spectral Evolution During UV Curing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Kinetic curves with model fits
    ax2 = axes[0, 1]
    kinetic_results = results['kinetic_analysis']
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'best_model' in kinetic_data and 'error' not in kinetic_data['best_model']:
            exp_data = kinetic_data['experimental_data']
            best_model = kinetic_data['best_model']
            
            # Plot experimental data
            ax2.scatter(exp_data['times'], exp_data['conversions'], 
                       label=f'{group_name} (exp)', s=50, alpha=0.7)
            
            # Plot fitted model
            ax2.plot(exp_data['times'], best_model['fitted_data'], 
                    label=f'{group_name} ({best_model["model_name"]})', linewidth=2)
    
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('Conversion')
    ax2.set_title('Kinetic Analysis with Model Fits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Chemical group conversion comparison
    ax3 = axes[1, 0]
    group_names = []
    final_conversions = []
    
    for group_name, group_data in results['chemical_groups'].items():
        if group_data['group_info']['reaction_role'] in ['primary_reactive_site', 'polymerization_indicator']:
            group_names.append(group_name.replace('_', '\n'))
            final_conversion = group_data['conversion_data']['Conversion'].iloc[-1]
            final_conversions.append(final_conversion * 100)
    
    bars = ax3.bar(group_names, final_conversions, color='skyblue', alpha=0.7)
    ax3.set_ylabel('Final Conversion (%)')
    ax3.set_title('Final Conversion by Chemical Group')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_conversions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Model comparison
    ax4 = axes[1, 1]
    model_names = []
    r_squared_values = []
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'all_models' in kinetic_data:
            for model_name, model_data in kinetic_data['all_models'].items():
                if 'error' not in model_data:
                    model_names.append(f'{group_name}\n{model_name}')
                    r_squared_values.append(model_data['r_squared'])
    
    if model_names:
        bars = ax4.bar(range(len(model_names)), r_squared_values, 
                      color='lightcoral', alpha=0.7)
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.set_ylabel('R² Value')
        ax4.set_title('Kinetic Model Performance Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, r_squared_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_ftir_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Visualizations saved as 'enhanced_ftir_analysis_results.png'")


if __name__ == "__main__":
    # Run the enhanced analysis demonstration
    results = demonstrate_enhanced_analysis()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("• Comprehensive data quality validation")
    print("• Chemical group-specific analysis with proper assignments")
    print("• Multiple kinetic model fitting and comparison")
    print("• Statistical validation of results")
    print("• Confidence interval calculation")
    print("• Publication-quality visualizations")
    print("\nThis enhanced analyzer addresses all critical points raised")
    print("in the scientific critique and provides rigorous analysis capabilities.")
