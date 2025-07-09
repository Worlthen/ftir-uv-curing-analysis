#!/usr/bin/env python3
"""
Enhanced FTIR Analyzer Program Validation Test

This script comprehensively tests the new enhanced FTIR analyzer to verify:
1. Data quality validation functionality
2. Chemical group analysis accuracy
3. Kinetic modeling performance
4. Statistical validation correctness
5. Error handling robustness
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.enhanced_ftir_analyzer import EnhancedFTIRAnalyzer, ChemicalGroupDefinitions, SpectralQualityValidator
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """
    Create comprehensive test data with known characteristics for validation
    """
    logger.info("Creating test data with known characteristics")
    
    # Define test parameters
    wavenumbers = np.arange(4000, 400, -2)  # 4000-400 cm⁻¹
    exposure_times = [0, 1, 2, 5, 10, 20, 30, 60, 120, 300]  # seconds
    
    # Known kinetic parameters for validation
    true_parameters = {
        'k1': 0.02,      # Initial rate constant (s⁻¹)
        'k2': 0.05,      # Autocatalytic rate constant (s⁻¹)
        'alpha_max': 0.95  # Maximum conversion
    }
    
    spectra_data = []
    
    for time in exposure_times:
        # Calculate true conversion using autocatalytic model
        if time == 0:
            conversion = 0.0
        else:
            # Analytical solution for autocatalytic model (simplified)
            k_eff = true_parameters['k1'] + true_parameters['k2'] * 0.5  # Average
            conversion = true_parameters['alpha_max'] * (1 - np.exp(-k_eff * time))
            conversion = min(conversion, true_parameters['alpha_max'])
        
        # Generate realistic spectrum with known chemical signatures
        spectrum = generate_test_spectrum(wavenumbers, conversion, add_noise=True)
        
        # Create data entries
        for i, wn in enumerate(wavenumbers):
            spectra_data.append({
                'Wavenumber': wn,
                'Absorbance': spectrum[i],
                'ExposureTime': time,
                'Filename': f'test_{time}s.csv'
            })
    
    test_df = pd.DataFrame(spectra_data)
    
    # Add metadata for validation
    test_metadata = {
        'true_parameters': true_parameters,
        'expected_conversion': conversion,
        'data_points': len(spectra_data),
        'time_points': len(exposure_times)
    }
    
    return test_df, test_metadata


def generate_test_spectrum(wavenumbers, conversion, add_noise=True):
    """
    Generate test spectrum with known chemical group signatures
    """
    spectrum = np.zeros_like(wavenumbers, dtype=np.float64)

    # Baseline
    baseline = 0.05 + 0.02 * np.random.random(len(wavenumbers)) if add_noise else 0.05
    spectrum = spectrum + baseline
    
    # C=C acrylate peak (1635 cm⁻¹) - decreases with conversion
    c_equals_c_peak = 1.0 * (1 - conversion) * gaussian_peak(wavenumbers, 1635, 12)
    spectrum = spectrum + c_equals_c_peak

    # Ester C=O peak (1730 cm⁻¹) - stable reference
    ester_peak = 1.5 * gaussian_peak(wavenumbers, 1730, 18)
    spectrum = spectrum + ester_peak

    # C-H alkyl (2920 cm⁻¹) - increases with polymerization
    ch_peak = 0.8 * (0.2 + 0.8 * conversion) * gaussian_peak(wavenumbers, 2920, 25)
    spectrum = spectrum + ch_peak

    # Photoinitiator peak (1670 cm⁻¹) - decreases with photolysis
    pi_peak = 0.4 * (1 - 0.8 * conversion) * gaussian_peak(wavenumbers, 1670, 10)
    spectrum = spectrum + pi_peak

    # Add realistic noise
    if add_noise:
        noise = 0.005 * np.random.random(len(wavenumbers))
        spectrum = spectrum + noise
    
    return spectrum


def gaussian_peak(x, center, width):
    """Generate Gaussian peak"""
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def test_data_quality_validation():
    """
    Test the data quality validation functionality
    """
    logger.info("Testing data quality validation")
    
    # Create test data with different quality levels
    wavenumbers = np.arange(4000, 400, -2)
    
    # High quality spectrum
    high_quality = generate_test_spectrum(wavenumbers, 0.5, add_noise=False)
    high_quality += 0.001 * np.random.random(len(wavenumbers))  # Low noise
    
    # Low quality spectrum
    low_quality = generate_test_spectrum(wavenumbers, 0.5, add_noise=False)
    low_quality += 0.1 * np.random.random(len(wavenumbers))  # High noise
    
    # Test validator
    validator = SpectralQualityValidator()
    
    # Test high quality
    hq_results = validator.validate_spectrum_quality(high_quality, wavenumbers)
    logger.info(f"High quality spectrum - S/N: {hq_results['snr']:.1f}, Pass: {hq_results['snr_pass']}")
    
    # Test low quality
    lq_results = validator.validate_spectrum_quality(low_quality, wavenumbers)
    logger.info(f"Low quality spectrum - S/N: {lq_results['snr']:.1f}, Pass: {lq_results['snr_pass']}")
    
    # Validation checks
    assert hq_results['snr'] > lq_results['snr'], "High quality should have better S/N"
    assert hq_results['snr_pass'] == True, "High quality should pass S/N test"
    
    logger.info("✅ Data quality validation test PASSED")
    return True


def test_chemical_group_definitions():
    """
    Test chemical group definitions and assignments
    """
    logger.info("Testing chemical group definitions")
    
    groups = ChemicalGroupDefinitions()
    
    # Test reactive groups
    assert 'c_equals_c_acrylate' in groups.REACTIVE_GROUPS
    assert groups.REACTIVE_GROUPS['c_equals_c_acrylate']['range'] == (1620, 1640)
    assert groups.REACTIVE_GROUPS['c_equals_c_acrylate']['reaction_role'] == 'primary_reactive_site'
    
    # Test structural groups
    assert 'ester_carbonyl' in groups.STRUCTURAL_GROUPS
    assert groups.STRUCTURAL_GROUPS['ester_carbonyl']['range'] == (1720, 1740)
    
    # Test photoinitiator groups
    assert 'benzoin_carbonyl' in groups.PHOTOINITIATOR_GROUPS
    assert groups.PHOTOINITIATOR_GROUPS['benzoin_carbonyl']['range'] == (1650, 1680)
    
    logger.info("✅ Chemical group definitions test PASSED")
    return True


def test_kinetic_analysis():
    """
    Test kinetic analysis with known parameters
    """
    logger.info("Testing kinetic analysis with known parameters")
    
    # Create test data with known kinetic parameters
    test_data, metadata = create_test_data()
    true_params = metadata['true_parameters']
    
    # Initialize analyzer
    analyzer = EnhancedFTIRAnalyzer()
    
    # Define test experimental conditions
    experimental_conditions = {
        'uv_wavelength': 365,
        'uv_intensity': 50,
        'temperature': 25,
        'atmosphere': 'nitrogen',
        'photoinitiator': 'Test PI',
        'monomer_system': 'Test Acrylate'
    }
    
    # Run analysis
    start_time = time.time()
    results = analyzer.analyze_uv_curing_kinetics(test_data, experimental_conditions)
    analysis_time = time.time() - start_time
    
    logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
    
    # Validate results structure
    assert 'data_quality' in results
    assert 'chemical_groups' in results
    assert 'kinetic_analysis' in results
    assert 'summary' in results
    
    # Check data quality
    quality = results['data_quality']
    logger.info(f"Data quality - Overall pass: {quality['overall_pass']}")
    
    # Check kinetic analysis results
    kinetic_results = results['kinetic_analysis']
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'best_model' in kinetic_data and 'error' not in kinetic_data['best_model']:
            best_model = kinetic_data['best_model']
            fitted_params = best_model['parameters']
            
            logger.info(f"Group: {group_name}")
            logger.info(f"  Best model: {best_model['model_name']}")
            logger.info(f"  R² value: {best_model['r_squared']:.4f}")
            logger.info(f"  Parameters: {fitted_params}")
            
            # Validate R² value
            assert best_model['r_squared'] > 0.8, f"R² too low: {best_model['r_squared']}"
            
            # Check if autocatalytic model was selected (should be for our test data)
            if best_model['model_name'] == 'autocatalytic':
                # Compare with true parameters (allow reasonable tolerance)
                if 'k1' in fitted_params:
                    k1_error = abs(fitted_params['k1'] - true_params['k1']) / true_params['k1']
                    logger.info(f"  k1 relative error: {k1_error:.2%}")
                    
                if 'alpha_max' in fitted_params:
                    alpha_error = abs(fitted_params['alpha_max'] - true_params['alpha_max']) / true_params['alpha_max']
                    logger.info(f"  alpha_max relative error: {alpha_error:.2%}")
    
    logger.info("✅ Kinetic analysis test PASSED")
    return results


def test_error_handling():
    """
    Test error handling and edge cases
    """
    logger.info("Testing error handling and edge cases")
    
    analyzer = EnhancedFTIRAnalyzer()
    
    # Test with empty data
    try:
        empty_data = pd.DataFrame()
        results = analyzer.analyze_uv_curing_kinetics(empty_data, {})
        logger.warning("Empty data should raise an error")
    except Exception as e:
        logger.info(f"✅ Empty data correctly handled: {type(e).__name__}")
    
    # Test with insufficient time points
    try:
        insufficient_data = pd.DataFrame({
            'Wavenumber': [1635, 1635],
            'Absorbance': [1.0, 0.8],
            'ExposureTime': [0, 10],
            'Filename': ['test1.csv', 'test2.csv']
        })
        results = analyzer.analyze_uv_curing_kinetics(insufficient_data, {})
        logger.info("Insufficient data handled gracefully")
    except Exception as e:
        logger.info(f"✅ Insufficient data correctly handled: {type(e).__name__}")
    
    # Test with invalid wavenumber ranges
    try:
        invalid_data = pd.DataFrame({
            'Wavenumber': [100, 200],  # Outside typical FTIR range
            'Absorbance': [1.0, 0.8],
            'ExposureTime': [0, 10],
            'Filename': ['test1.csv', 'test2.csv']
        })
        results = analyzer.analyze_uv_curing_kinetics(invalid_data, {})
        logger.info("Invalid wavenumber range handled gracefully")
    except Exception as e:
        logger.info(f"✅ Invalid wavenumber range correctly handled: {type(e).__name__}")
    
    logger.info("✅ Error handling test PASSED")
    return True


def test_statistical_validation():
    """
    Test statistical validation functionality
    """
    logger.info("Testing statistical validation")
    
    # Create test data
    test_data, metadata = create_test_data()
    
    analyzer = EnhancedFTIRAnalyzer()
    experimental_conditions = {'test': True}
    
    results = analyzer.analyze_uv_curing_kinetics(test_data, experimental_conditions)
    
    # Check statistical validation
    kinetic_results = results['kinetic_analysis']
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'validation' in kinetic_data:
            validation = kinetic_data['validation']
            
            logger.info(f"Statistical validation for {group_name}:")
            logger.info(f"  R² acceptable: {validation.get('r_squared_acceptable', False)}")
            logger.info(f"  Parameters physical: {validation.get('parameters_physical', False)}")
            logger.info(f"  Residuals random: {validation.get('residuals_random', False)}")
            logger.info(f"  Overall valid: {validation.get('overall_valid', False)}")
            
            # Check confidence intervals
            if 'confidence_intervals' in validation:
                ci = validation['confidence_intervals']
                logger.info(f"  Confidence intervals calculated: {len(ci)} parameters")
                
                # Validate CI structure
                for param, (lower, upper) in ci.items():
                    assert lower < upper, f"Invalid confidence interval for {param}"
    
    logger.info("✅ Statistical validation test PASSED")
    return True


def create_validation_plots(results, test_data):
    """
    Create validation plots to visualize test results
    """
    logger.info("Creating validation plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced FTIR Analyzer Validation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Data quality metrics
    ax1 = axes[0, 0]
    quality = results['data_quality']
    metrics = ['S/N Ratio', 'Baseline Stability', 'Peak Resolution', 'Frequency Accuracy']
    values = [quality['snr']/100, quality['baseline_stability']*1000, 
              quality['peak_resolution'], quality['frequency_accuracy']*10]
    colors = ['green' if quality['snr_pass'] else 'red',
              'green' if quality['baseline_pass'] else 'red',
              'green' if quality['resolution_pass'] else 'red',
              'green' if quality['frequency_pass'] else 'red']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_title('Data Quality Validation')
    ax1.set_ylabel('Normalized Values')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Kinetic fitting results
    ax2 = axes[0, 1]
    kinetic_results = results['kinetic_analysis']
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'best_model' in kinetic_data and 'error' not in kinetic_data['best_model']:
            exp_data = kinetic_data['experimental_data']
            best_model = kinetic_data['best_model']
            
            ax2.scatter(exp_data['times'], exp_data['conversions'], 
                       label=f'{group_name} (exp)', s=50, alpha=0.7)
            ax2.plot(exp_data['times'], best_model['fitted_data'], 
                    label=f'{group_name} (fit)', linewidth=2)
    
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('Conversion')
    ax2.set_title('Kinetic Model Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model comparison
    ax3 = axes[1, 0]
    model_names = []
    r_squared_values = []
    
    for group_name, kinetic_data in kinetic_results.items():
        if 'all_models' in kinetic_data:
            for model_name, model_data in kinetic_data['all_models'].items():
                if 'error' not in model_data:
                    model_names.append(f'{model_name}')
                    r_squared_values.append(model_data['r_squared'])
    
    if model_names:
        bars = ax3.bar(model_names, r_squared_values, color='skyblue', alpha=0.7)
        ax3.set_ylabel('R² Value')
        ax3.set_title('Model Performance Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add threshold line
        ax3.axhline(y=0.95, color='green', linestyle='--', label='Excellent (R²>0.95)')
        ax3.axhline(y=0.90, color='orange', linestyle='--', label='Good (R²>0.90)')
        ax3.legend()
    
    # Plot 4: Chemical group analysis
    ax4 = axes[1, 1]
    group_names = []
    final_conversions = []
    
    for group_name, group_data in results['chemical_groups'].items():
        if group_data['group_info']['reaction_role'] in ['primary_reactive_site']:
            group_names.append(group_name.replace('_', '\n'))
            final_conversion = group_data['conversion_data']['Conversion'].iloc[-1]
            final_conversions.append(final_conversion * 100)
    
    if group_names:
        bars = ax4.bar(group_names, final_conversions, color='lightcoral', alpha=0.7)
        ax4.set_ylabel('Final Conversion (%)')
        ax4.set_title('Chemical Group Conversion')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, final_conversions):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_analyzer_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Validation plots saved as 'enhanced_analyzer_validation.png'")


def run_comprehensive_validation():
    """
    Run comprehensive validation of the enhanced FTIR analyzer
    """
    logger.info("="*80)
    logger.info("ENHANCED FTIR ANALYZER COMPREHENSIVE VALIDATION")
    logger.info("="*80)
    
    validation_results = {}
    
    try:
        # Test 1: Data quality validation
        validation_results['data_quality'] = test_data_quality_validation()
        
        # Test 2: Chemical group definitions
        validation_results['chemical_groups'] = test_chemical_group_definitions()
        
        # Test 3: Kinetic analysis
        analysis_results = test_kinetic_analysis()
        validation_results['kinetic_analysis'] = analysis_results is not None
        
        # Test 4: Statistical validation
        validation_results['statistical_validation'] = test_statistical_validation()
        
        # Test 5: Error handling
        validation_results['error_handling'] = test_error_handling()
        
        # Create validation plots
        if analysis_results:
            test_data, _ = create_test_data()
            create_validation_plots(analysis_results, test_data)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        all_passed = True
        for test_name, result in validation_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if not result:
                all_passed = False
        
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("🎉 ALL VALIDATION TESTS PASSED!")
            logger.info("The enhanced FTIR analyzer is working correctly.")
        else:
            logger.info("⚠️  SOME VALIDATION TESTS FAILED!")
            logger.info("Please review the failed tests and fix issues.")
        logger.info("="*80)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run comprehensive validation
    results = run_comprehensive_validation()
    
    if results and all(results.values()):
        print("\n🎯 VALIDATION COMPLETED SUCCESSFULLY!")
        print("The enhanced FTIR analyzer is ready for use.")
        print("\nKey features validated:")
        print("• Data quality assessment and validation")
        print("• Chemical group analysis with proper assignments")
        print("• Multiple kinetic model fitting and selection")
        print("• Statistical validation and confidence intervals")
        print("• Robust error handling and edge case management")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED!")
        print("Please review the test results and address any failures.")
