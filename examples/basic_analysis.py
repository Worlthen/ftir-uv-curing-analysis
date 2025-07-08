#!/usr/bin/env python3
"""
Basic FTIR UV Curing Analysis Example
Demonstrates the fundamental usage of the FTIR analysis system
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from opus_reader import OPUSReader
from ftir_analyzer import FTIRUVCuringAnalyzer
from visualization import FTIRVisualizer

def basic_analysis_example():
    """
    Basic example showing the complete analysis workflow
    """
    print("="*60)
    print("BASIC FTIR UV CURING ANALYSIS EXAMPLE")
    print("="*60)
    
    # Step 1: Initialize components
    print("\n1. Initializing analysis components...")
    opus_reader = OPUSReader()
    analyzer = FTIRUVCuringAnalyzer()
    visualizer = FTIRVisualizer()
    
    # Step 2: Process OPUS files (if available)
    print("\n2. Looking for OPUS files...")
    opus_files = opus_reader.find_opus_files('.')
    
    if opus_files:
        print(f"Found {len(opus_files)} OPUS files")
        
        # Convert to CSV
        print("Converting OPUS files to CSV...")
        conversion_results = opus_reader.batch_convert('.', './csv_output')
        
        # Create integrated dataset
        if conversion_results['csv_files']:
            print("Creating integrated dataset...")
            integrated_file = opus_reader.create_integrated_dataset(
                conversion_results['csv_files'], 
                'integrated_spectra.csv'
            )
        else:
            print("No CSV files created, using existing data...")
            integrated_file = 'integrated_spectra.csv'
    else:
        print("No OPUS files found, looking for existing CSV data...")
        integrated_file = 'integrated_spectra.csv'
    
    # Step 3: Load data
    print(f"\n3. Loading data from {integrated_file}...")
    if not os.path.exists(integrated_file):
        print(f"Error: {integrated_file} not found!")
        print("Please ensure you have either:")
        print("  - OPUS files (.0, .1, .2, .3) in the current directory, or")
        print("  - An integrated_spectra.csv file")
        return
    
    success = analyzer.load_data(integrated_file)
    if not success:
        print("Failed to load data!")
        return
    
    print(f"Data loaded successfully:")
    print(f"  - Time points: {len(analyzer.exposure_times)}")
    print(f"  - Wavenumbers: {len(analyzer.wavenumbers)}")
    print(f"  - Time range: {min(analyzer.exposure_times):.1f} - {max(analyzer.exposure_times):.1f} s")
    print(f"  - Wavenumber range: {min(analyzer.wavenumbers):.0f} - {max(analyzer.wavenumbers):.0f} cm⁻¹")
    
    # Step 4: Preprocess data
    print("\n4. Preprocessing data...")
    processed_data = analyzer.preprocess_data(
        baseline_method='als',
        norm_method='max'
    )
    print("Data preprocessing completed")
    
    # Step 5: Analyze C=C consumption (key for UV curing)
    print("\n5. Analyzing C=C double bond consumption...")
    cc_results = analyzer.analyze_cc_consumption(wavenumber_range=(1620, 1640))
    
    print("C=C Analysis Results:")
    print(f"  - Final conversion: {max(cc_results['conversion_percent']):.2f}%")
    
    # Show kinetic model results
    if 'kinetic_models' in cc_results:
        print("  - Kinetic models:")
        for model_name, model_data in cc_results['kinetic_models'].items():
            if 'error' not in model_data:
                r2 = model_data.get('r_squared', 0)
                k = model_data.get('rate_constant', 0)
                print(f"    {model_name}: R² = {r2:.4f}, k = {k:.2e} s⁻¹")
    
    # Step 6: Perform PCA analysis
    print("\n6. Performing PCA analysis...")
    pca_results = analyzer.perform_pca_analysis()
    
    if pca_results:
        print("PCA Results:")
        print(f"  - PC1 explains {pca_results['explained_variance'][0]*100:.1f}% of variance")
        print(f"  - PC2 explains {pca_results['explained_variance'][1]*100:.1f}% of variance")
        print(f"  - First 3 PCs explain {pca_results['cumulative_variance'][2]*100:.1f}% of variance")
    
    # Step 7: Generate visualizations
    print("\n7. Generating visualizations...")
    
    # Create output directory
    output_dir = Path('./basic_analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    # Spectral evolution plot
    fig1 = visualizer.plot_spectral_evolution(
        processed_data,
        wavenumber_range=(1000, 3000),
        save_path=str(output_dir / 'spectral_evolution.png')
    )
    print(f"  - Spectral evolution plot saved")
    
    # Kinetic curve for C=C region
    region_results = {'acrylate_cc': cc_results}
    fig2 = visualizer.plot_kinetic_curves(
        region_results,
        save_path=str(output_dir / 'cc_kinetics.png')
    )
    print(f"  - C=C kinetics plot saved")
    
    # PCA plot
    if pca_results:
        fig3 = visualizer.plot_pca_analysis(
            pca_results,
            save_path=str(output_dir / 'pca_analysis.png')
        )
        print(f"  - PCA analysis plot saved")
    
    # Step 8: Summary
    print("\n8. Analysis Summary:")
    print("="*40)
    print(f"UV curing analysis completed successfully!")
    print(f"Key findings:")
    print(f"  - Maximum C=C conversion: {max(cc_results['conversion_percent']):.1f}%")
    
    if 'kinetic_models' in cc_results:
        # Find best model
        best_model = None
        best_r2 = -1
        for model_name, model_data in cc_results['kinetic_models'].items():
            if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                best_r2 = model_data['r_squared']
                best_model = model_name
        
        if best_model:
            print(f"  - Best kinetic model: {best_model} (R² = {best_r2:.4f})")
    
    if pca_results:
        print(f"  - Spectral variance explained by PC1: {pca_results['explained_variance'][0]*100:.1f}%")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


def analyze_specific_region_example():
    """
    Example showing how to analyze a specific wavenumber region
    """
    print("\n" + "="*60)
    print("SPECIFIC REGION ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FTIRUVCuringAnalyzer()
    
    # Load data
    if not analyzer.load_data('integrated_spectra.csv'):
        print("Could not load data for specific region example")
        return
    
    # Preprocess
    analyzer.preprocess_data()
    
    # Define custom regions of interest
    custom_regions = {
        'carbonyl_stretch': (1700, 1750),    # C=O stretch
        'aromatic_cc': (1580, 1620),         # Aromatic C=C
        'ch_aliphatic': (2800, 3000),        # Aliphatic C-H
        'oh_groups': (3200, 3600),           # O-H stretch
    }
    
    print(f"Analyzing {len(custom_regions)} custom regions...")
    
    # Analyze each region
    for region_name, wavenumber_range in custom_regions.items():
        print(f"\n{region_name.upper().replace('_', ' ')}:")
        print(f"  Wavenumber range: {wavenumber_range[0]}-{wavenumber_range[1]} cm⁻¹")
        
        # Analyze region
        results = analyzer.analyze_cc_consumption(wavenumber_range)
        
        if results:
            final_conversion = max(results['conversion_percent'])
            print(f"  Final conversion: {final_conversion:.2f}%")
            
            # Find best kinetic model
            if 'kinetic_models' in results:
                best_r2 = -1
                best_model = None
                for model_name, model_data in results['kinetic_models'].items():
                    if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                        best_r2 = model_data['r_squared']
                        best_model = model_name
                
                if best_model:
                    print(f"  Best model: {best_model} (R² = {best_r2:.4f})")


def difference_spectra_example():
    """
    Example showing difference spectra analysis
    """
    print("\n" + "="*60)
    print("DIFFERENCE SPECTRA ANALYSIS EXAMPLE")
    print("="*60)
    
    # Initialize components
    analyzer = FTIRUVCuringAnalyzer()
    visualizer = FTIRVisualizer()
    
    # Load and preprocess data
    if not analyzer.load_data('integrated_spectra.csv'):
        print("Could not load data for difference spectra example")
        return
    
    analyzer.preprocess_data()
    
    # Calculate difference spectra
    print("Calculating difference spectra...")
    diff_spectra = analyzer.calculate_difference_spectra()
    
    if not diff_spectra.empty:
        print(f"Difference spectra calculated for {len(diff_spectra['ExposureTime'].unique())} time points")
        
        # Identify significant peaks
        significant_peaks = analyzer.identify_significant_peaks(diff_spectra, threshold=0.01)
        
        print("\nSignificant peaks identified:")
        for time, peaks in significant_peaks.items():
            pos_peaks = peaks['positive_peaks']['wavenumbers']
            neg_peaks = peaks['negative_peaks']['wavenumbers']
            
            print(f"  Time {time}s:")
            if len(pos_peaks) > 0:
                print(f"    Positive peaks (product formation): {pos_peaks[:5]}")  # Show first 5
            if len(neg_peaks) > 0:
                print(f"    Negative peaks (reactant consumption): {neg_peaks[:5]}")  # Show first 5
        
        # Plot difference spectra
        output_dir = Path('./basic_analysis_output')
        output_dir.mkdir(exist_ok=True)
        
        fig = visualizer.plot_difference_spectra(
            diff_spectra,
            save_path=str(output_dir / 'difference_spectra.png')
        )
        print(f"\nDifference spectra plot saved to: {output_dir / 'difference_spectra.png'}")
    else:
        print("No difference spectra data available")


if __name__ == "__main__":
    try:
        # Run basic analysis example
        basic_analysis_example()
        
        # Run specific region analysis example
        analyze_specific_region_example()
        
        # Run difference spectra example
        difference_spectra_example()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Please ensure you have:")
        print("1. OPUS files (.0, .1, .2, .3) in the current directory, OR")
        print("2. An integrated_spectra.csv file with the required format")
        print("\nRequired CSV format:")
        print("  - Columns: Wavenumber, Absorbance, ExposureTime, Filename")
        print("  - Multiple time points with spectral data")
        
        import traceback
        traceback.print_exc()
