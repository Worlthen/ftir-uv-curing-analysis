#!/usr/bin/env python3
"""
Advanced FTIR UV Curing Analysis Example
Demonstrates advanced features and customization options
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from opus_reader import OPUSReader
from ftir_analyzer import FTIRUVCuringAnalyzer
from visualization import FTIRVisualizer
from report_generator import ReportGenerator

def advanced_multi_component_analysis():
    """
    Advanced example showing multi-component analysis with custom parameters
    """
    print("="*70)
    print("ADVANCED MULTI-COMPONENT UV CURING ANALYSIS")
    print("="*70)
    
    # Initialize analyzer with custom settings
    analyzer = FTIRUVCuringAnalyzer()
    visualizer = FTIRVisualizer()
    
    # Load data
    if not analyzer.load_data('integrated_spectra.csv'):
        print("Error: Could not load integrated_spectra.csv")
        return
    
    print(f"Loaded data: {len(analyzer.exposure_times)} time points")
    
    # Custom preprocessing with different methods
    print("\n1. Testing different preprocessing methods...")
    
    preprocessing_methods = [
        ('als', 'max'),
        ('als', 'area'),
        ('polynomial', 'max'),
    ]
    
    results_comparison = {}
    
    for baseline_method, norm_method in preprocessing_methods:
        print(f"   Processing with {baseline_method} baseline, {norm_method} normalization...")
        
        # Preprocess data
        processed_data = analyzer.preprocess_data(baseline_method, norm_method)
        
        # Analyze key region (C=C)
        cc_results = analyzer.analyze_cc_consumption((1620, 1640))
        final_conversion = max(cc_results['conversion_percent'])
        
        results_comparison[f"{baseline_method}_{norm_method}"] = {
            'final_conversion': final_conversion,
            'cc_results': cc_results
        }
        
        print(f"     Final C=C conversion: {final_conversion:.2f}%")
    
    # Find best preprocessing method
    best_method = max(results_comparison.keys(), 
                     key=lambda x: results_comparison[x]['final_conversion'])
    print(f"\n   Best preprocessing method: {best_method}")
    
    # Use best method for further analysis
    baseline_method, norm_method = best_method.split('_')
    analyzer.preprocess_data(baseline_method, norm_method)
    
    # 2. Comprehensive multi-region analysis
    print("\n2. Comprehensive multi-region analysis...")
    
    # Define comprehensive region set
    comprehensive_regions = {
        'acrylate_cc': (1620, 1640),      # Primary reaction
        'aromatic_cc': (1580, 1620),      # Aromatic involvement
        'carbonyl': (1700, 1750),         # Carbonyl changes
        'ch_aliphatic': (2800, 3000),     # Aliphatic C-H
        'ch_aromatic': (3000, 3100),      # Aromatic C-H
        'oh_groups': (3200, 3600),        # Hydroxyl groups
        'ether_co': (1000, 1300),         # Ether formation
        'aromatic_ring': (1500, 1600),    # Aromatic ring vibrations
        'ch_bending': (1400, 1500),       # C-H bending
        'fingerprint': (800, 1200),       # Fingerprint region
    }
    
    # Analyze all regions
    multi_region_results = analyzer.analyze_multiple_regions(comprehensive_regions)
    
    # Create comprehensive results summary
    print("\n   Multi-region analysis results:")
    print("   " + "-"*50)
    
    for region_name, results in multi_region_results.items():
        final_conv = max(results['conversion_percent'])
        
        # Find best kinetic model
        best_model = None
        best_r2 = -1
        if 'kinetic_models' in results:
            for model_name, model_data in results['kinetic_models'].items():
                if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                    best_r2 = model_data['r_squared']
                    best_model = model_name
        
        print(f"   {region_name:15s}: {final_conv:6.2f}% conversion", end="")
        if best_model:
            print(f" | Best model: {best_model:12s} (R² = {best_r2:.4f})")
        else:
            print()
    
    # 3. Advanced PCA analysis with interpretation
    print("\n3. Advanced PCA analysis...")
    
    pca_results = analyzer.perform_pca_analysis()
    
    if pca_results:
        # Analyze PCA loadings to identify important wavenumbers
        pc1_loadings = pca_results['loadings'][0, :]
        wavenumbers = pca_results['wavenumbers']
        
        # Find most important wavenumbers (highest absolute loadings)
        important_indices = np.argsort(np.abs(pc1_loadings))[-10:]  # Top 10
        important_wavenumbers = wavenumbers[important_indices]
        important_loadings = pc1_loadings[important_indices]
        
        print(f"   PC1 explains {pca_results['explained_variance'][0]*100:.1f}% of variance")
        print("   Most important wavenumbers for PC1:")
        for wn, loading in zip(important_wavenumbers, important_loadings):
            print(f"     {wn:7.1f} cm⁻¹: {loading:8.4f}")
        
        # Analyze temporal evolution in PC space
        scores = pca_results['scores']
        times = pca_results['exposure_times']
        
        # Calculate trajectory length in PC space
        pc_trajectory = np.sqrt(np.diff(scores[:, 0])**2 + np.diff(scores[:, 1])**2)
        total_trajectory = np.sum(pc_trajectory)
        
        print(f"   Total trajectory length in PC1-PC2 space: {total_trajectory:.4f}")
        
        # Identify phases of reaction
        if len(pc_trajectory) > 3:
            # Find phases based on trajectory speed
            trajectory_speed = pc_trajectory / np.diff(times)
            fast_phase_threshold = np.percentile(trajectory_speed, 75)
            
            fast_phases = np.where(trajectory_speed > fast_phase_threshold)[0]
            if len(fast_phases) > 0:
                print(f"   Fast reaction phases detected at times: {times[fast_phases+1]}")
    
    # 4. Custom kinetic modeling
    print("\n4. Custom kinetic modeling...")
    
    # Focus on the most reactive region
    most_reactive_region = max(multi_region_results.keys(), 
                              key=lambda x: max(multi_region_results[x]['conversion_percent']))
    
    print(f"   Most reactive region: {most_reactive_region}")
    
    # Extract kinetic data
    kinetic_data = multi_region_results[most_reactive_region]
    times = np.array(kinetic_data['exposure_times'])
    conversion = np.array(kinetic_data['conversion_percent'])
    
    # Custom autocatalytic model with inhibition
    def autocatalytic_with_inhibition(t, c_max, k, t_inhibition, n):
        """Autocatalytic model with inhibition period"""
        effective_time = np.maximum(0, t - t_inhibition)
        return c_max * (1 - np.exp(-k * effective_time**n))
    
    try:
        from scipy.optimize import curve_fit
        
        # Fit custom model
        popt, pcov = curve_fit(
            autocatalytic_with_inhibition, 
            times, conversion,
            bounds=([0, 0, 0, 0.1], [200, 10, max(times), 5]),
            maxfev=2000
        )
        
        c_max, k, t_inh, n = popt
        
        # Calculate R²
        y_pred = autocatalytic_with_inhibition(times, *popt)
        r2_custom = analyzer.calculate_r_squared(conversion, y_pred)
        
        print(f"   Custom autocatalytic model results:")
        print(f"     Maximum conversion: {c_max:.2f}%")
        print(f"     Rate constant: {k:.4f}")
        print(f"     Inhibition time: {t_inh:.2f} s")
        print(f"     Reaction order: {n:.2f}")
        print(f"     R² value: {r2_custom:.4f}")
        
    except Exception as e:
        print(f"   Custom model fitting failed: {str(e)}")
    
    # 5. Generate advanced visualizations
    print("\n5. Generating advanced visualizations...")
    
    output_dir = Path('./advanced_analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    # Create comprehensive summary plot
    fig = visualizer.generate_summary_plot(
        {
            'processed_data': analyzer.processed_data,
            'region_analysis': multi_region_results,
            'pca_analysis': pca_results,
            'difference_spectra': analyzer.calculate_difference_spectra(),
            'chemical_interpretation': analyzer.interpret_chemical_changes(
                multi_region_results, 
                analyzer.identify_significant_peaks(analyzer.calculate_difference_spectra())
            )
        },
        save_path=str(output_dir / 'comprehensive_summary.png')
    )
    plt.close(fig)
    
    # Create custom comparison plot
    create_preprocessing_comparison_plot(results_comparison, output_dir)
    
    # Create kinetic model comparison plot
    create_kinetic_comparison_plot(multi_region_results, output_dir)
    
    print(f"   Advanced visualizations saved to: {output_dir}")
    
    # 6. Generate comprehensive report
    print("\n6. Generating comprehensive report...")
    
    report_generator = ReportGenerator()
    
    # Prepare comprehensive results
    comprehensive_results = {
        'metadata': {
            'baseline_method': baseline_method,
            'normalization_method': norm_method,
            'analysis_regions': comprehensive_regions,
            'exposure_times': analyzer.exposure_times,
            'wavenumber_range': [min(analyzer.wavenumbers), max(analyzer.wavenumbers)]
        },
        'preprocessing_comparison': results_comparison,
        'region_analysis': multi_region_results,
        'pca_analysis': pca_results,
        'difference_spectra': analyzer.calculate_difference_spectra(),
        'chemical_interpretation': analyzer.interpret_chemical_changes(
            multi_region_results, 
            analyzer.identify_significant_peaks(analyzer.calculate_difference_spectra())
        )
    }
    
    # Generate reports
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    text_report = report_generator.generate_text_report(
        comprehensive_results,
        str(reports_dir / 'advanced_analysis_report.txt')
    )
    
    html_report = report_generator.generate_html_report(
        comprehensive_results,
        str(reports_dir / 'advanced_analysis_report.html')
    )
    
    excel_report = report_generator.generate_excel_report(
        comprehensive_results,
        str(reports_dir / 'advanced_analysis_data.xlsx')
    )
    
    print(f"   Reports generated:")
    print(f"     Text: {text_report}")
    print(f"     HTML: {html_report}")
    print(f"     Excel: {excel_report}")
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)


def create_preprocessing_comparison_plot(results_comparison, output_dir):
    """Create comparison plot for different preprocessing methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(results_comparison.keys())
    conversions = [results_comparison[method]['final_conversion'] for method in methods]
    
    # Bar plot of final conversions
    bars = ax1.bar(range(len(methods)), conversions, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_xlabel('Preprocessing Method')
    ax1.set_ylabel('Final C=C Conversion (%)')
    ax1.set_title('Preprocessing Method Comparison')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, conv in zip(bars, conversions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{conv:.1f}%', ha='center', va='bottom')
    
    # Kinetic curves comparison
    for i, method in enumerate(methods):
        cc_results = results_comparison[method]['cc_results']
        times = cc_results['exposure_times']
        conversion = cc_results['conversion_percent']
        
        ax2.plot(times, conversion, 'o-', label=method.replace('_', ' '), alpha=0.8)
    
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('C=C Conversion (%)')
    ax2.set_title('Kinetic Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_kinetic_comparison_plot(multi_region_results, output_dir):
    """Create comparison plot for kinetic models across regions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Model performance comparison
    regions = list(multi_region_results.keys())[:4]  # Show first 4 regions
    
    for idx, region in enumerate(regions):
        ax = axes[idx]
        results = multi_region_results[region]
        
        times = results['exposure_times']
        conversion = results['conversion_percent']
        
        # Plot experimental data
        ax.scatter(times, conversion, color='red', s=30, alpha=0.7, label='Experimental')
        
        # Plot fitted models
        if 'kinetic_models' in results:
            t_fit = np.linspace(min(times), max(times), 100)
            
            colors = ['blue', 'green', 'orange']
            for i, (model_name, model_data) in enumerate(results['kinetic_models'].items()):
                if 'error' not in model_data and i < len(colors):
                    if model_name == 'zero_order':
                        y_fit = model_data['rate_constant'] * t_fit
                    elif model_name == 'first_order':
                        y_fit = model_data['c_max'] * (1 - np.exp(-model_data['rate_constant'] * t_fit))
                    elif model_name == 'autocatalytic':
                        y_fit = (model_data['c_max'] * (t_fit**model_data['n']) / 
                               (model_data['t50']**model_data['n'] + t_fit**model_data['n']))
                    else:
                        continue
                    
                    ax.plot(t_fit, y_fit, '--', color=colors[i], alpha=0.8, 
                           label=f"{model_name} (R²={model_data['r_squared']:.3f})")
        
        ax.set_xlabel('Exposure Time (s)')
        ax.set_ylabel('Conversion (%)')
        ax.set_title(f'{region.replace("_", " ").title()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kinetic_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    try:
        advanced_multi_component_analysis()
        
    except Exception as e:
        print(f"\nError running advanced analysis: {str(e)}")
        print("Please ensure you have:")
        print("1. Run the basic analysis first to generate integrated_spectra.csv")
        print("2. All required dependencies installed")
        
        import traceback
        traceback.print_exc()
