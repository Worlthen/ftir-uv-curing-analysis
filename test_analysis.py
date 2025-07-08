#!/usr/bin/env python3
"""
FTIR Analysis System Test Script
Test core analysis functions without launching GUI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from system_analysis import FTIRAnalyzer

def test_analysis():
    """Test analysis functionality"""
    print("="*60)
    print("FTIR Spectral Analysis System Test")
    print("="*60)

    # Create analyzer
    analyzer = FTIRAnalyzer()

    # Load data
    print("\n1. Loading data...")
    if not analyzer.load_data('all_spectra.csv'):
        print("Data loading failed!")
        return

    # Execute analysis
    print("\n2. Performing comprehensive analysis...")
    try:
        # Select subset of time points for quick testing
        selected_times = [0, 2, 4, 8, 16]
        selected_wavenumbers = (1000, 2000)  # Select partial wavenumber range

        results = analyzer.perform_comprehensive_analysis(
            baseline_method='als',
            normalize_method='max',
            selected_times=selected_times,
            selected_wavenumbers=selected_wavenumbers
        )

        print("Analysis completed!")

    except Exception as e:
        print(f"Analysis failed: {e}")
        return

    # Display results
    print("\n3. Analysis Results Overview:")
    print("-"*40)

    if 'key_analysis' in results:
        print("\nKey Wavenumber Region Analysis Results:")
        for region, data in results['key_analysis'].items():
            if data['best_fit']:
                fit = data['best_fit']
                print(f"\n{region}:")
                print(f"  - Wavenumber Range: {data['wavenumber_range'][0]}-{data['wavenumber_range'][1]} cm⁻¹")
                print(f"  - Best Model: {fit['model']}")
                print(f"  - Rate Constant: {fit['k']:.2e} s⁻¹")
                print(f"  - R² Value: {fit['r2']:.4f}")
                print(f"  - Half-life: {fit['half_life']:.2f} seconds")

    if 'pca_results' in results:
        pca = results['pca_results']
        print(f"\nPrincipal Component Analysis:")
        for i, var in enumerate(pca['explained_variance_ratio']):
            print(f"  - PC{i+1}: {var*100:.1f}% variance explained")
        print(f"  - Cumulative variance explained: {pca['cumulative_variance'][-1]*100:.1f}%")

    # Generate report
    print("\n4. Generating analysis report...")
    analyzer.generate_report()

    # Create simple visualization
    print("\n5. Generating visualization plots...")
    create_simple_plots(analyzer, results)

    print("\nTest completed!")

def create_simple_plots(analyzer, results):
    """Create simple visualization plots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FTIR Spectral Analysis Results', fontsize=16)

        # 1. Spectral evolution plot
        ax1 = axes[0, 0]
        processed_data = results['processed_data']
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (time, spectrum) in enumerate(processed_data.items()):
            if i < len(colors):
                ax1.plot(spectrum['Wavenumber'], spectrum['Absorbance'],
                        color=colors[i], label=f'{time}s', linewidth=1.5)

        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Normalized Absorbance')
        ax1.set_title('Spectral Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Key wavenumber time series
        ax2 = axes[0, 1]
        if 'key_analysis' in results:
            for region, data in results['key_analysis'].items():
                if 'time_series' in data:
                    ts = data['time_series']
                    ax2.plot(ts['time'], ts['absorbance'], 'o-', label=region, linewidth=2)

        ax2.set_xlabel('Exposure Time (s)')
        ax2.set_ylabel('Average Absorbance')
        ax2.set_title('Key Wavenumber Region Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Difference spectra analysis
        ax3 = axes[1, 0]
        if 'difference_spectra' in results:
            diff_spectra = results['difference_spectra']
            for i, (time, diff_data) in enumerate(diff_spectra.items()):
                if i < len(colors):
                    ax3.plot(diff_data['Wavenumber'], diff_data['Difference'],
                            color=colors[i], label=f'{time}s', linewidth=1.5)

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Wavenumber (cm⁻¹)')
        ax3.set_ylabel('Absorbance Difference')
        ax3.set_title('Difference Spectra Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. PCA scores plot
        ax4 = axes[1, 1]
        if 'pca_results' in results:
            pca = results['pca_results']
            scores = pca['scores']
            times = results['analysis_times']

            ax4.scatter(scores[:, 0], scores[:, 1], c=times, s=100, cmap='viridis')

            # Add time labels
            for i, time in enumerate(times):
                ax4.annotate(f'{time}s', (scores[i, 0], scores[i, 1]),
                           xytext=(5, 5), textcoords='offset points')

            ax4.set_xlabel(f'PC1 ({pca["explained_variance_ratio"][0]*100:.1f}%)')
            ax4.set_ylabel(f'PC2 ({pca["explained_variance_ratio"][1]*100:.1f}%)')
            ax4.set_title('PCA Scores Plot')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ftir_analysis_results.png', dpi=300, bbox_inches='tight')
        print("Plot saved as: ftir_analysis_results.png")

    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    test_analysis()
