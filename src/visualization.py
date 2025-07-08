"""
FTIR Data Visualization Module
Comprehensive plotting and visualization functions for UV curing analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set matplotlib style
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class FTIRVisualizer:
    """
    FTIR data visualization class with comprehensive plotting capabilities
    """
    
    def __init__(self):
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def plot_spectral_evolution(self, data: pd.DataFrame, 
                               wavenumber_range: Tuple[float, float] = None,
                               save_path: str = None) -> plt.Figure:
        """
        Plot spectral evolution over time
        
        Args:
            data: Processed spectral data
            wavenumber_range: Wavenumber range to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter wavenumber range if specified
        if wavenumber_range:
            data = data[
                (data['Wavenumber'] >= wavenumber_range[0]) & 
                (data['Wavenumber'] <= wavenumber_range[1])
            ]
        
        # Plot each time point
        exposure_times = sorted(data['ExposureTime'].unique())
        
        for i, time in enumerate(exposure_times):
            time_data = data[data['ExposureTime'] == time].sort_values('Wavenumber')
            
            color = self.colors[i % len(self.colors)]
            ax.plot(time_data['Wavenumber'], time_data['ProcessedAbsorbance'], 
                   color=color, label=f'{time}s', linewidth=1.5)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Normalized Absorbance')
        ax.set_title('FTIR Spectral Evolution During UV Curing')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Reverse x-axis for FTIR convention
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_difference_spectra(self, diff_data: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
        """
        Plot difference spectra
        
        Args:
            diff_data: Difference spectra data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        exposure_times = sorted(diff_data['ExposureTime'].unique())
        
        for i, time in enumerate(exposure_times):
            time_data = diff_data[diff_data['ExposureTime'] == time].sort_values('Wavenumber')
            
            color = self.colors[i % len(self.colors)]
            ax.plot(time_data['Wavenumber'], time_data['DifferenceAbsorbance'], 
                   color=color, label=f'{time}s', linewidth=1.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Absorbance Difference')
        ax.set_title('Difference Spectra Analysis')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_kinetic_curves(self, region_results: Dict, save_path: str = None) -> plt.Figure:
        """
        Plot kinetic curves for multiple regions
        
        Args:
            region_results: Results from multiple region analysis
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        n_regions = len(region_results)
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_regions == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (region_name, results) in enumerate(region_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            times = results['exposure_times']
            conversion = results['conversion_percent']
            
            # Plot experimental data
            ax.scatter(times, conversion, color='red', s=50, alpha=0.7, label='Experimental')
            
            # Plot fitted models
            if 'kinetic_models' in results:
                t_fit = np.linspace(min(times), max(times), 100)
                
                for model_name, model_data in results['kinetic_models'].items():
                    if 'error' not in model_data:
                        if model_name == 'zero_order':
                            y_fit = model_data['rate_constant'] * t_fit
                        elif model_name == 'first_order':
                            y_fit = model_data['c_max'] * (1 - np.exp(-model_data['rate_constant'] * t_fit))
                        elif model_name == 'autocatalytic':
                            y_fit = (model_data['c_max'] * (t_fit**model_data['n']) / 
                                   (model_data['t50']**model_data['n'] + t_fit**model_data['n']))
                        else:
                            continue
                            
                        ax.plot(t_fit, y_fit, '--', alpha=0.8, 
                               label=f"{model_name} (R²={model_data['r_squared']:.3f})")
            
            ax.set_xlabel('Exposure Time (s)')
            ax.set_ylabel('Conversion (%)')
            ax.set_title(f'{region_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_regions, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_pca_analysis(self, pca_results: Dict, save_path: str = None) -> plt.Figure:
        """
        Plot PCA analysis results
        
        Args:
            pca_results: PCA analysis results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scores plot (PC1 vs PC2)
        ax1 = axes[0, 0]
        scores = pca_results['scores']
        times = pca_results['exposure_times']
        
        scatter = ax1.scatter(scores[:, 0], scores[:, 1], c=times, cmap='viridis', s=100)
        ax1.set_xlabel(f'PC1 ({pca_results["explained_variance"][0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca_results["explained_variance"][1]*100:.1f}%)')
        ax1.set_title('PCA Scores Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Exposure Time (s)')
        
        # Explained variance plot
        ax2 = axes[0, 1]
        n_components = min(10, len(pca_results['explained_variance']))
        ax2.bar(range(1, n_components+1), pca_results['explained_variance'][:n_components]*100)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.set_title('Explained Variance by PC')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative variance plot
        ax3 = axes[1, 0]
        ax3.plot(range(1, n_components+1), pca_results['cumulative_variance'][:n_components]*100, 'o-')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance (%)')
        ax3.set_title('Cumulative Explained Variance')
        ax3.grid(True, alpha=0.3)
        
        # Loadings plot (PC1)
        ax4 = axes[1, 1]
        wavenumbers = pca_results['wavenumbers']
        loadings_pc1 = pca_results['loadings'][0, :]
        
        ax4.plot(wavenumbers, loadings_pc1)
        ax4.set_xlabel('Wavenumber (cm⁻¹)')
        ax4.set_ylabel('PC1 Loadings')
        ax4.set_title('PC1 Loadings')
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_plot(self, data: pd.DataFrame) -> go.Figure:
        """
        Create interactive plotly visualization
        
        Args:
            data: Processed spectral data
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spectral Evolution', 'Kinetic Curves', 
                          'Difference Spectra', 'PCA Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Spectral evolution
        exposure_times = sorted(data['ExposureTime'].unique())
        for time in exposure_times:
            time_data = data[data['ExposureTime'] == time].sort_values('Wavenumber')
            fig.add_trace(
                go.Scatter(x=time_data['Wavenumber'], 
                          y=time_data['ProcessedAbsorbance'],
                          mode='lines', name=f'{time}s'),
                row=1, col=1
            )
        
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=1, autorange="reversed")
        fig.update_yaxes(title_text="Normalized Absorbance", row=1, col=1)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Interactive FTIR UV Curing Analysis")
        
        return fig
    
    def generate_summary_plot(self, analysis_results: Dict, save_path: str = None) -> plt.Figure:
        """
        Generate comprehensive summary plot
        
        Args:
            analysis_results: Complete analysis results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Spectral evolution (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'processed_data' in analysis_results:
            self._plot_spectral_evolution_subplot(ax1, analysis_results['processed_data'])
        
        # Plot 2: Key region kinetics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'region_analysis' in analysis_results and 'acrylate_cc' in analysis_results['region_analysis']:
            self._plot_kinetics_subplot(ax2, analysis_results['region_analysis']['acrylate_cc'])
        
        # Plot 3: Difference spectra (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'difference_spectra' in analysis_results:
            self._plot_difference_subplot(ax3, analysis_results['difference_spectra'])
        
        # Plot 4: PCA scores (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'pca_analysis' in analysis_results:
            self._plot_pca_subplot(ax4, analysis_results['pca_analysis'])
        
        # Plot 5: Conversion summary (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'region_analysis' in analysis_results:
            self._plot_conversion_summary(ax5, analysis_results['region_analysis'])
        
        # Plot 6: Chemical interpretation (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        if 'chemical_interpretation' in analysis_results:
            self._plot_chemical_summary(ax6, analysis_results['chemical_interpretation'])
        
        plt.suptitle('FTIR UV Curing Analysis Summary', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def _plot_spectral_evolution_subplot(self, ax, data):
        """Helper method for spectral evolution subplot"""
        exposure_times = sorted(data['ExposureTime'].unique())[:5]  # Show first 5 time points
        
        for i, time in enumerate(exposure_times):
            time_data = data[data['ExposureTime'] == time].sort_values('Wavenumber')
            ax.plot(time_data['Wavenumber'], time_data['ProcessedAbsorbance'], 
                   color=self.colors[i], label=f'{time}s', alpha=0.8)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Normalized Absorbance')
        ax.set_title('Spectral Evolution')
        ax.legend()
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
    
    def _plot_kinetics_subplot(self, ax, region_data):
        """Helper method for kinetics subplot"""
        times = region_data['exposure_times']
        conversion = region_data['conversion_percent']
        
        ax.scatter(times, conversion, color='red', s=30, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Conversion (%)')
        ax.set_title('C=C Consumption')
        ax.grid(True, alpha=0.3)
    
    def _plot_difference_subplot(self, ax, diff_data):
        """Helper method for difference spectra subplot"""
        if not diff_data.empty:
            time = diff_data['ExposureTime'].iloc[-1]  # Last time point
            time_data = diff_data[diff_data['ExposureTime'] == time].sort_values('Wavenumber')
            
            ax.plot(time_data['Wavenumber'], time_data['DifferenceAbsorbance'], 'b-')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Δ Absorbance')
            ax.set_title('Difference Spectrum')
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
    
    def _plot_pca_subplot(self, ax, pca_data):
        """Helper method for PCA subplot"""
        scores = pca_data['scores']
        times = pca_data['exposure_times']
        
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=times, cmap='viridis', s=50)
        ax.set_xlabel(f'PC1 ({pca_data["explained_variance"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_data["explained_variance"][1]*100:.1f}%)')
        ax.set_title('PCA Scores')
        ax.grid(True, alpha=0.3)
    
    def _plot_conversion_summary(self, ax, region_data):
        """Helper method for conversion summary"""
        regions = list(region_data.keys())[:5]  # Show first 5 regions
        final_conversions = [max(region_data[region]['conversion_percent']) for region in regions]
        
        bars = ax.bar(range(len(regions)), final_conversions, color=self.colors[:len(regions)])
        ax.set_xlabel('Chemical Region')
        ax.set_ylabel('Final Conversion (%)')
        ax.set_title('Conversion Summary')
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([r.replace('_', '\n') for r in regions], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    def _plot_chemical_summary(self, ax, chem_data):
        """Helper method for chemical interpretation summary"""
        ax.text(0.1, 0.8, 'Chemical Interpretation Summary:', fontsize=12, fontweight='bold', 
                transform=ax.transAxes)
        
        y_pos = 0.6
        if 'conversion_summary' in chem_data:
            for region, data in chem_data['conversion_summary'].items():
                text = f"• {region}: {data.get('final_conversion', 0):.1f}% conversion"
                ax.text(0.1, y_pos, text, fontsize=10, transform=ax.transAxes)
                y_pos -= 0.1
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
