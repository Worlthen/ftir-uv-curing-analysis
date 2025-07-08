"""
Report Generation Module
Automated generation of comprehensive analysis reports in multiple formats
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generate comprehensive analysis reports in multiple formats
    """
    
    def __init__(self):
        self.report_template = {
            'title': 'FTIR UV Curing Analysis Report',
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'software_version': '1.0.0'
        }
    
    def generate_text_report(self, analysis_results: Dict, output_path: str) -> str:
        """
        Generate comprehensive text report
        
        Args:
            analysis_results: Complete analysis results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "FTIR UV CURING ANALYSIS REPORT",
            "="*80,
            f"Generated: {self.report_template['generated_date']}",
            f"Software Version: {self.report_template['software_version']}",
            "",
        ])
        
        # Analysis Parameters
        if 'metadata' in analysis_results:
            metadata = analysis_results['metadata']
            report_lines.extend([
                "ANALYSIS PARAMETERS",
                "-" * 40,
                f"Baseline Method: {metadata.get('baseline_method', 'Unknown')}",
                f"Normalization Method: {metadata.get('normalization_method', 'Unknown')}",
                f"Number of Time Points: {len(metadata.get('exposure_times', []))}",
                f"Wavenumber Range: {metadata.get('wavenumber_range', 'Unknown')} cm⁻¹",
                "",
            ])
        
        # Region Analysis Results
        if 'region_analysis' in analysis_results:
            report_lines.extend([
                "CHEMICAL REGION ANALYSIS",
                "-" * 40,
            ])
            
            for region_name, region_data in analysis_results['region_analysis'].items():
                report_lines.extend([
                    f"\n{region_name.upper().replace('_', ' ')}:",
                    f"  Wavenumber Range: {region_data['wavenumber_range'][0]:.0f}-{region_data['wavenumber_range'][1]:.0f} cm⁻¹",
                    f"  Final Conversion: {max(region_data['conversion_percent']):.2f}%",
                ])
                
                # Kinetic models
                if 'kinetic_models' in region_data:
                    best_model = self._find_best_model(region_data['kinetic_models'])
                    if best_model:
                        model_data = region_data['kinetic_models'][best_model]
                        report_lines.extend([
                            f"  Best Kinetic Model: {best_model}",
                            f"  R² Value: {model_data.get('r_squared', 0):.4f}",
                            f"  Rate Constant: {model_data.get('rate_constant', 0):.2e} s⁻¹",
                        ])
            
            report_lines.append("")
        
        # PCA Analysis
        if 'pca_analysis' in analysis_results:
            pca_data = analysis_results['pca_analysis']
            report_lines.extend([
                "PRINCIPAL COMPONENT ANALYSIS",
                "-" * 40,
                f"PC1 Explained Variance: {pca_data['explained_variance'][0]*100:.1f}%",
                f"PC2 Explained Variance: {pca_data['explained_variance'][1]*100:.1f}%",
                f"PC3 Explained Variance: {pca_data['explained_variance'][2]*100:.1f}%",
                f"Cumulative Variance (3 PCs): {pca_data['cumulative_variance'][2]*100:.1f}%",
                "",
            ])
        
        # Chemical Interpretation
        if 'chemical_interpretation' in analysis_results:
            chem_data = analysis_results['chemical_interpretation']
            report_lines.extend([
                "CHEMICAL INTERPRETATION",
                "-" * 40,
            ])
            
            if 'conversion_summary' in chem_data:
                for region, data in chem_data['conversion_summary'].items():
                    report_lines.append(f"{region}: {data.get('final_conversion', 0):.1f}% conversion ({data.get('reaction_type', 'Unknown')})")
            
            if 'chemical_pathways' in chem_data:
                report_lines.append("\nIdentified Reaction Pathways:")
                for pathway in chem_data['chemical_pathways']:
                    report_lines.extend([
                        f"• {pathway['pathway']}",
                        f"  Evidence: {pathway['evidence']}",
                        f"  Wavenumber: {pathway['wavenumber_range']}",
                    ])
            
            report_lines.append("")
        
        # Summary and Conclusions
        report_lines.extend([
            "SUMMARY AND CONCLUSIONS",
            "-" * 40,
            self._generate_summary_text(analysis_results),
            "",
            "="*80,
            "END OF REPORT",
            "="*80,
        ])
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Text report generated: {output_path}")
        return output_path
    
    def generate_html_report(self, analysis_results: Dict, output_path: str) -> str:
        """
        Generate HTML report with embedded visualizations
        
        Args:
            analysis_results: Complete analysis results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FTIR UV Curing Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 5px; }}
        .parameter-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .parameter-table th, .parameter-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .parameter-table th {{ background-color: #f2f2f2; }}
        .results-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .result-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .footer {{ margin-top: 50px; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>FTIR UV Curing Analysis Report</h1>
        <p><strong>Generated:</strong> {self.report_template['generated_date']}</p>
        <p><strong>Software Version:</strong> {self.report_template['software_version']}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Parameters</h2>
        {self._generate_html_parameters_table(analysis_results)}
    </div>
    
    <div class="section">
        <h2>Chemical Region Analysis</h2>
        {self._generate_html_region_results(analysis_results)}
    </div>
    
    <div class="section">
        <h2>Statistical Analysis</h2>
        {self._generate_html_pca_results(analysis_results)}
    </div>
    
    <div class="section">
        <h2>Chemical Interpretation</h2>
        {self._generate_html_interpretation(analysis_results)}
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p>{self._generate_summary_text(analysis_results)}</p>
    </div>
    
    <div class="footer">
        <p>Report generated by FTIR UV Curing Analysis System</p>
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def generate_excel_report(self, analysis_results: Dict, output_path: str) -> str:
        """
        Generate Excel report with multiple sheets
        
        Args:
            analysis_results: Complete analysis results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = self._prepare_summary_data(analysis_results)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Region analysis sheet
                if 'region_analysis' in analysis_results:
                    region_data = self._prepare_region_data(analysis_results['region_analysis'])
                    region_df = pd.DataFrame(region_data)
                    region_df.to_excel(writer, sheet_name='Region_Analysis', index=False)
                
                # Kinetic parameters sheet
                kinetic_data = self._prepare_kinetic_data(analysis_results)
                if kinetic_data:
                    kinetic_df = pd.DataFrame(kinetic_data)
                    kinetic_df.to_excel(writer, sheet_name='Kinetic_Parameters', index=False)
                
                # PCA results sheet
                if 'pca_analysis' in analysis_results:
                    pca_data = analysis_results['pca_analysis']
                    pca_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(pca_data['explained_variance']))],
                        'Explained_Variance': pca_data['explained_variance'],
                        'Cumulative_Variance': pca_data['cumulative_variance']
                    })
                    pca_df.to_excel(writer, sheet_name='PCA_Results', index=False)
                
                # Raw data sheet (if available)
                if 'processed_data' in analysis_results:
                    processed_data = analysis_results['processed_data']
                    if isinstance(processed_data, pd.DataFrame) and not processed_data.empty:
                        # Sample the data if it's too large
                        if len(processed_data) > 10000:
                            sampled_data = processed_data.sample(n=10000, random_state=42)
                        else:
                            sampled_data = processed_data
                        sampled_data.to_excel(writer, sheet_name='Processed_Data', index=False)
            
            logger.info(f"Excel report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {str(e)}")
            return None
    
    def _find_best_model(self, models: Dict) -> Optional[str]:
        """Find the best kinetic model based on R-squared"""
        best_model = None
        best_r2 = -1
        
        for model_name, model_data in models.items():
            if 'r_squared' in model_data and model_data['r_squared'] > best_r2:
                best_r2 = model_data['r_squared']
                best_model = model_name
        
        return best_model
    
    def _generate_summary_text(self, analysis_results: Dict) -> str:
        """Generate summary text for the analysis"""
        summary_parts = []
        
        # Overall conversion
        if 'region_analysis' in analysis_results:
            max_conversion = 0
            for region_data in analysis_results['region_analysis'].values():
                conversion = max(region_data['conversion_percent'])
                max_conversion = max(max_conversion, conversion)
            
            summary_parts.append(f"Maximum conversion achieved: {max_conversion:.1f}%")
        
        # PCA summary
        if 'pca_analysis' in analysis_results:
            pca_data = analysis_results['pca_analysis']
            cum_var = pca_data['cumulative_variance'][2] * 100
            summary_parts.append(f"First three principal components explain {cum_var:.1f}% of spectral variance")
        
        # Chemical pathways
        if 'chemical_interpretation' in analysis_results:
            pathways = analysis_results['chemical_interpretation'].get('chemical_pathways', [])
            if pathways:
                summary_parts.append(f"Identified {len(pathways)} major reaction pathways")
        
        return ". ".join(summary_parts) + "." if summary_parts else "Analysis completed successfully."
    
    def _generate_html_parameters_table(self, analysis_results: Dict) -> str:
        """Generate HTML table for analysis parameters"""
        if 'metadata' not in analysis_results:
            return "<p>No parameter information available.</p>"
        
        metadata = analysis_results['metadata']
        
        html = '<table class="parameter-table">'
        html += '<tr><th>Parameter</th><th>Value</th></tr>'
        html += f'<tr><td>Baseline Method</td><td>{metadata.get("baseline_method", "Unknown")}</td></tr>'
        html += f'<tr><td>Normalization Method</td><td>{metadata.get("normalization_method", "Unknown")}</td></tr>'
        html += f'<tr><td>Number of Time Points</td><td>{len(metadata.get("exposure_times", []))}</td></tr>'
        html += f'<tr><td>Wavenumber Range</td><td>{metadata.get("wavenumber_range", "Unknown")} cm⁻¹</td></tr>'
        html += '</table>'
        
        return html
    
    def _generate_html_region_results(self, analysis_results: Dict) -> str:
        """Generate HTML for region analysis results"""
        if 'region_analysis' not in analysis_results:
            return "<p>No region analysis results available.</p>"
        
        html = '<div class="results-grid">'
        
        for region_name, region_data in analysis_results['region_analysis'].items():
            html += f'<div class="result-card">'
            html += f'<h3>{region_name.replace("_", " ").title()}</h3>'
            html += f'<p><strong>Wavenumber Range:</strong> {region_data["wavenumber_range"][0]:.0f}-{region_data["wavenumber_range"][1]:.0f} cm⁻¹</p>'
            html += f'<p><strong>Final Conversion:</strong> {max(region_data["conversion_percent"]):.2f}%</p>'
            
            if 'kinetic_models' in region_data:
                best_model = self._find_best_model(region_data['kinetic_models'])
                if best_model:
                    model_data = region_data['kinetic_models'][best_model]
                    html += f'<p><strong>Best Model:</strong> {best_model}</p>'
                    html += f'<p><strong>R² Value:</strong> {model_data.get("r_squared", 0):.4f}</p>'
            
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _generate_html_pca_results(self, analysis_results: Dict) -> str:
        """Generate HTML for PCA results"""
        if 'pca_analysis' not in analysis_results:
            return "<p>No PCA analysis results available.</p>"
        
        pca_data = analysis_results['pca_analysis']
        
        html = '<table class="parameter-table">'
        html += '<tr><th>Component</th><th>Explained Variance (%)</th><th>Cumulative Variance (%)</th></tr>'
        
        for i in range(min(5, len(pca_data['explained_variance']))):
            html += f'<tr><td>PC{i+1}</td><td>{pca_data["explained_variance"][i]*100:.1f}</td><td>{pca_data["cumulative_variance"][i]*100:.1f}</td></tr>'
        
        html += '</table>'
        return html
    
    def _generate_html_interpretation(self, analysis_results: Dict) -> str:
        """Generate HTML for chemical interpretation"""
        if 'chemical_interpretation' not in analysis_results:
            return "<p>No chemical interpretation available.</p>"
        
        chem_data = analysis_results['chemical_interpretation']
        html = ""
        
        if 'conversion_summary' in chem_data:
            html += '<h3>Conversion Summary</h3><ul>'
            for region, data in chem_data['conversion_summary'].items():
                html += f'<li>{region}: {data.get("final_conversion", 0):.1f}% conversion ({data.get("reaction_type", "Unknown")})</li>'
            html += '</ul>'
        
        if 'chemical_pathways' in chem_data:
            html += '<h3>Reaction Pathways</h3><ul>'
            for pathway in chem_data['chemical_pathways']:
                html += f'<li><strong>{pathway["pathway"]}</strong>: {pathway["evidence"]} ({pathway["wavenumber_range"]})</li>'
            html += '</ul>'
        
        return html
    
    def _prepare_summary_data(self, analysis_results: Dict) -> List[Dict]:
        """Prepare summary data for Excel export"""
        summary_data = []
        
        if 'metadata' in analysis_results:
            metadata = analysis_results['metadata']
            summary_data.extend([
                {'Parameter': 'Baseline Method', 'Value': metadata.get('baseline_method', 'Unknown')},
                {'Parameter': 'Normalization Method', 'Value': metadata.get('normalization_method', 'Unknown')},
                {'Parameter': 'Number of Time Points', 'Value': len(metadata.get('exposure_times', []))},
                {'Parameter': 'Wavenumber Range', 'Value': str(metadata.get('wavenumber_range', 'Unknown'))},
            ])
        
        return summary_data
    
    def _prepare_region_data(self, region_analysis: Dict) -> List[Dict]:
        """Prepare region analysis data for Excel export"""
        region_data = []
        
        for region_name, region_results in region_analysis.items():
            row = {
                'Region': region_name,
                'Wavenumber_Min': region_results['wavenumber_range'][0],
                'Wavenumber_Max': region_results['wavenumber_range'][1],
                'Final_Conversion': max(region_results['conversion_percent']),
            }
            
            if 'kinetic_models' in region_results:
                best_model = self._find_best_model(region_results['kinetic_models'])
                if best_model:
                    model_data = region_results['kinetic_models'][best_model]
                    row['Best_Model'] = best_model
                    row['R_Squared'] = model_data.get('r_squared', 0)
                    row['Rate_Constant'] = model_data.get('rate_constant', 0)
            
            region_data.append(row)
        
        return region_data
    
    def _prepare_kinetic_data(self, analysis_results: Dict) -> List[Dict]:
        """Prepare kinetic parameters data for Excel export"""
        kinetic_data = []
        
        if 'region_analysis' in analysis_results:
            for region_name, region_results in analysis_results['region_analysis'].items():
                if 'kinetic_models' in region_results:
                    for model_name, model_data in region_results['kinetic_models'].items():
                        if 'error' not in model_data:
                            row = {
                                'Region': region_name,
                                'Model': model_name,
                                'R_Squared': model_data.get('r_squared', 0),
                                'Rate_Constant': model_data.get('rate_constant', 0),
                            }
                            
                            # Add model-specific parameters
                            if 'c_max' in model_data:
                                row['C_Max'] = model_data['c_max']
                            if 't50' in model_data:
                                row['T50'] = model_data['t50']
                            if 'n' in model_data:
                                row['n'] = model_data['n']
                            
                            kinetic_data.append(row)
        
        return kinetic_data
