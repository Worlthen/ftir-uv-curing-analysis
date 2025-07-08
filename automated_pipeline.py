#!/usr/bin/env python3
"""
Automated FTIR UV Curing Analysis Pipeline
Complete automation from OPUS files to final analysis report
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from opus_reader import OPUSReader
from ftir_analyzer import FTIRUVCuringAnalyzer
from visualization import FTIRVisualizer
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ftir_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FTIRAutomatedPipeline:
    """
    Automated pipeline for complete FTIR UV curing analysis
    
    Pipeline steps:
    1. Find and read OPUS files
    2. Convert to CSV format
    3. Create integrated dataset
    4. Perform comprehensive analysis
    5. Generate visualizations
    6. Create analysis report
    """
    
    def __init__(self, input_dir: str = '.', output_dir: str = './results'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.opus_reader = OPUSReader()
        self.analyzer = FTIRUVCuringAnalyzer()
        self.visualizer = FTIRVisualizer()
        self.report_generator = ReportGenerator()
        
        # Analysis parameters
        self.analysis_params = {
            'baseline_method': 'als',
            'normalization_method': 'max',
            'wavenumber_range': (1000, 4000),
            'time_threshold': 0.1  # Minimum time difference for analysis
        }
        
        # Results storage
        self.results = {}
        
    def run_complete_pipeline(self) -> dict:
        """
        Execute the complete automated analysis pipeline
        
        Returns:
            Dictionary containing all analysis results and file paths
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED FTIR UV CURING ANALYSIS PIPELINE")
        logger.info("="*60)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'status': 'running',
            'steps_completed': [],
            'files_generated': [],
            'analysis_results': {}
        }
        
        try:
            # Step 1: Process OPUS files
            logger.info("STEP 1: Processing OPUS files")
            conversion_results = self._step1_process_opus_files()
            pipeline_results['steps_completed'].append('opus_conversion')
            pipeline_results['conversion_results'] = conversion_results
            
            if not conversion_results['csv_files']:
                raise Exception("No OPUS files were successfully converted")
            
            # Step 2: Create integrated dataset
            logger.info("STEP 2: Creating integrated dataset")
            integrated_file = self._step2_create_integrated_dataset(conversion_results['csv_files'])
            pipeline_results['steps_completed'].append('data_integration')
            pipeline_results['integrated_dataset'] = integrated_file
            
            # Step 3: Load and validate data
            logger.info("STEP 3: Loading and validating data")
            data_validation = self._step3_load_and_validate_data(integrated_file)
            pipeline_results['steps_completed'].append('data_validation')
            pipeline_results['data_validation'] = data_validation
            
            # Step 4: Perform comprehensive analysis
            logger.info("STEP 4: Performing comprehensive analysis")
            analysis_results = self._step4_comprehensive_analysis()
            pipeline_results['steps_completed'].append('analysis')
            pipeline_results['analysis_results'] = analysis_results
            
            # Step 5: Generate visualizations
            logger.info("STEP 5: Generating visualizations")
            visualization_files = self._step5_generate_visualizations(analysis_results)
            pipeline_results['steps_completed'].append('visualization')
            pipeline_results['visualization_files'] = visualization_files
            pipeline_results['files_generated'].extend(visualization_files)
            
            # Step 6: Create comprehensive report
            logger.info("STEP 6: Creating comprehensive report")
            report_files = self._step6_generate_reports(analysis_results)
            pipeline_results['steps_completed'].append('reporting')
            pipeline_results['report_files'] = report_files
            pipeline_results['files_generated'].extend(report_files)
            
            # Step 7: Save pipeline metadata
            logger.info("STEP 7: Saving pipeline metadata")
            metadata_file = self._step7_save_metadata(pipeline_results)
            pipeline_results['metadata_file'] = metadata_file
            pipeline_results['files_generated'].append(metadata_file)
            
            pipeline_results['status'] = 'completed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Files generated: {len(pipeline_results['files_generated'])}")
            logger.info("="*60)
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        
        return pipeline_results
    
    def _step1_process_opus_files(self) -> dict:
        """Step 1: Find and convert OPUS files to CSV"""
        logger.info(f"Searching for OPUS files in: {self.input_dir}")
        
        # Find OPUS files
        opus_files = self.opus_reader.find_opus_files(str(self.input_dir))
        logger.info(f"Found {len(opus_files)} OPUS files")
        
        if not opus_files:
            raise Exception(f"No OPUS files found in {self.input_dir}")
        
        # Convert to CSV
        csv_output_dir = self.output_dir / 'csv_files'
        csv_output_dir.mkdir(exist_ok=True)
        
        conversion_results = self.opus_reader.batch_convert(
            str(self.input_dir), 
            str(csv_output_dir)
        )
        
        logger.info(f"Conversion complete: {len(conversion_results['successful'])}/{conversion_results['total_files']} files")
        
        return conversion_results
    
    def _step2_create_integrated_dataset(self, csv_files: list) -> str:
        """Step 2: Create integrated dataset from CSV files"""
        integrated_file = self.output_dir / 'integrated_spectra.csv'
        
        result_file = self.opus_reader.create_integrated_dataset(
            csv_files, 
            str(integrated_file)
        )
        
        if result_file:
            logger.info(f"Integrated dataset created: {result_file}")
            return result_file
        else:
            raise Exception("Failed to create integrated dataset")
    
    def _step3_load_and_validate_data(self, integrated_file: str) -> dict:
        """Step 3: Load and validate the integrated dataset"""
        success = self.analyzer.load_data(integrated_file)
        
        if not success:
            raise Exception("Failed to load integrated dataset")
        
        validation_results = {
            'file_loaded': True,
            'num_time_points': len(self.analyzer.exposure_times),
            'num_wavenumbers': len(self.analyzer.wavenumbers),
            'time_range': [min(self.analyzer.exposure_times), max(self.analyzer.exposure_times)],
            'wavenumber_range': [min(self.analyzer.wavenumbers), max(self.analyzer.wavenumbers)]
        }
        
        logger.info(f"Data validation: {validation_results['num_time_points']} time points, "
                   f"{validation_results['num_wavenumbers']} wavenumbers")
        
        return validation_results
    
    def _step4_comprehensive_analysis(self) -> dict:
        """Step 4: Perform comprehensive FTIR analysis"""
        analysis_results = self.analyzer.run_automated_analysis(
            baseline_method=self.analysis_params['baseline_method'],
            norm_method=self.analysis_params['normalization_method']
        )
        
        # Add processed data to results for visualization
        analysis_results['processed_data'] = self.analyzer.processed_data
        
        logger.info("Comprehensive analysis completed")
        logger.info(f"Analyzed {len(analysis_results['region_analysis'])} chemical regions")
        
        return analysis_results
    
    def _step5_generate_visualizations(self, analysis_results: dict) -> list:
        """Step 5: Generate all visualization plots"""
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        visualization_files = []
        
        try:
            # 1. Spectral evolution plot
            if 'processed_data' in analysis_results:
                fig1 = self.visualizer.plot_spectral_evolution(
                    analysis_results['processed_data'],
                    save_path=str(viz_dir / 'spectral_evolution.png')
                )
                visualization_files.append(str(viz_dir / 'spectral_evolution.png'))
                plt.close(fig1)
            
            # 2. Difference spectra plot
            if 'difference_spectra' in analysis_results and not analysis_results['difference_spectra'].empty:
                fig2 = self.visualizer.plot_difference_spectra(
                    analysis_results['difference_spectra'],
                    save_path=str(viz_dir / 'difference_spectra.png')
                )
                visualization_files.append(str(viz_dir / 'difference_spectra.png'))
                plt.close(fig2)
            
            # 3. Kinetic curves plot
            if 'region_analysis' in analysis_results:
                fig3 = self.visualizer.plot_kinetic_curves(
                    analysis_results['region_analysis'],
                    save_path=str(viz_dir / 'kinetic_curves.png')
                )
                visualization_files.append(str(viz_dir / 'kinetic_curves.png'))
                plt.close(fig3)
            
            # 4. PCA analysis plot
            if 'pca_analysis' in analysis_results:
                fig4 = self.visualizer.plot_pca_analysis(
                    analysis_results['pca_analysis'],
                    save_path=str(viz_dir / 'pca_analysis.png')
                )
                visualization_files.append(str(viz_dir / 'pca_analysis.png'))
                plt.close(fig4)
            
            # 5. Summary plot
            fig5 = self.visualizer.generate_summary_plot(
                analysis_results,
                save_path=str(viz_dir / 'analysis_summary.png')
            )
            visualization_files.append(str(viz_dir / 'analysis_summary.png'))
            plt.close(fig5)
            
        except Exception as e:
            logger.warning(f"Some visualizations failed: {str(e)}")
        
        logger.info(f"Generated {len(visualization_files)} visualization files")
        return visualization_files
    
    def _step6_generate_reports(self, analysis_results: dict) -> list:
        """Step 6: Generate comprehensive analysis reports"""
        reports_dir = self.output_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_files = []
        
        try:
            # Generate text report
            text_report = self.report_generator.generate_text_report(
                analysis_results,
                str(reports_dir / 'analysis_report.txt')
            )
            report_files.append(text_report)
            
            # Generate HTML report
            html_report = self.report_generator.generate_html_report(
                analysis_results,
                str(reports_dir / 'analysis_report.html')
            )
            report_files.append(html_report)
            
            # Generate Excel report
            excel_report = self.report_generator.generate_excel_report(
                analysis_results,
                str(reports_dir / 'analysis_data.xlsx')
            )
            report_files.append(excel_report)
            
        except Exception as e:
            logger.warning(f"Some reports failed to generate: {str(e)}")
        
        logger.info(f"Generated {len(report_files)} report files")
        return report_files
    
    def _step7_save_metadata(self, pipeline_results: dict) -> str:
        """Step 7: Save pipeline metadata and results"""
        metadata_file = self.output_dir / 'pipeline_metadata.json'
        
        # Remove large data objects for JSON serialization
        metadata = pipeline_results.copy()
        if 'analysis_results' in metadata and 'processed_data' in metadata['analysis_results']:
            del metadata['analysis_results']['processed_data']
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline metadata saved: {metadata_file}")
        return str(metadata_file)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Automated FTIR UV Curing Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automated_pipeline.py --input_dir ./opus_files --output_dir ./results
  python automated_pipeline.py --input_dir . --baseline als --normalization max
        """
    )
    
    parser.add_argument('--input_dir', default='.', 
                       help='Input directory containing OPUS files (default: current directory)')
    parser.add_argument('--output_dir', default='./results', 
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--baseline', default='als', choices=['als', 'polynomial'],
                       help='Baseline correction method (default: als)')
    parser.add_argument('--normalization', default='max', choices=['max', 'area', 'snv'],
                       help='Normalization method (default: max)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = FTIRAutomatedPipeline(args.input_dir, args.output_dir)
    
    # Update analysis parameters
    pipeline.analysis_params['baseline_method'] = args.baseline
    pipeline.analysis_params['normalization_method'] = args.normalization
    
    try:
        # Run pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Input directory: {results['input_directory']}")
        print(f"Output directory: {results['output_directory']}")
        print(f"Steps completed: {len(results['steps_completed'])}")
        print(f"Files generated: {len(results['files_generated'])}")
        print("\nGenerated files:")
        for file_path in results['files_generated']:
            print(f"  - {file_path}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    sys.exit(main())
