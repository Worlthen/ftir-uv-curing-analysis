"""
Bruker OPUS File Reader Module
Automated conversion of Bruker OPUS files to CSV format with comprehensive error handling
"""

import os
import numpy as np
import pandas as pd
from brukeropusreader import read_file
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OPUSReader:
    """
    Bruker OPUS file reader with automated CSV conversion capabilities
    
    Features:
    - Batch processing of OPUS files
    - Automatic wavenumber extraction
    - Error handling and validation
    - Multiple data block support
    - Metadata preservation
    """
    
    def __init__(self):
        self.supported_extensions = ['.0', '.1', '.2', '.3']
        self.processed_files = []
        self.failed_files = []
        
    def find_opus_files(self, directory: str = '.') -> List[str]:
        """
        Find all OPUS files in the specified directory
        
        Args:
            directory: Directory path to search for OPUS files
            
        Returns:
            List of OPUS file paths
        """
        opus_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return opus_files
            
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                opus_files.append(str(file_path))
                
        logger.info(f"Found {len(opus_files)} OPUS files in {directory}")
        return sorted(opus_files)
    
    def extract_wavenumbers(self, opus_data: Dict) -> Optional[np.ndarray]:
        """
        Extract wavenumber array from OPUS data
        
        Args:
            opus_data: Dictionary containing OPUS file data
            
        Returns:
            Wavenumber array or None if extraction fails
        """
        try:
            # Try to get wavenumber information from different possible keys
            if 'AB' in opus_data and hasattr(opus_data['AB'], 'shape'):
                # Calculate wavenumbers from spectral parameters
                if 'AB_param' in opus_data:
                    params = opus_data['AB_param']
                    if 'FXV' in params and 'LXV' in params and 'NPT' in params:
                        first_x = params['FXV']
                        last_x = params['LXV']
                        num_points = params['NPT']
                        wavenumbers = np.linspace(first_x, last_x, num_points)
                        return wavenumbers
                
                # Fallback: create index-based wavenumbers
                data_length = len(opus_data['AB'])
                wavenumbers = np.arange(data_length)
                logger.warning("Using index-based wavenumbers - actual wavenumber calibration needed")
                return wavenumbers
                
        except Exception as e:
            logger.error(f"Error extracting wavenumbers: {str(e)}")
            
        return None
    
    def read_opus_file(self, file_path: str) -> Optional[Dict]:
        """
        Read a single OPUS file and extract spectral data
        
        Args:
            file_path: Path to the OPUS file
            
        Returns:
            Dictionary containing spectral data and metadata
        """
        try:
            # Read OPUS file
            opus_data = read_file(file_path)
            
            if 'AB' not in opus_data:
                logger.error(f"No absorbance data found in {file_path}")
                return None
                
            # Extract data
            absorbance = opus_data['AB']
            wavenumbers = self.extract_wavenumbers(opus_data)
            
            if wavenumbers is None:
                logger.error(f"Could not extract wavenumbers from {file_path}")
                return None
                
            # Ensure data consistency
            if len(wavenumbers) != len(absorbance):
                logger.error(f"Wavenumber and absorbance array length mismatch in {file_path}")
                return None
                
            # Extract metadata
            metadata = self.extract_metadata(opus_data, file_path)
            
            result = {
                'wavenumbers': wavenumbers,
                'absorbance': absorbance,
                'metadata': metadata,
                'file_path': file_path
            }
            
            logger.info(f"Successfully read {file_path}: {len(absorbance)} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None
    
    def extract_metadata(self, opus_data: Dict, file_path: str) -> Dict:
        """
        Extract metadata from OPUS file
        
        Args:
            opus_data: OPUS file data dictionary
            file_path: Original file path
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'filename': os.path.basename(file_path),
            'file_path': file_path
        }
        
        # Extract common parameters
        if 'AB_param' in opus_data:
            params = opus_data['AB_param']
            metadata.update({
                'resolution': params.get('RES', 'Unknown'),
                'scanner_velocity': params.get('VEL', 'Unknown'),
                'aperture': params.get('APT', 'Unknown'),
                'detector': params.get('DTC', 'Unknown'),
                'source': params.get('SRC', 'Unknown'),
                'beamsplitter': params.get('BMS', 'Unknown'),
                'measurement_date': params.get('DAT', 'Unknown'),
                'measurement_time': params.get('TIM', 'Unknown')
            })
            
        return metadata
    
    def convert_to_csv(self, opus_data: Dict, output_path: str = None) -> str:
        """
        Convert OPUS data to CSV format
        
        Args:
            opus_data: Dictionary containing OPUS spectral data
            output_path: Output CSV file path (optional)
            
        Returns:
            Path to the created CSV file
        """
        if output_path is None:
            base_name = os.path.splitext(opus_data['file_path'])[0]
            ext = os.path.splitext(opus_data['file_path'])[1]
            output_path = f"{base_name}{ext}.csv"
        
        # Create DataFrame
        df = pd.DataFrame({
            'Wavenumber': opus_data['wavenumbers'],
            'Absorbance': opus_data['absorbance']
        })
        
        # Add metadata as comments in the CSV
        metadata_lines = []
        for key, value in opus_data['metadata'].items():
            metadata_lines.append(f"# {key}: {value}")
        
        # Save CSV with metadata
        with open(output_path, 'w') as f:
            # Write metadata as comments
            for line in metadata_lines:
                f.write(line + '\n')
            f.write('# Data starts below\n')
            
        # Append data
        df.to_csv(output_path, mode='a', index=False)
        
        logger.info(f"Converted to CSV: {output_path}")
        return output_path
    
    def batch_convert(self, input_directory: str = '.', output_directory: str = None) -> Dict:
        """
        Batch convert all OPUS files in a directory to CSV
        
        Args:
            input_directory: Directory containing OPUS files
            output_directory: Directory for CSV output (optional)
            
        Returns:
            Dictionary with conversion results
        """
        if output_directory is None:
            output_directory = input_directory
            
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Find OPUS files
        opus_files = self.find_opus_files(input_directory)
        
        results = {
            'total_files': len(opus_files),
            'successful': [],
            'failed': [],
            'csv_files': []
        }
        
        for file_path in opus_files:
            try:
                # Read OPUS file
                opus_data = self.read_opus_file(file_path)
                
                if opus_data is not None:
                    # Generate output path
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    ext = os.path.splitext(file_path)[1]
                    csv_filename = f"{base_name}{ext}.csv"
                    csv_path = os.path.join(output_directory, csv_filename)
                    
                    # Convert to CSV
                    csv_file = self.convert_to_csv(opus_data, csv_path)
                    
                    results['successful'].append(file_path)
                    results['csv_files'].append(csv_file)
                else:
                    results['failed'].append(file_path)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results['failed'].append(file_path)
        
        # Log summary
        logger.info(f"Batch conversion complete:")
        logger.info(f"  Total files: {results['total_files']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        
        return results
    
    def extract_time_from_filename(self, filename: str) -> Optional[float]:
        """
        Extract exposure time from filename
        
        Args:
            filename: OPUS filename
            
        Returns:
            Exposure time in seconds or None
        """
        import re
        
        # Common patterns for time extraction
        patterns = [
            r'(\d+)s_',  # Pattern: "16s_"
            r'(\d+)sec',  # Pattern: "16sec"
            r't(\d+)',    # Pattern: "t16"
            r'time(\d+)', # Pattern: "time16"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
                    
        return None
    
    def create_integrated_dataset(self, csv_files: List[str], output_path: str = 'integrated_spectra.csv') -> str:
        """
        Create an integrated dataset from multiple CSV files
        
        Args:
            csv_files: List of CSV file paths
            output_path: Output path for integrated dataset
            
        Returns:
            Path to integrated dataset
        """
        all_data = []
        
        for csv_file in csv_files:
            try:
                # Read CSV file (skip comment lines)
                df = pd.read_csv(csv_file, comment='#')
                
                # Extract time from filename
                filename = os.path.basename(csv_file)
                exposure_time = self.extract_time_from_filename(filename)
                
                if exposure_time is not None:
                    df['ExposureTime'] = exposure_time
                    df['Filename'] = filename
                    all_data.append(df)
                else:
                    logger.warning(f"Could not extract time from {filename}")
                    
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {str(e)}")
        
        if all_data:
            # Combine all data
            integrated_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by exposure time and wavenumber
            integrated_df = integrated_df.sort_values(['ExposureTime', 'Wavenumber'])
            
            # Save integrated dataset
            integrated_df.to_csv(output_path, index=False)
            logger.info(f"Created integrated dataset: {output_path}")
            
            return output_path
        else:
            logger.error("No valid data found for integration")
            return None


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Bruker OPUS files to CSV')
    parser.add_argument('--input_dir', default='.', help='Input directory containing OPUS files')
    parser.add_argument('--output_dir', default=None, help='Output directory for CSV files')
    parser.add_argument('--integrate', action='store_true', help='Create integrated dataset')
    
    args = parser.parse_args()
    
    # Initialize reader
    reader = OPUSReader()
    
    # Batch convert files
    results = reader.batch_convert(args.input_dir, args.output_dir)
    
    # Create integrated dataset if requested
    if args.integrate and results['csv_files']:
        reader.create_integrated_dataset(results['csv_files'])
    
    print(f"Conversion complete: {len(results['successful'])}/{results['total_files']} files processed")


if __name__ == "__main__":
    main()
