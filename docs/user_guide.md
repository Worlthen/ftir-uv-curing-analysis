# FTIR UV Curing Analysis - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Analysis Workflow](#analysis-workflow)
5. [Understanding Results](#understanding-results)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux operating system
- Minimum 4GB RAM recommended

### Installation Steps

#### Method 1: Using pip (Recommended)
```bash
pip install ftir-uv-curing-analysis
```

#### Method 2: From Source
```bash
git clone https://github.com/your-username/ftir-uv-curing-analysis.git
cd ftir-uv-curing-analysis
pip install -r requirements.txt
pip install -e .
```

### Verify Installation
```bash
ftir-analysis --help
ftir-gui
```

## Quick Start

### Command Line Interface
```bash
# Process OPUS files in current directory
ftir-analysis --input_dir ./opus_files --output_dir ./results

# Use specific analysis parameters
ftir-analysis --baseline als --normalization max --verbose
```

### Graphical User Interface
```bash
# Launch GUI application
ftir-gui
```

### Python API
```python
from ftir_analyzer import FTIRUVCuringAnalyzer

# Initialize analyzer
analyzer = FTIRUVCuringAnalyzer()

# Load data
analyzer.load_data('integrated_spectra.csv')

# Run analysis
results = analyzer.run_automated_analysis()
```

## Data Preparation

### OPUS Files
The system supports Bruker OPUS files with extensions:
- `.0` - Primary spectrum
- `.1` - Secondary spectrum  
- `.2` - Tertiary spectrum
- `.3` - Quaternary spectrum

### File Naming Convention
For automatic time extraction, use these naming patterns:
- `0s_sample.0` - 0 seconds exposure
- `16s_sample.1` - 16 seconds exposure
- `32s_sample.2` - 32 seconds exposure

### CSV Data Format
If using pre-processed CSV data, ensure these columns:
- `Wavenumber`: Wavenumber values in cm⁻¹
- `Absorbance`: Absorbance values
- `ExposureTime`: Exposure time in seconds
- `Filename`: Original filename (optional)

Example CSV structure:
```csv
Wavenumber,Absorbance,ExposureTime,Filename
4000.0,0.001,0,0s_sample.0
3999.5,0.002,0,0s_sample.0
...
4000.0,0.003,16,16s_sample.1
```

## Analysis Workflow

### Step 1: Data Loading
1. **OPUS Files**: Place files in a directory and load using GUI or command line
2. **CSV Files**: Prepare integrated CSV file with required columns

### Step 2: Preprocessing
- **Baseline Correction**: 
  - ALS (Asymmetric Least Squares) - Recommended
  - Polynomial - Alternative method
- **Normalization**:
  - Max normalization - Recommended for UV curing
  - Area normalization - For quantitative analysis
  - SNV (Standard Normal Variate) - For removing scatter effects

### Step 3: Analysis Selection
- **Full Analysis**: Comprehensive automated analysis
- **C=C Analysis**: Focus on double bond consumption
- **PCA Analysis**: Principal component analysis
- **Custom Regions**: Define specific wavenumber ranges

### Step 4: Results Interpretation
- Review kinetic parameters
- Examine spectral changes
- Analyze statistical results
- Generate reports

## Understanding Results

### Kinetic Analysis

#### Zero-Order Kinetics
- **Equation**: C = k×t
- **Interpretation**: Constant reaction rate
- **Typical for**: Surface-limited reactions

#### First-Order Kinetics  
- **Equation**: C = C_max × (1 - exp(-k×t))
- **Interpretation**: Exponential approach to maximum
- **Typical for**: Diffusion-controlled reactions

#### Autocatalytic Model
- **Equation**: C = C_max × t^n / (t50^n + t^n)
- **Interpretation**: S-shaped curve with acceleration
- **Typical for**: Chain reactions with acceleration

### Key Parameters
- **Rate Constant (k)**: Reaction speed (s⁻¹)
- **R² Value**: Model fit quality (0-1, higher is better)
- **Maximum Conversion**: Theoretical maximum conversion (%)
- **Half-life**: Time to reach 50% conversion

### Chemical Regions

#### C=C Double Bonds (1620-1640 cm⁻¹)
- **Primary indicator** of UV curing progress
- **Decreases** during polymerization
- **Most important** for conversion calculation

#### Aromatic Rings (1500-1600 cm⁻¹)
- **Photoinitiator** involvement
- **Structural changes** in aromatic compounds
- **Secondary indicator** of reaction progress

#### Carbonyl Groups (1700-1750 cm⁻¹)
- **Photoinitiator** fragmentation
- **Side reactions** monitoring
- **Product formation** tracking

### PCA Analysis
- **PC1**: Usually represents main reaction progress
- **PC2**: Secondary variations (temperature, side reactions)
- **Explained Variance**: How much information each PC contains
- **Scores Plot**: Shows reaction trajectory over time

## Advanced Features

### Custom Region Analysis
```python
# Define custom regions
custom_regions = {
    'my_region': (1400, 1500),  # wavenumber range
    'another_region': (2800, 3000)
}

# Analyze custom regions
results = analyzer.analyze_multiple_regions(custom_regions)
```

### Batch Processing
```python
# Process multiple datasets
from automated_pipeline import FTIRAutomatedPipeline

pipeline = FTIRAutomatedPipeline('./data', './results')
results = pipeline.run_complete_pipeline()
```

### Custom Kinetic Models
```python
# Define custom kinetic function
def custom_model(t, a, b, c):
    return a * (1 - np.exp(-b * t**c))

# Fit custom model
from scipy.optimize import curve_fit
popt, pcov = curve_fit(custom_model, times, conversion)
```

### Advanced Visualization
```python
from visualization import FTIRVisualizer

visualizer = FTIRVisualizer()

# Create interactive plots
fig = visualizer.create_interactive_plot(data)

# Generate comprehensive summary
summary_fig = visualizer.generate_summary_plot(results)
```

## Troubleshooting

### Common Issues

#### "No OPUS files found"
- **Cause**: Files don't have correct extensions (.0, .1, .2, .3)
- **Solution**: Rename files or check file extensions

#### "Failed to extract wavenumbers"
- **Cause**: OPUS file format not recognized
- **Solution**: Check file integrity, try different files

#### "Kinetic fitting failed"
- **Cause**: Insufficient data points or poor data quality
- **Solution**: Check data quality, increase time points

#### "Memory error during analysis"
- **Cause**: Large datasets exceeding available RAM
- **Solution**: Reduce wavenumber range, increase system RAM

### Data Quality Issues

#### Noisy Spectra
- **Symptoms**: High R² variation, poor fits
- **Solutions**: 
  - Increase smoothing parameters
  - Check instrument stability
  - Improve sample preparation

#### Baseline Drift
- **Symptoms**: Curved baselines, poor normalization
- **Solutions**:
  - Use ALS baseline correction
  - Adjust baseline parameters
  - Check instrument calibration

#### Missing Time Points
- **Symptoms**: Gaps in kinetic curves
- **Solutions**:
  - Interpolate missing points
  - Exclude problematic time points
  - Repeat measurements

### Performance Optimization

#### Large Datasets
- Reduce wavenumber range to region of interest
- Use data decimation for preliminary analysis
- Process in batches

#### Slow Analysis
- Use multiprocessing for batch analysis
- Optimize baseline correction parameters
- Reduce PCA components

### Getting Help

#### Error Messages
- Check log files for detailed error information
- Enable verbose mode for debugging
- Review input data format

#### Support Resources
- GitHub Issues: Report bugs and request features
- Documentation: Comprehensive guides and examples
- Community: User discussions and tips

#### Reporting Issues
When reporting issues, include:
1. Error message and stack trace
2. Input data format and size
3. Analysis parameters used
4. System information (OS, Python version)
5. Minimal example to reproduce the issue

## Best Practices

### Data Collection
- Use consistent measurement conditions
- Include sufficient time points (minimum 5-7)
- Ensure good signal-to-noise ratio
- Document experimental conditions

### Analysis Parameters
- Start with default parameters
- Validate results with known samples
- Compare different preprocessing methods
- Document parameter choices

### Result Validation
- Check R² values for model quality
- Compare with literature values
- Validate with independent measurements
- Consider physical reasonableness

### Reporting
- Include all analysis parameters
- Show both experimental and fitted data
- Report confidence intervals
- Discuss limitations and assumptions
