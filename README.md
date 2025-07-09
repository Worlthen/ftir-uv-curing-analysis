# Automated FTIR Analysis for UV Curing Processes

A comprehensive Python application for automated analysis of Fourier Transform Infrared (FTIR) spectroscopy data from UV curing processes. This tool reads Bruker OPUS files, converts them to CSV format, and performs detailed kinetic and chemical analysis.

## 🚀 Features

### Core Functionality
- **Bruker OPUS File Reader**: Direct reading and conversion of Bruker OPUS files (.0, .1, .2, .3 extensions)
- **Automated CSV Conversion**: Batch processing of OPUS files to CSV format
- **UV Curing Analysis**: Specialized analysis for photopolymerization and UV curing processes
- **Kinetic Modeling**: Multiple reaction kinetic models (zero-order, first-order, second-order)
- **Statistical Analysis**: Principal Component Analysis (PCA) and multivariate statistics

### Advanced Features
- **Time-Resolved Analysis**: Track chemical changes over exposure time
- **Difference Spectroscopy**: Automated calculation and visualization of spectral differences
- **Chemical Interpretation**: Automated peak assignment and chemical mechanism inference
- **Interactive GUI**: User-friendly interface for data selection and analysis
- **Comprehensive Reporting**: Automated generation of analysis reports

## 📋 Requirements

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
brukeropusreader>=1.3.0
tkinter (usually included with Python)
```

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS compatible
- Minimum 4GB RAM recommended
- 1GB free disk space

## 🛠️ Installation

### Method 1: Clone Repository
```bash
git clone https://github.com/your-username/ftir-uv-curing-analysis.git
cd ftir-uv-curing-analysis
pip install -r requirements.txt
```

### Method 2: Direct Download
1. Download the repository as ZIP
2. Extract to your desired location
3. Install dependencies: `pip install -r requirements.txt`

## 🎯 Quick Start

### 1. Basic Usage - Command Line
```python
from ftir_analyzer import FTIRUVCuringAnalyzer

# Initialize analyzer
analyzer = FTIRUVCuringAnalyzer()

# Process OPUS files in current directory
analyzer.process_opus_directory()

# Run automated analysis
results = analyzer.run_automated_analysis()

# Generate report
analyzer.generate_report(results)
```

### 2. GUI Application
```bash
python gui_application.py
```

### 3. Batch Processing
```bash
python batch_processor.py --input_dir ./opus_files --output_dir ./results
```

## 📁 Project Structure

```
ftir-uv-curing-analysis/
├── src/
│   ├── opus_reader.py          # Bruker OPUS file reader
│   ├── ftir_analyzer.py        # Main analysis engine
│   ├── kinetic_models.py       # Reaction kinetic modeling
│   ├── chemical_interpreter.py # Chemical mechanism analysis
│   └── report_generator.py     # Automated reporting
├── gui/
│   ├── main_window.py          # Main GUI application
│   ├── analysis_panel.py       # Analysis control panel
│   └── visualization_panel.py  # Data visualization
├── examples/
│   ├── sample_data/            # Example OPUS files
│   ├── basic_analysis.py       # Basic usage examples
│   └── advanced_analysis.py    # Advanced analysis examples
├── tests/
│   ├── test_opus_reader.py     # Unit tests for OPUS reader
│   ├── test_analyzer.py        # Unit tests for analyzer
│   └── test_data/              # Test data files
├── docs/
│   ├── user_guide.md           # Detailed user guide
│   ├── api_reference.md        # API documentation
│   └── theory_background.md    # Theoretical background
├── requirements.txt            # Python dependencies
├── setup.py                   # Package installation
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🔬 Analysis Capabilities

### UV Curing Specific Analysis
- **Photopolymerization Kinetics**: Track C=C double bond consumption
- **Cross-linking Analysis**: Monitor network formation
- **Conversion Calculation**: Quantify reaction completion
- **Inhibition Period Detection**: Identify oxygen inhibition effects

### Chemical Characterization
- **Functional Group Analysis**: Automated peak assignment
- **Reaction Mechanism Inference**: Chemical pathway identification
- **Product Formation**: Track formation of new chemical bonds
- **Side Reaction Detection**: Identify unwanted reactions

### Statistical Analysis
- **Principal Component Analysis**: Dimensionality reduction
- **Kinetic Parameter Estimation**: Rate constants and reaction orders
- **Confidence Intervals**: Statistical significance testing
- **Correlation Analysis**: Inter-variable relationships
- **Positive Rate Constants**: Indicate product formation or new bond generation
- **R² > 0.95**: High confidence fitting
- **R² 0.90-0.95**: Medium confidence fitting
- **R² < 0.90**: Low confidence fitting

### PCA Analysis
- **PC1**: Usually represents main chemical changes
- **PC2**: Represents secondary variation patterns
- **Cumulative Variance**: Total variance explained by first few principal components

## Technical Features

### Algorithm Advantages
1. **Adaptive Baseline Correction**: ALS algorithm automatically handles baseline drift
2. **Multi-Model Comparison**: Automatically selects best kinetic model
3. **Data Alignment**: Automatically handles spectra of different lengths
4. **Outlier Detection**: 3σ criterion for filtering anomalous data points

### Performance Optimization
1. **Memory Efficiency**: Chunked processing for large datasets
2. **Computational Optimization**: Vectorized operations for improved speed
3. **Error Handling**: Comprehensive exception catching and handling

## Dependencies

```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
tkinter (Python standard library)
```

## Install Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## Important Notes

1. **Data Quality**: Ensure spectral data quality is good with sufficient signal-to-noise ratio
2. **Time Point Selection**: Recommend selecting representative time points for analysis
3. **Wavenumber Range**: Choose appropriate wavenumber range based on research objectives
4. **Model Selection**: Select best model based on R² values and physical significance

## Troubleshooting

### Common Issues
1. **Data Loading Failed**: Check CSV file format and column names
2. **Analysis Failed**: Ensure sufficient time points and data are selected
3. **Poor Fitting**: Try different preprocessing parameters
4. **Memory Insufficient**: Reduce analysis data range

### Support Contact
If you encounter issues, please check:
1. Whether data file format is correct
2. Whether dependencies are completely installed
3. Whether Python version is compatible (recommended 3.7+)

## Changelog

### v1.0.0 (Current Version)
- Complete FTIR analysis functionality
- GUI interface support
- Multiple kinetic models
- PCA analysis
- Visualization chart generation
- Automatic report generation
