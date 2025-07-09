# Enhanced FTIR UV Curing Analysis System - Technical Guide

## Based on Scientific Critique and Architecture Restructure

This guide provides comprehensive instructions for using the enhanced FTIR analysis system that addresses all critical points raised in the scientific critique.

---

## 1. SYSTEM OVERVIEW

### 1.1 Key Improvements

The enhanced system provides:

- **Scientific Rigor**: Proper chemical mechanistic understanding
- **Data Quality Validation**: Comprehensive spectral quality assessment
- **Statistical Validation**: Rigorous model selection and validation
- **Chemical Group Analysis**: Scientifically accurate group assignments
- **Kinetic Modeling**: Multiple models with proper statistical comparison
- **Uncertainty Quantification**: Confidence intervals and error propagation

### 1.2 Prerequisites

#### **Experimental Requirements**
- UV-curable photopolymer samples (acrylates, methacrylates, epoxies)
- FTIR spectrometer with MCT detector (resolution ≥ 4 cm⁻¹)
- UV light source (365 nm typical)
- Controlled atmosphere capability (N₂ purging)
- Temperature control (±0.1°C)

#### **Data Requirements**
- Spectral range: 4000-400 cm⁻¹
- Temporal resolution: ≤ 1 second
- Signal-to-noise ratio: > 1000:1
- Minimum 5-7 time points for kinetic analysis

---

## 2. INSTALLATION AND SETUP

### 2.1 Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### 2.2 Import and Initialize

```python
from src.enhanced_ftir_analyzer import EnhancedFTIRAnalyzer

# Initialize analyzer
analyzer = EnhancedFTIRAnalyzer()

# Configure analysis parameters
analyzer.analysis_params.update({
    'baseline_correction': 'als',      # 'als' or 'polynomial'
    'normalization': 'max',            # 'max', 'area', or 'none'
    'smoothing': 'savgol',             # 'savgol' or 'none'
    'kinetic_model': 'autocatalytic'   # Preferred model
})
```

---

## 3. DATA PREPARATION

### 3.1 Required Data Format

The system expects a pandas DataFrame with columns:
- `Wavenumber`: Wavenumber values (cm⁻¹)
- `Absorbance`: Absorbance values
- `ExposureTime`: UV exposure time (seconds)
- `Filename`: Original filename (optional)

### 3.2 Data Loading Example

```python
import pandas as pd

# Load from CSV files
spectra_data = pd.read_csv('integrated_spectra.csv')

# Or create from OPUS files using the OPUS reader
from src.opus_reader import OPUSReader

reader = OPUSReader()
spectra_data = reader.batch_convert_and_integrate('./opus_files')
```

### 3.3 Experimental Conditions

Define experimental conditions for proper analysis context:

```python
experimental_conditions = {
    'uv_wavelength': 365,              # nm
    'uv_intensity': 50,                # mW/cm²
    'temperature': 25,                 # °C
    'atmosphere': 'nitrogen',          # 'air' or 'nitrogen'
    'sample_thickness': 25,            # μm
    'photoinitiator': 'Irgacure 819',
    'photoinitiator_concentration': 2.0, # wt%
    'monomer_system': 'TMPTA'
}
```

---

## 4. ANALYSIS WORKFLOW

### 4.1 Complete Analysis

```python
# Perform comprehensive analysis
results = analyzer.analyze_uv_curing_kinetics(
    spectra_data, 
    experimental_conditions
)
```

### 4.2 Results Structure

The results dictionary contains:

```python
{
    'metadata': {
        'analysis_timestamp': '2025-01-09T...',
        'analysis_parameters': {...},
        'experimental_conditions': {...}
    },
    'data_quality': {
        'snr': 1250.5,
        'snr_pass': True,
        'baseline_stability': 0.0008,
        'baseline_pass': True,
        'overall_pass': True
    },
    'chemical_groups': {
        'c_equals_c_acrylate': {
            'group_info': {...},
            'conversion_data': DataFrame,
            'raw_data': DataFrame
        }
    },
    'kinetic_analysis': {
        'c_equals_c_acrylate': {
            'all_models': {...},
            'best_model': {...},
            'validation': {...}
        }
    },
    'summary': {...}
}
```

---

## 5. CHEMICAL GROUP ANALYSIS

### 5.1 Predefined Chemical Groups

The system includes scientifically validated chemical group definitions:

#### **Reactive Groups**
- **C=C Acrylate** (1620-1640 cm⁻¹): Primary reactive site
- **C=C Methacrylate** (1635-1645 cm⁻¹): Primary reactive site
- **Vinyl C-H** (3080-3120 cm⁻¹): Polymerization indicator

#### **Structural Groups**
- **Ester C=O** (1720-1740 cm⁻¹): Structural backbone
- **Ester C-O** (1000-1300 cm⁻¹): Crosslink formation

#### **Photoinitiator Groups**
- **Benzoin C=O** (1650-1680 cm⁻¹): Photoinitiator consumption
- **Acetophenone C=O** (1660-1690 cm⁻¹): Photoinitiator consumption

### 5.2 Accessing Group Analysis Results

```python
# Access chemical group analysis
group_results = results['chemical_groups']

for group_name, group_data in group_results.items():
    group_info = group_data['group_info']
    conversion_data = group_data['conversion_data']
    
    print(f"Group: {group_name}")
    print(f"Assignment: {group_info['assignment']}")
    print(f"Wavenumber Range: {group_info['range']}")
    print(f"Final Conversion: {conversion_data['Conversion'].iloc[-1]:.2%}")
```

---

## 6. KINETIC ANALYSIS

### 6.1 Available Kinetic Models

#### **Autocatalytic Model** (Recommended for UV curing)
```
dα/dt = (k₁ + k₂α)(α_max - α)
```
- **k₁**: Initial rate constant
- **k₂**: Autocatalytic rate constant
- **α_max**: Maximum conversion

#### **First-Order Model**
```
α(t) = α_max(1 - exp(-kt))
```
- **k**: Rate constant
- **α_max**: Maximum conversion

#### **Zero-Order Model**
```
α(t) = min(kt, α_max)
```
- **k**: Rate constant
- **α_max**: Maximum conversion

### 6.2 Model Selection and Validation

The system automatically:
1. Fits all available models
2. Compares using AIC/BIC criteria
3. Validates parameters physically
4. Tests residual randomness
5. Calculates confidence intervals

```python
# Access kinetic analysis results
kinetic_results = results['kinetic_analysis']

for group_name, kinetic_data in kinetic_results.items():
    if 'best_model' in kinetic_data:
        best_model = kinetic_data['best_model']
        validation = kinetic_data['validation']
        
        print(f"Group: {group_name}")
        print(f"Best Model: {best_model['model_name']}")
        print(f"R² Value: {best_model['r_squared']:.4f}")
        print(f"Parameters: {best_model['parameters']}")
        print(f"Validation: {'PASS' if validation['overall_valid'] else 'FAIL'}")
```

---

## 7. DATA QUALITY VALIDATION

### 7.1 Quality Metrics

The system validates:
- **Signal-to-Noise Ratio**: > 1000:1 required
- **Baseline Stability**: < 0.001 AU drift
- **Peak Resolution**: > 1.5 separation factor
- **Frequency Accuracy**: ± 0.1 cm⁻¹

### 7.2 Quality Assessment

```python
# Check data quality
quality = results['data_quality']

if quality['overall_pass']:
    print("Data quality: ACCEPTABLE")
else:
    print("Data quality: ISSUES DETECTED")
    if not quality['snr_pass']:
        print(f"  - Low S/N ratio: {quality['snr']:.1f}")
    if not quality['baseline_pass']:
        print(f"  - Baseline instability: {quality['baseline_stability']:.4f}")
```

---

## 8. VISUALIZATION AND REPORTING

### 8.1 Built-in Visualizations

```python
from examples.enhanced_analysis_demo import create_analysis_visualizations

# Create comprehensive plots
create_analysis_visualizations(results, spectra_data)
```

### 8.2 Custom Analysis

```python
import matplotlib.pyplot as plt

# Extract kinetic data for plotting
group_name = 'c_equals_c_acrylate'
kinetic_data = results['kinetic_analysis'][group_name]

if 'best_model' in kinetic_data:
    exp_data = kinetic_data['experimental_data']
    best_model = kinetic_data['best_model']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(exp_data['times'], exp_data['conversions'], 
               label='Experimental', s=50)
    plt.plot(exp_data['times'], best_model['fitted_data'], 
            label=f'{best_model["model_name"]} fit', linewidth=2)
    plt.xlabel('Exposure Time (s)')
    plt.ylabel('Conversion')
    plt.title(f'Kinetic Analysis: {group_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 9. ADVANCED FEATURES

### 9.1 Custom Chemical Groups

```python
# Define custom chemical group
custom_group = {
    'range': (1400, 1500),
    'assignment': 'Custom C-H bending',
    'reaction_role': 'structural_indicator',
    'extinction_coeff': 200,
    'baseline_method': 'als'
}

# Add to analyzer
analyzer.chemical_groups.CUSTOM_GROUPS = {
    'custom_ch_bending': custom_group
}
```

### 9.2 Parameter Optimization

```python
# Optimize analysis parameters
analyzer.analysis_params.update({
    'baseline_correction': 'polynomial',  # Try different method
    'normalization': 'area',              # Alternative normalization
})

# Re-run analysis with new parameters
optimized_results = analyzer.analyze_uv_curing_kinetics(
    spectra_data, experimental_conditions
)
```

---

## 10. TROUBLESHOOTING

### 10.1 Common Issues

#### **Low Data Quality**
- Check instrument calibration
- Verify sample preparation
- Ensure proper environmental control

#### **Poor Kinetic Fits**
- Increase number of time points
- Check for experimental artifacts
- Verify chemical group assignments

#### **Parameter Validation Failures**
- Review experimental conditions
- Check for outliers in data
- Validate chemical mechanism assumptions

### 10.2 Error Handling

The system provides detailed error messages and logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with detailed logging
results = analyzer.analyze_uv_curing_kinetics(spectra_data, experimental_conditions)
```

---

## 11. BEST PRACTICES

### 11.1 Experimental Design
- Use minimum 5-7 time points for kinetic analysis
- Include early time points (< 5 seconds) for initiation kinetics
- Ensure consistent environmental conditions
- Use appropriate photoinitiator concentrations (0.5-5 wt%)

### 11.2 Data Analysis
- Always validate data quality before analysis
- Compare multiple kinetic models
- Check parameter physical reasonableness
- Report confidence intervals
- Document experimental conditions

### 11.3 Result Interpretation
- Consider chemical mechanism when interpreting results
- Validate results with independent measurements
- Report limitations and assumptions
- Use appropriate statistical significance levels

---

This enhanced system provides scientifically rigorous FTIR analysis capabilities that address all critical points raised in the original critique while maintaining practical usability for research and industrial applications.
