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
- **正速率常数**: 表示产物形成或新键生成
- **R² > 0.95**: 高置信度拟合
- **R² 0.90-0.95**: 中等置信度拟合
- **R² < 0.90**: 低置信度拟合

### PCA分析
- **PC1**: 通常代表主要的化学变化
- **PC2**: 代表次要的变化模式
- **累积方差**: 前几个主成分解释的总方差

## 技术特点

### 算法优势
1. **自适应基线校正**: ALS算法自动处理基线漂移
2. **多模型比较**: 自动选择最佳动力学模型
3. **数据对齐**: 自动处理不同长度的光谱数据
4. **异常值处理**: 3σ准则筛选异常数据点

### 性能优化
1. **内存效率**: 分块处理大数据集
2. **计算优化**: 向量化操作提高速度
3. **错误处理**: 完善的异常捕获和处理

## 依赖库

```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
tkinter (Python标准库)
```

## 安装依赖

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## 注意事项

1. **数据质量**: 确保光谱数据质量良好，信噪比足够
2. **时间点选择**: 建议选择有代表性的时间点进行分析
3. **波数范围**: 根据研究目标选择合适的波数范围
4. **模型选择**: 根据R²值和物理意义选择最佳模型

## 故障排除

### 常见问题
1. **数据加载失败**: 检查CSV文件格式和列名
2. **分析失败**: 确保选择了足够的时间点和数据
3. **拟合效果差**: 尝试不同的预处理参数
4. **内存不足**: 减少分析的数据范围

### 联系支持
如有问题，请检查：
1. 数据文件格式是否正确
2. 依赖库是否完整安装
3. Python版本是否兼容 (推荐3.7+)

## 更新日志

### v1.0.0 (当前版本)
- 完整的FTIR分析功能
- GUI界面支持
- 多种动力学模型
- PCA分析
- 可视化图表生成
- 自动报告生成
