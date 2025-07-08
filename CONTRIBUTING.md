# Contributing to FTIR UV Curing Analysis

We welcome contributions to the FTIR UV Curing Analysis project! This document provides guidelines for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Development Standards](#development-standards)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

### Our Standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of FTIR spectroscopy
- Familiarity with Python scientific libraries (NumPy, Pandas, SciPy)

### Areas for Contribution
- **Bug fixes**: Fix reported issues
- **New features**: Add analysis capabilities
- **Documentation**: Improve guides and examples
- **Testing**: Add unit tests and integration tests
- **Performance**: Optimize algorithms and memory usage
- **UI/UX**: Improve the graphical interface

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/your-username/ftir-uv-curing-analysis.git
cd ftir-uv-curing-analysis
```

### 2. Create Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Verify Setup
```bash
# Run tests
pytest

# Run example analysis
python examples/basic_analysis.py

# Launch GUI
python gui_application.py
```

## Contributing Guidelines

### Branch Naming
- `feature/description` - New features
- `bugfix/issue-number` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages
Use clear, descriptive commit messages:
```
Add support for custom kinetic models

- Implement user-defined kinetic functions
- Add parameter validation
- Update documentation with examples
- Add unit tests for new functionality
```

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate
- Keep functions focused and modular

### Documentation
- Update relevant documentation for new features
- Include examples in docstrings
- Add user guide sections for significant features
- Update API reference for new functions

## Pull Request Process

### 1. Prepare Your Changes
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Add and commit changes
git add .
git commit -m "Descriptive commit message"

# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create Pull Request
1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the pull request template

### 3. Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### 4. Review Process
- Maintainers will review your pull request
- Address any requested changes
- Once approved, your changes will be merged

## Issue Reporting

### Bug Reports
Use the bug report template:
```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Load data file '...'
2. Run analysis with parameters '...'
3. See error

**Expected Behavior**
What you expected to happen

**Screenshots/Output**
If applicable, add screenshots or error output

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.0]

**Additional Context**
Any other context about the problem
```

### Feature Requests
Use the feature request template:
```markdown
**Feature Description**
Clear description of the desired feature

**Use Case**
Describe the problem this feature would solve

**Proposed Solution**
Describe your preferred solution

**Alternatives**
Describe alternative solutions considered

**Additional Context**
Any other context or screenshots
```

## Development Standards

### Code Quality
- Write clean, readable code
- Follow SOLID principles
- Use appropriate design patterns
- Minimize code duplication
- Handle errors gracefully

### Testing
- Write unit tests for new functions
- Maintain test coverage above 80%
- Include integration tests for workflows
- Test edge cases and error conditions
- Use meaningful test names

### Performance
- Profile code for performance bottlenecks
- Optimize memory usage for large datasets
- Use vectorized operations where possible
- Consider algorithmic complexity
- Document performance characteristics

### Documentation
- Write clear, comprehensive docstrings
- Include parameter descriptions and types
- Provide usage examples
- Document return values and exceptions
- Keep documentation up to date

### Example Code Structure
```python
def analyze_kinetic_data(
    times: np.ndarray, 
    conversion: np.ndarray,
    model_type: str = 'first_order'
) -> Dict[str, float]:
    """
    Analyze kinetic data using specified model.
    
    Parameters
    ----------
    times : np.ndarray
        Array of time points in seconds
    conversion : np.ndarray
        Array of conversion percentages
    model_type : str, optional
        Type of kinetic model ('zero_order', 'first_order', 'autocatalytic')
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing fitted parameters:
        - 'rate_constant': Rate constant in s⁻¹
        - 'r_squared': Coefficient of determination
        - 'max_conversion': Maximum conversion percentage
        
    Raises
    ------
    ValueError
        If input arrays have different lengths or invalid model_type
    RuntimeError
        If fitting fails to converge
        
    Examples
    --------
    >>> times = np.array([0, 10, 20, 30])
    >>> conversion = np.array([0, 25, 45, 60])
    >>> results = analyze_kinetic_data(times, conversion)
    >>> print(f"Rate constant: {results['rate_constant']:.2e} s⁻¹")
    """
    # Implementation here
    pass
```

## Release Process

### Version Numbering
We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Deploy to PyPI

## Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support
- **Email**: Direct contact for sensitive issues

### Resources
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Working code examples for common tasks
- **Tests**: Unit tests demonstrate expected behavior
- **Code**: Well-commented source code

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments
- GitHub contributor statistics

Thank you for contributing to the FTIR UV Curing Analysis project!
