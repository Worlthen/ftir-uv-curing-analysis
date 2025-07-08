#!/usr/bin/env python3
"""
Setup script for FTIR UV Curing Analysis System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ftir-uv-curing-analysis",
    version="1.0.0",
    author="FTIR Analysis Team",
    author_email="contact@ftir-analysis.com",
    description="Automated FTIR analysis for UV curing processes",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ftir-uv-curing-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "ftir-analysis=automated_pipeline:main",
            "ftir-gui=gui_application:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="ftir spectroscopy uv-curing analysis chemistry",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ftir-uv-curing-analysis/issues",
        "Source": "https://github.com/your-username/ftir-uv-curing-analysis",
        "Documentation": "https://github.com/your-username/ftir-uv-curing-analysis/docs",
    },
)
