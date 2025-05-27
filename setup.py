#!/usr/bin/env python3
"""
ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Machine Learning Environmental Impact Analysis Framework"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="ml-ecolyzer",
    version="1.0.0",
    author="Jose Marie Antonio Minoza, Rex Gregor Laylo, Christian Villarin, Sebastian Ibanez",
    author_email="contact@cair.ph",
    description="Machine Learning Environmental Impact Analysis Framework",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/JomaMinoza/ml-ecolyzer",
    project_urls={
        "Bug Reports": "https://github.com/JomaMinoza/ml-ecolyzer/issues",
        "Source": "https://github.com/JomaMinoza/ml-ecolyzer",
        "Documentation": "",
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "Topic :: Software Development :: Testing",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
            "pynvml>=11.0.0",
        ],
        "audio": [
            "librosa>=0.8.0",
            "soundfile>=0.10.0",
            "jiwer>=2.0.0",
        ],
        "vision": [
            "opencv-python>=4.5.0",
            "pillow>=8.0.0",
        ],
        "all": [
            "nvidia-ml-py3>=7.352.0",
            "pynvml>=11.0.0",
            "librosa>=0.8.0",
            "soundfile>=0.10.0",
            "jiwer>=2.0.0",
            "opencv-python>=4.5.0",
            "pillow>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlecolyzer=mlecolyzer.cli.main:main",
            "ml-ecolyzer=mlecolyzer.cli.main:main",
            "mlecolyzer-research=mlecolyzer.cli.main:research_main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning ml environmental-impact carbon-emissions sustainability pytorch huggingface neurips iclr",
)
