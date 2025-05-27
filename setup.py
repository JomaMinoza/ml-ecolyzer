#!/usr/bin/env python3
"""
ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework
Setup configuration for package installation supporting multiple ML frameworks
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Machine Learning Environmental Impact Analysis Framework supporting HuggingFace, scikit-learn, and PyTorch"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

# Core dependencies (minimal for basic functionality)
CORE_REQUIREMENTS = [
    "numpy>=1.21.0",
    "psutil>=5.8.0", 
    "codecarbon>=2.0.0",
    "pandas>=1.3.0",
    "tqdm>=4.60.0",
    "pyyaml>=5.4.0",
    "click>=8.0.0",
    "typing-extensions>=4.0.0",
    "scikit-learn>=1.0.0",
    "joblib>=1.0.0",
]

# Framework-specific dependencies
HUGGINGFACE_REQUIREMENTS = [
    "torch>=1.9.0",
    "transformers>=4.20.0",
    "datasets>=2.0.0", 
    "diffusers>=0.10.0",
    "evaluate>=0.3.0",
    "sacrebleu>=2.0.0",
    "accelerate>=0.20.0",
]

SKLEARN_REQUIREMENTS = [
    "scikit-learn>=1.0.0",
    "imbalanced-learn>=0.8.0",
    "yellowbrick>=1.4.0",
    "openpyxl>=3.0.0",
    "xlrd>=2.0.0",
]

PYTORCH_REQUIREMENTS = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "torchaudio>=0.9.0",
    "pytorch-lightning>=1.5.0",
    "torchmetrics>=0.6.0",
]

GPU_REQUIREMENTS = [
    "nvidia-ml-py3>=7.352.0",
    "pynvml>=11.0.0",
]

AUDIO_REQUIREMENTS = [
    "librosa>=0.8.0",
    "soundfile>=0.10.0",
    "jiwer>=2.0.0",
    "torchaudio>=0.9.0",
]

VISION_REQUIREMENTS = [
    "opencv-python>=4.5.0",
    "pillow>=8.0.0",
    "torchvision>=0.10.0",
]

TRACKING_REQUIREMENTS = [
    "wandb>=0.13.0",
    "rich>=12.0.0",
]

METRICS_REQUIREMENTS = [
    "sentence-transformers>=2.0.0",
    "scipy>=1.7.0",
]

VISUALIZATION_REQUIREMENTS = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "ipywidgets>=7.6.0",
]

DEV_REQUIREMENTS = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "pytest-asyncio>=0.21.0",
    "black>=21.0",
    "flake8>=3.8",
    "isort>=5.0",
    "mypy>=0.900",
    "pre-commit>=2.15.0",
    "memory-profiler>=0.60.0",
]

DOCS_REQUIREMENTS = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "myst-parser>=0.15",
    "sphinx-autodoc-typehints>=1.12",
    "nbsphinx>=0.8.0",
    "jupyter>=1.0.0",
]

CONFIG_REQUIREMENTS = [
    "omegaconf>=2.1.0",
    "hydra-core>=1.1.0",
]

RESEARCH_REQUIREMENTS = [
    "jupyter>=1.0.0", 
    "notebook>=6.4.0",
    "ipykernel>=6.0.0",
    "requests>=2.25.0",
]

# Comprehensive "all" dependencies
ALL_REQUIREMENTS = list(set(
    HUGGINGFACE_REQUIREMENTS +
    SKLEARN_REQUIREMENTS +
    PYTORCH_REQUIREMENTS +
    GPU_REQUIREMENTS +
    AUDIO_REQUIREMENTS +
    VISION_REQUIREMENTS +
    TRACKING_REQUIREMENTS +
    METRICS_REQUIREMENTS +
    VISUALIZATION_REQUIREMENTS +
    CONFIG_REQUIREMENTS +
    RESEARCH_REQUIREMENTS +
    ["memory-profiler>=0.60.0"]
))

setup(
    name="ml-ecolyzer",
    version="1.0.0",
    author="Jose Marie Antonio Minoza, Rex Gregor Laylo, Christian Villarin, Sebastian Ibanez",
    author_email="contact@cair.ph",
    description="Machine Learning Environmental Impact Analysis Framework supporting HuggingFace, scikit-learn, and PyTorch",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ml-ecolyzer/ml-ecolyzer",
    project_urls={
        "Bug Reports": "https://github.com/ml-ecolyzer/ml-ecolyzer/issues",
        "Source": "https://github.com/ml-ecolyzer/ml-ecolyzer",
        "Documentation": "https://ml-ecolyzer.readthedocs.io",
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",
        "PyPI": "https://pypi.org/project/ml-ecolyzer",
        "Discussions": "https://github.com/ml-ecolyzer/ml-ecolyzer/discussions",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers", 
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
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
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        # Framework-specific installations
        "huggingface": HUGGINGFACE_REQUIREMENTS,
        "sklearn": SKLEARN_REQUIREMENTS,
        "pytorch": PYTORCH_REQUIREMENTS,
        
        # Hardware and monitoring
        "gpu": GPU_REQUIREMENTS,
        "tracking": TRACKING_REQUIREMENTS,
        
        # Domain-specific
        "audio": AUDIO_REQUIREMENTS,
        "vision": VISION_REQUIREMENTS,
        "metrics": METRICS_REQUIREMENTS,
        "visualization": VISUALIZATION_REQUIREMENTS,
        
        # Development and documentation
        "dev": DEV_REQUIREMENTS,
        "docs": DOCS_REQUIREMENTS,
        "config": CONFIG_REQUIREMENTS,
        "research": RESEARCH_REQUIREMENTS,
        
        # Convenience bundles
        "all": ALL_REQUIREMENTS,
        "minimal": [],  # Just core requirements
        
        # Task-focused bundles
        "text": HUGGINGFACE_REQUIREMENTS,
        "cv": PYTORCH_REQUIREMENTS + VISION_REQUIREMENTS,
        "classical": SKLEARN_REQUIREMENTS,
        
        # Legacy compatibility
        "audio_full": AUDIO_REQUIREMENTS,
        "vision_full": VISION_REQUIREMENTS,
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
    keywords=[
        "machine-learning", "ml", "environmental-impact", "carbon-emissions", 
        "sustainability", "pytorch", "huggingface", "scikit-learn", "sklearn",
        "benchmarking", "green-ai", "carbon-footprint", "neural-networks",
        "transformers", "deep-learning", "climate-change", "energy-efficiency"
    ],
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
    
    # Package data
    package_data={
        "mlecolyzer": [
            "*.yaml", 
            "*.yml", 
            "*.json",
            "configs/**/*",
            "examples/**/*", 
            "templates/**/*"
        ],
    },
    
    # Dependency links for development versions
    dependency_links=[],    
)

# Post-installation checks and information
def post_install_check():
    """
    Check framework availability after installation
    """
    print("\n🔧 ML-EcoLyzer Installation Complete!")
    print("=" * 50)
    
    # Check which frameworks are available
    framework_status = {}
    
    try:
        import sklearn
        framework_status["scikit-learn"] = "✅ Available"
    except ImportError:
        framework_status["scikit-learn"] = "❌ Not Available"
    
    try:
        import torch
        framework_status["PyTorch"] = "✅ Available"
    except ImportError:
        framework_status["PyTorch"] = "❌ Not Available"
    
    try:
        import transformers
        framework_status["HuggingFace"] = "✅ Available"
    except ImportError:
        framework_status["HuggingFace"] = "❌ Not Available"
    
    print("Framework Support:")
    for framework, status in framework_status.items():
        print(f"  {framework}: {status}")
    
    print("\n💡 Quick Start:")
    print("  mlecolyzer info                    # Check system capabilities")
    print("  mlecolyzer init --template sklearn # Generate sklearn config")
    print("  mlecolyzer analyze --help          # See analysis options")
    
    print("\n📚 Documentation:")
    print("  https://ml-ecolyzer.readthedocs.io")
    
    print("\n🚀 Install additional frameworks:")
    print("  pip install ml-ecolyzer[huggingface]  # HuggingFace support")
    print("  pip install ml-ecolyzer[pytorch]      # PyTorch support") 
    print("  pip install ml-ecolyzer[all]          # All frameworks")
    print("=" * 50)

if __name__ == "__main__":
    # Run post-installation check if installing
    import sys
    if "install" in sys.argv:
        try:
            post_install_check()
        except Exception:
            pass  # Don't fail installation if post-check fails