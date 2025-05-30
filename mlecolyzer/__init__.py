"""
ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework

A scientific framework for analyzing and quantifying the environmental impact of machine 
learning systems across diverse hardware configurations and deployment scenarios.

ML-EcoLyzer provides comprehensive measurement and analysis of:
- Carbon emissions and energy consumption
- Hardware utilization and thermal impact  
- Model efficiency optimization opportunities
- Quantization and deployment recommendations

The framework supports multiple ML frameworks and adapts to various hardware setups 
from datacenter GPUs to edge devices, providing scientifically rigorous measurements 
for sustainable AI research and deployment.

Supported Frameworks:
- HuggingFace: Transformers, diffusers, datasets
- scikit-learn: Classical ML algorithms
- PyTorch: Deep learning models and torchvision

Example:
    HuggingFace text generation:
    
    >>> from mlecolyzer import EcoLyzer
    >>> config = {
    ...     "project": "sustainability_study",
    ...     "models": [{"name": "gpt2", "task": "text", "framework": "huggingface"}],
    ...     "datasets": [{"name": "wikitext", "task": "text", "framework": "huggingface"}]
    ... }
    >>> analyzer = EcoLyzer(config)
    >>> results = analyzer.run()

    scikit-learn classification:
    
    >>> config = {
    ...     "project": "sklearn_carbon_study",
    ...     "models": [{"name": "RandomForestClassifier", "task": "classification", "framework": "sklearn"}],
    ...     "datasets": [{"name": "iris", "task": "classification", "framework": "sklearn"}]
    ... }
    >>> analyzer = EcoLyzer(config)
    >>> results = analyzer.run()

    PyTorch image classification:
    
    >>> config = {
    ...     "project": "pytorch_vision_study",
    ...     "models": [{"name": "resnet18", "task": "image", "framework": "pytorch", "pretrained": True}],
    ...     "datasets": [{"name": "CIFAR10", "task": "image", "framework": "pytorch", "limit": 1000}]
    ... }
    >>> analyzer = EcoLyzer(config)
    >>> results = analyzer.run()

    Multi-framework research:
    
    >>> from mlecolyzer import run_comprehensive_analysis
    >>> research_config = {
    ...     "project": "multi_framework_carbon_study",
    ...     "models": [
    ...         {"name": "gpt2", "task": "text", "framework": "huggingface"},
    ...         {"name": "RandomForestClassifier", "task": "classification", "framework": "sklearn"},
    ...         {"name": "resnet18", "task": "image", "framework": "pytorch", "pretrained": True}
    ...     ],
    ...     "datasets": [
    ...         {"name": "wikitext", "task": "text", "framework": "huggingface", "limit": 500},
    ...         {"name": "iris", "task": "classification", "framework": "sklearn"},
    ...         {"name": "CIFAR10", "task": "image", "framework": "pytorch", "limit": 300}
    ...     ]
    ... }
    >>> results = run_comprehensive_analysis(research_config)

References:
    - Strubell et al. (2019) "Energy and Policy Considerations for Deep Learning in NLP"
    - Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training"
    - Schwartz et al. (2020) "Green AI" (Communications of the ACM)
    - Henderson et al. (2020) "Towards the Systematic Reporting of Energy and Carbon Footprints"
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0"

# Main analysis classes and functions
from .core.runner import EcoLyzer
from .core.research import run_comprehensive_analysis
from .core.config import AnalysisConfig, ModelConfig, DatasetConfig

# Import key monitoring classes
from .monitoring.environmental import AdaptiveEnvironmentalTracker
from .monitoring.hardware import HardwareCapabilities, detect_hardware_capabilities


# Import utilities
from .utils.validation import validate_config
from .utils.helpers import setup_logging, get_default_config

# Public API
__all__ = [
    # Main classes and functions
    "EcoLyzer",
    "run_comprehensive_analysis", 
    
    # Configuration classes
    "AnalysisConfig",
    "ModelConfig", 
    "DatasetConfig",
    
    # Monitoring classes
    "AdaptiveEnvironmentalTracker",
    "HardwareCapabilities",
    "detect_hardware_capabilities",
    
    # Utilities
    "validate_config",
    "setup_logging",
    "get_default_config",
        
    # Version
    "__version__",
]

# Package metadata
__author__ = "Center for AI Research PH"
__email__ = "contact@cair.ph"
__license__ = "MIT"
__description__ = "Machine Learning Environmental Impact Analysis Framework"

# Scientific references
__references__ = [
    "Strubell et al. (2019) 'Energy and Policy Considerations for Deep Learning in NLP'",
    "Patterson et al. (2021) 'Carbon Emissions and Large Neural Network Training'", 
    "Schwartz et al. (2020) 'Green AI' (Communications of the ACM)",
    "Henderson et al. (2020) 'Towards the Systematic Reporting of Energy and Carbon Footprints'"
]

# Configure logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_info():
    """Get package information including version and capabilities."""
    from .monitoring.hardware import detect_hardware_capabilities
    
    capabilities = detect_hardware_capabilities()
    
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "hardware_capabilities": capabilities.__dict__ if capabilities else None,
        "supported_frameworks": ["huggingface", "sklearn", "pytorch"],
        "references": __references__
    }

def create_analysis_config(**kwargs):
    """Create an analysis configuration with sensible defaults."""
    return AnalysisConfig(**kwargs)

def quick_analysis(model_name: str, dataset_name: str, framework: str = "huggingface", 
                   task: str = "text", **kwargs):
    """
    Run a quick environmental impact analysis with minimal configuration.
    
    Args:
        model_name: Model identifier
        dataset_name: Dataset identifier
        framework: Framework type (huggingface, sklearn, pytorch)
        task: Task type
        **kwargs: Additional configuration options
        
    Returns:
        Dict with analysis results
    """
    config = {
        "project": f"quick_analysis_{framework}_{model_name.replace('/', '_')}",
        "models": [{"name": model_name, "task": task, "framework": framework}],
        "datasets": [{"name": dataset_name, "task": task, "framework": framework, "limit": 100}],
        **kwargs
    }
    
    analyzer = EcoLyzer(config)
    return analyzer.run()

def quick_sklearn_analysis(model_name: str = "RandomForestClassifier", 
                          dataset_name: str = "iris", task: str = "classification", **kwargs):
    """
    Run a quick sklearn environmental impact analysis.
    
    Args:
        model_name: sklearn model class name
        dataset_name: sklearn dataset name
        task: classification or regression
        **kwargs: Additional configuration options
        
    Returns:
        Dict with analysis results
    """
    return quick_analysis(model_name, dataset_name, "sklearn", task, **kwargs)

def quick_pytorch_analysis(model_name: str = "resnet18", dataset_name: str = "CIFAR10", 
                          task: str = "image", **kwargs):
    """
    Run a quick PyTorch environmental impact analysis.
    
    Args:
        model_name: PyTorch model name
        dataset_name: PyTorch dataset name
        task: Task type
        **kwargs: Additional configuration options
        
    Returns:
        Dict with analysis results
    """
    kwargs.setdefault("pretrained", True)
    kwargs.setdefault("limit", 200)
    return quick_analysis(model_name, dataset_name, "pytorch", task, **kwargs)

def compare_frameworks(models_config: dict, dataset_config: dict, project: str = "framework_comparison"):
    """
    Compare environmental impact across different frameworks.
    
    Args:
        models_config: Dict with framework -> model_name mapping
        dataset_config: Dict with framework -> dataset_name mapping
        project: Project name
        
    Returns:
        Dict with analysis results
        
    Example:
        >>> models = {
        ...     "sklearn": "RandomForestClassifier",
        ...     "pytorch": "resnet18"
        ... }
        >>> datasets = {
        ...     "sklearn": "iris", 
        ...     "pytorch": "CIFAR10"
        ... }
        >>> results = compare_frameworks(models, datasets)
    """
    config = {
        "project": project,
        "models": [],
        "datasets": []
    }
    
    # Add models for each framework
    for framework, model_name in models_config.items():
        task = "classification" if framework == "sklearn" else "image"
        config["models"].append({
            "name": model_name,
            "task": task,
            "framework": framework
        })
    
    # Add datasets for each framework
    for framework, dataset_name in dataset_config.items():
        task = "classification" if framework == "sklearn" else "image"
        config["datasets"].append({
            "name": dataset_name,
            "task": task,
            "framework": framework,
            "limit": 100 if framework != "sklearn" else None
        })
    
    analyzer = EcoLyzer(config)
    return analyzer.run()

def create_multiframework_research_config(project: str = "multiframework_research"):
    """
    Create a comprehensive multi-framework research configuration.
    
    Args:
        project: Project name
        
    Returns:
        Complete research configuration
    """
    config = {
        "project": project,
        "models": [
            # HuggingFace models
            {"name": "gpt2", "task": "text", "framework": "huggingface"},
            {"name": "distilbert-base-uncased", "task": "text", "framework": "huggingface"},
            
            # scikit-learn models
            {"name": "RandomForestClassifier", "task": "classification", "framework": "sklearn"},
            {"name": "LogisticRegression", "task": "classification", "framework": "sklearn"},
            {"name": "LinearRegression", "task": "regression", "framework": "sklearn"},
            
            # PyTorch models
            {"name": "resnet18", "task": "image", "framework": "pytorch", "pretrained": True},
            {"name": "mobilenet_v2", "task": "image", "framework": "pytorch", "pretrained": True}
        ],
        "datasets": [
            # HuggingFace datasets
            {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "framework": "huggingface", "limit": 500},
            {"name": "imdb", "task": "text", "framework": "huggingface", "limit": 400},
            
            # scikit-learn datasets
            {"name": "iris", "task": "classification", "framework": "sklearn"},
            {"name": "wine", "task": "classification", "framework": "sklearn"},
            {"name": "diabetes", "task": "regression", "framework": "sklearn"},
            
            # PyTorch datasets
            {"name": "CIFAR10", "task": "image", "framework": "pytorch", "limit": 300},
            {"name": "MNIST", "task": "image", "framework": "pytorch", "limit": 300}
        ],
        "enable_quantization_analysis": True,
        "enable_wandb": True,
        "monitoring_duration": 600,
        "wandb_project": f"{project}_comprehensive"
    }
    
    return config

def get_framework_examples():
    """
    Get example configurations for different frameworks.
    
    Returns:
        Dict with example configurations for each framework
    """
    examples = {
        "huggingface": {
            "text_generation": {
                "project": "huggingface_text_generation",
                "models": [{"name": "gpt2", "task": "text", "framework": "huggingface"}],
                "datasets": [{"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "framework": "huggingface", "limit": 100}]
            },
            "image_classification": {
                "project": "huggingface_image_classification", 
                "models": [{"name": "microsoft/resnet-50", "task": "image", "framework": "huggingface"}],
                "datasets": [{"name": "imagenet-1k", "task": "image", "framework": "huggingface", "limit": 100}]
            }
        },
        
        "sklearn": {
            "classification": {
                "project": "sklearn_classification",
                "models": [
                    {"name": "RandomForestClassifier", "task": "classification", "framework": "sklearn"},
                    {"name": "LogisticRegression", "task": "classification", "framework": "sklearn"}
                ],
                "datasets": [
                    {"name": "iris", "task": "classification", "framework": "sklearn"},
                    {"name": "wine", "task": "classification", "framework": "sklearn"}
                ]
            },
            "regression": {
                "project": "sklearn_regression",
                "models": [
                    {"name": "LinearRegression", "task": "regression", "framework": "sklearn"},
                    {"name": "RandomForestRegressor", "task": "regression", "framework": "sklearn"}
                ],
                "datasets": [
                    {"name": "diabetes", "task": "regression", "framework": "sklearn"},
                    {"name": "boston", "task": "regression", "framework": "sklearn"}
                ]
            }
        },
        
        "pytorch": {
            "image_classification": {
                "project": "pytorch_image_classification",
                "models": [
                    {"name": "resnet18", "task": "image", "framework": "pytorch", "pretrained": True},
                    {"name": "mobilenet_v2", "task": "image", "framework": "pytorch", "pretrained": True}
                ],
                "datasets": [
                    {"name": "CIFAR10", "task": "image", "framework": "pytorch", "limit": 200},
                    {"name": "MNIST", "task": "image", "framework": "pytorch", "limit": 200}
                ]
            }
        }
    }
    
    return examples

def validate_framework_support():
    """
    Validate which frameworks are available in the current environment.
    
    Returns:
        Dict with framework availability status
    """
    from .utils.validation import (
        validate_huggingface_dependencies,
        validate_sklearn_dependencies, 
        validate_pytorch_dependencies
    )
    
    return {
        "huggingface": validate_huggingface_dependencies(),
        "sklearn": validate_sklearn_dependencies(),
        "pytorch": validate_pytorch_dependencies()
    }

# Maintain backward compatibility
create_benchmark_config = create_analysis_config
quick_benchmark = quick_analysis