"""
ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework

A scientific framework for analyzing and quantifying the environmental impact of machine 
learning systems across diverse hardware configurations and deployment scenarios.

ML-EcoLyzer provides comprehensive measurement and analysis of:
- Carbon emissions and energy consumption
- Hardware utilization and thermal impact  
- Model efficiency optimization opportunities
- Quantization and deployment recommendations

The framework adapts to various hardware setups from datacenter GPUs to edge devices,
providing scientifically rigorous measurements for sustainable AI research and deployment.

Example:
    Basic usage:
    
    >>> from mlecolyzer import EcoLyzer
    >>> config = {
    ...     "project": "sustainability_study",
    ...     "models": [{"name": "gpt2", "task": "text"}],
    ...     "datasets": [{"name": "wikitext", "task": "text"}]
    ... }
    >>> analyzer = EcoLyzer(config)
    >>> results = analyzer.run()

    Comprehensive research:
    
    >>> from mlecolyzer import run_comprehensive_analysis
    >>> research_config = {
    ...     "project": "carbon_footprint_study",
    ...     "models": [...],
    ...     "datasets": [...]
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
    __version__ = "unknown"

# Main analysis classes and functions
from .core.runner import EcoLyzer
from .core.research import run_comprehensive_analysis
from .core.config import BenchmarkConfig, ModelConfig, DatasetConfig

# Import key monitoring classes
from .monitoring.environmental import AdaptiveEnvironmentalTracker
from .monitoring.hardware import HardwareCapabilities, detect_hardware_capabilities

# Import utilities
from .utils.validation import validate_config
from .utils.helpers import setup_logging, get_default_config

# Backward compatibility aliases
GenBenchRunner = EcoLyzer  # For backward compatibility
run_comprehensive_carbon_research = run_comprehensive_analysis
BenchmarkConfig = AnalysisConfig

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
    
    # Backward compatibility
    "GenBenchRunner",
    "run_comprehensive_carbon_research",
    "BenchmarkConfig",
    
    # Version
    "__version__",
]

# Package metadata
__author__ = "ML-EcoLyzer Research Team"
__email__ = "contact@ml-ecolyzer.org"
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
        "references": __references__
    }

def create_analysis_config(**kwargs):
    """Create an analysis configuration with sensible defaults."""
    return AnalysisConfig(**kwargs)

def quick_analysis(model_name: str, dataset_name: str, **kwargs):
    """
    Run a quick environmental impact analysis with minimal configuration.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: HuggingFace dataset identifier
        **kwargs: Additional configuration options
        
    Returns:
        Dict with analysis results
    """
    config = {
        "project": f"quick_analysis_{model_name.replace('/', '_')}",
        "models": [{"name": model_name, "task": "text"}],
        "datasets": [{"name": dataset_name, "task": "text", "limit": 100}],
        **kwargs
    }
    
    analyzer = EcoLyzer(config)
    return analyzer.run()

# Maintain backward compatibility
create_benchmark_config = create_analysis_config
quick_benchmark = quick_analysis