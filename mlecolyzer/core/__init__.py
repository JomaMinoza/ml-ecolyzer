"""
Core ML-EcoLyzer Analysis Module

This module contains the core analysis functionality for ML-EcoLyzer.
"""

from .runner import EcoLyzer
from .research import run_comprehensive_analysis, analyze_research_results
from .config import (
    BenchmarkConfig, ModelConfig, DatasetConfig, 
    MonitoringConfig, HardwareConfig, OutputConfig,
    create_quick_config, create_research_config
)

__all__ = [
    # Main classes
    "EcoLyzer",
    "run_comprehensive_analysis",
    "analyze_research_results",
    
    # Configuration classes
    "BenchmarkConfig",
    "ModelConfig", 
    "DatasetConfig",
    "MonitoringConfig",
    "HardwareConfig", 
    "OutputConfig",
    
    # Configuration helpers
    "create_quick_config",
    "create_research_config"
]