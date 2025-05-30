"""
Utility Functions Module

This module provides helper functions for configuration, validation, and logging.
"""

from .validation import (
    validate_config, validate_research_config, validate_project_name,
    validate_models_config, validate_datasets_config, get_config_summary
)
from .helpers import (
    setup_logging, get_default_config, load_config_from_file, save_config_to_file,
    merge_configs, create_timestamped_dir, format_duration, format_bytes,
    get_system_info, print_banner, safe_json_serialize, estimate_runtime,
    check_disk_space
)

from .water_utilities import (
    calculate_water_equivalents,
    compare_water_footprints,
    estimate_water_savings_from_quantization,
    water_intensity_lookup,
    generate_water_efficiency_report,
    calculate_regional_water_impact_comparison
)

__all__ = [
    # Validation functions
    "validate_config",
    "validate_research_config", 
    "validate_project_name",
    "validate_models_config",
    "validate_datasets_config",
    "get_config_summary",

    # Water utilities functions
    "calculate_water_equivalents",
    "compare_water_footprints",
    "estimate_water_savings_from_quantization",
    "water_intensity_lookup",
    "generate_water_efficiency_report",
    "calculate_regional_water_impact_comparison",
    
    # Helper functions
    "setup_logging",
    "get_default_config",
    "load_config_from_file",
    "save_config_to_file", 
    "merge_configs",
    "create_timestamped_dir",
    "format_duration",
    "format_bytes",
    "get_system_info",
    "print_banner",
    "safe_json_serialize",
    "estimate_runtime",
    "check_disk_space"
]