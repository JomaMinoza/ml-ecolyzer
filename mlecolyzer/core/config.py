"""
Configuration Classes Module

This module provides structured configuration classes for GenBench.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration for a single model
    
    Attributes:
        name: Model name (HuggingFace identifier or local path)
        task: Task type (text, image, image_generation, audio)
        model_type: Specific model type (optional)
        max_length: Maximum sequence length for text models
        quantization: Quantization configuration
        custom_args: Additional model-specific arguments
    """
    name: str
    task: str
    model_type: Optional[str] = None
    max_length: int = 1024
    quantization: Optional[Dict[str, Any]] = None
    custom_args: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_tasks = ["text", "image", "image_generation", "audio"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{self.task}'. Must be one of: {valid_tasks}")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class DatasetConfig:
    """
    Configuration for a single dataset
    
    Attributes:
        name: Dataset name (HuggingFace identifier)
        task: Task type (text, image, image_generation, audio)
        subset: Dataset subset/configuration name
        split: Primary split to load
        fallback_splits: Alternative splits if primary fails
        strict_split: Only use specified split, no fallbacks
        limit: Maximum number of samples to load
        label_key: Key for extracting labels/text from samples
        custom_args: Additional dataset-specific arguments
    """
    name: str
    task: str
    subset: Optional[str] = None
    split: str = "test"
    fallback_splits: List[str] = field(default_factory=lambda: ["train", "validation", "dev"])
    strict_split: bool = False
    limit: Optional[int] = None
    label_key: str = "text"
    custom_args: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_tasks = ["text", "image", "image_generation", "audio"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{self.task}'. Must be one of: {valid_tasks}")
        
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class MonitoringConfig:
    """
    Configuration for environmental monitoring
    
    Attributes:
        duration_seconds: Monitoring duration
        frequency_hz: Sampling frequency
        enable_quantization_analysis: Include quantization analysis
        enable_frequency_analysis: Include frequency analysis
        enable_wandb: Enable wandb tracking
        wandb_project: Wandb project name
        custom_device_profiles: Custom device power profiles
    """
    duration_seconds: float = 300
    frequency_hz: float = 1.0
    enable_quantization_analysis: bool = True
    enable_frequency_analysis: bool = True
    enable_wandb: Union[bool, str] = "auto"
    wandb_project: Optional[str] = None
    custom_device_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        
        if self.duration_seconds > 3600:
            raise ValueError("duration_seconds too long (max 3600 seconds)")


@dataclass
class HardwareConfig:
    """
    Configuration for hardware settings
    
    Attributes:
        device_profile: Device profile to use
        force_cpu: Force CPU-only execution
        force_gpu: Force GPU execution
        gpu_devices: Specific GPU devices to use
        precision: Model precision (float32, float16, bfloat16)
        memory_optimization: Enable memory optimizations
    """
    device_profile: str = "auto"
    force_cpu: bool = False
    force_gpu: bool = False
    gpu_devices: Optional[List[int]] = None
    precision: str = "auto"
    memory_optimization: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_profiles = ["auto", "datacenter", "desktop_gpu", "desktop_cpu", "mobile", "edge"]
        if self.device_profile not in valid_profiles:
            raise ValueError(f"Invalid device_profile '{self.device_profile}'. Must be one of: {valid_profiles}")
        
        valid_precisions = ["auto", "float32", "float16", "bfloat16"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision '{self.precision}'. Must be one of: {valid_precisions}")


@dataclass
class OutputConfig:
    """
    Configuration for output and logging
    
    Attributes:
        output_dir: Output directory for results
        cache_dir: Cache directory for datasets and models
        emissions_file: Emissions tracking file name
        save_intermediate: Save intermediate results
        export_formats: Export formats for results
        log_level: Logging level
        enable_progress_bars: Show progress bars
    """
    output_dir: str = "./genbench_results"
    cache_dir: Optional[str] = None
    emissions_file: str = "emissions.csv"
    save_intermediate: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    log_level: str = "INFO"
    enable_progress_bars: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level '{self.log_level}'. Must be one of: {valid_levels}")
        
        valid_formats = ["json", "csv", "html", "wandb"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid export format '{fmt}'. Must be one of: {valid_formats}")


@dataclass
class BenchmarkConfig:
    """
    Complete benchmark configuration
    
    Attributes:
        project: Project name
        models: List of model configurations
        datasets: List of dataset configurations
        monitoring: Monitoring configuration
        hardware: Hardware configuration
        output: Output configuration
        metadata: Additional metadata
    """
    project: str
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.project or not self.project.strip():
            raise ValueError("project name cannot be empty")
        
        if not self.models:
            raise ValueError("at least one model must be specified")
        
        if not self.datasets:
            raise ValueError("at least one dataset must be specified")
        
        # Convert dict configs to dataclasses if needed
        self.models = [
            ModelConfig.from_dict(model) if isinstance(model, dict) else model
            for model in self.models
        ]
        
        self.datasets = [
            DatasetConfig.from_dict(dataset) if isinstance(dataset, dict) else dataset
            for dataset in self.datasets
        ]
        
        if isinstance(self.monitoring, dict):
            self.monitoring = MonitoringConfig(**self.monitoring)
        
        if isinstance(self.hardware, dict):
            self.hardware = HardwareConfig(**self.hardware)
        
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "project": self.project,
            "models": [model.to_dict() for model in self.models],
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "monitoring": asdict(self.monitoring),
            "hardware": asdict(self.hardware),
            "output": asdict(self.output),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create from dictionary"""
        # Handle nested configurations
        config_data = data.copy()
        
        # Convert models
        if "models" in config_data:
            config_data["models"] = [
                ModelConfig.from_dict(model) if isinstance(model, dict) else model
                for model in config_data["models"]
            ]
        
        # Convert datasets
        if "datasets" in config_data:
            config_data["datasets"] = [
                DatasetConfig.from_dict(dataset) if isinstance(dataset, dict) else dataset
                for dataset in config_data["datasets"]
            ]
        
        # Convert monitoring config
        if "monitoring" in config_data and isinstance(config_data["monitoring"], dict):
            config_data["monitoring"] = MonitoringConfig(**config_data["monitoring"])
        
        # Convert hardware config
        if "hardware" in config_data and isinstance(config_data["hardware"], dict):
            config_data["hardware"] = HardwareConfig(**config_data["hardware"])
        
        # Convert output config
        if "output" in config_data and isinstance(config_data["output"], dict):
            config_data["output"] = OutputConfig(**config_data["output"])
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'BenchmarkConfig':
        """Load configuration from file"""
        from ..utils.helpers import load_config_from_file
        config_dict = load_config_from_file(config_path)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file"""
        from ..utils.helpers import save_config_to_file
        save_config_to_file(self.to_dict(), config_path, format)
    
    def add_model(self, name: str, task: str, **kwargs) -> ModelConfig:
        """Add a model configuration"""
        model_config = ModelConfig(name=name, task=task, **kwargs)
        self.models.append(model_config)
        return model_config
    
    def add_dataset(self, name: str, task: str, **kwargs) -> DatasetConfig:
        """Add a dataset configuration"""
        dataset_config = DatasetConfig(name=name, task=task, **kwargs)
        self.datasets.append(dataset_config)
        return dataset_config
    
    def get_total_combinations(self) -> int:
        """Get total number of model-dataset combinations"""
        return len(self.models) * len(self.datasets)
    
    def estimate_runtime(self, avg_time_per_combination: float = 300) -> Dict[str, float]:
        """Estimate total runtime"""
        from ..utils.helpers import estimate_runtime
        return estimate_runtime(self.get_total_combinations(), avg_time_per_combination)
    
    def validate(self) -> None:
        """Validate the entire configuration"""
        from ..utils.validation import validate_config
        validate_config(self.to_dict())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "project": self.project,
            "total_models": len(self.models),
            "total_datasets": len(self.datasets),
            "total_combinations": self.get_total_combinations(),
            "model_names": [model.name for model in self.models],
            "dataset_names": [dataset.name for dataset in self.datasets],
            "monitoring_duration": self.monitoring.duration_seconds,
            "output_directory": self.output.output_dir,
            "wandb_enabled": self.monitoring.enable_wandb
        }


def create_quick_config(model_name: str, dataset_name: str, task: str = "text", 
                       project: Optional[str] = None, **kwargs) -> BenchmarkConfig:
    """
    Create a quick benchmark configuration for single model-dataset pair
    
    Args:
        model_name: Model name (HuggingFace identifier)
        dataset_name: Dataset name (HuggingFace identifier)
        task: Task type
        project: Project name (auto-generated if None)
        **kwargs: Additional configuration options
        
    Returns:
        BenchmarkConfig: Complete configuration
    """
    if project is None:
        project = f"quick_{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}"
    
    config = BenchmarkConfig(project=project)
    config.add_model(model_name, task)
    config.add_dataset(dataset_name, task, limit=kwargs.get("limit", 100))
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if key == "limit":
            continue  # Already handled
        elif hasattr(config.monitoring, key):
            setattr(config.monitoring, key, value)
        elif hasattr(config.hardware, key):
            setattr(config.hardware, key, value)
        elif hasattr(config.output, key):
            setattr(config.output, key, value)
    
    return config


def create_research_config(models: List[Dict[str, Any]], datasets: List[Dict[str, Any]], 
                          project: str, **kwargs) -> BenchmarkConfig:
    """
    Create a research configuration for comprehensive studies
    
    Args:
        models: List of model configurations
        datasets: List of dataset configurations
        project: Project name
        **kwargs: Additional configuration options
        
    Returns:
        BenchmarkConfig: Complete research configuration
    """
    config = BenchmarkConfig(project=project)
    
    # Add models
    for model_data in models:
        config.add_model(**model_data)
    
    # Add datasets
    for dataset_data in datasets:
        config.add_dataset(**dataset_data)
    
    # Configure for research (longer monitoring, comprehensive analysis)
    config.monitoring.duration_seconds = kwargs.get("monitoring_duration", 600)
    config.monitoring.enable_quantization_analysis = True
    config.monitoring.enable_frequency_analysis = True
    config.output.save_intermediate = True
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if key == "monitoring_duration":
            continue  # Already handled
        elif hasattr(config.monitoring, key):
            setattr(config.monitoring, key, value)
        elif hasattr(config.hardware, key):
            setattr(config.hardware, key, value)
        elif hasattr(config.output, key):
            setattr(config.output, key, value)
    
    return config


def load_example_configs() -> Dict[str, BenchmarkConfig]:
    """
    Load example configurations for different use cases
    
    Returns:
        Dictionary of example configurations
    """
    examples = {}
    
    # Basic text generation benchmark
    examples["basic_text"] = create_quick_config(
        "gpt2", "wikitext", "text",
        project="basic_text_benchmark",
        limit=100
    )
    
    # Multi-model comparison
    models = [
        {"name": "gpt2", "task": "text"},
        {"name": "distilbert-base-uncased", "task": "text"},
        {"name": "microsoft/DialoGPT-medium", "task": "text"}
    ]
    datasets = [
        {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 500},
        {"name": "imdb", "task": "text", "limit": 500}
    ]
    examples["model_comparison"] = create_research_config(
        models, datasets, "model_comparison_study"
    )
    
    # Comprehensive research
    research_models = [
        {"name": "gpt2", "task": "text"},
        {"name": "microsoft/DialoGPT-medium", "task": "text"},
        {"name": "facebook/bart-base", "task": "text"}
    ]
    research_datasets = [
        {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 1000},
        {"name": "squad", "task": "text", "limit": 800},
        {"name": "imdb", "task": "text", "limit": 600}
    ]
    examples["comprehensive_research"] = create_research_config(
        research_models, research_datasets, "comprehensive_carbon_study",
        monitoring_duration=900,
        enable_wandb=True
    )
    
    return examples