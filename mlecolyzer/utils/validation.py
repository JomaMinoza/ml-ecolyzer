"""
Configuration Validation Module

This module provides comprehensive validation for ML-EcoLyzer configurations
supporting HuggingFace, scikit-learn, and PyTorch frameworks.
"""

from typing import Dict, Any, List, Optional
import os


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate ML-EcoLyzer configuration dictionary
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Basic structure validation
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Required fields
    required_fields = ["project"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from configuration")
    
    # Validate project name
    validate_project_name(config["project"])
    
    # Models validation
    if "models" in config:
        validate_models_config(config["models"])
    
    # Datasets validation
    if "datasets" in config:
        validate_datasets_config(config["datasets"])
    
    # Output directory validation
    if "output_dir" in config:
        validate_output_dir(config["output_dir"])
    
    # Monitoring configuration validation
    validate_monitoring_config(config)
    
    # Hardware configuration validation
    validate_hardware_config(config)
    
    # Framework compatibility validation
    validate_framework_compatibility(config)


def validate_project_name(project_name: str) -> None:
    """
    Validate project name
    
    Args:
        project_name: Project name to validate
        
    Raises:
        ValueError: If project name is invalid
    """
    if not isinstance(project_name, str):
        raise ValueError("Project name must be a string")
    
    if not project_name.strip():
        raise ValueError("Project name cannot be empty")
    
    if len(project_name) > 100:
        raise ValueError("Project name too long (max 100 characters)")
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
    if any(char in project_name for char in invalid_chars):
        raise ValueError(f"Project name contains invalid characters: {invalid_chars}")


def validate_models_config(models: List[Dict[str, Any]]) -> None:
    """
    Validate models configuration
    
    Args:
        models: List of model configurations
        
    Raises:
        ValueError: If models configuration is invalid
    """
    if not isinstance(models, list):
        raise ValueError("Models configuration must be a list")
    
    if not models:
        raise ValueError("At least one model must be specified")
    
    for i, model in enumerate(models):
        validate_single_model_config(model, i)


def validate_single_model_config(model: Dict[str, Any], index: int) -> None:
    """
    Validate single model configuration
    
    Args:
        model: Model configuration dictionary
        index: Index of model in list (for error messages)
        
    Raises:
        ValueError: If model configuration is invalid
    """
    if not isinstance(model, dict):
        raise ValueError(f"Model {index} configuration must be a dictionary")
    
    # Required fields
    required_fields = ["name", "task"]
    for field in required_fields:
        if field not in model:
            raise ValueError(f"Model {index} missing required field '{field}'")
    
    # Validate model name
    if not isinstance(model["name"], str) or not model["name"].strip():
        raise ValueError(f"Model {index} name must be a non-empty string")
    
    # Validate framework
    framework = model.get("framework", "huggingface")
    valid_frameworks = ["huggingface", "sklearn", "pytorch"]
    if framework not in valid_frameworks:
        raise ValueError(f"Model {index} framework '{framework}' not in valid frameworks: {valid_frameworks}")
    
    # Validate task type
    valid_tasks = ["text", "image", "image_generation", "audio", "classification", "regression"]
    if model["task"] not in valid_tasks:
        raise ValueError(f"Model {index} task '{model['task']}' not in valid tasks: {valid_tasks}")
    
    # Framework-specific validation
    validate_framework_specific_model_config(model, index, framework)
    
    # Validate optional fields
    if "model_type" in model and not isinstance(model["model_type"], str):
        raise ValueError(f"Model {index} model_type must be a string")
    
    if "max_length" in model:
        if not isinstance(model["max_length"], int) or model["max_length"] <= 0:
            raise ValueError(f"Model {index} max_length must be a positive integer")
    
    # Validate quantization config if present
    if "quantization" in model:
        validate_quantization_config(model["quantization"], index)
    
    # Validate model parameters
    if "model_params" in model and not isinstance(model["model_params"], dict):
        raise ValueError(f"Model {index} model_params must be a dictionary")


def validate_framework_specific_model_config(model: Dict[str, Any], index: int, framework: str) -> None:
    """
    Validate framework-specific model configuration
    
    Args:
        model: Model configuration
        index: Model index for error messages
        framework: Framework type
        
    Raises:
        ValueError: If framework-specific configuration is invalid
    """
    if framework == "sklearn":
        # sklearn-specific validation
        task = model["task"]
        sklearn_tasks = ["classification", "regression"]
        if task not in sklearn_tasks:
            raise ValueError(f"Model {index} sklearn framework only supports tasks: {sklearn_tasks}")
        
        # Validate sklearn model names
        sklearn_models = [
            "RandomForestClassifier", "RandomForestRegressor",
            "LogisticRegression", "LinearRegression",
            "SVC", "SVR", "GaussianNB",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "KNeighborsClassifier", "KNeighborsRegressor",
            "MLPClassifier", "MLPRegressor"
        ]
        
        model_name = model["name"]
        if not model_name.endswith(('.pkl', '.joblib')) and model_name not in sklearn_models:
            print(f"   ⚠️ Warning: Model {index} '{model_name}' not in known sklearn models: {sklearn_models}")
    
    elif framework == "pytorch":
        # pytorch-specific validation
        if "model_class" in model and model.get("model_class") is not None:
            # Custom model class provided - should be callable
            try:
                # Check if it's a class (not an instance)
                if not callable(model["model_class"]):
                    raise ValueError(f"Model {index} model_class must be callable (a class)")
            except:
                pass  # Allow for serialized classes or other formats
    
    elif framework == "huggingface":
        # huggingface-specific validation
        hf_tasks = ["text", "image", "image_generation", "audio"]
        task = model["task"]
        if task not in hf_tasks:
            raise ValueError(f"Model {index} huggingface framework only supports tasks: {hf_tasks}")


def validate_quantization_config(quantization: Dict[str, Any], model_index: int) -> None:
    """
    Validate quantization configuration
    
    Args:
        quantization: Quantization configuration
        model_index: Model index for error messages
        
    Raises:
        ValueError: If quantization configuration is invalid
    """
    if not isinstance(quantization, dict):
        raise ValueError(f"Model {model_index} quantization config must be a dictionary")
    
    # Validate enabled flag
    if "enabled" in quantization and not isinstance(quantization["enabled"], bool):
        raise ValueError(f"Model {model_index} quantization enabled must be a boolean")
    
    # Validate quantization method
    if "method" in quantization:
        valid_methods = ["dynamic", "static", "qat"]
        if quantization["method"] not in valid_methods:
            raise ValueError(f"Model {model_index} quantization method '{quantization['method']}' not in valid methods: {valid_methods}")
    
    # Validate target dtype
    if "target_dtype" in quantization:
        valid_dtypes = ["int8", "int4", "float16"]
        if quantization["target_dtype"] not in valid_dtypes:
            raise ValueError(f"Model {model_index} quantization target_dtype '{quantization['target_dtype']}' not in valid types: {valid_dtypes}")


def validate_datasets_config(datasets: List[Dict[str, Any]]) -> None:
    """
    Validate datasets configuration
    
    Args:
        datasets: List of dataset configurations
        
    Raises:
        ValueError: If datasets configuration is invalid
    """
    if not isinstance(datasets, list):
        raise ValueError("Datasets configuration must be a list")
    
    if not datasets:
        raise ValueError("At least one dataset must be specified")
    
    for i, dataset in enumerate(datasets):
        validate_single_dataset_config(dataset, i)


def validate_single_dataset_config(dataset: Dict[str, Any], index: int) -> None:
    """
    Validate single dataset configuration
    
    Args:
        dataset: Dataset configuration dictionary
        index: Index of dataset in list (for error messages)
        
    Raises:
        ValueError: If dataset configuration is invalid
    """
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset {index} configuration must be a dictionary")
    
    # Required fields
    required_fields = ["name", "task"]
    for field in required_fields:
        if field not in dataset:
            raise ValueError(f"Dataset {index} missing required field '{field}'")
    
    # Validate dataset name
    if not isinstance(dataset["name"], str) or not dataset["name"].strip():
        raise ValueError(f"Dataset {index} name must be a non-empty string")
    
    # Validate framework
    framework = dataset.get("framework", "huggingface")
    valid_frameworks = ["huggingface", "sklearn", "pytorch"]
    if framework not in valid_frameworks:
        raise ValueError(f"Dataset {index} framework '{framework}' not in valid frameworks: {valid_frameworks}")
    
    # Validate task type
    valid_tasks = ["text", "image", "image_generation", "audio", "classification", "regression"]
    if dataset["task"] not in valid_tasks:
        raise ValueError(f"Dataset {index} task '{dataset['task']}' not in valid tasks: {valid_tasks}")
    
    # Framework-specific validation
    validate_framework_specific_dataset_config(dataset, index, framework)
    
    # Validate optional fields
    if "subset" in dataset and not isinstance(dataset["subset"], str):
        raise ValueError(f"Dataset {index} subset must be a string")
    
    if "split" in dataset and not isinstance(dataset["split"], str):
        raise ValueError(f"Dataset {index} split must be a string")
    
    if "limit" in dataset:
        if not isinstance(dataset["limit"], int) or dataset["limit"] <= 0:
            raise ValueError(f"Dataset {index} limit must be a positive integer")
    
    if "fallback_splits" in dataset:
        if not isinstance(dataset["fallback_splits"], list):
            raise ValueError(f"Dataset {index} fallback_splits must be a list")
        
        for split in dataset["fallback_splits"]:
            if not isinstance(split, str):
                raise ValueError(f"Dataset {index} fallback_splits must contain only strings")
    
    if "strict_split" in dataset and not isinstance(dataset["strict_split"], bool):
        raise ValueError(f"Dataset {index} strict_split must be a boolean")
    
    if "label_key" in dataset and not isinstance(dataset["label_key"], str):
        raise ValueError(f"Dataset {index} label_key must be a string")
    
    # Validate data parameters
    if "data_params" in dataset and not isinstance(dataset["data_params"], dict):
        raise ValueError(f"Dataset {index} data_params must be a dictionary")


def validate_framework_specific_dataset_config(dataset: Dict[str, Any], index: int, framework: str) -> None:
    """
    Validate framework-specific dataset configuration
    
    Args:
        dataset: Dataset configuration
        index: Dataset index for error messages
        framework: Framework type
        
    Raises:
        ValueError: If framework-specific configuration is invalid
    """
    if framework == "sklearn":
        # sklearn-specific validation
        task = dataset["task"]
        sklearn_tasks = ["classification", "regression"]
        if task not in sklearn_tasks:
            raise ValueError(f"Dataset {index} sklearn framework only supports tasks: {sklearn_tasks}")
        
        # Validate sklearn dataset names
        sklearn_datasets = [
            "iris", "wine", "breast_cancer", "diabetes", "boston",
            "20newsgroups", "fetch_olivetti_faces", "fetch_lfw_people"
        ]
        
        dataset_name = dataset["name"]
        # Check if it's a file path or sklearn dataset
        if not (dataset_name.endswith(('.csv', '.xlsx', '.xls', '.pkl', '.joblib')) or 
                dataset_name in sklearn_datasets or 
                os.path.exists(dataset_name)):
            print(f"   ⚠️ Warning: Dataset {index} '{dataset_name}' not in known sklearn datasets: {sklearn_datasets}")
    
    elif framework == "pytorch":
        # pytorch-specific validation
        torchvision_datasets = [
            "CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "ImageNet",
            "COCO", "VOC", "CelebA", "STL10", "SVHN"
        ]
        
        dataset_name = dataset["name"]
        # Check if it's a file path or torchvision dataset
        if not (dataset_name.endswith(('.pt', '.pth', '.csv', '.xlsx', '.xls')) or 
                dataset_name in torchvision_datasets or 
                os.path.exists(dataset_name) or
                "data" in dataset.get("data_params", {})):
            print(f"   ⚠️ Warning: Dataset {index} '{dataset_name}' not in known PyTorch datasets: {torchvision_datasets}")
    
    elif framework == "huggingface":
        # huggingface-specific validation
        hf_tasks = ["text", "image", "image_generation", "audio"]
        task = dataset["task"]
        if task not in hf_tasks:
            raise ValueError(f"Dataset {index} huggingface framework only supports tasks: {hf_tasks}")


def validate_framework_compatibility(config: Dict[str, Any]) -> None:
    """
    Validate that model and dataset frameworks are compatible
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If framework combinations are invalid
    """
    models = config.get("models", [])
    datasets = config.get("datasets", [])
    
    if not models or not datasets:
        return  # Skip if no models or datasets
    
    compatible_combinations = 0
    
    for model in models:
        model_framework = model.get("framework", "huggingface")
        model_task = model.get("task", "")
        
        for dataset in datasets:
            dataset_framework = dataset.get("framework", "huggingface")
            dataset_task = dataset.get("task", "")
            
            # Check if frameworks match
            if model_framework == dataset_framework and model_task == dataset_task:
                compatible_combinations += 1
    
    if compatible_combinations == 0:
        raise ValueError(
            "No compatible model-dataset combinations found. "
            "Models and datasets must have matching frameworks and tasks."
        )
    
    print(f"   ✅ Found {compatible_combinations} compatible model-dataset combinations")


def validate_output_dir(output_dir: str) -> None:
    """
    Validate output directory
    
    Args:
        output_dir: Output directory path
        
    Raises:
        ValueError: If output directory is invalid
    """
    if not isinstance(output_dir, str):
        raise ValueError("Output directory must be a string")
    
    if not output_dir.strip():
        raise ValueError("Output directory cannot be empty")
    
    # Try to create directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory '{output_dir}': {e}")
    
    # Check if directory is writable
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory '{output_dir}' is not writable")


def validate_monitoring_config(config: Dict[str, Any]) -> None:
    """
    Validate monitoring configuration
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If monitoring configuration is invalid
    """
    # Validate monitoring duration
    if "monitoring_duration" in config:
        duration = config["monitoring_duration"]
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("Monitoring duration must be a positive number")
        
        if duration > 3600:  # 1 hour
            raise ValueError("Monitoring duration too long (max 3600 seconds)")
    
    # Validate boolean flags
    boolean_flags = [
        "enable_quantization_analysis",
        "enable_frequency_analysis",
        "enable_wandb",
        "cache_datasets",
        "validate_datasets"
    ]
    
    for flag in boolean_flags:
        if flag in config and not isinstance(config[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    
    # Validate wandb configuration
    if "wandb_project" in config:
        if not isinstance(config["wandb_project"], str) or not config["wandb_project"].strip():
            raise ValueError("wandb_project must be a non-empty string")


def validate_hardware_config(config: Dict[str, Any]) -> None:
    """
    Validate hardware configuration
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If hardware configuration is invalid
    """
    # Validate device profile
    if "device_profile" in config:
        valid_profiles = ["auto", "datacenter", "desktop_gpu", "desktop_cpu", "mobile", "edge"]
        if config["device_profile"] not in valid_profiles:
            raise ValueError(f"Device profile '{config['device_profile']}' not in valid profiles: {valid_profiles}")
    
    # Validate custom device profiles
    if "custom_device_profiles" in config:
        profiles = config["custom_device_profiles"]
        if not isinstance(profiles, dict):
            raise ValueError("Custom device profiles must be a dictionary")
        
        for profile_name, profile_config in profiles.items():
            validate_device_profile_config(profile_config, profile_name)


def validate_device_profile_config(profile: Dict[str, Any], name: str) -> None:
    """
    Validate device profile configuration
    
    Args:
        profile: Device profile configuration
        name: Profile name for error messages
        
    Raises:
        ValueError: If device profile configuration is invalid
    """
    if not isinstance(profile, dict):
        raise ValueError(f"Device profile '{name}' must be a dictionary")
    
    # Validate power settings
    power_fields = ["idle", "cpu_load", "gpu_load", "max_power"]
    for field in power_fields:
        if field in profile:
            value = profile[field]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Device profile '{name}' {field} must be a non-negative number")


def validate_research_config(config: Dict[str, Any]) -> None:
    """
    Validate comprehensive research configuration
    
    Args:
        config: Research configuration dictionary
        
    Raises:
        ValueError: If research configuration is invalid
    """
    # First validate as regular config
    validate_config(config)
    
    # Additional validation for research mode
    if not config.get("models") or not config.get("datasets"):
        raise ValueError("Research configuration must include both models and datasets")
    
    # Count compatible combinations
    models = config["models"]
    datasets = config["datasets"]
    
    compatible_combinations = 0
    for model in models:
        model_framework = model.get("framework", "huggingface")
        model_task = model.get("task", "")
        
        for dataset in datasets:
            dataset_framework = dataset.get("framework", "huggingface")
            dataset_task = dataset.get("task", "")
            
            if model_framework == dataset_framework and model_task == dataset_task:
                compatible_combinations += 1
    
    if compatible_combinations < 2:
        raise ValueError("Research configuration must have at least 2 compatible model-dataset combinations")
    
    # Warn about large research configurations
    if compatible_combinations > 50:
        print(f"⚠️ Warning: Large research configuration with {compatible_combinations} combinations")
        print("   This may take a very long time to complete")


def validate_file_paths(config: Dict[str, Any]) -> None:
    """
    Validate file paths in configuration
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If file paths are invalid
    """
    # Check cache directory
    if "cache_dir" in config:
        cache_dir = config["cache_dir"]
        if not isinstance(cache_dir, str):
            raise ValueError("Cache directory must be a string")
        
        # Create cache directory if it doesn't exist
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create cache directory '{cache_dir}': {e}")
    
    # Check emissions file path
    if "emissions_file" in config:
        emissions_file = config["emissions_file"]
        if not isinstance(emissions_file, str):
            raise ValueError("Emissions file must be a string")
        
        # Check if directory exists
        emissions_dir = os.path.dirname(emissions_file)
        if emissions_dir and not os.path.exists(emissions_dir):
            try:
                os.makedirs(emissions_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create directory for emissions file '{emissions_file}': {e}")


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of configuration for logging/display
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Summary dictionary with key configuration details
    """
    # Count compatible combinations
    models = config.get("models", [])
    datasets = config.get("datasets", [])
    
    compatible_combinations = 0
    frameworks_used = set()
    
    for model in models:
        model_framework = model.get("framework", "huggingface")
        model_task = model.get("task", "")
        frameworks_used.add(model_framework)
        
        for dataset in datasets:
            dataset_framework = dataset.get("framework", "huggingface")
            dataset_task = dataset.get("task", "")
            frameworks_used.add(dataset_framework)
            
            if model_framework == dataset_framework and model_task == dataset_task:
                compatible_combinations += 1
    
    summary = {
        "project": config.get("project", "unknown"),
        "num_models": len(models),
        "num_datasets": len(datasets),
        "total_combinations": len(models) * len(datasets),
        "compatible_combinations": compatible_combinations,
        "frameworks_used": list(frameworks_used),
        "monitoring_duration": config.get("monitoring_duration", 300),
        "quantization_analysis": config.get("enable_quantization_analysis", True),
        "wandb_enabled": config.get("enable_wandb", "auto"),
        "device_profile": config.get("device_profile", "auto"),
        "output_dir": config.get("output_dir", ".")
    }
    
    # Add model names by framework
    model_names_by_framework = {}
    for model in models:
        framework = model.get("framework", "huggingface")
        if framework not in model_names_by_framework:
            model_names_by_framework[framework] = []
        model_names_by_framework[framework].append(model.get("name", "unknown"))
    summary["model_names_by_framework"] = model_names_by_framework
    
    # Add dataset names by framework
    dataset_names_by_framework = {}
    for dataset in datasets:
        framework = dataset.get("framework", "huggingface")
        if framework not in dataset_names_by_framework:
            dataset_names_by_framework[framework] = []
        dataset_names_by_framework[framework].append(dataset.get("name", "unknown"))
    summary["dataset_names_by_framework"] = dataset_names_by_framework
    
    return summary


def validate_sklearn_dependencies() -> bool:
    """
    Validate that sklearn dependencies are available
    
    Returns:
        True if sklearn is available and properly configured
    """
    try:
        import sklearn
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        
        # Test basic functionality
        data = load_iris()
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(data.data[:10], data.target[:10])
        
        return True
    except ImportError:
        return False
    except Exception:
        return False


def validate_pytorch_dependencies() -> bool:
    """
    Validate that PyTorch dependencies are available
    
    Returns:
        True if PyTorch is available and properly configured
    """
    try:
        import torch
        import torchvision
        
        # Test basic functionality
        x = torch.randn(2, 3)
        y = torch.sum(x)
        
        return True
    except ImportError:
        return False
    except Exception:
        return False


def validate_huggingface_dependencies() -> bool:
    """
    Validate that HuggingFace dependencies are available
    
    Returns:
        True if HuggingFace is available and properly configured
    """
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        
        return True
    except ImportError:
        return False
    except Exception:
        return False