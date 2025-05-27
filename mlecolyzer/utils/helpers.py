"""
Helper Utilities Module

This module provides utility functions for logging, configuration, and common operations.
"""

import logging
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_logging(level: str = "INFO", use_rich: bool = None) -> logging.Logger:
    """
    Setup logging with optional rich formatting
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_rich: Whether to use rich formatting (auto-detect if None)
        
    Returns:
        Configured logger
    """
    # Auto-detect rich availability
    if use_rich is None:
        use_rich = HAS_RICH
    
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Clear existing handlers
    logger = logging.getLogger("genbench")
    logger.handlers.clear()
    
    if use_rich:
        # Rich handler for pretty formatting
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
    else:
        # Standard handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    
    return logger


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for GenBench
    
    Returns:
        Default configuration dictionary
    """
    return {
        "project": "genbench_default",
        "models": [],
        "datasets": [],
        "output_dir": "./genbench_results",
        "monitoring_duration": 300,
        "enable_quantization_analysis": True,
        "enable_frequency_analysis": True,
        "enable_wandb": "auto",  # Auto-detect based on capabilities
        "cache_datasets": True,
        "validate_datasets": True,
        "max_length": 1024,
        "device_profile": "auto",
        "log_level": "INFO"
    }


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file cannot be loaded or parsed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def save_config_to_file(config: Dict[str, Any], config_path: Union[str, Path], format: str = "yaml") -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format ('yaml' or 'json')
        
    Raises:
        ValueError: If save fails
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
    except Exception as e:
        raise ValueError(f"Failed to save configuration to {config_path}: {e}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries with override taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def create_timestamped_dir(base_dir: Union[str, Path], prefix: str = "run") -> Path:
    """
    Create a timestamped directory for organizing results
    
    Args:
        base_dir: Base directory path
        prefix: Prefix for directory name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    
    result_dir = Path(base_dir) / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and logging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    system_info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix
        },
        "environment": {
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "hostname": platform.node(),
            "cwd": os.getcwd()
        }
    }
    
    # Add GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            system_info["gpu"] = {
                "available": True,
                "count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                system_info["gpu"]["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor
                })
        else:
            system_info["gpu"] = {"available": False}
    except ImportError:
        system_info["gpu"] = {"available": False, "torch_not_installed": True}
    
    # Add memory information if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["memory"] = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        }
        
        # CPU information
        system_info["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "usage_percent": psutil.cpu_percent(interval=1)
        }
    except ImportError:
        pass
    
    return system_info


def create_progress_bar(description: str = "Processing") -> Optional[object]:
    """
    Create a progress bar if rich is available
    
    Args:
        description: Description for the progress bar
        
    Returns:
        Progress bar object or None if rich not available
    """
    if HAS_RICH:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=Console()
        )
    return None


def print_banner(title: str, subtitle: str = "", width: int = 80) -> None:
    """
    Print a banner with title and subtitle
    
    Args:
        title: Main title
        subtitle: Subtitle (optional)
        width: Banner width
    """
    if HAS_RICH:
        console = Console()
        console.print(f"\n[bold blue]{'='*width}[/bold blue]")
        console.print(f"[bold green]{title.center(width)}[/bold green]")
        if subtitle:
            console.print(f"[dim]{subtitle.center(width)}[/dim]")
        console.print(f"[bold blue]{'='*width}[/bold blue]\n")
    else:
        print(f"\n{'='*width}")
        print(f"{title.center(width)}")
        if subtitle:
            print(f"{subtitle.center(width)}")
        print(f"{'='*width}\n")


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize object to JSON-compatible format
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # Objects with attributes
        return safe_json_serialize(obj.__dict__)
    else:
        return str(obj)


def estimate_runtime(num_combinations: int, avg_time_per_combination: float = 300) -> Dict[str, float]:
    """
    Estimate runtime for benchmarking
    
    Args:
        num_combinations: Number of model-dataset combinations
        avg_time_per_combination: Average time per combination in seconds
        
    Returns:
        Dictionary with runtime estimates
    """
    total_seconds = num_combinations * avg_time_per_combination
    
    return {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "formatted": format_duration(total_seconds),
        "combinations": num_combinations,
        "avg_per_combination": avg_time_per_combination
    }


def check_disk_space(path: Union[str, Path], required_gb: float = 1.0) -> Dict[str, Any]:
    """
    Check available disk space
    
    Args:
        path: Path to check
        required_gb: Required space in GB
        
    Returns:
        Dictionary with disk space information
    """
    try:
        import shutil
        
        path = Path(path)
        if not path.exists():
            path = path.parent
        
        total, used, free = shutil.disk_usage(path)
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        
        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "sufficient": free_gb >= required_gb,
            "required_gb": required_gb,
            "usage_percent": (used / total) * 100
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "sufficient": False,
            "required_gb": required_gb
        }


def generate_report_html(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Generate HTML report from results
    
    Args:
        results: Benchmark results
        output_path: Path to save HTML report
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GenBench Results Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }
            .error { border-left-color: #cc0000; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>GenBench Results Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <h2>Summary</h2>
        {summary_html}
        
        <h2>Detailed Results</h2>
        {results_html}
    </body>
    </html>
    """
    
    # Generate summary HTML
    summary_html = "<p>Report generation completed</p>"
    
    # Generate results HTML
    results_html = "<p>Detailed results would be rendered here</p>"
    
    # Fill template
    html_content = html_template.format(
        timestamp=datetime.now().isoformat(),
        summary_html=summary_html,
        results_html=results_html
    )
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def cleanup_temp_files(temp_dir: Union[str, Path]) -> None:
    """
    Clean up temporary files and directories
    
    Args:
        temp_dir: Temporary directory to clean up
    """
    try:
        import shutil
        temp_path = Path(temp_dir)
        
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(temp_path)
            
    except Exception as e:
        print(f"⚠️ Failed to cleanup temporary files: {e}")


def get_git_info() -> Optional[Dict[str, str]]:
    """
    Get git repository information for reproducibility
    
    Returns:
        Dictionary with git information or None if not in git repo
    """
    try:
        import subprocess
        
        def run_git_command(command):
            result = subprocess.run(
                ["git"] + command,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        
        commit_hash = run_git_command(["rev-parse", "HEAD"])
        branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        remote_url = run_git_command(["config", "--get", "remote.origin.url"])
        is_dirty = run_git_command(["status", "--porcelain"]) != ""
        
        if commit_hash:
            return {
                "commit_hash": commit_hash,
                "branch": branch or "unknown",
                "remote_url": remote_url or "unknown",
                "is_dirty": is_dirty
            }
            
    except Exception:
        pass
    
    return None