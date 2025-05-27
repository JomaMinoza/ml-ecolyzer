"""
Hardware Detection and Capabilities Module

This module provides comprehensive hardware detection and capability assessment
for adaptive environmental monitoring across different device categories.
"""

import platform
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


@dataclass
class HardwareCapabilities:
    """
    Hardware capabilities and configuration information
    
    Attributes:
        platform: Operating system platform
        device_category: Hardware category (datacenter, desktop_gpu, desktop_cpu, mobile, edge)
        has_gpu: Whether GPU is available
        gpu_count: Number of available GPUs
        gpu_names: List of GPU names
        has_battery: Whether device has battery
        has_gpu_monitoring: Whether GPU monitoring is available
        monitoring_methods: List of available monitoring methods
        cpu_info: CPU information
        memory_info: Memory information
        thermal_sensors: Available thermal sensors
        power_sensors: Available power monitoring
    """
    platform: str
    device_category: str
    has_gpu: bool
    gpu_count: int
    gpu_names: List[str]
    has_battery: bool
    has_gpu_monitoring: bool
    monitoring_methods: List[str]
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    thermal_sensors: List[str]
    power_sensors: List[str]


def detect_hardware_capabilities() -> HardwareCapabilities:
    """
    Detect and assess hardware capabilities for environmental monitoring
    
    Returns:
        HardwareCapabilities: Comprehensive hardware information
    """
    # Basic platform detection
    platform_name = platform.system().lower()
    
    # GPU detection
    has_gpu, gpu_count, gpu_names, has_gpu_monitoring = _detect_gpu_capabilities()
    
    # Battery detection
    has_battery = _detect_battery()
    
    # CPU information
    cpu_info = _get_cpu_info()
    
    # Memory information
    memory_info = _get_memory_info()
    
    # Thermal sensors
    thermal_sensors = _detect_thermal_sensors()
    
    # Power sensors
    power_sensors = _detect_power_sensors()
    
    # Available monitoring methods
    monitoring_methods = _detect_monitoring_methods(
        has_gpu, has_gpu_monitoring, has_battery, thermal_sensors, power_sensors
    )
    
    # Device category classification
    device_category = _classify_device_category(
        has_gpu, gpu_count, gpu_names, has_battery, cpu_info, memory_info
    )
    
    return HardwareCapabilities(
        platform=platform_name,
        device_category=device_category,
        has_gpu=has_gpu,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        has_battery=has_battery,
        has_gpu_monitoring=has_gpu_monitoring,
        monitoring_methods=monitoring_methods,
        cpu_info=cpu_info,
        memory_info=memory_info,
        thermal_sensors=thermal_sensors,
        power_sensors=power_sensors
    )


def _detect_gpu_capabilities() -> tuple[bool, int, List[str], bool]:
    """Detect GPU capabilities and monitoring availability"""
    has_gpu = False
    gpu_count = 0
    gpu_names = []
    has_gpu_monitoring = False
    
    if HAS_TORCH:
        has_gpu = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_gpu else 0
        
        # Get GPU names
        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_names.append(props.name)
            except:
                gpu_names.append(f"GPU_{i}")
    
    # Check for GPU monitoring capabilities
    if has_gpu and HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            has_gpu_monitoring = True
        except:
            has_gpu_monitoring = False
    
    return has_gpu, gpu_count, gpu_names, has_gpu_monitoring


def _detect_battery() -> bool:
    """Detect if device has a battery"""
    if not HAS_PSUTIL:
        return False
    
    try:
        battery = psutil.sensors_battery()
        return battery is not None
    except:
        return False


def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU information"""
    cpu_info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
    }
    
    if HAS_PSUTIL:
        try:
            cpu_info.update({
                "physical_cores": psutil.cpu_count(logical=False),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            })
        except:
            pass
    
    return cpu_info


def _get_memory_info() -> Dict[str, Any]:
    """Get memory information"""
    memory_info = {}
    
    if HAS_PSUTIL:
        try:
            virtual_memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": virtual_memory.total / (1024**3),
                "available_gb": virtual_memory.available / (1024**3),
                "used_percent": virtual_memory.percent
            }
            
            # Swap memory
            swap = psutil.swap_memory()
            memory_info.update({
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_percent": swap.percent
            })
        except:
            pass
    
    return memory_info


def _detect_thermal_sensors() -> List[str]:
    """Detect available thermal sensors"""
    thermal_sensors = []
    
    if HAS_PSUTIL:
        try:
            temps = psutil.sensors_temperatures()
            thermal_sensors = list(temps.keys())
        except:
            pass
    
    # Platform-specific thermal detection
    platform_name = platform.system().lower()
    
    if platform_name == "linux":
        # Check for common Linux thermal zones
        thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/proc/acpi/thermal_zone/*/temperature"
        ]
        
        for path in thermal_paths:
            if os.path.exists(path):
                thermal_sensors.append(f"linux_{path.split('/')[-2]}")
    
    elif platform_name == "darwin":
        # macOS thermal detection would go here
        thermal_sensors.append("macos_thermal")
    
    return thermal_sensors


def _detect_power_sensors() -> List[str]:
    """Detect available power monitoring capabilities"""
    power_sensors = []
    
    platform_name = platform.system().lower()
    
    # Battery power monitoring
    if _detect_battery():
        power_sensors.append("battery")
    
    # Platform-specific power monitoring
    if platform_name == "linux":
        # Check for RAPL (Running Average Power Limit)
        rapl_paths = [
            "/sys/class/powercap/intel-rapl",
            "/sys/devices/virtual/powercap/intel-rapl"
        ]
        
        for path in rapl_paths:
            if os.path.exists(path):
                power_sensors.append("rapl")
                break
        
        # Check for other power monitoring
        if os.path.exists("/proc/acpi/battery"):
            power_sensors.append("acpi_battery")
    
    elif platform_name == "darwin":
        # macOS power monitoring
        power_sensors.append("macos_power")
    
    elif platform_name == "windows":
        # Windows power monitoring
        power_sensors.append("windows_power")
    
    return power_sensors


def _detect_monitoring_methods(has_gpu: bool, has_gpu_monitoring: bool, has_battery: bool,
                              thermal_sensors: List[str], power_sensors: List[str]) -> List[str]:
    """Detect available monitoring methods"""
    methods = ["basic"]  # Always available
    
    if HAS_PSUTIL:
        methods.append("psutil")
    
    if has_gpu:
        methods.append("gpu_basic")
        
        if has_gpu_monitoring:
            methods.append("gpu_advanced")
    
    if has_battery:
        methods.append("battery")
    
    if thermal_sensors:
        methods.append("thermal")
    
    if power_sensors:
        methods.append("power")
    
    # Platform-specific methods
    platform_name = platform.system().lower()
    if platform_name == "linux":
        methods.append("linux_sysfs")
    elif platform_name == "darwin":
        methods.append("macos_system")
    elif platform_name == "windows":
        methods.append("windows_wmi")
    
    return methods


def _classify_device_category(has_gpu: bool, gpu_count: int, gpu_names: List[str],
                             has_battery: bool, cpu_info: Dict[str, Any],
                             memory_info: Dict[str, Any]) -> str:
    """Classify device into appropriate category for monitoring optimization"""
    
    # Mobile/Edge devices
    if has_battery:
        total_memory_gb = memory_info.get("total_gb", 0)
        if total_memory_gb < 8:
            return "mobile"
        else:
            return "edge"
    
    # Desktop/Server classification
    if has_gpu:
        # Check for high-end datacenter GPUs
        datacenter_gpu_patterns = [
            "A100", "V100", "H100", "A40", "A6000", "Quadro", "Tesla"
        ]
        
        is_datacenter = any(
            any(pattern in gpu_name for pattern in datacenter_gpu_patterns)
            for gpu_name in gpu_names
        )
        
        if is_datacenter or gpu_count > 1:
            return "datacenter"
        else:
            return "desktop_gpu"
    
    # CPU-only systems
    logical_cores = cpu_info.get("logical_cores", 0)
    total_memory_gb = memory_info.get("total_gb", 0)
    
    if logical_cores > 16 and total_memory_gb > 64:
        return "server_cpu"
    else:
        return "desktop_cpu"


def get_device_power_profile(capabilities: HardwareCapabilities) -> Dict[str, float]:
    """
    Get power consumption profile for device category
    
    Returns estimated power consumption in watts for different states
    """
    profiles = {
        "datacenter": {
            "idle": 250.0,
            "cpu_load": 400.0,
            "gpu_load": 600.0,
            "max_power": 800.0
        },
        "desktop_gpu": {
            "idle": 80.0,
            "cpu_load": 150.0,
            "gpu_load": 300.0,
            "max_power": 400.0
        },
        "desktop_cpu": {
            "idle": 50.0,
            "cpu_load": 120.0,
            "gpu_load": 120.0,  # Same as CPU for CPU-only
            "max_power": 150.0
        },
        "server_cpu": {
            "idle": 150.0,
            "cpu_load": 250.0,
            "gpu_load": 250.0,
            "max_power": 350.0
        },
        "edge": {
            "idle": 15.0,
            "cpu_load": 25.0,
            "gpu_load": 35.0,
            "max_power": 45.0
        },
        "mobile": {
            "idle": 2.0,
            "cpu_load": 8.0,
            "gpu_load": 12.0,
            "max_power": 15.0
        }
    }
    
    return profiles.get(capabilities.device_category, profiles["desktop_cpu"])


def estimate_cooling_overhead(capabilities: HardwareCapabilities) -> float:
    """
    Estimate cooling overhead factor based on device category
    
    Returns multiplication factor for power consumption due to cooling
    """
    cooling_factors = {
        "datacenter": 1.4,  # High cooling overhead
        "desktop_gpu": 1.2,
        "desktop_cpu": 1.1,
        "server_cpu": 1.3,
        "edge": 1.05,
        "mobile": 1.0  # No active cooling overhead
    }
    
    return cooling_factors.get(capabilities.device_category, 1.1)


def get_carbon_intensity_region() -> float:
    """
    Estimate carbon intensity based on geographic location (simplified)
    
    Returns kg CO2 per kWh
    """
    # This is a simplified implementation
    # In practice, this would use IP geolocation and regional carbon data
    
    default_carbon_intensity = 0.5  # kg CO2/kWh (global average)
    
    try:
        # Could integrate with APIs like:
        # - WattTime API for real-time carbon intensity
        # - electricityMap API for regional data
        # - Cloud provider carbon intensity APIs
        
        return default_carbon_intensity
    except:
        return default_carbon_intensity