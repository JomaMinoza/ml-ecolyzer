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
        water_intensity_factor: Water consumption per kWh (liters/kWh)
        region: Estimated region for water/carbon intensity
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
    water_intensity_factor: float
    region: str


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
    
    # Water intensity and region detection
    water_intensity_factor, region = _detect_water_intensity_factor()
    
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
        power_sensors=power_sensors,
        water_intensity_factor=water_intensity_factor,
        region=region
    )


def _detect_water_intensity_factor() -> tuple[float, str]:
    """
    Detect water intensity factor based on region and energy mix
    
    Returns:
        Tuple of (water_intensity_liters_per_kwh, region_name)
    """
    # Default global average water intensity
    default_water_intensity = 2.5  # liters per kWh (global average)
    default_region = "global_average"
    
    try:
        # Attempt to detect region based on various indicators
        region = _detect_region()
        
        # Water intensity factors by region (liters per kWh)
        # Based on energy mix and cooling requirements
        water_intensity_map = {
            # High water intensity (coal/nuclear heavy, dry climates)
            "us_southwest": 4.2,      # Arizona, Nevada, California
            "us_texas": 3.8,          # Texas (coal + natural gas)
            "china_north": 4.5,       # Northern China (coal heavy)
            "india": 3.9,             # India (coal + hot climate)
            "australia": 4.1,         # Australia (coal + dry climate)
            "middle_east": 4.8,       # Middle East (oil + very hot/dry)
            
            # Medium water intensity
            "us_east": 3.2,           # US East Coast
            "us_central": 3.0,        # US Central states
            "germany": 2.8,           # Germany (mixed energy)
            "uk": 2.5,                # UK (natural gas + renewables)
            "japan": 3.1,             # Japan (mixed, some nuclear)
            "south_korea": 3.3,       # South Korea
            "france": 2.2,            # France (nuclear heavy, low water)
            "canada": 2.1,            # Canada (hydro heavy)
            "brazil": 2.0,            # Brazil (hydro heavy)
            
            # Low water intensity (renewables heavy, cooler climates)
            "scandinavia": 1.5,       # Norway, Sweden, Denmark (hydro/wind)
            "iceland": 1.2,           # Iceland (geothermal/hydro + cold)
            "switzerland": 1.8,       # Switzerland (hydro + nuclear)
            "new_zealand": 1.6,       # New Zealand (hydro/geothermal)
            "costa_rica": 1.4,        # Costa Rica (renewables heavy)
            
            # Cloud provider regions (optimized data centers)
            "aws_us_west": 3.5,       # AWS US West
            "aws_us_east": 3.0,       # AWS US East  
            "aws_eu_west": 2.3,       # AWS EU West
            "azure_europe": 2.4,      # Azure Europe
            "gcp_us": 3.2,            # Google Cloud US
            "gcp_europe": 2.2,        # Google Cloud Europe
        }
        
        return water_intensity_map.get(region, default_water_intensity), region
        
    except Exception:
        return default_water_intensity, default_region


def _detect_region() -> str:
    """
    Detect geographic region for environmental factors
    
    Returns:
        Region identifier string
    """
    try:
        # Method 1: Check environment variables (useful for cloud deployments)
        cloud_region = _detect_cloud_region()
        if cloud_region:
            return cloud_region
        
        # Method 2: Platform-specific detection
        platform_name = platform.system().lower()
        
        if platform_name == "darwin":  # macOS
            return _detect_macos_region()
        elif platform_name == "linux":
            return _detect_linux_region()
        elif platform_name == "windows":
            return _detect_windows_region()
        
        # Method 3: Timezone-based fallback
        return _detect_region_from_timezone()
        
    except Exception:
        return "global_average"


def _detect_cloud_region() -> Optional[str]:
    """Detect cloud provider region from environment variables"""
    # AWS region detection
    aws_region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if aws_region:
        if aws_region.startswith("us-west"):
            return "aws_us_west"
        elif aws_region.startswith("us-east"):
            return "aws_us_east"
        elif aws_region.startswith("eu-"):
            return "aws_eu_west"
    
    # Azure region detection
    azure_region = os.environ.get("AZURE_REGION")
    if azure_region and "europe" in azure_region.lower():
        return "azure_europe"
    
    # Google Cloud region detection
    gcp_zone = os.environ.get("GOOGLE_CLOUD_ZONE") or os.environ.get("GCP_ZONE")
    if gcp_zone:
        if gcp_zone.startswith("us-"):
            return "gcp_us"
        elif gcp_zone.startswith("europe-"):
            return "gcp_europe"
    
    return None


def _detect_macos_region() -> str:
    """Detect region on macOS"""
    try:
        import subprocess
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleLocale"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            locale = result.stdout.strip()
            if locale.startswith("en_US"):
                return "us_east"  # Default US region
            elif locale.startswith("en_CA"):
                return "canada"
            elif locale.startswith("en_GB"):
                return "uk"
            elif locale.startswith("de_"):
                return "germany"
            elif locale.startswith("fr_"):
                return "france"
    except:
        pass
    
    return "us_east"  # macOS default


def _detect_linux_region() -> str:
    """Detect region on Linux"""
    try:
        # Check locale
        locale = os.environ.get("LANG", "")
        
        if "en_US" in locale:
            return "us_east"
        elif "en_CA" in locale:
            return "canada"
        elif "en_GB" in locale:
            return "uk"
        elif "de_" in locale:
            return "germany"
        elif "fr_" in locale:
            return "france"
        elif "ja_" in locale:
            return "japan"
        elif "ko_" in locale:
            return "south_korea"
        elif "zh_CN" in locale:
            return "china_north"
        elif "pt_BR" in locale:
            return "brazil"
        
        # Check timezone
        return _detect_region_from_timezone()
        
    except:
        pass
    
    return "global_average"


def _detect_windows_region() -> str:
    """Detect region on Windows"""
    try:
        # Try to use Windows locale information
        import locale
        system_locale = locale.getdefaultlocale()
        
        if system_locale and system_locale[0]:
            lang_code = system_locale[0]
            
            if lang_code.startswith("en_US"):
                return "us_east"
            elif lang_code.startswith("en_CA"):
                return "canada"
            elif lang_code.startswith("en_GB"):
                return "uk"
            elif lang_code.startswith("de"):
                return "germany"
            elif lang_code.startswith("fr"):
                return "france"
            elif lang_code.startswith("ja"):
                return "japan"
            elif lang_code.startswith("ko"):
                return "south_korea"
            elif lang_code.startswith("zh_CN"):
                return "china_north"
    except:
        pass
    
    return "us_east"  # Windows default


def _detect_region_from_timezone() -> str:
    """Detect region from timezone information"""
    try:
        import time
        
        # Get timezone offset
        timezone_offset = time.timezone
        
        # Rough mapping of timezone offsets to regions
        if -8 <= timezone_offset // 3600 <= -5:  # US timezones
            return "us_east"
        elif timezone_offset // 3600 == 0:  # GMT
            return "uk"
        elif 1 <= timezone_offset // 3600 <= 2:  # Central Europe
            return "germany"
        elif timezone_offset // 3600 == 9:  # Japan
            return "japan"
        elif timezone_offset // 3600 == 8:  # China
            return "china_north"
        
    except:
        pass
    
    return "global_average"


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


def get_device_water_profile(capabilities: HardwareCapabilities) -> Dict[str, float]:
    """
    Get water consumption profile for device category
    
    Returns estimated water consumption in liters for different operations
    """
    # Base water intensity from regional factors
    base_intensity = capabilities.water_intensity_factor
    
    # Device-specific water consumption profiles (liters per hour of operation)
    # Accounts for cooling, data center infrastructure, etc.
    profiles = {
        "datacenter": {
            "idle_liters_per_hour": base_intensity * 0.25,  # 250W baseline
            "cpu_load_liters_per_hour": base_intensity * 0.4,  # 400W
            "gpu_load_liters_per_hour": base_intensity * 0.6,  # 600W
            "max_liters_per_hour": base_intensity * 0.8,  # 800W
            "cooling_overhead": 1.4,  # Data center cooling overhead
            "infrastructure_overhead": 1.2  # Additional infrastructure water usage
        },
        "desktop_gpu": {
            "idle_liters_per_hour": base_intensity * 0.08,  # 80W
            "cpu_load_liters_per_hour": base_intensity * 0.15,  # 150W
            "gpu_load_liters_per_hour": base_intensity * 0.3,  # 300W
            "max_liters_per_hour": base_intensity * 0.4,  # 400W
            "cooling_overhead": 1.1,  # Minimal cooling overhead
            "infrastructure_overhead": 1.0  # No additional infrastructure
        },
        "desktop_cpu": {
            "idle_liters_per_hour": base_intensity * 0.05,  # 50W
            "cpu_load_liters_per_hour": base_intensity * 0.12,  # 120W
            "gpu_load_liters_per_hour": base_intensity * 0.12,  # Same as CPU
            "max_liters_per_hour": base_intensity * 0.15,  # 150W
            "cooling_overhead": 1.05,  # Minimal cooling
            "infrastructure_overhead": 1.0
        },
        "server_cpu": {
            "idle_liters_per_hour": base_intensity * 0.15,  # 150W
            "cpu_load_liters_per_hour": base_intensity * 0.25,  # 250W
            "gpu_load_liters_per_hour": base_intensity * 0.25,
            "max_liters_per_hour": base_intensity * 0.35,  # 350W
            "cooling_overhead": 1.3,  # Server cooling
            "infrastructure_overhead": 1.1
        },
        "edge": {
            "idle_liters_per_hour": base_intensity * 0.015,  # 15W
            "cpu_load_liters_per_hour": base_intensity * 0.025,  # 25W
            "gpu_load_liters_per_hour": base_intensity * 0.035,  # 35W
            "max_liters_per_hour": base_intensity * 0.045,  # 45W
            "cooling_overhead": 1.0,  # Passive cooling
            "infrastructure_overhead": 1.0
        },
        "mobile": {
            "idle_liters_per_hour": base_intensity * 0.002,  # 2W
            "cpu_load_liters_per_hour": base_intensity * 0.008,  # 8W
            "gpu_load_liters_per_hour": base_intensity * 0.012,  # 12W
            "max_liters_per_hour": base_intensity * 0.015,  # 15W
            "cooling_overhead": 1.0,  # No active cooling
            "infrastructure_overhead": 1.0
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


def calculate_water_footprint_from_energy(energy_kwh: float, capabilities: HardwareCapabilities) -> Dict[str, float]:
    """
    Calculate water footprint from energy consumption
    
    Args:
        energy_kwh: Energy consumption in kWh
        capabilities: Hardware capabilities with water intensity factor
        
    Returns:
        Dictionary with water footprint metrics
    """
    # Direct water consumption from energy
    direct_water_liters = energy_kwh * capabilities.water_intensity_factor
    
    # Get device-specific water profile
    water_profile = get_device_water_profile(capabilities)
    
    # Apply cooling and infrastructure overhead
    cooling_overhead = water_profile.get("cooling_overhead", 1.0)
    infrastructure_overhead = water_profile.get("infrastructure_overhead", 1.0)
    
    # Total water footprint
    total_water_liters = direct_water_liters * cooling_overhead * infrastructure_overhead
    
    # Water footprint breakdown
    cooling_water_liters = direct_water_liters * (cooling_overhead - 1.0)
    infrastructure_water_liters = direct_water_liters * cooling_overhead * (infrastructure_overhead - 1.0)
    
    return {
        "direct_water_liters": direct_water_liters,
        "cooling_water_liters": cooling_water_liters,
        "infrastructure_water_liters": infrastructure_water_liters,
        "total_water_liters": total_water_liters,
        "water_intensity_factor": capabilities.water_intensity_factor,
        "region": capabilities.region,
        "cooling_overhead_factor": cooling_overhead,
        "infrastructure_overhead_factor": infrastructure_overhead
    }