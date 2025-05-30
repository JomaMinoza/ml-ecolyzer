"""
Adaptive Environmental Tracking Module

This module provides comprehensive environmental metrics collection with adaptive
monitoring that works across different hardware configurations.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from .hardware import (
    HardwareCapabilities, get_device_power_profile, get_device_water_profile,
    estimate_cooling_overhead, calculate_water_footprint_from_energy
)


@dataclass
class EnvironmentalMetrics:
    """Container for environmental monitoring metrics"""
    timestamp: float
    power_consumption_watts: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    water_consumption_liters_per_hour: float
    cumulative_water_liters: float
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_utilization_percent: Optional[float] = None
    gpu_temperature_celsius: Optional[float] = None
    cpu_temperature_celsius: Optional[float] = None
    battery_level_percent: Optional[float] = None
    battery_time_remaining_seconds: Optional[float] = None


class AdaptiveEnvironmentalTracker:
    """
    Adaptive environmental tracking system that works across hardware configurations
    
    This class provides comprehensive environmental monitoring with graceful degradation
    based on available hardware capabilities and monitoring APIs.
    """

    def __init__(self, config: Dict[str, Any], capabilities: Optional[HardwareCapabilities] = None):
        """
        Initialize adaptive environmental tracker
        
        Args:
            config: Configuration dictionary
            capabilities: Hardware capabilities (auto-detected if None)
        """
        self.config = config
        
        if capabilities is None:
            from .hardware import detect_hardware_capabilities
            capabilities = detect_hardware_capabilities()
        
        self.capabilities = capabilities
        self.power_profile = get_device_power_profile(capabilities)
        self.water_profile = get_device_water_profile(capabilities)
        self.cooling_factor = estimate_cooling_overhead(capabilities)
        
        # Initialize monitoring components
        self._init_gpu_monitoring()
        self._init_thermal_monitoring()
        self._init_power_monitoring()
        
        # Tracking state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._metrics_history: List[EnvironmentalMetrics] = []
        self._monitoring_lock = threading.Lock()
        
        # Water tracking state
        self._cumulative_water_liters = 0.0
        self._last_water_calculation_time = None

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        self.gpu_monitoring_available = False
        
        if self.capabilities.has_gpu and self.capabilities.has_gpu_monitoring and HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.gpu_monitoring_available = True
                self.gpu_handles = []
                
                for i in range(self.capabilities.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    
            except Exception as e:
                print(f"⚠️ GPU monitoring initialization failed: {e}")
                self.gpu_monitoring_available = False

    def _init_thermal_monitoring(self):
        """Initialize thermal monitoring if available"""
        self.thermal_monitoring_available = len(self.capabilities.thermal_sensors) > 0

    def _init_power_monitoring(self):
        """Initialize power monitoring if available"""
        self.power_monitoring_available = len(self.capabilities.power_sensors) > 0

    def collect_current_metrics(self) -> EnvironmentalMetrics:
        """
        Collect current environmental metrics including water footprint
        
        Returns:
            EnvironmentalMetrics: Current system metrics with water footprint
        """
        timestamp = time.time()
        
        # CPU and memory metrics
        cpu_utilization = self._get_cpu_utilization()
        memory_utilization = self._get_memory_utilization()
        
        # Power consumption estimation
        power_consumption = self._estimate_power_consumption(cpu_utilization)
        
        # Water consumption calculation
        water_consumption_per_hour, cumulative_water = self._calculate_water_consumption(
            power_consumption, timestamp
        )
        
        # GPU metrics
        gpu_utilization = None
        gpu_memory_utilization = None
        gpu_temperature = None
        
        if self.gpu_monitoring_available:
            gpu_metrics = self._get_gpu_metrics()
            gpu_utilization = gpu_metrics.get("utilization")
            gpu_memory_utilization = gpu_metrics.get("memory_utilization")
            gpu_temperature = gpu_metrics.get("temperature")

        # Thermal metrics
        cpu_temperature = self._get_cpu_temperature()
        
        # Battery metrics
        battery_level = None
        battery_time_remaining = None
        
        if self.capabilities.has_battery:
            battery_metrics = self._get_battery_metrics()
            battery_level = battery_metrics.get("level")
            battery_time_remaining = battery_metrics.get("time_remaining")

        return EnvironmentalMetrics(
            timestamp=timestamp,
            power_consumption_watts=power_consumption,
            cpu_utilization_percent=cpu_utilization,
            memory_utilization_percent=memory_utilization,
            water_consumption_liters_per_hour=water_consumption_per_hour,
            cumulative_water_liters=cumulative_water,
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_utilization_percent=gpu_memory_utilization,
            gpu_temperature_celsius=gpu_temperature,
            cpu_temperature_celsius=cpu_temperature,
            battery_level_percent=battery_level,
            battery_time_remaining_seconds=battery_time_remaining
        )

    def _calculate_water_consumption(self, power_watts: float, timestamp: float) -> tuple[float, float]:
        """
        Calculate water consumption based on power usage
        
        Args:
            power_watts: Current power consumption in watts
            timestamp: Current timestamp
            
        Returns:
            Tuple of (liters_per_hour, cumulative_liters)
        """
        # Convert power to energy consumption rate (kW)
        power_kw = power_watts / 1000.0
        
        # Water consumption rate (liters per hour)
        water_liters_per_hour = power_kw * self.capabilities.water_intensity_factor
        
        # Apply device-specific overhead factors
        cooling_overhead = self.water_profile.get("cooling_overhead", 1.0)
        infrastructure_overhead = self.water_profile.get("infrastructure_overhead", 1.0)
        
        total_water_per_hour = water_liters_per_hour * cooling_overhead * infrastructure_overhead
        
        # Update cumulative water consumption
        if self._last_water_calculation_time is not None:
            time_delta_hours = (timestamp - self._last_water_calculation_time) / 3600.0
            self._cumulative_water_liters += total_water_per_hour * time_delta_hours
        
        self._last_water_calculation_time = timestamp
        
        return total_water_per_hour, self._cumulative_water_liters

    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage"""
        if HAS_PSUTIL:
            try:
                return psutil.cpu_percent(interval=0.1)
            except:
                pass
        
        # Fallback estimation based on load average (Unix-like systems)
        try:
            import os
            load_avg = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return min(100.0, (load_avg / cpu_count) * 100)
        except:
            return 50.0  # Conservative fallback

    def _get_memory_utilization(self) -> float:
        """Get memory utilization percentage"""
        if HAS_PSUTIL:
            try:
                return psutil.virtual_memory().percent
            except:
                pass
        
        return 50.0  # Conservative fallback

    def _estimate_power_consumption(self, cpu_utilization: float) -> float:
        """
        Estimate current power consumption based on utilization
        
        Args:
            cpu_utilization: CPU utilization percentage
            
        Returns:
            Estimated power consumption in watts
        """
        # Base power consumption
        idle_power = self.power_profile["idle"]
        max_power = self.power_profile["max_power"]
        
        # Estimate power based on CPU utilization
        utilization_factor = cpu_utilization / 100.0
        estimated_power = idle_power + (max_power - idle_power) * utilization_factor
        
        # Apply cooling overhead
        estimated_power *= self.cooling_factor
        
        return estimated_power

    def _get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get GPU metrics if available"""
        if not self.gpu_monitoring_available:
            return {}
        
        try:
            # Use first GPU for primary metrics
            handle = self.gpu_handles[0]
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            # Memory utilization
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_util = (memory_info.used / memory_info.total) * 100
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            return {
                "utilization": gpu_util,
                "memory_utilization": memory_util,
                "temperature": temperature
            }
            
        except Exception as e:
            return {}

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        if not HAS_PSUTIL:
            return None
        
        try:
            temps = psutil.sensors_temperatures()
            
            # Look for CPU temperature sensors
            for name, entries in temps.items():
                if "cpu" in name.lower() or "core" in name.lower():
                    if entries:
                        return entries[0].current
            
            # Fallback to any available temperature sensor
            for name, entries in temps.items():
                if entries:
                    return entries[0].current
                    
        except:
            pass
        
        return None

    def _get_battery_metrics(self) -> Dict[str, Optional[float]]:
        """Get battery metrics if available"""
        if not self.capabilities.has_battery or not HAS_PSUTIL:
            return {}
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "level": battery.percent,
                    "time_remaining": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
        except:
            pass
        
        return {}

    def start_monitoring(self, frequency_hz: float = 1.0):
        """
        Start continuous monitoring in background thread
        
        Args:
            frequency_hz: Monitoring frequency in Hz
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._metrics_history.clear()
        self._cumulative_water_liters = 0.0
        self._last_water_calculation_time = None
        
        def monitoring_loop():
            interval = 1.0 / frequency_hz
            
            while self._monitoring_active:
                try:
                    metrics = self.collect_current_metrics()
                    
                    with self._monitoring_lock:
                        self._metrics_history.append(metrics)
                        
                        # Limit history size to prevent memory issues
                        if len(self._metrics_history) > 10000:
                            self._metrics_history = self._metrics_history[-5000:]
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    time.sleep(interval)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

    def get_metrics_history(self) -> List[EnvironmentalMetrics]:
        """Get copy of metrics history"""
        with self._monitoring_lock:
            return self._metrics_history.copy()

    def collect_comprehensive_metrics(self, duration_seconds: float = 300, 
                                    frequency_hz: float = 1.0,
                                    include_quantization_analysis: bool = True) -> Dict[str, Any]:
        """
        Collect comprehensive environmental metrics over specified duration
        
        Args:
            duration_seconds: Duration to monitor
            frequency_hz: Sampling frequency
            include_quantization_analysis: Whether to include quantization analysis
            
        Returns:
            Dict with comprehensive environmental analysis including water footprint
        """
        print(f"🔍 Starting {duration_seconds}s environmental monitoring at {frequency_hz} Hz...")
        print(f"💧 Water intensity factor: {self.capabilities.water_intensity_factor:.2f} L/kWh ({self.capabilities.region})")
        
        # Start monitoring
        self.start_monitoring(frequency_hz)
        
        # Wait for monitoring duration
        time.sleep(duration_seconds)
        
        # Stop monitoring and get results
        self.stop_monitoring()
        metrics_history = self.get_metrics_history()
        
        if not metrics_history:
            return {
                "error": "No metrics collected during monitoring period",
                "assessment_metadata": {
                    "duration_seconds": duration_seconds,
                    "frequency_hz": frequency_hz,
                    "samples_collected": 0
                }
            }
        
        # Analyze collected metrics
        analysis = self._analyze_metrics_history(metrics_history)
        
        # Add metadata
        analysis["assessment_metadata"] = {
            "duration_seconds": duration_seconds,
            "frequency_hz": frequency_hz,
            "samples_collected": len(metrics_history),
            "monitoring_capabilities": self.capabilities.monitoring_methods,
            "device_category": self.capabilities.device_category,
            "water_intensity_factor": self.capabilities.water_intensity_factor,
            "region": self.capabilities.region
        }
        
        # Add quantization analysis if requested
        if include_quantization_analysis:
            analysis["quantization_analysis"] = self._analyze_quantization_potential(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Overall assessment
        analysis["integrated_assessment"] = self._generate_integrated_assessment(analysis)
        
        return analysis

    def _analyze_metrics_history(self, metrics_history: List[EnvironmentalMetrics]) -> Dict[str, Any]:
        """Analyze collected metrics history including water footprint"""
        if not metrics_history:
            return {"error": "No metrics to analyze"}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        power_consumption = [m.power_consumption_watts for m in metrics_history]
        water_consumption = [m.water_consumption_liters_per_hour for m in metrics_history]
        cumulative_water = [m.cumulative_water_liters for m in metrics_history]
        cpu_utilization = [m.cpu_utilization_percent for m in metrics_history]
        memory_utilization = [m.memory_utilization_percent for m in metrics_history]
        
        # GPU metrics (if available)
        gpu_utilization = [m.gpu_utilization_percent for m in metrics_history if m.gpu_utilization_percent is not None]
        gpu_memory = [m.gpu_memory_utilization_percent for m in metrics_history if m.gpu_memory_utilization_percent is not None]
        gpu_temp = [m.gpu_temperature_celsius for m in metrics_history if m.gpu_temperature_celsius is not None]
        
        # Thermal metrics
        cpu_temp = [m.cpu_temperature_celsius for m in metrics_history if m.cpu_temperature_celsius is not None]
        
        # Battery metrics
        battery_level = [m.battery_level_percent for m in metrics_history if m.battery_level_percent is not None]
        
        analysis = {
            "power_analysis": {
                "average_watts": sum(power_consumption) / len(power_consumption),
                "max_watts": max(power_consumption),
                "min_watts": min(power_consumption),
                "total_energy_wh": self._calculate_energy_consumption(timestamps, power_consumption),
                "power_efficiency": self._calculate_power_efficiency(cpu_utilization, power_consumption)
            },
            
            "water_analysis": {
                "average_liters_per_hour": sum(water_consumption) / len(water_consumption),
                "max_liters_per_hour": max(water_consumption),
                "min_liters_per_hour": min(water_consumption),
                "total_water_liters": cumulative_water[-1] if cumulative_water else 0,
                "water_intensity_factor": self.capabilities.water_intensity_factor,
                "region": self.capabilities.region,
                "water_efficiency": self._calculate_water_efficiency(cpu_utilization, water_consumption),
                "water_footprint_breakdown": self._calculate_water_footprint_breakdown(),
                "water_equivalent_bottles": self._calculate_water_equivalent_bottles(cumulative_water[-1] if cumulative_water else 0)
            },
            
            "resource_analysis": {
                "avg_cpu_utilization": sum(cpu_utilization) / len(cpu_utilization),
                "max_cpu_utilization": max(cpu_utilization),
                "avg_memory_utilization": sum(memory_utilization) / len(memory_utilization),
                "max_memory_utilization": max(memory_utilization),
                "resource_efficiency": self._calculate_resource_efficiency(cpu_utilization, memory_utilization)
            }
        }
        
        # Add GPU analysis if available
        if gpu_utilization:
            analysis["gpu_analysis"] = {
                "avg_gpu_utilization": sum(gpu_utilization) / len(gpu_utilization),
                "max_gpu_utilization": max(gpu_utilization),
                "avg_gpu_memory": sum(gpu_memory) / len(gpu_memory) if gpu_memory else None,
                "max_gpu_memory": max(gpu_memory) if gpu_memory else None,
                "gpu_efficiency": self._calculate_gpu_efficiency(gpu_utilization, gpu_memory)
            }
        
        # Add thermal analysis if available
        if cpu_temp or gpu_temp:
            analysis["thermal_analysis"] = {
                "avg_cpu_temp": sum(cpu_temp) / len(cpu_temp) if cpu_temp else None,
                "max_cpu_temp": max(cpu_temp) if cpu_temp else None,
                "avg_gpu_temp": sum(gpu_temp) / len(gpu_temp) if gpu_temp else None,
                "max_gpu_temp": max(gpu_temp) if gpu_temp else None,
                "thermal_efficiency": self._calculate_thermal_efficiency(cpu_temp, gpu_temp)
            }
        
        # Add battery analysis if available
        if battery_level and self.capabilities.has_battery:
            analysis["battery_analysis"] = {
                "avg_battery_level": sum(battery_level) / len(battery_level),
                "battery_drain_rate": self._calculate_battery_drain_rate(timestamps, battery_level),
                "estimated_runtime_hours": self._estimate_battery_runtime(battery_level, timestamps),
                "water_per_battery_percent": self._calculate_water_per_battery_percent(water_consumption, battery_level)
            }
        
        return analysis

    def _calculate_water_efficiency(self, cpu_utilization: List[float], water_consumption: List[float]) -> float:
        """Calculate water efficiency score (0-1)"""
        if not cpu_utilization or not water_consumption:
            return 0.5
        
        avg_utilization = sum(cpu_utilization) / len(cpu_utilization)
        avg_water = sum(water_consumption) / len(water_consumption)
        
        # Efficiency is high when utilization is high but water consumption is relatively low
        max_expected_water = self.water_profile.get("max_liters_per_hour", avg_water * 2)
        
        if max_expected_water > 0:
            water_ratio = avg_water / max_expected_water
            utilization_ratio = avg_utilization / 100.0
            
            if utilization_ratio > 0:
                efficiency = utilization_ratio / water_ratio
                return min(1.0, efficiency)
        
        return 0.5

    def _calculate_water_footprint_breakdown(self) -> Dict[str, float]:
        """Calculate detailed water footprint breakdown"""
        total_water = self._cumulative_water_liters
        
        # Calculate breakdown based on device profile
        cooling_overhead = self.water_profile.get("cooling_overhead", 1.0)
        infrastructure_overhead = self.water_profile.get("infrastructure_overhead", 1.0)
        
        # Base water consumption (direct energy-to-water conversion)
        base_water = total_water / (cooling_overhead * infrastructure_overhead)
        
        # Breakdown components
        cooling_water = base_water * (cooling_overhead - 1.0)
        infrastructure_water = base_water * cooling_overhead * (infrastructure_overhead - 1.0)
        
        return {
            "direct_energy_water_liters": base_water,
            "cooling_water_liters": cooling_water,
            "infrastructure_water_liters": infrastructure_water,
            "total_water_liters": total_water,
            "cooling_percentage": (cooling_water / total_water * 100) if total_water > 0 else 0,
            "infrastructure_percentage": (infrastructure_water / total_water * 100) if total_water > 0 else 0
        }

    def _calculate_water_equivalent_bottles(self, water_liters: float) -> Dict[str, float]:
        """Calculate water consumption in terms of everyday equivalents"""
        # Standard bottle size: 0.5 liters
        bottles_500ml = water_liters / 0.5
        
        # Standard glass: 0.25 liters
        glasses = water_liters / 0.25
        
        # Standard gallon: 3.785 liters
        gallons = water_liters / 3.785
        
        # Coffee cup: 0.24 liters
        coffee_cups = water_liters / 0.24
        
        return {
            "bottles_500ml": bottles_500ml,
            "glasses_250ml": glasses,
            "gallons": gallons,
            "coffee_cups": coffee_cups
        }

    def _calculate_water_per_battery_percent(self, water_consumption: List[float], battery_levels: List[float]) -> Optional[float]:
        """Calculate water consumption per battery percentage for mobile devices"""
        if not water_consumption or not battery_levels or len(battery_levels) < 2:
            return None
        
        avg_water_per_hour = sum(water_consumption) / len(water_consumption)
        
        # Calculate battery drain rate
        battery_start = battery_levels[0]
        battery_end = battery_levels[-1]
        battery_drop = battery_start - battery_end
        
        if battery_drop > 0:
            # Water consumption per 1% battery drop
            return avg_water_per_hour / battery_drop
        
        return None

    def _calculate_energy_consumption(self, timestamps: List[float], power_watts: List[float]) -> float:
        """Calculate total energy consumption in Wh"""
        if len(timestamps) < 2:
            return 0.0
        
        total_energy = 0.0
        for i in range(1, len(timestamps)):
            dt_hours = (timestamps[i] - timestamps[i-1]) / 3600.0
            avg_power = (power_watts[i] + power_watts[i-1]) / 2.0
            total_energy += avg_power * dt_hours
        
        return total_energy

    def _calculate_power_efficiency(self, cpu_utilization: List[float], power_watts: List[float]) -> float:
        """Calculate power efficiency score (0-1)"""
        if not cpu_utilization or not power_watts:
            return 0.5
        
        avg_utilization = sum(cpu_utilization) / len(cpu_utilization)
        avg_power = sum(power_watts) / len(power_watts)
        
        # Efficiency is high when utilization is high but power is relatively low
        max_expected_power = self.power_profile["max_power"]
        
        if max_expected_power > 0:
            power_ratio = avg_power / max_expected_power
            utilization_ratio = avg_utilization / 100.0
            
            if utilization_ratio > 0:
                efficiency = utilization_ratio / power_ratio
                return min(1.0, efficiency)
        
        return 0.5

    def _calculate_resource_efficiency(self, cpu_util: List[float], memory_util: List[float]) -> float:
        """Calculate resource utilization efficiency"""
        avg_cpu = sum(cpu_util) / len(cpu_util)
        avg_memory = sum(memory_util) / len(memory_util)
        
        # Balanced utilization is considered efficient
        balance_score = 1.0 - abs(avg_cpu - avg_memory) / 100.0
        utilization_score = (avg_cpu + avg_memory) / 200.0
        
        return (balance_score + utilization_score) / 2.0

    def _calculate_gpu_efficiency(self, gpu_util: List[float], gpu_memory: List[float]) -> Optional[float]:
        """Calculate GPU efficiency score"""
        if not gpu_util:
            return None
        
        avg_gpu_util = sum(gpu_util) / len(gpu_util)
        
        if gpu_memory:
            avg_gpu_memory = sum(gpu_memory) / len(gpu_memory)
            # Good GPU efficiency means high utilization without excessive memory usage
            memory_efficiency = 1.0 - max(0, avg_gpu_memory - 80) / 20.0  # Penalty for >80% memory
            utilization_efficiency = avg_gpu_util / 100.0
            return (memory_efficiency + utilization_efficiency) / 2.0
        else:
            return avg_gpu_util / 100.0

    def _calculate_thermal_efficiency(self, cpu_temp: List[float], gpu_temp: List[float]) -> float:
        """Calculate thermal efficiency (lower temperatures = higher efficiency)"""
        temps = []
        
        if cpu_temp:
            temps.extend(cpu_temp)
        if gpu_temp:
            temps.extend(gpu_temp)
        
        if not temps:
            return 0.5
        
        avg_temp = sum(temps) / len(temps)
        
        # Thermal efficiency decreases as temperature increases
        # Assume 70°C is optimal, efficiency drops above that
        optimal_temp = 70.0
        max_temp = 95.0
        
        if avg_temp <= optimal_temp:
            return 1.0
        else:
            efficiency = 1.0 - (avg_temp - optimal_temp) / (max_temp - optimal_temp)
            return max(0.0, efficiency)

    def _calculate_battery_drain_rate(self, timestamps: List[float], battery_levels: List[float]) -> Optional[float]:
        """Calculate battery drain rate in percent per hour"""
        if len(timestamps) < 2 or len(battery_levels) < 2:
            return None
        
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600.0
        level_change = battery_levels[0] - battery_levels[-1]
        
        if duration_hours > 0:
            return level_change / duration_hours
        
        return None

    def _estimate_battery_runtime(self, battery_levels: List[float], timestamps: List[float]) -> Optional[float]:
        """Estimate remaining battery runtime in hours"""
        if len(battery_levels) < 2:
            return None
        
        drain_rate = self._calculate_battery_drain_rate(timestamps, battery_levels)
        current_level = battery_levels[-1]
        
        if drain_rate and drain_rate > 0:
            return current_level / drain_rate
        
        return None

    def _analyze_quantization_potential(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential benefits of model quantization including water savings"""
        power_analysis = analysis.get("power_analysis", {})
        water_analysis = analysis.get("water_analysis", {})
        
        avg_power = power_analysis.get("average_watts", 0)
        total_water = water_analysis.get("total_water_liters", 0)
        
        # Estimate quantization benefits based on device category and current usage
        if self.capabilities.device_category in ["mobile", "edge"]:
            potential_power_savings = 0.3  # 30% power savings
            potential_water_savings = 0.35  # 35% water savings (higher due to cooling reduction)
            recommended = True
        elif self.capabilities.device_category in ["desktop_gpu", "desktop_cpu"]:
            potential_power_savings = 0.2  # 20% power savings
            potential_water_savings = 0.25  # 25% water savings
            recommended = avg_power > 200  # Recommend if power usage is high
        else:  # datacenter
            potential_power_savings = 0.15  # 15% power savings
            potential_water_savings = 0.20  # 20% water savings (cooling benefits)
            recommended = avg_power > 400
        
        return {
            "potential_power_savings_percent": potential_power_savings * 100,
            "potential_water_savings_percent": potential_water_savings * 100,
            "estimated_energy_reduction_wh": power_analysis.get("total_energy_wh", 0) * potential_power_savings,
            "estimated_water_reduction_liters": total_water * potential_water_savings,
            "water_bottles_saved": (total_water * potential_water_savings) / 0.5,  # 500ml bottles
            "recommended": recommended,
            "quantization_methods": self._recommend_quantization_methods()
        }

    def _recommend_quantization_methods(self) -> List[str]:
        """Recommend appropriate quantization methods based on hardware"""
        methods = []
        
        if self.capabilities.has_gpu:
            methods.extend(["dynamic_quantization", "static_quantization", "qat"])
        else:
            methods.extend(["dynamic_quantization", "static_quantization"])
        
        if self.capabilities.device_category in ["mobile", "edge"]:
            methods.append("aggressive_pruning")
        
        return methods

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis including water efficiency"""
        recommendations = []
        
        # Power recommendations
        power_analysis = analysis.get("power_analysis", {})
        avg_power = power_analysis.get("average_watts", 0)
        power_efficiency = power_analysis.get("power_efficiency", 0.5)
        
        # Water recommendations
        water_analysis = analysis.get("water_analysis", {})
        total_water = water_analysis.get("total_water_liters", 0)
        water_efficiency = water_analysis.get("water_efficiency", 0.5)
        
        if power_efficiency < 0.3:
            recommendations.append("Consider model optimization or hardware upgrade for better power efficiency")
        
        if water_efficiency < 0.3:
            recommendations.append("High water consumption detected - consider model quantization or workload optimization")
        
        if avg_power > self.power_profile["max_power"] * 0.8:
            recommendations.append("High power consumption detected - consider workload distribution or cooling optimization")
        
        # Water-specific recommendations
        if total_water > 1.0:  # More than 1 liter used
            water_bottles = total_water / 0.5
            recommendations.append(f"Water footprint equivalent to {water_bottles:.1f} bottles - consider green computing practices")
        
        if self.capabilities.device_category == "datacenter" and total_water > 5.0:
            recommendations.append("High data center water usage - investigate cooling efficiency and renewable energy sources")
        
        # Resource recommendations
        resource_analysis = analysis.get("resource_analysis", {})
        cpu_util = resource_analysis.get("avg_cpu_utilization", 0)
        memory_util = resource_analysis.get("avg_memory_utilization", 0)
        
        if cpu_util > 90:
            recommendations.append("High CPU utilization - consider parallelization or hardware scaling")
        
        if memory_util > 85:
            recommendations.append("High memory utilization - consider batch size reduction or memory optimization")
        
        # GPU recommendations
        gpu_analysis = analysis.get("gpu_analysis", {})
        if gpu_analysis:
            gpu_util = gpu_analysis.get("avg_gpu_utilization", 0)
            gpu_memory = gpu_analysis.get("avg_gpu_memory", 0)
            
            if gpu_util < 50:
                recommendations.append("Low GPU utilization - consider increasing batch size or model complexity")
            
            if gpu_memory and gpu_memory > 90:
                recommendations.append("High GPU memory usage - consider gradient checkpointing or model sharding")
        
        # Battery recommendations
        battery_analysis = analysis.get("battery_analysis", {})
        if battery_analysis:
            drain_rate = battery_analysis.get("battery_drain_rate", 0)
            if drain_rate and drain_rate > 10:  # >10% per hour
                recommendations.append("High battery drain rate - consider power-saving mode or model optimization")
        
        # Thermal recommendations
        thermal_analysis = analysis.get("thermal_analysis", {})
        if thermal_analysis:
            max_cpu_temp = thermal_analysis.get("max_cpu_temp", 0)
            max_gpu_temp = thermal_analysis.get("max_gpu_temp", 0)
            
            if max_cpu_temp and max_cpu_temp > 80:
                recommendations.append("High CPU temperature - improve cooling or reduce workload intensity")
            
            if max_gpu_temp and max_gpu_temp > 85:
                recommendations.append("High GPU temperature - improve cooling or enable thermal throttling")
        
        return recommendations

    def _generate_integrated_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall integrated environmental assessment including water efficiency"""
        # Calculate component scores
        power_score = analysis.get("power_analysis", {}).get("power_efficiency", 0.5)
        water_score = analysis.get("water_analysis", {}).get("water_efficiency", 0.5)
        resource_score = analysis.get("resource_analysis", {}).get("resource_efficiency", 0.5)
        
        scores = [power_score, water_score, resource_score]
        
        # Add GPU score if available
        gpu_analysis = analysis.get("gpu_analysis", {})
        if gpu_analysis and gpu_analysis.get("gpu_efficiency") is not None:
            scores.append(gpu_analysis["gpu_efficiency"])
        
        # Add thermal score if available
        thermal_analysis = analysis.get("thermal_analysis", {})
        if thermal_analysis and thermal_analysis.get("thermal_efficiency") is not None:
            scores.append(thermal_analysis["thermal_efficiency"])
        
        # Calculate overall efficiency score
        overall_efficiency = sum(scores) / len(scores)
        
        # Determine efficiency category
        if overall_efficiency >= 0.8:
            efficiency_category = "excellent"
        elif overall_efficiency >= 0.6:
            efficiency_category = "good"
        elif overall_efficiency >= 0.4:
            efficiency_category = "moderate"
        else:
            efficiency_category = "poor"
        
        # Water impact classification
        total_water = analysis.get("water_analysis", {}).get("total_water_liters", 0)
        if total_water < 0.1:
            water_impact = "minimal"
        elif total_water < 0.5:
            water_impact = "low"
        elif total_water < 2.0:
            water_impact = "moderate"
        elif total_water < 10.0:
            water_impact = "high"
        else:
            water_impact = "very_high"
        
        return {
            "overall_efficiency_score": overall_efficiency,
            "efficiency_category": efficiency_category,
            "water_impact_category": water_impact,
            "component_scores": {
                "power_efficiency": power_score,
                "water_efficiency": water_score,
                "resource_efficiency": resource_score,
                "gpu_efficiency": gpu_analysis.get("gpu_efficiency") if gpu_analysis else None,
                "thermal_efficiency": thermal_analysis.get("thermal_efficiency") if thermal_analysis else None
            },
            "assessment_quality": self._assess_measurement_quality(analysis)
        }

    def _assess_measurement_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and reliability of measurements"""
        quality_factors = []
        
        # Check availability of different monitoring methods
        if "gpu_analysis" in analysis and self.gpu_monitoring_available:
            quality_factors.append("gpu_monitoring")
        
        if "thermal_analysis" in analysis:
            quality_factors.append("thermal_monitoring")
        
        if "battery_analysis" in analysis and self.capabilities.has_battery:
            quality_factors.append("battery_monitoring")
        
        if "water_analysis" in analysis:
            quality_factors.append("water_footprint_tracking")
        
        if HAS_PSUTIL:
            quality_factors.append("system_monitoring")
        
        # Determine overall quality
        quality_score = len(quality_factors) / max(len(self.capabilities.monitoring_methods), 1)
        
        if quality_score >= 0.8:
            overall_quality = "high"
        elif quality_score >= 0.5:
            overall_quality = "moderate"
        else:
            overall_quality = "low"
        
        return {
            "overall_quality": overall_quality,
            "quality_score": quality_score,
            "available_monitoring": quality_factors,
            "missing_monitoring": [m for m in self.capabilities.monitoring_methods if m not in quality_factors],
            "water_tracking_available": True,
            "water_intensity_region": self.capabilities.region
        }