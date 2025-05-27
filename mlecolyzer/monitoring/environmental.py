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

from .hardware import HardwareCapabilities, get_device_power_profile, estimate_cooling_overhead


@dataclass
class EnvironmentalMetrics:
    """Container for environmental monitoring metrics"""
    timestamp: float
    power_consumption_watts: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
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
        Collect current environmental metrics
        
        Returns:
            EnvironmentalMetrics: Current system metrics
        """
        timestamp = time.time()
        
        # CPU and memory metrics
        cpu_utilization = self._get_cpu_utilization()
        memory_utilization = self._get_memory_utilization()
        
        # Power consumption estimation
        power_consumption = self._estimate_power_consumption(cpu_utilization)
        
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
            gpu_utilization_percent=gpu_utilization,
            gpu_memory_utilization_percent=gpu_memory_utilization,
            gpu_temperature_celsius=gpu_temperature,
            cpu_temperature_celsius=cpu_temperature,
            battery_level_percent=battery_level,
            battery_time_remaining_seconds=battery_time_remaining
        )

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
            Dict with comprehensive environmental analysis
        """
        print(f"🔍 Starting {duration_seconds}s environmental monitoring at {frequency_hz} Hz...")
        
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
            "device_category": self.capabilities.device_category
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
        """Analyze collected metrics history"""
        if not metrics_history:
            return {"error": "No metrics to analyze"}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        power_consumption = [m.power_consumption_watts for m in metrics_history]
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
                "estimated_runtime_hours": self._estimate_battery_runtime(battery_level, timestamps)
            }
        
        return analysis

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
        """Analyze potential benefits of model quantization"""
        # This is a simplified analysis - in practice would be more sophisticated
        power_analysis = analysis.get("power_analysis", {})
        avg_power = power_analysis.get("average_watts", 0)
        
        # Estimate quantization benefits based on device category and current power usage
        if self.capabilities.device_category in ["mobile", "edge"]:
            potential_savings = 0.3  # 30% power savings
            recommended = True
        elif self.capabilities.device_category in ["desktop_gpu", "desktop_cpu"]:
            potential_savings = 0.2  # 20% power savings
            recommended = avg_power > 200  # Recommend if power usage is high
        else:  # datacenter
            potential_savings = 0.15  # 15% power savings
            recommended = avg_power > 400
        
        return {
            "potential_power_savings_percent": potential_savings * 100,
            "estimated_energy_reduction_wh": power_analysis.get("total_energy_wh", 0) * potential_savings,
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
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Power recommendations
        power_analysis = analysis.get("power_analysis", {})
        avg_power = power_analysis.get("average_watts", 0)
        power_efficiency = power_analysis.get("power_efficiency", 0.5)
        
        if power_efficiency < 0.3:
            recommendations.append("Consider model optimization or hardware upgrade for better power efficiency")
        
        if avg_power > self.power_profile["max_power"] * 0.8:
            recommendations.append("High power consumption detected - consider workload distribution or cooling optimization")
        
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
        """Generate overall integrated environmental assessment"""
        # Calculate component scores
        power_score = analysis.get("power_analysis", {}).get("power_efficiency", 0.5)
        resource_score = analysis.get("resource_analysis", {}).get("resource_efficiency", 0.5)
        
        scores = [power_score, resource_score]
        
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
        
        return {
            "overall_efficiency_score": overall_efficiency,
            "efficiency_category": efficiency_category,
            "component_scores": {
                "power_efficiency": power_score,
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
        
        if HAS_PSUTIL:
            quality_factors.append("system_monitoring")
        
        # Determine overall quality
        quality_score = len(quality_factors) / len(self.capabilities.monitoring_methods)
        
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
            "missing_monitoring": [m for m in self.capabilities.monitoring_methods if m not in quality_factors]
        }