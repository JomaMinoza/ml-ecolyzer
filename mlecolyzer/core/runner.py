"""
Core ML-EcoLyzer Runner Module

This module contains the main EcoLyzer class that orchestrates the comprehensive
environmental impact analysis process across multiple frameworks.
"""

import time
import platform
import json
import os
import gc
import psutil

from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import numpy as np
import wandb
from codecarbon import EmissionsTracker

# Import our monitoring and utility modules
from ..monitoring.environmental import AdaptiveEnvironmentalTracker
from ..monitoring.hardware import detect_hardware_capabilities, calculate_water_footprint_from_energy
from ..models.loader import ModelLoader
from ..datasets.loader import DatasetLoader
from ..metrics.accuracy import AccuracyMetrics
from ..utils.validation import validate_config
from ..utils.helpers import setup_logging


class EcoLyzer:
    """
    ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework

    This framework analyzes and quantifies the environmental impact of machine learning 
    systems with adaptive monitoring that works across all hardware configurations and
    multiple ML frameworks (HuggingFace, scikit-learn, PyTorch).

    Key Features:
    - Multi-framework support (HuggingFace, scikit-learn, PyTorch)
    - Adaptive hardware detection and monitoring stack initialization
    - Comprehensive environmental metrics (power, thermal, emissions, battery, water)
    - Scientific quantization analysis with empirical validation
    - Robust dataset handling with intelligent fallback mechanisms
    - Cross-platform compatibility with graceful degradation
    - Integration with wandb for advanced monitoring (when beneficial)
    - Extensive error handling and quality assessment
    - Water footprint tracking and analysis

    Environmental Impact Methodology:
    The framework measures multiple dimensions of environmental impact:
    1. Carbon Emissions: Direct CO2 emissions from energy consumption
    2. Water Footprint: Direct and indirect water consumption from energy and cooling
    3. Energy Consumption: Power usage patterns and efficiency analysis
    4. Thermal Impact: Heat generation and cooling requirements
    5. Resource Utilization: Hardware efficiency and optimization potential
    6. Quantization Benefits: Power and water savings from model optimization

    Scientific Validation:
    All measurements and calculations are based on established standards:
    - IEEE standards for electrical measurements
    - JEDEC standards for battery and semiconductor specifications
    - Academic literature for thermal and power modeling
    - Industry benchmarks for quantization impact assessment
    - Regional water intensity factors for accurate water footprint calculation

    References:
    - Strubell et al. (2019) "Energy and Policy Considerations for Deep Learning in NLP"
    - Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training"
    - Schwartz et al. (2020) "Green AI" (Communications of the ACM)
    - Henderson et al. (2020) "Towards the Systematic Reporting of Energy and Carbon Footprints"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML-EcoLyzer environmental analysis framework

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - project: Project name for tracking and organization
                - models: List of model configurations to benchmark
                - datasets: List of dataset configurations to evaluate
                - device_profile: Override device detection (optional)
                - enable_wandb: Whether to use wandb for monitoring (default: auto)
                - enable_quantization_analysis: Include quantization impact (default: True)
                - monitoring_duration: Default monitoring duration in seconds
                - custom_device_profiles: Override default power profiles (optional)
        """
        # Validate configuration
        validate_config(config)
        
        self.config = config
        self.results = {}

        # Setup logging
        setup_logging(self.config.get("log_level", "INFO"))

        # Initialize adaptive environmental tracking system
        print("🔍 Initializing Adaptive Environmental Tracking System...")

        # Detect hardware capabilities
        self.capabilities = detect_hardware_capabilities()

        # Initialize environmental tracker
        try:
            self.environmental_tracker = AdaptiveEnvironmentalTracker(config, self.capabilities)
        except Exception as e:
            print(f"⚠️ Environmental tracking initialization failed: {e}")
            self.environmental_tracker = self._create_minimal_tracker(config)

        # Initialize primary emissions tracker (CodeCarbon)
        self.primary_tracker = EmissionsTracker(
            project_name=config.get("project", "mlecolyzer"),
            output_dir=config.get("output_dir", "."),
            output_file=config.get("emissions_file", "emissions.csv")
        )

        # Initialize wandb if enabled and beneficial
        self.wandb_enabled = self._should_enable_wandb()
        if self.wandb_enabled:
            self._initialize_wandb()

        # Configuration parameters
        self.monitoring_duration = config.get("monitoring_duration", 300)
        self.enable_quantization_analysis = config.get("enable_quantization_analysis", True)
        self.enable_frequency_analysis = config.get("enable_frequency_analysis", True)

        # Initialize specialized components
        self.model_loader = ModelLoader(self.capabilities)
        self.dataset_loader = DatasetLoader(config)
        self.accuracy_metrics = AccuracyMetrics()

        # Initialize specialized trackers based on hardware capabilities
        self._initialize_specialized_trackers()

        # For tracking current processing state
        self._current_dataset_name = ""
        self._current_dataset_split = ""
        self._current_dataset_size = 0
        self._current_model_name = ""
        self._current_framework = ""

        print(f"✅ EcoLyzer initialized successfully!")
        print(f"   Hardware Category: {self.capabilities.device_category}")
        print(f"   Monitoring Methods: {len(self.capabilities.monitoring_methods)}")
        print(f"   Water Intensity: {self.capabilities.water_intensity_factor:.2f} L/kWh ({self.capabilities.region})")
        print(f"   wandb Integration: {'Enabled' if self.wandb_enabled else 'Disabled'}")

    def _create_minimal_tracker(self, config: Dict[str, Any]):
        """Create minimal tracker fallback when full environmental classes unavailable"""
        class MinimalTracker:
            def __init__(self):
                pass

            def collect_comprehensive_metrics(self, duration_seconds=300, frequency_hz=1.0, include_quantization_analysis=True):
                return {
                    "assessment_metadata": {"minimal_mode": True},
                    "integrated_assessment": {"overall_efficiency_score": 0.5},
                    "water_analysis": {"total_water_liters": 0.0, "water_efficiency": 0.5},
                    "recommendations": ["Full environmental tracking unavailable - install required dependencies"]
                }

        return MinimalTracker()

    def _should_enable_wandb(self) -> bool:
        """
        Determine if wandb should be enabled based on hardware capabilities and user preferences
        """
        # Check explicit user preference
        if "enable_wandb" in self.config:
            return self.config["enable_wandb"]

        # Disable on very resource-constrained systems
        if self.capabilities.device_category in ["mobile", "edge"]:
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                if total_ram_gb < 2:  # Less than 2GB RAM
                    return False
            except:
                return False

        # Enable on capable systems
        if self.capabilities.device_category in ["datacenter", "desktop_gpu"]:
            return True

        # Default: Enable for most systems
        return True

    def _initialize_wandb(self):
        """Initialize wandb with hardware-appropriate configuration"""
        try:
            wandb_config = {
                "project": self.config.get("project", "mlecolyzer"),
                "config": {
                    **self.config,
                    "hardware_capabilities": {
                        "platform": self.capabilities.platform,
                        "device_category": self.capabilities.device_category,
                        "has_gpu": self.capabilities.has_gpu,
                        "gpu_count": self.capabilities.gpu_count,
                        "monitoring_methods": self.capabilities.monitoring_methods,
                        "water_intensity_factor": self.capabilities.water_intensity_factor,
                        "region": self.capabilities.region
                    }
                }
            }

            self.wandb_run = wandb.init(**wandb_config)

            # Log GPU specifications if available
            if self.capabilities.has_gpu:
                for i in range(self.capabilities.gpu_count):
                    try:
                        gpu_props = torch.cuda.get_device_properties(i)
                        wandb.config.update({
                            f"gpu_{i}_name": gpu_props.name,
                            f"gpu_{i}_memory_gb": gpu_props.total_memory / 1e9,
                            f"gpu_{i}_compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                        })
                    except:
                        pass

            print("✅ wandb monitoring initialized")

        except Exception as e:
            print(f"⚠️ wandb initialization failed: {e}")
            print("   Continuing with local monitoring only...")
            self.wandb_enabled = False

    def _initialize_specialized_trackers(self):
        """Initialize specialized trackers based on hardware capabilities"""
        self.specialized_trackers = {}

        # Always available: Power profiles and basic system monitoring
        self.specialized_trackers["system"] = True

        # Thermal tracking (always available with fallback estimation)
        self.specialized_trackers["thermal"] = True

        # Water footprint tracking (always available)
        self.specialized_trackers["water"] = True
        print("💧 Water footprint tracking enabled")

        # Battery tracking (mobile/laptop devices)
        if self.capabilities.has_battery:
            self.specialized_trackers["battery"] = True
            print("🔋 Battery monitoring enabled")

        # GPU monitoring (if GPU present and monitoring available)
        if self.capabilities.has_gpu and hasattr(self.capabilities, 'has_gpu_monitoring') and self.capabilities.has_gpu_monitoring:
            self.specialized_trackers["gpu"] = True
            print("🎮 GPU monitoring enabled")

        # Quantization analysis (always available)
        if self.enable_quantization_analysis:
            self.specialized_trackers["quantization"] = True
            print("⚖️ Quantization analysis enabled")

    def load_model(self, model_name: str, task_type: str, framework: str, 
                   model_type: Optional[str] = None, **kwargs) -> Tuple[Any, Any]:
        """
        Load model and processor/tokenizer with comprehensive error handling
        """
        print(f"📥 Loading {framework} model: {model_name} for task: {task_type}")
        self._current_model_name = model_name
        self._current_framework = framework

        return self.model_loader.load_model(model_name, task_type, model_type, framework, **kwargs)

    def load_dataset(self, dataset_name: str, framework: str, subset: Optional[str] = None,
                    split: Optional[str] = None, limit: Optional[int] = None,
                    fallback_splits: Optional[List[str]] = None,
                    strict_split: bool = False, **kwargs):
        """
        Load dataset with comprehensive fallback mechanisms and validation
        """
        dataset = self.dataset_loader.load_dataset(
            dataset_name, subset, split, limit, fallback_splits, strict_split, framework, **kwargs
        )

        # Store metadata
        self._current_dataset_name = dataset_name
        self._current_dataset_split = self.dataset_loader.successful_split
        self._current_dataset_size = len(dataset)

        return dataset

    def run_inference(self, model: Any, processor: Any, dataset: Any, task: str,
                     label_key: str, model_name: str, framework: str) -> Tuple[List[Any], List[Any], Optional[str]]:
        """
        Run inference on dataset with comprehensive monitoring and error handling
        """
        print(f"🔄 Running {framework} inference on {len(dataset)} samples...")

        predictions = []
        references = []
        processing_stats = {
            "processed": 0,
            "successful": 0,
            "errors": 0,
            "skipped": 0
        }

        # Determine model type for specialized handling
        model_type = self._determine_model_type(model_name, task, framework)

        # Process samples with progress tracking
        start_time = time.time()

        # Handle different dataset types
        if framework == "sklearn":
            return self._run_sklearn_inference(model, dataset, task, processing_stats)
        elif framework == "pytorch":
            return self._run_pytorch_inference(model, dataset, task, processing_stats)
        else:  # huggingface
            return self._run_huggingface_inference(model, processor, dataset, task, label_key, 
                                                  model_name, processing_stats)

    def _run_sklearn_inference(self, model, dataset, task: str, processing_stats: Dict[str, int]) -> Tuple[List[Any], List[Any], Optional[str]]:
        """Run inference for sklearn models"""
        try:
            if hasattr(dataset, 'X') and hasattr(dataset, 'y'):
                X, y = dataset.X, dataset.y
            else:
                # Handle other dataset formats
                X = np.array([item[0] if isinstance(item, (tuple, list)) else item for item in dataset])
                y = np.array([item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None for item in dataset])
            
            predictions = self.model_loader.predict_sklearn(model, X, task)
            references = y.tolist() if y is not None else [None] * len(predictions)
            
            processing_stats["processed"] = len(X)
            processing_stats["successful"] = len(predictions)
            
            print(f"   ✅ sklearn inference complete: {len(predictions)} predictions")
            
            return predictions.tolist(), references, "sklearn_model"
            
        except Exception as e:
            print(f"   ❌ sklearn inference error: {e}")
            return [], [], None

    def _run_pytorch_inference(self, model, dataset, task: str, processing_stats: Dict[str, int]) -> Tuple[List[Any], List[Any], Optional[str]]:
        """Run inference for PyTorch models"""
        try:
            predictions = []
            references = []
            
            # Handle different dataset types
            if hasattr(dataset, '__getitem__'):
                for i in range(len(dataset)):
                    try:
                        sample = dataset[i]
                        if isinstance(sample, (tuple, list)) and len(sample) >= 1:
                            X = sample[0]
                            y = sample[1] if len(sample) > 1 else None
                        else:
                            X = sample
                            y = None
                        
                        pred = self.model_loader.predict_pytorch(model, X.unsqueeze(0) if X.dim() > 0 else X, task)
                        predictions.append(pred[0] if len(pred) > 0 else pred)
                        references.append(y.item() if isinstance(y, torch.Tensor) and y.numel() == 1 else y)
                        
                        processing_stats["successful"] += 1
                        
                    except Exception as e:
                        processing_stats["errors"] += 1
                        continue
                    
                    processing_stats["processed"] += 1
            
            print(f"   ✅ PyTorch inference complete: {len(predictions)} predictions")
            
            return predictions, references, "pytorch_model"
            
        except Exception as e:
            print(f"   ❌ PyTorch inference error: {e}")
            return [], [], None

    def _run_huggingface_inference(self, model: Any, processor: Any, dataset: Any, task: str,
                                  label_key: str, model_name: str, processing_stats: Dict[str, int]) -> Tuple[List[Any], List[Any], Optional[str]]:
        """Run inference for HuggingFace models"""
        predictions = []
        references = []
        
        # Determine model type for specialized handling
        model_type = self._determine_model_type(model_name, task, "huggingface")

        # Process samples with progress tracking
        start_time = time.time()

        for i, sample in enumerate(dataset):
            try:
                # Progress reporting
                if i > 0 and i % max(1, len(dataset) // 10) == 0:
                    progress = (i / len(dataset)) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / i) * (len(dataset) - i)
                    print(f"   Progress: {progress:.1f}% ({i}/{len(dataset)}) - ETA: {eta:.1f}s")

                processing_stats["processed"] += 1

                # Process single sample using dataset loader
                pred, ref = self.dataset_loader.process_single_sample(
                    sample, model, processor, task, label_key, model_name, self._current_dataset_name, "huggingface"
                )

                if pred is not None and ref is not None:
                    predictions.append(pred)
                    references.append(ref)
                    processing_stats["successful"] += 1

                    # Log to wandb if enabled
                    if self.wandb_enabled and i % 10 == 0:  # Log every 10th sample
                        wandb.log({
                            "sample_idx": i,
                            "prediction_preview": str(pred)[:100],
                            "reference_preview": str(ref)[:100],
                            "processing_success_rate": processing_stats["successful"] / processing_stats["processed"]
                        })
                else:
                    processing_stats["skipped"] += 1

            except Exception as e:
                processing_stats["errors"] += 1
                print(f"   ❌ Error processing sample {i}: {e}")

                # Stop if too many errors
                if processing_stats["errors"] > len(dataset) * 0.1:  # More than 10% errors
                    print(f"   🛑 Stopping due to high error rate ({processing_stats['errors']} errors)")
                    break

                continue

        # Final processing summary
        total_time = time.time() - start_time
        success_rate = processing_stats["successful"] / processing_stats["processed"] * 100 if processing_stats["processed"] > 0 else 0

        print(f"   ✅ Inference complete:")
        print(f"      Successful: {processing_stats['successful']}/{processing_stats['processed']} ({success_rate:.1f}%)")
        print(f"      Processing time: {total_time:.1f}s ({total_time/len(dataset):.2f}s/sample)")

        if processing_stats["successful"] == 0:
            print(f"   ⚠️ Warning: No successful predictions generated")

        return predictions, references, model_type

    def _determine_model_type(self, model_name: str, task_type: str, framework: str) -> Optional[str]:
        """Determine specific model type for specialized handling"""
        if framework == "sklearn":
            return f"sklearn_{model_name}"
        elif framework == "pytorch":
            return f"pytorch_{model_name}"
        else:  # huggingface
            model_name_lower = model_name.lower()

            if task_type == "audio":
                if "wav2vec2" in model_name_lower or "whisper" in model_name_lower:
                    return "asr"  # Automatic Speech Recognition
                else:
                    return "classification"
            elif task_type == "text":
                if any(arch in model_name_lower for arch in ["gpt", "llama", "mistral", "phi"]):
                    return "causal_lm"  # Causal Language Model
                elif any(arch in model_name_lower for arch in ["bert", "roberta", "distilbert"]):
                    return "masked_lm"  # Masked Language Model
            elif task_type == "image":
                if any(arch in model_name_lower for arch in ["vit", "deit", "swin"]):
                    return "vision_transformer"
                elif any(arch in model_name_lower for arch in ["resnet", "efficientnet", "mobilenet"]):
                    return "cnn"

        return None

    def compute_metrics(self, predictions: List[Any], references: List[Any],
                       task: str, model_type: Optional[str] = None, framework: str = "huggingface") -> Dict[str, Any]:
        """Compute appropriate metrics based on task type and framework"""
        return self.accuracy_metrics.compute_metrics(predictions, references, task, model_type, framework)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete comprehensive environmental impact analysis

        Main Analysis Process:
        1. Initialize monitoring systems and validate configuration
        2. For each model-dataset combination:
           a. Load model with hardware optimizations
           b. Load dataset with robust fallback handling
           c. Run comprehensive environmental monitoring during inference
           d. Compute accuracy metrics and environmental impact including water footprint
           e. Perform quantization analysis and optimization recommendations
        3. Aggregate results and generate comprehensive report
        4. Save results with full traceability and citations

        Returns:
            Dict[str, Any]: Comprehensive analysis results with environmental impact assessment
        """
        print(f"\n🚀 Starting Comprehensive ML Environmental Impact Analysis")
        print(f"{'='*80}")

        # Initialize wandb if enabled
        if self.wandb_enabled:
            self.wandb_run = wandb.init(
                project=self.config["project"],
                config=self.config,
                reinit=True
            )

        # Validate configuration
        if not self.config.get("models") or not self.config.get("datasets"):
            raise ValueError("Configuration must include 'models' and 'datasets' lists")

        # Count compatible combinations
        compatible_combinations = 0
        for model_cfg in self.config["models"]:
            for dataset_cfg in self.config["datasets"]:
                if (model_cfg.get("task") == dataset_cfg.get("task") and 
                    model_cfg.get("framework", "huggingface") == dataset_cfg.get("framework", "huggingface")):
                    compatible_combinations += 1

        print(f"📊 Evaluating {compatible_combinations} compatible model-dataset combinations")
        print(f"🖥️ Hardware: {self.capabilities.device_category}")
        print(f"⚡ Monitoring: {', '.join(self.capabilities.monitoring_methods)}")
        print(f"💧 Water tracking: {self.capabilities.water_intensity_factor:.2f} L/kWh ({self.capabilities.region})")
        print(f"{'='*80}")

        combination_count = 0

        # Main evaluation loop
        for model_cfg in self.config["models"]:
            model_name = model_cfg["name"]
            task_type = model_cfg["task"]
            framework = model_cfg.get("framework", "huggingface")
            specified_model_type = model_cfg.get("model_type")
            model_params = model_cfg.get("model_params", {})

            for dataset_cfg in self.config["datasets"]:
                # Extract dataset configuration
                dataset_name = dataset_cfg["name"]
                dataset_framework = dataset_cfg.get("framework", "huggingface")
                dataset_task = dataset_cfg["task"]
                
                # Skip if framework or task mismatch
                if framework != dataset_framework or task_type != dataset_task:
                    print(f"⏭️ Skipping {model_name} on {dataset_name} (framework/task mismatch)")
                    continue


                combination_count += 1

                # Extract dataset configuration
                subset = dataset_cfg.get("subset")
                split = dataset_cfg.get("split", "test")
                fallback_splits = dataset_cfg.get("fallback_splits", ["train", "validation"])
                strict_split = dataset_cfg.get("strict_split", False)
                label_key = dataset_cfg.get("label_key", "text")
                limit = dataset_cfg.get("limit")
                data_params = dataset_cfg.get("data_params", {})

                print(f"\n📈 Evaluation {combination_count}/{compatible_combinations}")
                print(f"{'='*60}")
                print(f"Framework: {framework}")
                print(f"Model: {model_name}")
                print(f"Dataset: {dataset_name}")
                print(f"Task: {task_type}")
                if subset:
                    print(f"Subset: {subset}")
                print(f"Split: {split} ({'strict' if strict_split else 'with fallbacks'})")
                if limit:
                    print(f"Sample limit: {limit}")
                print(f"{'='*60}")

                try:

                    # Implement memory cleanup before each run
                    self._cleanup_memory()                             

                    # Phase 1: Model and dataset loading with emissions tracking
                    print("📥 Phase 1: Loading model and dataset...")
                    self.primary_tracker.start()

                    # Load model
                    model, processor = self.load_model(
                        model_name, task_type, framework, specified_model_type, **model_params
                    )

                    # Load dataset
                    dataset = self.load_dataset(
                        dataset_name=dataset_name,
                        framework=framework,
                        subset=subset,
                        split=split,
                        limit=limit,
                        fallback_splits=fallback_splits,
                        strict_split=strict_split,
                        **data_params
                    )

                    loading_emissions = self.primary_tracker.stop()
                    print(f"✅ Loading complete - Emissions: {loading_emissions:.6f} kg CO2")

                    # Log to wandb if enabled
                    if self.wandb_enabled:
                        wandb.log({
                            "loading_emissions": loading_emissions,
                            "dataset_split_used": self._current_dataset_split,
                            "dataset_size": self._current_dataset_size,
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "framework": framework
                        })

                    # Phase 2: Comprehensive environmental monitoring during inference
                    print("🌍 Phase 2: Environmental monitoring during inference...")
                    self.primary_tracker.start()

                    # Run inference with monitoring
                    predictions, references, model_type = self.run_inference(
                        model, processor, dataset, task_type, label_key, model_name, framework
                    )

                    inference_emissions = self.primary_tracker.stop()
                    print(f"✅ Inference complete - Emissions: {inference_emissions:.6f} kg CO2")

                    if not predictions:
                        print("⚠️ Warning: No valid predictions generated - skipping analysis")
                        continue

                    # Phase 3: Comprehensive environmental assessment
                    print("📊 Phase 3: Comprehensive environmental assessment...")

                    # Collect comprehensive environmental metrics
                    environmental_metrics = self.environmental_tracker.collect_comprehensive_metrics(
                        duration_seconds=min(300, len(predictions) * 2),  # Adaptive duration
                        frequency_hz=len(predictions) / 300,  # Realistic frequency
                        include_quantization_analysis=self.enable_quantization_analysis
                    )

                    # Phase 4: Accuracy metrics computation
                    print("🎯 Phase 4: Computing accuracy metrics...")
                    self.primary_tracker.start()

                    accuracy_metrics = self.compute_metrics(predictions, references, task_type, model_type, framework)

                    metrics_emissions = self.primary_tracker.stop()
                    print(f"✅ Metrics computed - Emissions: {metrics_emissions:.6f} kg CO2")

                    # Phase 5: Results compilation and analysis
                    print("📋 Phase 5: Compiling comprehensive results...")

                    # Calculate total environmental impact
                    total_emissions = loading_emissions + inference_emissions + metrics_emissions

                    # Get PUE factor for infrastructure overhead
                    pue_factor = 1.2 if self.capabilities.device_category == "datacenter" else 1.0
                    pue_adjusted_emissions = total_emissions * pue_factor

                    # Calculate water footprint from total energy consumption
                    total_energy_kwh = (loading_emissions + inference_emissions + metrics_emissions) / 0.5  # Rough conversion
                    water_footprint = calculate_water_footprint_from_energy(total_energy_kwh, self.capabilities)

                    # Generate unique result key
                    result_key = f"{framework}__{model_name}__{dataset_name}"
                    if subset:
                        result_key += f"__{subset}"
                    result_key += f"__{self._current_dataset_split}"

                    # Comprehensive result compilation
                    comprehensive_result = {
                        # Basic identification
                        "framework": framework,
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "subset": subset,
                        "task": task_type,
                        "model_type": model_type,

                        # Dataset information
                        "split_used": self._current_dataset_split,
                        "split_requested": split,
                        "strict_split_mode": strict_split,
                        "fallback_splits_configured": fallback_splits,
                        "dataset_size": self._current_dataset_size,
                        "successful_predictions": len(predictions),

                        # Hardware and environment context
                        "hardware_profile": {
                            "device_category": self.capabilities.device_category,
                            "platform": self.capabilities.platform,
                            "has_gpu": self.capabilities.has_gpu,
                            "gpu_count": self.capabilities.gpu_count,
                            "monitoring_methods": self.capabilities.monitoring_methods,
                            "water_intensity_factor": self.capabilities.water_intensity_factor,
                            "region": self.capabilities.region
                        },

                        # Accuracy metrics
                        "accuracy_metrics": accuracy_metrics,

                        # Emissions breakdown
                        "emissions_analysis": {
                            "loading_kg_co2": loading_emissions,
                            "inference_kg_co2": inference_emissions,
                            "metrics_computation_kg_co2": metrics_emissions,
                            "total_kg_co2": total_emissions,
                            "pue_adjusted_kg_co2": pue_adjusted_emissions,
                            "pue_factor": pue_factor,
                            "emissions_per_sample": total_emissions / len(predictions) if predictions else 0
                        },

                        # Water footprint analysis
                        "water_analysis": {
                            "total_water_liters": water_footprint["total_water_liters"],
                            "direct_water_liters": water_footprint["direct_water_liters"],
                            "cooling_water_liters": water_footprint["cooling_water_liters"],
                            "infrastructure_water_liters": water_footprint["infrastructure_water_liters"],
                            "water_per_sample": water_footprint["total_water_liters"] / len(predictions) if predictions else 0,
                            "water_intensity_factor": water_footprint["water_intensity_factor"],
                            "region": water_footprint["region"],
                            "water_equivalent_bottles": water_footprint["total_water_liters"] / 0.5,  # 500ml bottles
                            "cooling_overhead_factor": water_footprint["cooling_overhead_factor"],
                            "infrastructure_overhead_factor": water_footprint["infrastructure_overhead_factor"]
                        },

                        # Comprehensive environmental assessment
                        "environmental_assessment": environmental_metrics,

                        # Quality and reliability indicators
                        "assessment_quality": {
                            "monitoring_duration": environmental_metrics.get("assessment_metadata", {}).get("duration_seconds", 0),
                            "measurement_quality": environmental_metrics.get("assessment_quality", {}),
                            "data_sources": self.capabilities.monitoring_methods,
                            "wandb_enabled": self.wandb_enabled,
                            "water_tracking_enabled": True
                        },

                        # Timing information
                        "timing": {
                            "evaluation_timestamp": time.time(),
                            "hardware_detection_methods": len(self.capabilities.monitoring_methods),
                            "environmental_monitoring_duration": environmental_metrics.get("assessment_metadata", {}).get("duration_seconds", 0)
                        },

                        # Recommendations
                        "recommendations": environmental_metrics.get("recommendations", []),

                        # Reproducibility information
                        "reproducibility": {
                            "framework_version": "ML-EcoLyzer v1.0",
                            "hardware_capabilities": self.capabilities.__dict__,
                            "configuration": {
                                "monitoring_duration": self.monitoring_duration,
                                "quantization_analysis_enabled": self.enable_quantization_analysis,
                                "wandb_enabled": self.wandb_enabled
                            }
                        }
                    }

                    # Store results
                    self.results[result_key] = comprehensive_result

                    # Log comprehensive results to wandb if enabled
                    if self.wandb_enabled:
                        wandb.log({
                            "combination_complete": combination_count,
                            "framework": framework,
                            "total_emissions": total_emissions,
                            "pue_adjusted_emissions": pue_adjusted_emissions,
                            "total_water_liters": water_footprint["total_water_liters"],
                            "water_bottles_equivalent": water_footprint["total_water_liters"] / 0.5,
                            "accuracy_score": accuracy_metrics.get("accuracy", accuracy_metrics.get("bleu_score", 0)),
                            "environmental_efficiency_score": environmental_metrics.get("integrated_assessment", {}).get("overall_efficiency_score", 0),
                            "water_efficiency_score": environmental_metrics.get("water_analysis", {}).get("water_efficiency", 0.5),
                            "model_dataset_key": result_key
                        })

                        # Log detailed metrics
                        wandb.log(accuracy_metrics)
                        if environmental_metrics.get("power_analysis"):
                            wandb.log(environmental_metrics["power_analysis"])
                        if environmental_metrics.get("water_analysis"):
                            wandb.log(environmental_metrics["water_analysis"])

                    # Print summary for this combination
                    print(f"✅ Evaluation {combination_count} Complete:")
                    print(f"   Framework: {framework}")
                    print(f"   Total CO2: {total_emissions:.6f} kg (PUE-adjusted: {pue_adjusted_emissions:.6f} kg)")
                    print(f"   Total Water: {water_footprint['total_water_liters']:.3f} L ({water_footprint['total_water_liters']/0.5:.1f} bottles)")
                    print(f"   Accuracy: {accuracy_metrics.get('accuracy', accuracy_metrics.get('bleu_score', 'N/A'))}")
                    print(f"   Environmental Score: {environmental_metrics.get('integrated_assessment', {}).get('overall_efficiency_score', 'N/A')}")
                    print(f"   Water Efficiency: {environmental_metrics.get('water_analysis', {}).get('water_efficiency', 'N/A')}")
                    print(f"   Samples Processed: {len(predictions)}/{self._current_dataset_size}")

                except Exception as e:
                    print(f"❌ Error evaluating {framework} {model_name} on {dataset_name}: {e}")

                    # Log error to results for debugging
                    error_key = f"ERROR__{framework}__{model_name}__{dataset_name}"
                    self.results[error_key] = {
                        "error": str(e),
                        "framework": framework,
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "timestamp": time.time(),
                        "hardware_profile": self.capabilities.__dict__
                    }

                    continue
                finally:
                    # Ensure primary tracker is stopped
                    self.primary_tracker.stop()
                    
                    # Aggressive memory cleanup after each combination
                    self._cleanup_memory()  
                    
                    
        # Finalize wandb run if enabled\

        # Generate final comprehensive report
        final_report = self._generate_final_report()

        # Save results
        self.save_results()

        print(f"\n🎉 Comprehensive Environmental Analysis Complete!")
        print(f"{'='*80}")
        print(f"✅ Evaluated {len([k for k in self.results.keys() if not k.startswith('ERROR')])} combinations successfully")
        print(f"❌ {len([k for k in self.results.keys() if k.startswith('ERROR')])} combinations failed")
        print(f"💾 Results saved to '{self.config.get('output_dir', '.')}/ml_ecolyzer_results.json'")
        print(f"📊 Final report available in results['final_report']")
        if self.wandb_enabled:
            print(f"🔗 wandb dashboard: {wandb.run.url}")
        print(f"{'='*80}")

        return self.results
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Log memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"🔧 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")    

    def _monitor_resources(self) -> Dict[str, float]:
        """Monitor system resources"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            }
            
            # GPU Memory
            if torch.cuda.is_available():
                resources.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1e9,
                    "gpu_utilization": self._get_gpu_utilization()
                })
                
            return resources
        except Exception as e:
            return {"error": str(e)}

    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            gpu_pressure = False
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_pressure = (allocated / total) > 0.8
                
            return memory.percent > 85 or gpu_pressure
        except:
            return False

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report with analysis and insights including water footprint"""
        successful_results = {k: v for k, v in self.results.items() if not k.startswith('ERROR')}

        if not successful_results:
            return {"error": "No successful evaluations to analyze"}

        # Aggregate statistics
        total_emissions = sum(r["emissions_analysis"]["total_kg_co2"] for r in successful_results.values())
        avg_emissions = total_emissions / len(successful_results)
        
        # Water footprint statistics
        total_water = sum(r["water_analysis"]["total_water_liters"] for r in successful_results.values())
        avg_water = total_water / len(successful_results)

        # Framework analysis
        framework_stats = {}
        for result in successful_results.values():
            framework = result.get("framework", "unknown")
            if framework not in framework_stats:
                framework_stats[framework] = {
                    "count": 0, 
                    "total_emissions": 0, 
                    "total_water": 0,
                    "efficiency_scores": [],
                    "water_efficiency_scores": []
                }
            
            framework_stats[framework]["count"] += 1
            framework_stats[framework]["total_emissions"] += result["emissions_analysis"]["total_kg_co2"]
            framework_stats[framework]["total_water"] += result["water_analysis"]["total_water_liters"]
            
            efficiency = result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0)
            framework_stats[framework]["efficiency_scores"].append(efficiency)
            
            water_efficiency = result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", 0.5)
            framework_stats[framework]["water_efficiency_scores"].append(water_efficiency)

        # Environmental efficiency analysis
        efficiency_scores = [
            r["environmental_assessment"].get("integrated_assessment", {}).get("overall_efficiency_score", 0)
            for r in successful_results.values()
        ]

        final_report = {
            "analysis_summary": {
                "total_evaluations": len(successful_results),
                "failed_evaluations": len([k for k in self.results.keys() if k.startswith('ERROR')]),
                "total_co2_emissions_kg": total_emissions,
                "average_co2_per_evaluation_kg": avg_emissions,
                "total_water_liters": total_water,
                "average_water_per_evaluation_liters": avg_water,
                "water_bottles_equivalent": total_water / 0.5,
                "hardware_category": self.capabilities.device_category,
                "monitoring_capabilities": self.capabilities.monitoring_methods,
                "frameworks_analyzed": list(framework_stats.keys()),
                "water_intensity_factor": self.capabilities.water_intensity_factor,
                "region": self.capabilities.region
            },

            "framework_analysis": {
                framework: {
                    "evaluation_count": stats["count"],
                    "total_emissions_kg": stats["total_emissions"],
                    "avg_emissions_kg": stats["total_emissions"] / stats["count"],
                    "total_water_liters": stats["total_water"],
                    "avg_water_liters": stats["total_water"] / stats["count"],
                    "avg_efficiency_score": sum(stats["efficiency_scores"]) / len(stats["efficiency_scores"]) if stats["efficiency_scores"] else 0,
                    "avg_water_efficiency_score": sum(stats["water_efficiency_scores"]) / len(stats["water_efficiency_scores"]) if stats["water_efficiency_scores"] else 0
                }
                for framework, stats in framework_stats.items()
            },

            "environmental_analysis": {
                "overall_efficiency_score": np.mean(efficiency_scores) if efficiency_scores else 0,
                "efficiency_score_std": np.std(efficiency_scores) if efficiency_scores else 0,
                "best_efficiency": max(efficiency_scores) if efficiency_scores else 0,
                "worst_efficiency": min(efficiency_scores) if efficiency_scores else 0,
                "water_intensity_used": self.capabilities.water_intensity_factor,
                "region_analyzed": self.capabilities.region
            },

            "water_impact_analysis": {
                "total_water_footprint_liters": total_water,
                "average_water_per_model_liters": avg_water,
                "water_bottles_equivalent": total_water / 0.5,
                "water_gallons_equivalent": total_water / 3.785,
                "regional_water_intensity": self.capabilities.water_intensity_factor,
                "water_efficiency_recommendations": self._generate_water_recommendations(total_water, avg_water)
            },

            "methodology_validation": {
                "monitoring_method_coverage": self.capabilities.monitoring_methods,
                "water_tracking_enabled": True,
                "scientific_standards_applied": [
                    "IEEE 754 floating-point standards",
                    "JEDEC battery specifications",
                    "ASHRAE thermal guidelines",
                    "CodeCarbon emissions methodology",
                    "Regional water intensity factors",
                    "Data center PUE standards"
                ]
            }
        }

        # Add to results
        self.results["final_report"] = final_report

        return final_report

    def _generate_water_recommendations(self, total_water: float, avg_water: float) -> List[str]:
        """Generate water-specific recommendations"""
        recommendations = []
        
        if total_water > 5.0:  # More than 5 liters total
            recommendations.append("Consider model quantization to reduce water footprint")
        
        if avg_water > 1.0:  # More than 1 liter per evaluation
            recommendations.append("High water usage per evaluation - optimize model size and inference efficiency")
        
        if self.capabilities.device_category == "datacenter" and total_water > 10.0:
            recommendations.append("Data center water usage is high - investigate cooling system efficiency")
        
        if self.capabilities.water_intensity_factor > 3.0:
            recommendations.append(f"Regional water intensity is high ({self.capabilities.water_intensity_factor:.1f} L/kWh) - consider renewable energy sources")
        
        return recommendations

    def save_results(self, output_dir: str = None, filename: str = None) -> None:
        """Save comprehensive results with metadata and citations"""
        output_dir = output_dir or self.config.get("output_dir", "results")
        filename = filename or "ml_ecolyzer_comprehensive_results.json"

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Add metadata to results
        results_with_metadata = {
            "framework_metadata": {
                "framework_version": "ML-EcoLyzer v1.0 - Multi-Framework Environmental Impact Analysis",
                "timestamp": time.time(),
                "hardware_profile": self.capabilities.__dict__,
                "configuration": self.config,
                "total_evaluations": len([k for k in self.results.keys() if not k.startswith('ERROR')]),
                "failed_evaluations": len([k for k in self.results.keys() if k.startswith('ERROR')]),
                "monitoring_methods_used": self.capabilities.monitoring_methods,
                "water_tracking_enabled": True,
                "scientific_standards": [
                    "IEEE 754-2019: Floating-point arithmetic standard",
                    "JEDEC No. 21-C: Li-ion battery specifications",
                    "ASHRAE TC 9.9: Data center thermal guidelines",
                    "CodeCarbon: Carbon emissions tracking methodology",
                    "Regional water intensity factors: Data center and energy mix analysis"
                ]
            },
            "results": self.results
        }

        with open(filepath, "w") as f:
            json.dump(results_with_metadata, f, indent=2, default=str)

        print(f"💾 Comprehensive results saved to: {filepath}")

        # Also save a summary CSV for easy analysis
        self._save_summary_csv(output_dir)

    def _save_summary_csv(self, output_dir: str):
        """Save summary results in CSV format for easy analysis"""
        try:
            import pandas as pd

            summary_data = []
            for key, result in self.results.items():
                if key.startswith('ERROR') or key == 'final_report':
                    continue

                summary_row = {
                    "framework": result.get("framework", ""),
                    "model_name": result.get("model_name", ""),
                    "dataset_name": result.get("dataset_name", ""),
                    "task": result.get("task", ""),
                    "device_category": result.get("hardware_profile", {}).get("device_category", ""),
                    "total_co2_kg": result.get("emissions_analysis", {}).get("total_kg_co2", 0),
                    "total_water_liters": result.get("water_analysis", {}).get("total_water_liters", 0),
                    "water_bottles_equivalent": result.get("water_analysis", {}).get("water_equivalent_bottles", 0),
                    "efficiency_score": result.get("environmental_assessment", {}).get("integrated_assessment", {}).get("overall_efficiency_score", 0),
                    "water_efficiency_score": result.get("environmental_assessment", {}).get("water_analysis", {}).get("water_efficiency", 0),
                    "accuracy": result.get("accuracy_metrics", {}).get("accuracy", result.get("accuracy_metrics", {}).get("bleu_score", 0)),
                    "samples_processed": result.get("successful_predictions", 0),
                    "dataset_size": result.get("dataset_size", 0),
                    "region": result.get("hardware_profile", {}).get("region", ""),
                    "water_intensity_factor": result.get("hardware_profile", {}).get("water_intensity_factor", 0)
                }
                summary_data.append(summary_row)

            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_path = os.path.join(output_dir, "ml_ecolyzer_summary.csv")
                df.to_csv(csv_path, index=False)
                print(f"📊 Summary CSV saved to: {csv_path}")

        except ImportError:
            print("⚠️ pandas not available - skipping CSV export")
        except Exception as e:
            print(f"⚠️ CSV export failed: {e}")