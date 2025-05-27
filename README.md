# ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ml-ecolyzer.svg)](https://badge.fury.io/py/ml-ecolyzer)

A scientific framework for analyzing and quantifying the environmental impact of machine learning systems with adaptive monitoring across diverse hardware configurations.

## 🌟 Key Features

- **Comprehensive Environmental Analysis**: Carbon emissions, energy consumption, thermal impact, and resource utilization
- **Adaptive Hardware Detection**: Automatically optimizes for datacenter GPUs, laptops, mobile devices, and edge hardware
- **Scientific Rigor**: Based on IEEE, JEDEC, and ASHRAE standards with peer-reviewed methodology
- **Model Optimization Insights**: Quantization analysis and efficiency recommendations
- **Cross-Platform Compatibility**: Works seamlessly across Windows, macOS, and Linux
- **Research-Grade Accuracy**: Designed for reproducible scientific studies and publications
- **Integration Ready**: wandb support, multiple export formats, and extensible APIs

## 🔬 Scientific Foundation

ML-EcoLyzer is built on established scientific standards and research methodologies:

- **IEEE 754-2019**: Floating-point arithmetic standards
- **JEDEC No. 21-C**: Li-ion battery specifications  
- **ASHRAE TC 9.9**: Data center thermal guidelines
- **CodeCarbon**: Carbon emissions tracking methodology

### 📚 References

- Strubell et al. (2019) "Energy and Policy Considerations for Deep Learning in NLP"
- Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training"
- Schwartz et al. (2020) "Green AI" (Communications of the ACM)
- Henderson et al. (2020) "Towards the Systematic Reporting of Energy and Carbon Footprints"

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install ml-ecolyzer

# With all optional dependencies
pip install ml-ecolyzer[all]

# For GPU monitoring
pip install ml-ecolyzer[gpu]

# For audio/vision models
pip install ml-ecolyzer[audio,vision]

# Development installation
pip install ml-ecolyzer[dev]
```

### Basic Usage

```python
from mlecolyzer import EcoLyzer

# Simple environmental impact analysis
config = {
    "project": "sustainability_study",
    "models": [
        {"name": "gpt2", "task": "text"},
        {"name": "distilbert-base-uncased", "task": "text"}
    ],
    "datasets": [
        {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text"},
        {"name": "imdb", "task": "text"}
    ]
}

# Run analysis
analyzer = EcoLyzer(config)
results = analyzer.run()

# Access results
print(f"Total CO2 emissions: {results['final_report']['analysis_summary']['total_co2_emissions_kg']:.6f} kg")
```

### Comprehensive Research Study

```python
from mlecolyzer import run_comprehensive_analysis

# Large-scale research configuration
research_config = {
    "project": "comprehensive_ml_carbon_study",
    "models": [
        {"name": "gpt2", "task": "text"},
        {"name": "microsoft/DialoGPT-medium", "task": "text"},
        {"name": "facebook/bart-base", "task": "text"}
    ],
    "datasets": [
        {"name": "wikitext", "subset": "wikitext-2-raw-v1", "task": "text", "limit": 1000},
        {"name": "squad", "task": "text", "limit": 500},
        {"name": "imdb", "task": "text", "limit": 800}
    ],
    "enable_wandb": True,
    "enable_quantization_analysis": True
}

# Run comprehensive analysis
results = run_comprehensive_analysis(research_config)

# Analyze results
from mlecolyzer.core.research import analyze_research_results
analysis = analyze_research_results(results)
print("Key findings:", analysis["insights"])
```

### Command Line Interface

```bash
# Quick analysis
mlecolyzer analyze --model gpt2 --dataset wikitext --project my_study

# Comprehensive research
mlecolyzer research --config research_config.yaml

# Generate configuration template
mlecolyzer init --template research --output my_config.yaml

# Check system capabilities
mlecolyzer info
```

## 📊 Environmental Metrics

ML-EcoLyzer provides comprehensive environmental impact analysis:

### 1. Carbon Footprint Analysis
- **Direct CO2 emissions** from computational energy consumption
- **PUE-adjusted emissions** accounting for cooling and infrastructure
- **Regional carbon intensity** factors for accurate footprint calculation

### 2. Energy Consumption Profiling
- **Real-time power monitoring** across CPU, GPU, and system components
- **Energy efficiency analysis** per model parameter and dataset sample
- **Optimization opportunity identification** for sustainable deployment

### 3. Thermal Impact Assessment
- **Heat generation measurement** and thermal efficiency analysis
- **Cooling requirement estimation** for different deployment scenarios
- **Thermal throttling impact** on performance and sustainability

### 4. Resource Utilization Optimization
- **Hardware efficiency analysis** across different model architectures
- **Memory usage optimization** recommendations
- **Batch size and parallelization** impact on environmental footprint

### 5. Model Optimization Insights
- **Quantization benefits analysis** for reduced precision deployment
- **Model compression impact** on accuracy vs. sustainability trade-offs
- **Edge deployment feasibility** assessment

## 🔧 Configuration

### Model Configuration

```python
model_config = {
    "name": "microsoft/DialoGPT-medium",
    "task": "text",
    "model_type": "causal_lm",
    "max_length": 512,
    "quantization": {
        "enabled": True,
        "method": "dynamic",
        "target_dtype": "int8"
    }
}
```

### Dataset Configuration

```python
dataset_config = {
    "name": "squad",
    "task": "text",
    "subset": "v1.1",
    "split": "validation",
    "fallback_splits": ["train", "test"],
    "limit": 1000,
    "label_key": "text"
}
```

### Advanced Configuration

```yaml
project: "advanced_sustainability_analysis"

models:
  - name: "gpt2"
    task: "text"
    quantization:
      enabled: true
      method: "dynamic"
      target_dtype: "int8"

datasets:
  - name: "wikitext"
    task: "text"
    limit: 1000

# Environmental monitoring
monitoring:
  duration_seconds: 600
  frequency_hz: 2.0
  enable_quantization_analysis: true
  enable_wandb: true
  wandb_project: "ml_sustainability_research"

# Hardware optimization
hardware:
  device_profile: "auto"  # auto, datacenter, desktop, mobile, edge
  precision: "auto"       # auto, float32, float16, bfloat16
  memory_optimization: true

# Output configuration
output:
  output_dir: "./sustainability_results"
  export_formats: ["json", "csv", "html"]
  log_level: "INFO"
```

## 🏗️ Architecture

ML-EcoLyzer uses a modular, research-oriented architecture:

```
mlecolyzer/
├── core/                   # Core analysis functionality
│   ├── runner.py          # EcoLyzer main class
│   ├── research.py        # Comprehensive research functions
│   └── config.py          # Configuration management
├── monitoring/            # Environmental monitoring
│   ├── environmental.py   # Environmental metrics collection
│   ├── hardware.py        # Hardware detection and capabilities
│   └── tracking.py        # Real-time tracking utilities
├── models/                # Model loading and optimization
│   ├── loader.py          # Optimized model loading
│   └── inference.py       # Efficient inference routines
├── datasets/              # Dataset handling
│   ├── loader.py          # Robust dataset loading
│   └── processor.py       # Data preprocessing and validation
├── metrics/               # Comprehensive metrics
│   ├── accuracy.py        # Task-specific accuracy metrics
│   └── environmental.py   # Environmental impact calculations
└── utils/                 # Research utilities
    ├── validation.py      # Configuration validation
    └── helpers.py         # Logging and utility functions
```

## 🔍 Hardware Support

ML-EcoLyzer automatically detects and optimizes for various hardware configurations:

### 🏢 Datacenter
- High-end GPUs (A100, H100, V100)
- Multi-GPU setups with advanced monitoring
- PUE-adjusted emissions calculations
- Comprehensive power and thermal analysis

### 🖥️ Desktop
- Consumer GPUs (RTX, GTX series)
- CPU-only configurations
- Standard power monitoring
- Thermal efficiency analysis

### 📱 Mobile/Edge
- ARM processors and mobile SoCs
- Battery life impact analysis
- Memory-optimized monitoring
- Edge deployment recommendations

### ☁️ Cloud
- AWS, GCP, Azure instance optimization
- Regional carbon intensity integration
- Spot instance efficiency analysis
- Cost-aware sustainability metrics

## 📈 Results and Analysis

### Individual Analysis Results
```python
result = {
    "model_name": "gpt2",
    "dataset_name": "wikitext",
    "emissions_analysis": {
        "total_kg_co2": 0.001234,
        "emissions_per_sample": 0.000001234,
        "pue_adjusted_kg_co2": 0.001481
    },
    "accuracy_metrics": {
        "bleu_score": 23.45,
        "perplexity": 18.2
    },
    "environmental_assessment": {
        "overall_efficiency_score": 0.78,
        "power_analysis": {...},
        "thermal_analysis": {...},
        "optimization_recommendations": [...]
    }
}
```

### Research Study Reports
```python
final_report = {
    "analysis_summary": {
        "total_evaluations": 12,
        "total_co2_emissions_kg": 0.0147,
        "average_efficiency_score": 0.72
    },
    "optimization_insights": [
        "BERT-family models show 23% better efficiency than GPT-family",
        "Quantization reduces emissions by 31% with <2% accuracy loss",
        "Edge deployment viable for models under 1B parameters"
    ],
    "research_contributions": [
        "First comprehensive analysis of Transformer efficiency across tasks",
        "Novel quantization impact methodology for environmental assessment"
    ]
}
```

## 🧪 Research Examples

### Example 1: Model Architecture Comparison

```python
from mlecolyzer import EcoLyzer

config = {
    "project": "architecture_comparison_study",
    "models": [
        {"name": "bert-base-uncased", "task": "text"},
        {"name": "roberta-base", "task": "text"},
        {"name": "distilbert-base-uncased", "task": "text"},
        {"name": "albert-base-v2", "task": "text"}
    ],
    "datasets": [
        {"name": "glue", "subset": "sst2", "task": "text", "limit": 1000}
    ]
}

analyzer = EcoLyzer(config)
results = analyzer.run()

# Compare architectures
for key, result in results.items():
    if not key.startswith('ERROR'):
        model = result['model_name']
        co2 = result['emissions_analysis']['total_kg_co2']
        efficiency = result['environmental_assessment']['overall_efficiency_score']
        print(f"{model}: {co2:.6f} kg CO2, efficiency: {efficiency:.3f}")
```

### Example 2: Quantization Impact Study

```python
config = {
    "project": "quantization_impact_study",
    "models": [
        {"name": "gpt2", "task": "text"},
        {"name": "gpt2", "task": "text", "quantization": {"enabled": True, "method": "dynamic"}},
        {"name": "gpt2", "task": "text", "quantization": {"enabled": True, "method": "static"}}
    ],
    "datasets": [{"name": "wikitext", "task": "text", "limit": 500}],
    "enable_quantization_analysis": True
}

results = run_comprehensive_analysis(config)
```

### Example 3: Cross-Platform Deployment Analysis

```python
config = {
    "project": "deployment_platform_study",
    "models": [{"name": "distilbert-base-uncased", "task": "text"}],
    "datasets": [{"name": "imdb", "task": "text", "limit": 200}],
    "hardware": {
        "device_profile": "mobile"  # Test mobile deployment
    }
}

analyzer = EcoLyzer(config)
results = analyzer.run()

# Analyze deployment feasibility
deployment_analysis = results['final_report']['deployment_recommendations']
print("Mobile deployment insights:", deployment_analysis)
```

## 🛠️ Development and Research

### Setting up Research Environment

```bash
# Clone repository
git clone https://github.com/ml-ecolyzer/ml-ecolyzer.git
cd ml-ecolyzer

# Create research environment
python -m venv research_env
source research_env/bin/activate

# Install for research
pip install -e .[dev,all]

# Run tests
pytest tests/ --cov=mlecolyzer

# Run research experiments
python research_scripts/comprehensive_study.py
```

### Contributing to Research

We welcome research contributions! Areas of active development:

- **Multi-modal environmental impact analysis**
- **Real-time carbon intensity integration**
- **Advanced quantization impact modeling**
- **Edge device optimization strategies**
- **Distributed training environmental assessment**

See our [Contributing Guide](CONTRIBUTING.md) for research collaboration guidelines.

## 📄 Citation

If you use ML-EcoLyzer in your research, please cite our paper:

```bibtex
@inproceedings{mlecolyzer2024,
  title={ML-EcoLyzer: Comprehensive Environmental Impact Analysis for Machine Learning Systems},
  author={ML-EcoLyzer Research Team},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://github.com/ml-ecolyzer/ml-ecolyzer}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [https://ml-ecolyzer.readthedocs.io](https://ml-ecolyzer.readthedocs.io)
- **Paper**: [https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)
- **PyPI**: [https://pypi.org/project/ml-ecolyzer](https://pypi.org/project/ml-ecolyzer)
- **GitHub**: [https://github.com/ml-ecolyzer/ml-ecolyzer](https://github.com/ml-ecolyzer/ml-ecolyzer)
- **Issues**: [https://github.com/ml-ecolyzer/ml-ecolyzer/issues](https://github.com/ml-ecolyzer/ml-ecolyzer/issues)

## 💬 Support

- **Documentation**: [Comprehensive research documentation](https://ml-ecolyzer.readthedocs.io)
- **GitHub Issues**: Report bugs, request features, or ask research questions
- **Research Discussions**: Join our community for methodology discussions
- **Email**: contact@ml-ecolyzer.org

---

**ML-EcoLyzer** - Advancing sustainable AI through rigorous environmental impact analysis. 🌱🔬