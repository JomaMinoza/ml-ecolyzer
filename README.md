# ML-EcoLyzer: Machine Learning Environmental Impact Analysis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ml-ecolyzer.svg)](https://badge.fury.io/py/ml-ecolyzer)

**ML-EcoLyzer** is a reproducible benchmarking and analysis framework for quantifying the environmental cost of machine learning inference. It supports modern transformers, vision models, and classical ML pipelines, adaptable to both edge and datacenter-scale deployments.

---

![ML-EcoLyzer Overview](docs/assets/ml_ecolyzer.png)  
*Environmental profiling across tasks, models, and hardware tiers.*

---

## 🌍 Key Features

- **Inference-Centric Analysis**: Quantifies CO₂ emissions, energy use, and water impact from real-time inference
- **Cross-Hardware Profiling**: Supports A100, T4, RTX, GTX, and CPU-only setups
- **Model-Agnostic Framework**: Runs LLMs, ViTs, audio models, and traditional ML
- **ESS Metric**: Introduces the Environmental Sustainability Score for normalized emissions comparison
- **Quantization Insights**: Analyzes FP16 and INT8 savings for sustainable deployment
- **Frequency-Aware Monitoring**: Adjusts sampling dynamically for short and long-running workloads
- **Lightweight and Extensible**: Runs on mobile, edge, and low-resource devices

---

## 📊 What It Measures

### ✅ CO₂ Emissions  
- Based on PUE, regional carbon intensity, and power consumption  
- Adaptive to cloud, desktop, or edge scenarios

### ✅ Energy Usage  
- Instantaneous power profiling via NVIDIA-SMI, `psutil`, or RAPL  
- Sample-level granularity for each inference configuration

### ✅ Water Footprint  
- Derived from power-to-water coefficients by tier (e.g., datacenter vs. mobile)

### ✅ Environmental Sustainability Score (ESS) Metric  

$$\text{ESS} = \frac{\text{Effective Parameters (M)}}{\text{CO₂ (g)}}$$

- A normalized environmental efficiency metric for sustainable ML comparisons

---

## 🔧 Installation

```bash
pip install ml-ecolyzer
```

---

## 🚀 Quick Example

```python
from mlecolyzer import EcoLyzer

config = {
    "project": "sustainability_demo",
    "models": [{"name": "gpt2", "task": "text"}],
    "datasets": [{"name": "wikitext", "task": "text"}]
}

eco = EcoLyzer(config)
results = eco.run()

print(f"CO₂: {results['final_report']['analysis_summary']['total_co2_emissions_kg']:.6f} kg")
```

---

## 📚 Scientific Foundation

Built on rigorously defined environmental assessment literature and standards:

- IEEE 754 (numeric precision)
- ASHRAE TC 9.9 (thermal/infra cooling)
- JEDEC JESD51 (thermal/power envelopes)
- Strubell et al. (2019), Patterson et al. (2021), Henderson et al. (2020), Lacoste et al. (2019)

---

## 🔬 Benchmark Coverage

- 1,500+ inference runs
- 4 hardware tiers: GTX 1650, RTX 4090, Tesla T4, A100
- Tasks: text, audio, vision, classification, regression
- Model families: GPT-2, OPT, Qwen, LLaMA, Phi, Whisper, ViT, etc.
- Precisions: FP32, FP16, INT8

---

## 🛠️ Configuration Template

```yaml
project: "ml_sustainability_benchmark"

models:
  - name: "facebook/opt-2.7b"
    task: "text"
    quantization:
      enabled: true
      method: "dynamic"
      target_dtype: "int8"

datasets:
  - name: "wikitext"
    task: "text"
    limit: 1000

monitoring:
  frequency_hz: 5
  enable_quantization_analysis: true

hardware:
  device_profile: "auto"

output:
  export_formats: ["json", "csv"]
  output_dir: "./results"
```

---

## 🧪 Research Insights

### Quantization Efficiency

```text
INT8 models show up to 74% higher ESS than FP32 equivalents.
```

### Hardware Utilization

```text
A100 performs worst in ESS when underutilized; RTX/T4 yield better emissions-per-parameter for single-batch workloads.
```

### Task-Wise Trends

```text
Traditional models like SVC or Logistic Regression incur high ECEP due to small parameter count, despite low energy.
```

---

## 📜 Citation

```bibtex
@inproceedings{mlecolyzer2025,
  title={ML-EcoLyzer: Comprehensive Environmental Impact Analysis for Machine Learning Systems},
  author={Minoza, Jose Marie Antonio and Laylo, Rex Gregor and Villarin, Christian and Ibanez, Sebastian},
  booktitle={Proceedings of the Asian Conference on Machine Learning (ACML)},
  year={2025}
}
```

---
**ML-EcoLyzer** — Advancing sustainable inference in resource-constrained and production-scale deployments. 🌱
