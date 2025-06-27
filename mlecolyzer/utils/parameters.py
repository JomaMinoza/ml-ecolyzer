"""
Parameter Estimation and ESS Calculation Module

This module provides comprehensive parameter counting and Environmental Sustainability Score (ESS)
calculation for ML models across multiple frameworks with multi-tier detection strategy.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParameterResult:
    """Result of parameter counting with metadata"""
    count: int
    source: str  # "config", "pytorch", "huggingface", "fallback"
    confidence: str  # "high", "medium", "low"
    details: Dict[str, Any]


class ParameterEstimator:
    """
    Multi-tier parameter estimation system with ESS calculation
    
    Detection Strategy:
    1. User Config (if provided)
    2. PyTorch Introspection (for loaded models)
    3. HuggingFace Config (from model config)
    4. Name-based Fallback (sophisticated pattern matching)
    """

    def __init__(self):
        """Initialize parameter estimator"""
        self.ess_thresholds = {
            "excellent": 1.0,     # ESS > 1.0
            "efficient": 0.5,     # ESS > 0.5
            "moderate": 0.1,      # ESS > 0.1
            "poor": 0.01,         # ESS > 0.01
            "very_poor": 0.0      # ESS <= 0.01
        }

    def count_parameters(self, model: Any, model_config: Dict[str, Any], 
                        model_name: str, framework: str) -> ParameterResult:
        """
        Multi-tier parameter counting with comprehensive fallback
        
        Args:
            model: Loaded model object
            model_config: Model configuration dictionary
            model_name: Model name/identifier
            framework: Framework type (huggingface, sklearn, pytorch)
            
        Returns:
            ParameterResult with count, source, and metadata
        """
        
        # Tier 1: Explicit user configuration
        if model_config.get("parameter_count"):
            return ParameterResult(
                count=int(model_config["parameter_count"]),
                source="config",
                confidence="high",
                details={"user_specified": True}
            )
        
        # Tier 2: PyTorch introspection (most accurate for loaded models)
        if framework in ["pytorch", "huggingface"] and hasattr(model, 'parameters'):
            try:
                param_result = self._count_pytorch_parameters(model)
                if param_result.count > 0:
                    return param_result
            except Exception as e:
                print(f"   ⚠️ PyTorch parameter counting failed: {e}")
        
        # Tier 3: HuggingFace config introspection
        if framework == "huggingface" and hasattr(model, 'config'):
            try:
                param_result = self._count_huggingface_parameters(model)
                if param_result.count > 0:
                    return param_result
            except Exception as e:
                print(f"   ⚠️ HuggingFace config parameter counting failed: {e}")
        
        # Tier 4: Name-based fallback estimation
        param_result = self._estimate_parameters_from_name(model_name, framework)
        
        return param_result

    def _count_pytorch_parameters(self, model) -> ParameterResult:
        """Count parameters from PyTorch model"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            details = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                "parameter_breakdown": self._get_layer_breakdown(model)
            }
            
            return ParameterResult(
                count=total_params,
                source="pytorch",
                confidence="high",
                details=details
            )
            
        except Exception as e:
            return ParameterResult(count=0, source="pytorch_failed", confidence="low", details={"error": str(e)})

    def _get_layer_breakdown(self, model) -> Dict[str, int]:
        """Get parameter breakdown by layer type"""
        try:
            breakdown = {}
            for name, param in model.named_parameters():
                layer_type = name.split('.')[0] if '.' in name else name
                if layer_type not in breakdown:
                    breakdown[layer_type] = 0
                breakdown[layer_type] += param.numel()
            
            # Sort by parameter count
            return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}

    def _count_huggingface_parameters(self, model) -> ParameterResult:
        """Count parameters from HuggingFace model config"""
        try:
            config = model.config
            
            # Try to get parameter count from config
            if hasattr(config, 'num_parameters'):
                param_count = config.num_parameters
            else:
                # Estimate from architecture parameters
                param_count = self._estimate_from_hf_config(config)
            
            details = {
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                "intermediate_size": getattr(config, 'intermediate_size', None),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
                "model_type": getattr(config, 'model_type', None)
            }
            
            return ParameterResult(
                count=param_count,
                source="huggingface",
                confidence="medium",
                details=details
            )
            
        except Exception as e:
            return ParameterResult(count=0, source="huggingface_failed", confidence="low", details={"error": str(e)})

    def _estimate_from_hf_config(self, config) -> int:
        """Estimate parameters from HuggingFace config attributes"""
        try:
            # For transformer models, rough estimation
            vocab_size = getattr(config, 'vocab_size', 50000)
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 12)
            intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Rough parameter estimation for transformer
            # Embedding layer
            embedding_params = vocab_size * hidden_size
            
            # Each transformer layer
            attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
            ffn_params = hidden_size * intermediate_size + intermediate_size * hidden_size
            layer_params = attention_params + ffn_params
            
            # Total estimation
            total_params = embedding_params + (num_layers * layer_params)
            
            return int(total_params)
            
        except:
            return 0

    def _estimate_parameters_from_name(self, model_name: str, framework: str) -> ParameterResult:
        """Sophisticated name-based parameter estimation using the enhanced logic from your analysis"""
        
        if pd.isna(model_name) or model_name is None:
            return ParameterResult(count=10_000_000, source="fallback", confidence="low", details={"reason": "null_model_name"})
        
        name = model_name.lower()
        
        # ================================
        # SKLEARN/TRADITIONAL ML MODELS
        # ================================
        
        if framework == "sklearn" or 'saved_models' in name and '.joblib' in name:
            param_count = self._estimate_sklearn_parameters(name)
            return ParameterResult(
                count=param_count,
                source="fallback",
                confidence="medium",
                details={"framework": "sklearn", "estimation_method": "algorithm_specific"}
            )
        
        # ================================
        # AUDIO MODELS (Speech Recognition, TTS, etc.)
        # ================================
        
        audio_models = {
            'whisper-tiny': 39_000_000,
            'whisper-base': 74_000_000, 
            'whisper-small': 244_000_000,
            'whisper-medium': 769_000_000,
            'whisper-large': 1_550_000_000,
            'wav2vec2-base': 95_000_000,
            'wav2vec2-large': 317_000_000,
            'hubert-base': 95_000_000,
            'hubert-large': 317_000_000,
            'wavlm-base': 95_000_000
        }
        
        for model_key, params in audio_models.items():
            if model_key in name:
                return ParameterResult(
                    count=params,
                    source="fallback", 
                    confidence="high",
                    details={"model_family": "audio", "model_type": model_key}
                )
        
        # ================================
        # TEXT LANGUAGE MODELS (LLMs)
        # ================================
        
        # GPT-2 models (must come before generic 'gpt' checks)
        if 'gpt2-xl' in name:
            param_count = 1_500_000_000
        elif 'gpt2-large' in name:
            param_count = 774_000_000
        elif 'gpt2-medium' in name:
            param_count = 355_000_000
        elif 'distilgpt2' in name:
            param_count = 82_000_000
        elif name == 'gpt2' or name.endswith('/gpt2'):
            param_count = 124_000_000
        
        # Gemma models
        elif 'gemma-2-2b' in name or 'gemma-2b' in name:
            param_count = 2_000_000_000
        elif 'gemma-2-9b' in name or 'gemma-9b' in name:
            param_count = 9_000_000_000
        elif 'gemma-7b' in name:
            param_count = 7_000_000_000
        
        # Qwen models
        elif 'qwen2-0.5b' in name or 'qwen1.5-0.5b' in name:
            param_count = 500_000_000
        elif 'qwen2-1.5b' in name or 'qwen1.5-1.8b' in name:
            param_count = 1_500_000_000
        elif 'qwen2-7b' in name or 'qwen1.5-4b' in name:
            param_count = 7_000_000_000
        elif 'qwen1.5-7b' in name:
            param_count = 7_000_000_000
        
        # Phi models
        elif 'phi-1_5' in name or 'phi-1.5' in name:
            param_count = 1_300_000_000
        elif 'phi-2' in name and 'phi-3' not in name:
            param_count = 2_700_000_000
        elif 'phi-3-mini' in name:
            param_count = 3_820_000_000
        elif 'phi-3-medium' in name:
            param_count = 14_000_000_000
        
        # LLaMA models
        elif 'llama-2-7b' in name or 'llama-3-8b' in name:
            param_count = 7_000_000_000
        elif 'llama-2-13b' in name:
            param_count = 13_000_000_000
        
        # OPT models
        elif 'opt-350m' in name:
            param_count = 350_000_000
        elif 'opt-1.3b' in name:
            param_count = 1_300_000_000
        elif 'opt-2.7b' in name:
            param_count = 2_700_000_000
        elif 'opt-6.7b' in name:
            param_count = 6_700_000_000
        
        # Pythia models
        elif 'pythia-70m' in name:
            param_count = 70_000_000
        elif 'pythia-160m' in name:
            param_count = 160_000_000
        elif 'pythia-410m' in name:
            param_count = 410_000_000
        
        # DialoGPT models
        elif 'dialogpt-small' in name:
            param_count = 117_000_000
        elif 'dialogpt-medium' in name:
            param_count = 355_000_000
        
        # Other models
        elif 'biogpt' in name:
            param_count = 1_500_000_000
        elif 'godel' in name:
            param_count = 220_000_000
        elif 'olmo-1b' in name:
            param_count = 1_000_000_000
        elif 'olmo-7b' in name:
            param_count = 7_000_000_000
        
        # ================================
        # VISION MODELS
        # ================================
        
        elif 'vit-base' in name:
            param_count = 86_000_000
        elif 'resnet-50' in name:
            param_count = 25_000_000
        elif 'deit-base' in name:
            param_count = 86_000_000
        elif 'swin-base' in name:
            param_count = 88_000_000
        elif 'efficientnet-b3' in name:
            param_count = 12_000_000
        
        # Image Generation Models
        elif 'stable-diffusion-v1-5' in name:
            param_count = 860_000_000
        elif 'playground-v2' in name:
            param_count = 2_500_000_000
        elif 'opendalle' in name:
            param_count = 1_300_000_000
        elif 'kandinsky' in name:
            param_count = 1_200_000_000
        elif 'openjourney' in name:
            param_count = 860_000_000
        
        # Size-based detection (fallback)
        elif '70m' in name:
            param_count = 70_000_000
        elif '160m' in name:
            param_count = 160_000_000
        elif '350m' in name:
            param_count = 350_000_000
        elif '410m' in name:
            param_count = 410_000_000
        elif '1b' in name and '11b' not in name:
            param_count = 1_000_000_000
        elif '2b' in name:
            param_count = 2_000_000_000
        elif '7b' in name or '8b' in name:
            param_count = 7_000_000_000
        elif '9b' in name:
            param_count = 9_000_000_000
        elif '13b' in name:
            param_count = 13_000_000_000
        
        # Final fallback based on framework
        elif framework == 'sklearn':
            param_count = 100_000
        elif framework == 'huggingface':
            param_count = 1_000_000_000
        else:
            param_count = 10_000_000
        
        model_family = self._get_model_family(name)
        confidence = "high" if param_count != 10_000_000 else "low"
        
        return ParameterResult(
            count=param_count,
            source="fallback",
            confidence=confidence,
            details={
                "model_family": model_family,
                "estimation_method": "pattern_matching"
            }
        )

    def _estimate_sklearn_parameters(self, name: str) -> int:
        """Estimate parameters for sklearn models"""
        if 'logistic regression' in name:
            return 1_000
        elif 'linear regression' in name:
            return 500
        elif 'support vector' in name:
            return 50_000
        elif 'random forest' in name:
            return 500_000
        elif 'xgboost' in name:
            return 1_000_000
        elif 'lightgbm' in name:
            return 800_000
        else:
            return 100_000

    def _get_model_family(self, name: str) -> str:
        """Get model family for categorization"""
        if 'gpt' in name:
            return 'GPT'
        elif 'bert' in name:
            return 'BERT'
        elif 'llama' in name:
            return 'LLaMA'
        elif 'opt' in name:
            return 'OPT'
        elif 'whisper' in name:
            return 'Whisper'
        elif 'wav2vec' in name:
            return 'Wav2Vec2'
        elif 'vit' in name:
            return 'Vision Transformer'
        elif 'resnet' in name:
            return 'ResNet'
        else:
            return 'Other'

    def calculate_ess(self, parameter_count: int, co2_kg: float) -> float:
        """
        Calculate Environmental Sustainability Score (ESS)
        
        ESS = Effective Parameters (M) / CO₂ (g)
        
        Args:
            parameter_count: Model parameter count
            co2_kg: CO₂ emissions in kg
            
        Returns:
            ESS score (higher is better)
        """
        if co2_kg <= 0 or parameter_count <= 0:
            return 0.0
        
        effective_params_m = parameter_count / 1_000_000
        co2_grams = co2_kg * 1000
        
        return effective_params_m / co2_grams

    def categorize_ess(self, ess_score: float) -> str:
        """Categorize ESS score into efficiency levels"""
        return None

        # Disable the categorization logic for now
        if ess_score >= self.ess_thresholds["excellent"]:
            return "excellent"
        elif ess_score >= self.ess_thresholds["efficient"]:
            return "efficient"
        elif ess_score >= self.ess_thresholds["moderate"]:
            return "moderate"
        elif ess_score >= self.ess_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"

    def get_ess_insights(self, ess_score: float, parameter_count: int, co2_kg: float) -> Dict[str, Any]:
        """Generate insights about ESS score"""
        category = self.categorize_ess(ess_score)
        effective_params_m = parameter_count / 1_000_000
        
        insights = {
            "ess_score": ess_score,
            "ess_category": category,
            "effective_parameters_m": effective_params_m,
            "co2_grams": co2_kg * 1000,
            "parameters_per_gram_co2": effective_params_m / (co2_kg * 1000) if co2_kg > 0 else 0
        }
        
        # Add contextual messages
        if category is None:
            insights["message"] = "ESS categorization is disabled"
        
        return insights

        if category == "excellent":
            insights["message"] = f"Outstanding efficiency! {effective_params_m:.1f}M parameters with only {co2_kg*1000:.2f}g CO₂"
        elif category == "efficient":
            insights["message"] = f"Good efficiency: {effective_params_m:.1f}M parameters with {co2_kg*1000:.2f}g CO₂"
        elif category == "moderate":
            insights["message"] = f"Moderate efficiency: Consider optimization for {effective_params_m:.1f}M parameter model"
        else:
            insights["message"] = f"Low efficiency: {effective_params_m:.1f}M parameters produced {co2_kg*1000:.2f}g CO₂"
        
        return insights

    def compare_ess_scores(self, ess_results: Dict[str, float]) -> Dict[str, Any]:
        """Compare ESS scores across multiple models"""
        if not ess_results:
            return {}
        
        sorted_results = sorted(ess_results.items(), key=lambda x: x[1], reverse=True)
        
        best_model, best_ess = sorted_results[0]
        worst_model, worst_ess = sorted_results[-1]
        
        avg_ess = sum(ess_results.values()) / len(ess_results)
        
        return {
            "model_ranking": [
                {"rank": i+1, "model": model, "ess": ess, "category": self.categorize_ess(ess)}
                for i, (model, ess) in enumerate(sorted_results)
            ],
            "summary": {
                "best_model": {"name": best_model, "ess": best_ess},
                "worst_model": {"name": worst_model, "ess": worst_ess},
                "average_ess": avg_ess,
                "ess_range": best_ess - worst_ess,
                "total_models": len(ess_results)
            },
            "insights": self._generate_comparative_insights(sorted_results, avg_ess)
        }

    def _generate_comparative_insights(self, sorted_results: list, avg_ess: float) -> list:
        """Generate insights from ESS comparison"""
        insights = []
        
        if len(sorted_results) < 2:
            return insights
        
        best_model, best_ess = sorted_results[0]
        worst_model, worst_ess = sorted_results[-1]
        
        if worst_ess > 0:
            improvement_factor = best_ess / worst_ess
            insights.append(f"{best_model} is {improvement_factor:.1f}x more efficient than {worst_model}")
        
        efficient_models = sum(1 for _, ess in sorted_results if ess >= self.ess_thresholds["efficient"])
        insights.append(f"{efficient_models}/{len(sorted_results)} models achieve efficient ESS (≥{self.ess_thresholds['efficient']})")
        
        if avg_ess >= self.ess_thresholds["efficient"]:
            insights.append("Overall model selection shows good environmental efficiency")
        else:
            insights.append("Consider model optimization or selection of more efficient architectures")
        
        return insights