"""
Model Loading Module

This module provides robust model loading with hardware optimizations and error handling
for HuggingFace, scikit-learn, and PyTorch models.
"""

import torch
import pickle
import joblib
import numpy as np
from typing import Any, Tuple, Optional, Union, Dict
from pathlib import Path

# HuggingFace imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor,
    AutoModelForImageClassification, AutoProcessor, AutoModelForCTC,
    AutoModelForAudioClassification
)
from diffusers import StableDiffusionPipeline, DiffusionPipeline

# scikit-learn imports
try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    BaseEstimator = object

from ..monitoring.hardware import HardwareCapabilities


class ModelLoader:
    """
    Model loading with hardware-appropriate optimizations for multiple frameworks
    """

    def __init__(self, capabilities: HardwareCapabilities):
        """
        Initialize model loader with hardware capabilities
        
        Args:
            capabilities: Hardware capabilities for optimization
        """
        self.capabilities = capabilities

    def load_model(self, model_name: str, task_type: str, model_type: Optional[str] = None,
                   framework: str = "huggingface", **kwargs) -> Tuple[Any, Any]:
        """
        Load model and processor/tokenizer with comprehensive error handling

        Model Loading Strategy:
        1. Determine framework (huggingface, sklearn, pytorch)
        2. Load model with appropriate method
        3. Apply hardware-appropriate optimizations
        4. Validate model compatibility with task type

        Args:
            model_name (str): Model identifier, file path, or class name
            task_type (str): Task type (text, image, image_generation, audio, classification, regression)
            model_type (Optional[str]): Specific model type for specialized handling
            framework (str): Framework type (huggingface, sklearn, pytorch)
            **kwargs: Additional framework-specific arguments

        Returns:
            Tuple[Any, Any]: (model, processor/tokenizer/None) pair

        Raises:
            ValueError: If model loading fails or framework is unsupported
        """
        print(f"📥 Loading {framework} model: {model_name} for task: {task_type}")

        try:
            if framework.lower() == "huggingface":
                return self._load_huggingface_model(model_name, task_type, model_type)
            elif framework.lower() == "sklearn":
                return self._load_sklearn_model(model_name, task_type, model_type, **kwargs)
            elif framework.lower() == "pytorch":
                return self._load_pytorch_model(model_name, task_type, model_type, **kwargs)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        except Exception as e:
            print(f"❌ Failed to load {framework} model {model_name}: {e}")
            raise

    def _load_huggingface_model(self, model_name: str, task_type: str, 
                               model_type: Optional[str] = None) -> Tuple[Any, Any]:
        """Load HuggingFace model with optimization"""
        try:
            if task_type == "text":
                return self._load_text_model(model_name)
            elif task_type == "image":
                return self._load_image_model(model_name)
            elif task_type == "image_generation":
                return self._load_image_generation_model(model_name)
            elif task_type == "audio":
                return self._load_audio_model(model_name, model_type)
            else:
                raise ValueError(f"Unsupported HuggingFace task type: {task_type}")

        except Exception as e:
            print(f"   Error in HuggingFace model loading: {e}")
            raise

    def _load_sklearn_model(self, model_name: str, task_type: str, 
                           model_type: Optional[str] = None, **kwargs) -> Tuple[Any, Any]:
        """Load scikit-learn model"""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for sklearn models")

        try:
            # Check if it's a file path
            if isinstance(model_name, (str, Path)) and Path(model_name).exists():
                model = self._load_sklearn_from_file(model_name)
            else:
                # Create model from class name
                model = self._create_sklearn_model(model_name, task_type, **kwargs)

            print(f"   ✅ Loaded sklearn model: {type(model).__name__}")
            return model, None  # sklearn models don't need processors

        except Exception as e:
            print(f"   Error in sklearn model loading: {e}")
            raise

    def _load_pytorch_model(self, model_name: str, task_type: str,
                           model_type: Optional[str] = None, **kwargs) -> Tuple[Any, Any]:
        """Load PyTorch model"""
        try:
            # Check if it's a file path
            if isinstance(model_name, (str, Path)) and Path(model_name).exists():
                model = self._load_pytorch_from_file(model_name, **kwargs)
            else:
                # Create model from class or torchvision
                model = self._create_pytorch_model(model_name, task_type, **kwargs)

            # Move to appropriate device
            if self.capabilities.has_gpu:
                model = model.to("cuda")
                print(f"   Model moved to GPU")

            print(f"   ✅ Loaded PyTorch model")
            return model, None  # Return None for processor

        except Exception as e:
            print(f"   Error in PyTorch model loading: {e}")
            raise

    def _load_text_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load text generation/classification model with optimization"""
        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=False
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with hardware-appropriate settings
            model_kwargs = {"torch_dtype": torch.float32}

            # Use FP16 on compatible hardware for efficiency
            if self.capabilities.has_gpu:
                try:
                    model_kwargs["torch_dtype"] = torch.float16
                    print("   Using FP16 precision for GPU efficiency")
                except:
                    model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
                trust_remote_code=False
            )

            # Move to GPU if available
            if self.capabilities.has_gpu:
                model = model.to("cuda")
                print(f"   Model moved to GPU")

            return model, tokenizer

        except Exception as e:
            print(f"   Error in text model loading: {e}")
            raise

    def _load_image_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load image classification model"""
        try:
            processor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)

            if self.capabilities.has_gpu:
                model = model.to("cuda")

            return model, processor

        except Exception as e:
            print(f"   Error in image model loading: {e}")
            raise

    def _load_image_generation_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load image generation model (Stable Diffusion, etc.)"""
        try:
            torch_dtype = torch.float16 if self.capabilities.has_gpu else torch.float32

            if "stable-diffusion" in model_name.lower() or "sd-" in model_name.lower():
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype
                )

            if self.capabilities.has_gpu:
                pipeline = pipeline.to("cuda")
                try:
                    pipeline.enable_attention_slicing()
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("   Memory optimizations enabled")
                except:
                    pass

            return pipeline, None

        except Exception as e:
            print(f"   Error in image generation model loading: {e}")
            raise

    def _load_audio_model(self, model_name: str, model_type: Optional[str] = None) -> Tuple[Any, Any]:
        """Load audio processing model"""
        try:
            if "whisper" in model_name.lower():
                processor = AutoProcessor.from_pretrained(model_name, sampling_rate=16000)
            else:
                processor = AutoProcessor.from_pretrained(model_name)
                    
            if model_type == "ctc" or "wav2vec2" in model_name.lower():
                model = AutoModelForCTC.from_pretrained(model_name)
            else:
                try:
                    model = AutoModelForAudioClassification.from_pretrained(model_name)
                except (OSError, ValueError):
                    model = AutoModelForCTC.from_pretrained(model_name)

            if self.capabilities.has_gpu:
                model = model.to("cuda")

            return model, processor

        except Exception as e:
            print(f"   Error in audio model loading: {e}")
            raise

    def _load_sklearn_from_file(self, file_path: Union[str, Path]) -> BaseEstimator:
        """Load sklearn model from file"""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            elif file_path.suffix == '.joblib':
                model = joblib.load(file_path)
            else:
                # Try joblib first, then pickle
                try:
                    model = joblib.load(file_path)
                except:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
            
            if not isinstance(model, BaseEstimator):
                raise ValueError("Loaded object is not a sklearn model")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load sklearn model from {file_path}: {e}")

    def _create_sklearn_model(self, model_name: str, task_type: str, **kwargs) -> BaseEstimator:
        """Create sklearn model from class name"""
        # Default parameters for different models
        model_defaults = {
            'RandomForestClassifier': {'n_estimators': 100, 'random_state': 42},
            'RandomForestRegressor': {'n_estimators': 100, 'random_state': 42},
            'LogisticRegression': {'random_state': 42, 'max_iter': 1000},
            'LinearRegression': {},
            'SVC': {'random_state': 42},
            'SVR': {},
            'GaussianNB': {},
            'DecisionTreeClassifier': {'random_state': 42},
            'DecisionTreeRegressor': {'random_state': 42},
            'KNeighborsClassifier': {'n_neighbors': 5},
            'KNeighborsRegressor': {'n_neighbors': 5},
            'MLPClassifier': {'random_state': 42, 'max_iter': 500},
            'MLPRegressor': {'random_state': 42, 'max_iter': 500}
        }

        # Model mapping
        model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'LogisticRegression': LogisticRegression,
            'LinearRegression': LinearRegression,
            'SVC': SVC,
            'SVR': SVR,
            'GaussianNB': GaussianNB,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'KNeighborsClassifier': KNeighborsClassifier,
            'KNeighborsRegressor': KNeighborsRegressor,
            'MLPClassifier': MLPClassifier,
            'MLPRegressor': MLPRegressor
        }

        if model_name not in model_classes:
            raise ValueError(f"Unsupported sklearn model: {model_name}")

        # Merge default parameters with user parameters
        params = model_defaults.get(model_name, {})
        params.update(kwargs)

        # Create and return model
        model_class = model_classes[model_name]
        return model_class(**params)

    def _load_pytorch_from_file(self, file_path: Union[str, Path], **kwargs) -> torch.nn.Module:
        """Load PyTorch model from file"""
        file_path = Path(file_path)
        
        try:
            # Load checkpoint
            if self.capabilities.has_gpu:
                checkpoint = torch.load(file_path)
            else:
                checkpoint = torch.load(file_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, torch.nn.Module):
                # Direct model save
                model = checkpoint
            elif isinstance(checkpoint, dict):
                # State dict format
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Need model architecture - check if provided in kwargs
                if 'model_class' in kwargs:
                    model = kwargs['model_class'](**kwargs.get('model_kwargs', {}))
                    model.load_state_dict(state_dict)
                else:
                    raise ValueError("Model architecture not provided for state_dict loading")
            else:
                raise ValueError("Unsupported checkpoint format")
            
            model.eval()  # Set to evaluation mode
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model from {file_path}: {e}")

    def _create_pytorch_model(self, model_name: str, task_type: str, **kwargs) -> torch.nn.Module:
        """Create PyTorch model from torchvision or custom class"""
        try:
            # Try torchvision models first
            import torchvision.models as models
            
            if hasattr(models, model_name):
                model_fn = getattr(models, model_name)
                pretrained = kwargs.get('pretrained', True)
                
                # Handle different torchvision versions
                try:
                    if pretrained:
                        model = model_fn(weights='DEFAULT')
                    else:
                        model = model_fn()
                except TypeError:
                    # Older torchvision versions
                    model = model_fn(pretrained=pretrained)
                
                return model
            else:
                # Custom model class
                if 'model_class' in kwargs:
                    model_class = kwargs['model_class']
                    model_kwargs = kwargs.get('model_kwargs', {})
                    return model_class(**model_kwargs)
                else:
                    raise ValueError(f"Unknown PyTorch model: {model_name}")
                    
        except ImportError:
            raise ImportError("torchvision is required for standard PyTorch models")

    def generate_text(self, model: Any, tokenizer: Any, prompt: str, max_length: int) -> str:
        """
        Generate text continuation from prompt with enhanced quality control
        """
        if not prompt or len(prompt.strip()) < 10:
            return "[Skipped: prompt too short]"

        try:
            # Tokenize input with proper handling
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(512, max_length // 2),
                padding=False
            )

            # Move to appropriate device
            if self.capabilities.has_gpu:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generation parameters optimized for environmental efficiency
            generation_kwargs = {
                "max_length": min(max_length, inputs["input_ids"].shape[1] + 100),
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "early_stopping": True
            }

            # Adjust parameters for hardware efficiency
            if self.capabilities.device_category in ["mobile", "edge"]:
                generation_kwargs["max_length"] = min(generation_kwargs["max_length"],
                                                    inputs["input_ids"].shape[1] + 50)
                generation_kwargs["temperature"] = 0.8

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            # Extract generated text (excluding input prompt)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Quality validation
            generated_text = generated_text.strip()
            if len(generated_text) < 5:
                return "[Skipped: generation too short]"

            return generated_text

        except Exception as e:
            print(f"   ⚠️ Text generation error: {e}")
            return f"[Generation failed: {str(e)[:50]}]"

    def predict_sklearn(self, model: BaseEstimator, X: Union[np.ndarray, list], 
                       task_type: str) -> Union[np.ndarray, list]:
        """
        Run prediction with sklearn model
        
        Args:
            model: Trained sklearn model
            X: Input features
            task_type: classification or regression
            
        Returns:
            Predictions
        """
        try:
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            
            if task_type == "classification":
                # Use predict_proba if available for better metrics
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(X)
                    # Return class predictions for compatibility
                    return np.argmax(predictions, axis=1)
                else:
                    return model.predict(X)
            else:  # regression
                return model.predict(X)
                
        except Exception as e:
            print(f"   ⚠️ Sklearn prediction error: {e}")
            raise

    def predict_pytorch(self, model: torch.nn.Module, X: Union[torch.Tensor, np.ndarray, list],
                       task_type: str) -> Union[torch.Tensor, np.ndarray]:
        """
        Run prediction with PyTorch model
        
        Args:
            model: PyTorch model
            X: Input data
            task_type: classification, regression, or other
            
        Returns:
            Predictions
        """
        try:
            # Convert to tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(np.array(X), dtype=torch.float32)
            
            # Move to appropriate device
            if self.capabilities.has_gpu and next(model.parameters()).is_cuda:
                X = X.to("cuda")
            
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                
                # Handle different output formats
                if task_type == "classification":
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        # Multi-class classification
                        predictions = torch.argmax(outputs, dim=1)
                    else:
                        # Binary classification
                        predictions = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    # Regression or other tasks
                    predictions = outputs
                
                return predictions.cpu().numpy()
                
        except Exception as e:
            print(f"   ⚠️ PyTorch prediction error: {e}")
            raise