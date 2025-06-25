"""
Dataset Loading Module

This module provides robust dataset loading with intelligent fallback mechanisms
for HuggingFace, scikit-learn, and PyTorch datasets.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# HuggingFace datasets
from datasets import load_dataset as hf_load_dataset

# scikit-learn datasets
try:
    import sklearn.datasets as sklearn_datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# PyTorch datasets
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


class DatasetLoader:
    """
    Dataset loading with comprehensive fallback mechanisms and validation
    for multiple frameworks
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.successful_split = None
        
        # Dataset handling configuration
        self.dataset_config = {
            "default_fallback_splits": ["train", "validation", "dev", "test"],
            "max_retries": 3,
            "timeout_seconds": 30,
            "cache_datasets": config.get("cache_datasets", True),
            "validate_on_load": config.get("validate_datasets", True)
        }

    def load_dataset(self, dataset_name: str, subset: Optional[str] = None,
                    split: Optional[str] = None, limit: Optional[int] = None,
                    fallback_splits: Optional[List[str]] = None,
                    strict_split: bool = False, framework: str = "huggingface",
                    **kwargs):
        """
        Load dataset with comprehensive fallback mechanisms and validation

        Enhanced Dataset Loading Features:
        1. Multi-framework support (HuggingFace, sklearn, PyTorch)
        2. Intelligent split detection and fallback
        3. Automatic dataset validation and quality checks
        4. Configurable sample limiting for quick testing
        5. Robust error handling with detailed diagnostics
        6. Cross-platform compatibility

        Args:
            dataset_name (str): Dataset identifier or file path
            subset (Optional[str]): Dataset configuration/subset name
            split (Optional[str]): Primary split to load
            limit (Optional[int]): Maximum samples to load (for testing)
            fallback_splits (Optional[List[str]]): Alternative splits if primary fails
            strict_split (bool): Only use specified split, no fallbacks
            framework (str): Framework type (huggingface, sklearn, pytorch)
            **kwargs: Framework-specific arguments

        Returns:
            Dataset: Loaded and validated dataset

        Raises:
            ValueError: If dataset cannot be loaded with any available split
        """
        print(f"📚 Loading {framework} dataset: {dataset_name}")
        if subset:
            print(f"   Subset: {subset}")

        try:
            if framework.lower() == "huggingface":
                return self._load_huggingface_dataset(
                    dataset_name, subset, split, limit, fallback_splits, strict_split
                )
            elif framework.lower() == "sklearn":
                return self._load_sklearn_dataset(
                    dataset_name, split, limit, **kwargs
                )
            elif framework.lower() == "pytorch":
                return self._load_pytorch_dataset(
                    dataset_name, split, limit, **kwargs
                )
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        except Exception as e:
            print(f"❌ Failed to load {framework} dataset {dataset_name}: {e}")
            raise

    def _load_huggingface_dataset(self, dataset_name: str, subset: Optional[str] = None,
                                 split: Optional[str] = None, limit: Optional[int] = None,
                                 fallback_splits: Optional[List[str]] = None,
                                 strict_split: bool = True):
        """Load HuggingFace dataset with fallback mechanisms and efficient limiting"""
        
        # Handle case where no split is specified - load entire dataset
        if split is None:
            print(f"   No split specified - loading entire dataset")
            return self._load_huggingface_no_split(dataset_name, subset, limit)
        
        # Set default split if not provided was handled above, so split is not None here
        # Configure fallback behavior
        if strict_split or (fallback_splits is not None and len(fallback_splits) == 0):
            splits_to_try = [split]
            print(f"   Strict mode: Only attempting split '{split}'")
        else:
            if fallback_splits is None:
                fallback_splits = self.dataset_config["default_fallback_splits"]

            splits_to_try = [split] + [s for s in fallback_splits if s != split]
            print(f"   Fallback mode: {splits_to_try}")

        dataset = None
        successful_split = None
        loading_errors = []

        # Attempt to load with each split
        for current_split in splits_to_try:
            try:
                print(f"   Attempting split: {current_split}")

                loading_start_time = time.time()

                # Choose loading strategy based on limit size
                if limit and limit < 10000:  # For small limits, use streaming + take
                    print(f"   Using streaming mode for efficient loading (limit={limit})")
                    
                    if subset:
                        stream_dataset = hf_load_dataset(
                            dataset_name,
                            subset,
                            split=current_split,
                            streaming=True,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )
                    else:
                        stream_dataset = hf_load_dataset(
                            dataset_name,
                            split=current_split,
                            streaming=True,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )
                    
                    # Take only the required number of examples
                    limited_stream = stream_dataset.take(limit)
                    
                    # Convert streaming dataset to regular Dataset for compatibility
                    try:
                        from datasets import Dataset
                        dataset = Dataset.from_generator(lambda: limited_stream)
                    except Exception as convert_e:
                        print(f"   ⚠️ Streaming conversion failed, using list approach: {convert_e}")
                        # Fallback: convert to list then to Dataset
                        examples = list(limited_stream)
                        dataset = Dataset.from_list(examples)
                    
                elif limit:
                    # For larger limits, use split slicing syntax
                    print(f"   Using split slicing for limit={limit}")
                    split_str = f"{current_split}[:{limit}]"
                    
                    if subset:
                        dataset = hf_load_dataset(
                            dataset_name,
                            subset,
                            split=split_str,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )
                    else:
                        dataset = hf_load_dataset(
                            dataset_name,
                            split=split_str,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )
                        
                else:
                    # No limit - load entire split
                    print(f"   Loading entire split: {current_split}")
                    
                    if subset:
                        dataset = hf_load_dataset(
                            dataset_name,
                            subset,
                            split=current_split,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )
                    else:
                        dataset = hf_load_dataset(
                            dataset_name,
                            split=current_split,
                            cache_dir=self.config.get("cache_dir"),
                            trust_remote_code=True
                        )

                loading_time = time.time() - loading_start_time
                successful_split = current_split
                self.successful_split = successful_split
                
                dataset_size = len(dataset)
                print(f"   ✅ Loaded {dataset_size} samples in {loading_time:.1f}s")
                
                if limit and dataset_size != limit and dataset_size < limit:
                    print(f"   ⚠️ Warning: Requested {limit} samples but only {dataset_size} available")
                
                break

            except Exception as e:
                error_msg = f"Split '{current_split}': {str(e)[:100]}..."
                loading_errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                continue

        # Handle loading failure
        if dataset is None:
            if strict_split:
                raise ValueError(
                    f"Failed to load dataset '{dataset_name}' with split '{split}' in strict mode.\n"
                    f"Error: {loading_errors[0] if loading_errors else 'Unknown error'}"
                )
            else:
                error_summary = "\n".join(loading_errors)
                raise ValueError(
                    f"Failed to load dataset '{dataset_name}' with any split.\n"
                    f"Attempted: {splits_to_try}\n"
                    f"Errors:\n{error_summary}"
                )

        # Final dataset size check (redundant with streaming approach but kept for safety)
        original_size = len(dataset)
        final_size = original_size
        
        # This section is now mostly redundant due to efficient loading above,
        # but keeping as a safety fallback for edge cases
        if limit and limit < len(dataset) and not (limit < 10000):
            # This should only happen in edge cases where split slicing didn't work
            print(f"   📏 Applying final limit: {limit} samples (from {original_size})")
            dataset = dataset.select(range(limit))
            final_size = len(dataset)

        # Dataset validation
        if self.dataset_config["validate_on_load"]:
            self._validate_dataset(dataset, dataset_name, successful_split)

        print(f"   🎯 Final dataset: {final_size} samples from split '{successful_split}'")
        
        return dataset

    def _load_huggingface_no_split(self, dataset_name: str, subset: Optional[str] = None, 
                                   limit: Optional[int] = None):
        """Load HuggingFace dataset without specifying split, with optional limit"""
        try:
            loading_start_time = time.time()
            
            # Choose loading strategy based on limit size
            if limit and limit < 10000:  # For small limits, use streaming + take
                print(f"   Using streaming mode for efficient loading (limit={limit})")
                
                if subset:
                    stream_dataset = hf_load_dataset(
                        dataset_name,
                        subset,
                        streaming=True,
                        cache_dir=self.config.get("cache_dir"),
                        trust_remote_code=True
                    )
                else:
                    stream_dataset = hf_load_dataset(
                        dataset_name,
                        streaming=True,
                        cache_dir=self.config.get("cache_dir"),
                        trust_remote_code=True
                    )
                
                # Get the first available split from streaming dataset
                available_splits = list(stream_dataset.keys()) if hasattr(stream_dataset, 'keys') else ['train']
                first_split = available_splits[0]
                print(f"   Using first available split: '{first_split}' from {available_splits}")
                
                # Access the specific split
                split_dataset = stream_dataset[first_split] if hasattr(stream_dataset, 'keys') else stream_dataset
                
                # Take only the required number of examples
                limited_stream = split_dataset.take(limit)
                
                # Convert streaming dataset to regular Dataset for compatibility
                try:
                    from datasets import Dataset
                    dataset = Dataset.from_generator(lambda: limited_stream)
                except Exception as convert_e:
                    print(f"   ⚠️ Streaming conversion failed, using list approach: {convert_e}")
                    # Fallback: convert to list then to Dataset
                    examples = list(limited_stream)
                    dataset = Dataset.from_list(examples)
                
                self.successful_split = first_split
                
            else:
                # For larger limits or no limit, load entire dataset
                print(f"   Loading entire dataset" + (f" with limit={limit}" if limit else ""))
                
                if subset:
                    dataset_dict = hf_load_dataset(
                        dataset_name,
                        subset,
                        cache_dir=self.config.get("cache_dir"),
                        trust_remote_code=True
                    )
                else:
                    dataset_dict = hf_load_dataset(
                        dataset_name,
                        cache_dir=self.config.get("cache_dir"),
                        trust_remote_code=True
                    )
                
                # Get the first available split
                if hasattr(dataset_dict, 'keys'):
                    available_splits = list(dataset_dict.keys())
                    first_split = available_splits[0]
                    print(f"   Using first available split: '{first_split}' from {available_splits}")
                    dataset = dataset_dict[first_split]
                else:
                    # Dataset is already a single split
                    dataset = dataset_dict
                    first_split = "train"  # Default assumption
                
                self.successful_split = first_split
                
                # Apply limit if specified
                if limit and limit < len(dataset):
                    print(f"   Applying limit: {limit} samples (from {len(dataset)})")
                    dataset = dataset.select(range(limit))

            loading_time = time.time() - loading_start_time
            dataset_size = len(dataset)
            
            print(f"   ✅ Loaded {dataset_size} samples in {loading_time:.1f}s")
            
            if limit and dataset_size != limit and dataset_size < limit:
                print(f"   ⚠️ Warning: Requested {limit} samples but only {dataset_size} available")
            
            # Dataset validation
            if self.dataset_config["validate_on_load"]:
                self._validate_dataset(dataset, dataset_name, self.successful_split)

            print(f"   🎯 Final dataset: {dataset_size} samples from split '{self.successful_split}'")
            
            return dataset
            
        except Exception as e:
            print(f"   ❌ Error loading dataset without split: {e}")
            raise ValueError(f"Failed to load dataset '{dataset_name}' without split specification: {e}")

    def _load_sklearn_dataset(self, dataset_name: str, split: Optional[str] = None,
                             limit: Optional[int] = None, **kwargs):
        """Load sklearn dataset"""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for sklearn datasets")

        try:
            # Check if it's a built-in sklearn dataset
            if hasattr(sklearn_datasets, f'load_{dataset_name}'):
                return self._load_sklearn_builtin(dataset_name, split, limit, **kwargs)
            elif hasattr(sklearn_datasets, f'fetch_{dataset_name}'):
                return self._load_sklearn_fetch(dataset_name, split, limit, **kwargs)
            elif Path(dataset_name).exists():
                return self._load_sklearn_from_file(dataset_name, split, limit, **kwargs)
            else:
                raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        except Exception as e:
            print(f"   Error in sklearn dataset loading: {e}")
            raise

    def _load_pytorch_dataset(self, dataset_name: str, split: Optional[str] = None,
                             limit: Optional[int] = None, **kwargs):
        """Load PyTorch dataset"""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for pytorch datasets")

        try:
            # Check if it's a torchvision dataset
            if hasattr(torchvision.datasets, dataset_name):
                return self._load_torchvision_dataset(dataset_name, split, limit, **kwargs)
            elif Path(dataset_name).exists():
                return self._load_pytorch_from_file(dataset_name, split, limit, **kwargs)
            elif 'data' in kwargs or 'X' in kwargs:
                return self._create_pytorch_dataset(kwargs, split, limit)
            else:
                raise ValueError(f"Unknown PyTorch dataset: {dataset_name}")

        except Exception as e:
            print(f"   Error in PyTorch dataset loading: {e}")
            raise

    def _load_sklearn_builtin(self, dataset_name: str, split: Optional[str] = None,
                             limit: Optional[int] = None, **kwargs):
        """Load built-in sklearn dataset"""
        # Load the dataset
        load_func = getattr(sklearn_datasets, f'load_{dataset_name}')
        data = load_func(**kwargs)
        
        X, y = data.data, data.target
        
        # Handle different splits
        if split in ["train", "test"] and split != "full":
            test_size = 0.2 if split == "test" else 0.8
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            if split == "train":
                X, y = X_train, y_train
            else:
                X, y = X_test, y_test
        
        self.successful_split = split or "full"
        
        # Apply limit
        if limit and limit < len(X):
            X = X[:limit]
            y = y[:limit]
        
        print(f"   ✅ Loaded {len(X)} samples")
        
        # Return as sklearn-compatible format
        return SklearnDataset(X, y, feature_names=data.feature_names, target_names=data.target_names)

    def _load_sklearn_fetch(self, dataset_name: str, split: Optional[str] = None,
                           limit: Optional[int] = None, **kwargs):
        """Load sklearn fetch dataset (like 20newsgroups)"""
        fetch_func = getattr(sklearn_datasets, f'fetch_{dataset_name}')
        
        # Handle subset parameter for fetch datasets
        subset = split if split in ["train", "test"] else "train"
        
        data = fetch_func(subset=subset, **kwargs)
        X, y = data.data, data.target
        
        self.successful_split = subset
        
        # Apply limit
        if limit and limit < len(X):
            X = X[:limit]
            y = y[:limit]
        
        print(f"   ✅ Loaded {len(X)} samples")
        
        return SklearnDataset(X, y, feature_names=getattr(data, 'feature_names', None),
                             target_names=getattr(data, 'target_names', None))

    def _load_sklearn_from_file(self, file_path: str, split: Optional[str] = None,
                               limit: Optional[int] = None, **kwargs):
        """Load sklearn dataset from file (CSV, etc.)"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Assume last column is target, unless specified
        target_column = kwargs.get('target_column', df.columns[-1])
        feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Handle categorical targets
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Handle train/test split
        if split in ["train", "test"]:
            test_size = 0.2 if split == "test" else 0.8
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            if split == "train":
                X, y = X_train, y_train
            else:
                X, y = X_test, y_test
        
        self.successful_split = split or "full"
        
        # Apply limit
        if limit and limit < len(X):
            X = X[:limit]
            y = y[:limit]
        
        print(f"   ✅ Loaded {len(X)} samples from {file_path}")
        
        return SklearnDataset(X, y, feature_names=feature_columns)

    def _load_torchvision_dataset(self, dataset_name: str, split: Optional[str] = None,
                                 limit: Optional[int] = None, **kwargs):
        """Load torchvision dataset"""
        dataset_class = getattr(torchvision.datasets, dataset_name)
        
        # Default transform
        if 'transform' not in kwargs:
            kwargs['transform'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        # Handle train/test split
        train = split != "test" if split in ["train", "test"] else True
        
        dataset = dataset_class(
            root=kwargs.get('root', './data'),
            train=train,
            download=kwargs.get('download', True),
            **{k: v for k, v in kwargs.items() if k not in ['root', 'download']}
        )
        
        self.successful_split = "train" if train else "test"
        
        # Apply limit
        if limit and limit < len(dataset):
            indices = list(range(limit))
            dataset = torch.utils.data.Subset(dataset, indices)
        
        print(f"   ✅ Loaded {len(dataset)} samples")
        
        return dataset

    def _load_pytorch_from_file(self, file_path: str, split: Optional[str] = None,
                               limit: Optional[int] = None, **kwargs):
        """Load PyTorch dataset from file"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() in ['.pt', '.pth']:
            # PyTorch tensor file
            data = torch.load(file_path)
            
            if isinstance(data, dict):
                X = data.get('X', data.get('data'))
                y = data.get('y', data.get('targets', data.get('labels')))
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                X, y = data
            else:
                raise ValueError("Unsupported tensor file format")
                
        else:
            # Load as DataFrame and convert to tensors
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            target_column = kwargs.get('target_column', df.columns[-1])
            feature_columns = [col for col in df.columns if col != target_column]
            
            X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
            y = torch.tensor(df[target_column].values, dtype=torch.long)
        
        # Handle train/test split
        if split in ["train", "test"]:
            total_size = len(X)
            test_size = int(0.2 * total_size)
            
            if split == "test":
                X = X[-test_size:]
                y = y[-test_size:]
            else:
                X = X[:-test_size]
                y = y[:-test_size]
        
        self.successful_split = split or "full"
        
        # Apply limit
        if limit and limit < len(X):
            X = X[:limit]
            y = y[:limit]
        
        print(f"   ✅ Loaded {len(X)} samples from {file_path}")
        
        return TensorDataset(X, y)

    def _create_pytorch_dataset(self, data_dict: Dict[str, Any], split: Optional[str] = None,
                               limit: Optional[int] = None):
        """Create PyTorch dataset from provided data"""
        X = data_dict.get('X', data_dict.get('data'))
        y = data_dict.get('y', data_dict.get('targets', data_dict.get('labels')))
        
        if X is None:
            raise ValueError("No data (X) provided")
        
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.tensor(np.array(y), dtype=torch.long)
        
        # Handle train/test split
        if split in ["train", "test"] and y is not None:
            total_size = len(X)
            test_size = int(0.2 * total_size)
            
            if split == "test":
                X = X[-test_size:]
                y = y[-test_size:]
            else:
                X = X[:-test_size]
                y = y[:-test_size]
        
        self.successful_split = split or "full"
        
        # Apply limit
        if limit and limit < len(X):
            X = X[:limit]
            if y is not None:
                y = y[:limit]
        
        print(f"   ✅ Created dataset with {len(X)} samples")
        
        if y is not None:
            return TensorDataset(X, y)
        else:
            return TensorDataset(X)

    def _validate_dataset(self, dataset, dataset_name: str, split: str):
        """Validate dataset quality and compatibility"""
        try:
            # Basic validation
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")

            # Sample validation for HuggingFace datasets
            if hasattr(dataset, '__getitem__'):
                sample = dataset[0]
                if not isinstance(sample, dict):
                    print("   ⚠️ Warning: Unexpected dataset sample format")

                # Check for common required fields
                if isinstance(sample, dict):
                    common_fields = ["text", "label", "input", "output", "image"]
                    available_fields = list(sample.keys())

                    if not any(field in available_fields for field in common_fields):
                        print(f"   ⚠️ Warning: No common fields found. Available: {available_fields[:5]}")

            print(f"   ✅ Dataset validation passed")

        except Exception as e:
            print(f"   ⚠️ Dataset validation warning: {e}")

    def process_single_sample(self, sample: Dict[str, Any], model: Any, processor: Any,
                             task: str, label_key: str, model_name: str, dataset_name: str,
                             framework: str = "huggingface") -> Tuple[Optional[Any], Optional[Any]]:
        """
        Process a single sample with framework-specific handling

        Sample Processing Strategy:
        1. Extract and validate input data
        2. Apply framework-specific preprocessing
        3. Run inference with error handling
        4. Validate and format outputs
        5. Return prediction-reference pair
        """
        try:
            if framework == "huggingface":
                if task == "text":
                    return self._process_text_sample(sample, model, processor, label_key, dataset_name)
                elif task == "image":
                    return self._process_image_sample(sample, model, processor, label_key)
                elif task == "image_generation":
                    return self._process_image_generation_sample(sample, model, processor, label_key)
                elif task == "audio":
                    return self._process_audio_sample(sample, model, processor, label_key)
            
            elif framework == "sklearn":
                return self._process_sklearn_sample(sample, model, task)
            
            elif framework == "pytorch":
                return self._process_pytorch_sample(sample, model, task)
            
            else:
                print(f"   ⚠️ Unsupported framework: {framework}")
                return None, None

        except Exception as e:
            # Log error but don't crash the entire batch
            return None, None

    def _process_sklearn_sample(self, sample, model, task: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Process sklearn sample"""
        try:
            if isinstance(sample, SklearnDataset):
                # Get a random sample
                idx = np.random.randint(0, len(sample.X))
                X = sample.X[idx:idx+1]  # Keep 2D shape
                y = sample.y[idx] if sample.y is not None else None
            elif isinstance(sample, (list, tuple)) and len(sample) == 2:
                X, y = sample
                X = np.array(X).reshape(1, -1) if np.array(X).ndim == 1 else np.array(X)
            else:
                return None, None
            
            # Make prediction
            from ..models.loader import ModelLoader
            from ..monitoring.hardware import detect_hardware_capabilities
            
            capabilities = detect_hardware_capabilities()
            model_loader = ModelLoader(capabilities)
            
            prediction = model_loader.predict_sklearn(model, X, task)
            reference = y
            
            return prediction[0] if len(prediction) > 0 else None, reference
            
        except Exception as e:
            return None, None

    def _process_pytorch_sample(self, sample, model, task: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Process PyTorch sample"""
        try:
            if isinstance(sample, (list, tuple)) and len(sample) >= 1:
                X = sample[0]
                y = sample[1] if len(sample) > 1 else None
            else:
                X = sample
                y = None
            
            # Ensure tensor format
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            
            # Add batch dimension if needed
            if X.dim() == 1 or (X.dim() == 3 and task == "image"):  # For images, keep 3D
                X = X.unsqueeze(0)
            
            # Make prediction
            from ..models.loader import ModelLoader
            from ..monitoring.hardware import detect_hardware_capabilities
            
            capabilities = detect_hardware_capabilities()
            model_loader = ModelLoader(capabilities)
            
            prediction = model_loader.predict_pytorch(model, X, task)
            reference = y.item() if isinstance(y, torch.Tensor) and y.numel() == 1 else y
            
            return prediction[0] if len(prediction) > 0 else prediction, reference
            
        except Exception as e:
            return None, None

    def _process_text_sample(self, sample: Dict[str, Any], model: Any, tokenizer: Any, 
                           label_key: str, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Process HuggingFace text sample for generation or classification"""
        # Extract text content
        text = self.extract_text_from_complex_dataset(sample, dataset_name, label_key)
        if not text or len(text.strip()) < 50:
            return None, None

        # Create prompt-completion pairs
        prompt, completion = self.create_instruction_prompt_completion_pairs(text, dataset_name)
        if prompt is None or completion is None:
            return None, None

        # Generate text using model loader's generate_text method
        from ..models.loader import ModelLoader
        from ..monitoring.hardware import detect_hardware_capabilities
        
        # Create a temporary model loader for text generation
        capabilities = detect_hardware_capabilities()
        model_loader = ModelLoader(capabilities)
        
        max_length = self.config.get("max_length", 1024)
        output = model_loader.generate_text(model, tokenizer, prompt, max_length)

        if output.startswith("[Skipped:") or output.startswith("[Generation failed:"):
            return None, None

        return output, completion

    def _process_image_sample(self, sample: Dict[str, Any], model: Any, processor: Any, label_key: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Process image sample for classification
        
        Args:
            sample: Dataset sample containing image and label
            model: Loaded image classification model (AutoModelForImageClassification)
            processor: Image processor/feature extractor (AutoFeatureExtractor)
            label_key: Key for extracting labels from samples
            
        Returns:
            Tuple of (predicted_class_id, true_class_id)
        """

        import torch

        try:

            # Extract image from sample
            image = None
            image_keys = ["image", "img", "picture", "photo", "pixel_values"]
            
            for key in image_keys:
                if key in sample and sample[key] is not None:
                    image = sample[key]
                    break
            
            if image is None:
                print(f"   ⚠️ No image found in sample keys: {list(sample.keys())}")
                return None, None
            
            # Extract label/ground truth
            true_label = None
            label_keys = [label_key, "label", "labels", "target", "class", "category"]
            
            for key in label_keys:
                if key in sample and sample[key] is not None:
                    true_label = sample[key]
                    break
            
            if true_label is None:
                print(f"   ⚠️ No label found in sample keys: {list(sample.keys())}")
                return None, None
            
            # Convert label to integer if needed
            if isinstance(true_label, str):
                # Try to convert string label to integer
                try:
                    true_label = int(true_label)
                except ValueError:
                    # If string label, we might need label mapping
                    # For now, use hash-based approach as fallback
                    label_hash = hash(true_label) % 1000  # Keep reasonable range
                    true_label = abs(label_hash)
                    print(f"   ⚠️ String label converted to hash-based ID: {true_label}")
            
            # Ensure true_label is integer
            if not isinstance(true_label, (int, float)):
                print(f"   ⚠️ Invalid label type: {type(true_label)}")
                return None, None
            
            true_label = int(true_label)
            
            print(f"   🖼️ Processing image classification sample (true_label: {true_label})")
            
            # Process image with the processor/feature extractor
            try:
                # Handle different image formats
                if hasattr(image, 'convert'):
                    # PIL Image
                    processed_image = image.convert('RGB')
                elif isinstance(image, dict) and 'path' in image:
                    # Image path format (some datasets)
                    from PIL import Image
                    processed_image = Image.open(image['path']).convert('RGB')
                elif isinstance(image, str):
                    # Direct path string
                    from PIL import Image
                    processed_image = Image.open(image).convert('RGB')
                elif hasattr(image, 'shape'):
                    # Numpy array or tensor
                    import numpy as np
                    from PIL import Image
                    
                    if isinstance(image, np.ndarray):
                        # Handle different numpy array formats
                        if image.dtype != np.uint8:
                            # Normalize to 0-255 range
                            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        
                        if len(image.shape) == 3:
                            processed_image = Image.fromarray(image)
                        elif len(image.shape) == 2:
                            processed_image = Image.fromarray(image, mode='L').convert('RGB')
                        else:
                            print(f"   ❌ Unsupported image array shape: {image.shape}")
                            return None, None
                    else:
                        # Tensor format
                        import torch
                        if isinstance(image, torch.Tensor):
                            # Convert tensor to PIL Image
                            if image.dim() == 4:  # Batch dimension
                                image = image.squeeze(0)
                            if image.dim() == 3:
                                # CHW to HWC
                                if image.shape[0] in [1, 3]:  # Channels first
                                    image = image.permute(1, 2, 0)
                            
                            image_np = image.cpu().numpy()
                            if image_np.dtype != np.uint8:
                                image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
                            
                            processed_image = Image.fromarray(image_np)
                        else:
                            print(f"   ❌ Unsupported tensor type: {type(image)}")
                            return None, None
                else:
                    print(f"   ❌ Unsupported image format: {type(image)}")
                    return None, None
                
                # Apply processor/feature extractor
                if processor is not None:
                    inputs = processor(processed_image, return_tensors="pt")
                else:
                    print(f"   ⚠️ No processor available, using raw image")
                    return None, None
                
                # Move inputs to appropriate device
                if hasattr(model, 'device') and model.device.type == 'cuda':
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Extract predictions
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'prediction_scores'):
                    logits = outputs.prediction_scores
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    print(f"   ❌ Unexpected model output format: {type(outputs)}")
                    return None, None
                
                # Get predicted class
                predicted_class = torch.argmax(logits, dim=-1).item()
                
                print(f"   ✅ Image classification complete - Predicted: {predicted_class}, True: {true_label}")
                
                return predicted_class, true_label
                
            except Exception as proc_e:
                print(f"   ❌ Image processing error: {proc_e}")
                return None, None
        
        except Exception as e:
            print(f"   ❌ Error processing image classification sample: {e}")
            return None, None


    def _extract_image_features(self, image) -> Dict[str, Any]:
        """
        Extract basic features from image for analysis
        
        Args:
            image: PIL Image or image array
            
        Returns:
            Dictionary with image features
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                if hasattr(image, 'shape'):
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                    else:
                        # Tensor
                        import torch
                        if isinstance(image, torch.Tensor):
                            image_np = image.cpu().numpy()
                            if image_np.dtype != np.uint8:
                                image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
                            image = Image.fromarray(image_np)
            
            if not isinstance(image, Image.Image):
                return {"error": "Could not convert to PIL Image"}
            
            # Extract basic features
            img_array = np.array(image)
            
            features = {
                "width": image.size[0],
                "height": image.size[1],
                "mode": image.mode,
                "channels": len(img_array.shape) if len(img_array.shape) == 2 else img_array.shape[2],
                "mean_pixel": float(np.mean(img_array)),
                "std_pixel": float(np.std(img_array)),
                "min_pixel": float(np.min(img_array)),
                "max_pixel": float(np.max(img_array)),
                "total_pixels": img_array.size
            }
            
            # Color analysis for RGB images
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                features.update({
                    "mean_red": float(np.mean(img_array[:, :, 0])),
                    "mean_green": float(np.mean(img_array[:, :, 1])),
                    "mean_blue": float(np.mean(img_array[:, :, 2])),
                    "color_variance": float(np.var([np.mean(img_array[:, :, i]) for i in range(3)]))
                })
            
            # Simple texture analysis
            if len(img_array.shape) >= 2:
                # Edge detection approximation
                edges_x = np.abs(np.diff(img_array, axis=1))
                edges_y = np.abs(np.diff(img_array, axis=0))
                features.update({
                    "edge_intensity_x": float(np.mean(edges_x)),
                    "edge_intensity_y": float(np.mean(edges_y)),
                    "texture_complexity": float(np.std(img_array))
                })
            
            return features
            
        except Exception as e:
            return {"error": f"Feature extraction failed: {str(e)}"}


    def _validate_image_classification_setup(self, model: Any, processor: Any) -> bool:
        """
        Validate that model and processor are properly set up for image classification
        
        Args:
            model: Image classification model
            processor: Image processor/feature extractor
            
        Returns:
            True if setup is valid, False otherwise
        """
        try:
            # Check model type
            if not hasattr(model, '__call__'):
                print(f"   ❌ Model is not callable")
                return False
            
            # Check processor
            if processor is None:
                print(f"   ⚠️ No processor provided")
                return False
            
            if not hasattr(processor, '__call__'):
                print(f"   ❌ Processor is not callable")
                return False
            
            # Check if model is in eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            print(f"   ✅ Image classification setup validated")
            return True
            
        except Exception as e:
            print(f"   ❌ Setup validation failed: {e}")
            return False


    def _handle_image_classification_batch(self, samples: List[Dict[str, Any]], model: Any, 
                                        processor: Any, label_key: str) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Process a batch of image samples for better efficiency
        
        Args:
            samples: List of image samples
            model: Image classification model
            processor: Image processor
            label_key: Key for extracting labels
            
        Returns:
            List of (predicted_class, true_class) tuples
        """
        try:
            if not samples:
                return []
            
            # For batch processing, we'd need to modify the individual sample processing
            # For now, process individually to maintain compatibility
            results = []
            
            for i, sample in enumerate(samples):
                try:
                    pred, true = self._process_image_sample(sample, model, processor, label_key)
                    results.append((pred, true))
                    
                    if i > 0 and i % 10 == 0:
                        print(f"   📊 Processed {i}/{len(samples)} image samples")
                        
                except Exception as e:
                    print(f"   ❌ Error in batch sample {i}: {e}")
                    results.append((None, None))
            
            return results
            
        except Exception as e:
            print(f"   ❌ Batch processing error: {e}")
            return [(None, None)] * len(samples)


    def _get_class_names_from_model(self, model: Any) -> Optional[List[str]]:
        """
        Extract class names from model configuration if available
        
        Args:
            model: Image classification model
            
        Returns:
            List of class names or None if not available
        """
        try:
            # Try different ways to get class names
            if hasattr(model, 'config'):
                config = model.config
                
                # Check for id2label mapping
                if hasattr(config, 'id2label') and config.id2label:
                    return [config.id2label.get(i, f"class_{i}") for i in sorted(config.id2label.keys())]
                
                # Check for label2id mapping
                if hasattr(config, 'label2id') and config.label2id:
                    id2label = {v: k for k, v in config.label2id.items()}
                    return [id2label.get(i, f"class_{i}") for i in sorted(id2label.keys())]
                
                # Check for num_labels
                if hasattr(config, 'num_labels'):
                    return [f"class_{i}" for i in range(config.num_labels)]
            
            print(f"   ⚠️ Could not extract class names from model")
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error extracting class names: {e}")
            return None

    def _process_image_generation_sample(self, sample: Dict[str, Any], model: Any, processor: Any, label_key: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Process sample for image generation
        
        Args:
            sample: Dataset sample containing prompt/text
            model: Loaded diffusion pipeline (StableDiffusionPipeline or DiffusionPipeline)
            processor: Not used for image generation (set to None)
            label_key: Key for extracting labels from samples
            
        Returns:
            Tuple of (generated_image, prompt_text)
        """
        try:
            # Extract prompt from sample based on dataset format
            prompt = None
            
            # Try different prompt keys based on common dataset formats
            prompt_keys = [
                label_key,  # User-specified label key
                "prompt", "text", "Prompt", "Text", 
                "caption", "description", "input"
            ]
            
            for key in prompt_keys:
                if key in sample and sample[key]:
                    prompt = sample[key]
                    break
            
            if not prompt or not isinstance(prompt, str):
                print(f"   ⚠️ No valid prompt found in sample keys: {list(sample.keys())}")
                return None, None
            
            # Clean and validate prompt
            prompt = prompt.strip()
            if len(prompt) < 3:
                print(f"   ⚠️ Prompt too short: '{prompt}'")
                return None, None
            
            # Truncate very long prompts
            if len(prompt) > 500:
                prompt = prompt[:497] + "..."
                print(f"   ⚠️ Prompt truncated to 500 characters")
            
            print(f"   🎨 Generating image from prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            
            # Generate image using the diffusion pipeline
            # Standard parameters for consistent generation
            generation_kwargs = {
                "prompt": prompt,
                "num_inference_steps": 20,  # Balanced speed vs quality
                "guidance_scale": 7.5,      # Standard guidance
                "height": 512,              # Standard resolution
                "width": 512,
                "num_images_per_prompt": 1,
                "output_type": "pil"        # Return PIL Image
            }
            
            # Adjust parameters based on model type
            model_name = getattr(model, 'config', {}).get('_name_or_path', str(model.__class__))
            
            if "turbo" in model_name.lower():
                # Turbo models need fewer steps and lower guidance
                generation_kwargs.update({
                    "num_inference_steps": 4,
                    "guidance_scale": 0.0
                })
            elif "sdxl" in model_name.lower():
                # SDXL models prefer higher resolution
                generation_kwargs.update({
                    "height": 1024,
                    "width": 1024,
                    "num_inference_steps": 25
                })
            elif "tiny-sd" in model_name.lower():
                # Tiny models are optimized for speed
                generation_kwargs.update({
                    "num_inference_steps": 10,
                    "height": 512,
                    "width": 512
                })
            
            # Generate image with error handling
            try:
                with torch.no_grad():
                    result = model(**generation_kwargs)
                    
                # Extract image from result
                if hasattr(result, 'images') and result.images:
                    generated_image = result.images[0]
                    print(f"   ✅ Image generated successfully ({generated_image.size})")
                    
                    # For ML-EcoLyzer, we return the image and prompt
                    # The image could be saved or converted to tensor for analysis
                    return generated_image, prompt
                    
                else:
                    print(f"   ❌ No images in generation result")
                    return None, None
                    
            except torch.cuda.OutOfMemoryError:
                print(f"   ❌ CUDA out of memory during generation")
                # Try with smaller resolution
                try:
                    generation_kwargs.update({"height": 256, "width": 256})
                    with torch.no_grad():
                        result = model(**generation_kwargs)
                    generated_image = result.images[0] if result.images else None
                    if generated_image:
                        print(f"   ✅ Image generated at reduced resolution ({generated_image.size})")
                        return generated_image, prompt
                except Exception as fallback_e:
                    print(f"   ❌ Fallback generation failed: {fallback_e}")
                
                return None, None
                
            except Exception as gen_e:
                print(f"   ❌ Generation error: {gen_e}")
                return None, None
        
        except Exception as e:
            print(f"   ❌ Error processing image generation sample: {e}")
            return None, None


    def _save_generated_image(self, image, prompt: str, output_dir: str, sample_idx: int) -> str:
        """
        Helper method to save generated images for analysis
        
        Args:
            image: PIL Image object
            prompt: Original prompt used
            output_dir: Directory to save images
            sample_idx: Sample index for unique naming
            
        Returns:
            Path to saved image file
        """
        import os
        from PIL import Image
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create safe filename from prompt
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            
            filename = f"generated_{sample_idx:04d}_{safe_prompt}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            if isinstance(image, Image.Image):
                image.save(filepath, "PNG")
                print(f"   💾 Image saved: {filepath}")
                return filepath
            else:
                print(f"   ⚠️ Invalid image type for saving: {type(image)}")
                return ""
                
        except Exception as e:
            print(f"   ❌ Error saving image: {e}")
            return ""


    def _extract_image_generation_metrics(self, generated_image, prompt: str) -> Dict[str, Any]:
        """
        Extract basic metrics from generated image
        
        Args:
            generated_image: PIL Image object
            prompt: Original prompt
            
        Returns:
            Dictionary with image metrics
        """
        try:
            from PIL import Image
            import numpy as np
            
            if not isinstance(generated_image, Image.Image):
                return {"error": "Invalid image type"}
            
            # Convert to numpy array
            img_array = np.array(generated_image)
            
            # Basic image statistics
            metrics = {
                "image_size": generated_image.size,
                "image_mode": generated_image.mode,
                "image_format": generated_image.format or "PIL",
                "prompt_length": len(prompt),
                "prompt_word_count": len(prompt.split()),
                
                # Image statistics
                "mean_pixel_value": float(np.mean(img_array)) if img_array.size > 0 else 0,
                "std_pixel_value": float(np.std(img_array)) if img_array.size > 0 else 0,
                "min_pixel_value": float(np.min(img_array)) if img_array.size > 0 else 0,
                "max_pixel_value": float(np.max(img_array)) if img_array.size > 0 else 0,
            }
            
            # Color analysis for RGB images
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                metrics.update({
                    "mean_red": float(np.mean(img_array[:, :, 0])),
                    "mean_green": float(np.mean(img_array[:, :, 1])),
                    "mean_blue": float(np.mean(img_array[:, :, 2])),
                })
            
            return metrics
            
        except Exception as e:
            return {"error": f"Metrics extraction failed: {str(e)}"}

    def _compute_image_generation_quality_score(self, generated_image, prompt: str) -> float:
        """
        Compute a simple quality score for generated image
        
        This is a placeholder for more sophisticated metrics like:
        - CLIP Score (text-image alignment)
        - FID (Fréchet Inception Distance)
        - IS (Inception Score)
        - Aesthetic score
        
        Args:
            generated_image: PIL Image object
            prompt: Original prompt
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            import numpy as np
            from PIL import Image
            
            if not isinstance(generated_image, Image.Image):
                return 0.0
            
            # Convert to array
            img_array = np.array(generated_image)
            
            if img_array.size == 0:
                return 0.0
            
            # Simple heuristic quality metrics
            # 1. Check if image is not just noise (has structure)
            variance = np.var(img_array)
            variance_score = min(variance / 10000.0, 1.0)  # Normalize
            
            # 2. Check color diversity (not monochrome)
            if len(img_array.shape) == 3:
                color_std = np.std([np.std(img_array[:,:,i]) for i in range(3)])
                color_score = min(color_std / 50.0, 1.0)
            else:
                color_score = 0.5
            
            # 3. Check dynamic range
            pixel_range = np.max(img_array) - np.min(img_array)
            range_score = min(pixel_range / 255.0, 1.0)
            
            # 4. Simple prompt-length correlation (longer prompts often produce better results)
            prompt_score = min(len(prompt.split()) / 20.0, 1.0)
            
            # Combine scores
            quality_score = (variance_score + color_score + range_score + prompt_score) / 4.0
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"   ⚠️ Quality score computation failed: {e}")
            return 0.5  # Default neutral score

    def _process_audio_sample(self, sample: Dict[str, Any], model: Any, processor: Any, label_key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process audio sample for recognition or classification
        
        Args:
            sample: Dataset sample containing audio and label
            model: Loaded audio model (AutoModelForCTC, AutoModelForAudioClassification)
            processor: Audio processor (AutoProcessor, Wav2Vec2Processor)
            label_key: Key for extracting labels/transcripts from samples
            
        Returns:
            Tuple of (predicted_text_or_class, reference_text_or_class)
        """

        import torch

        try:
            # Extract audio from sample
            audio = None
            audio_keys = ["audio", "speech", "waveform", "input_values", "path"]
            
            for key in audio_keys:
                if key in sample and sample[key] is not None:
                    audio = sample[key]
                    break
            
            if audio is None:
                print(f"   ⚠️ No audio found in sample keys: {list(sample.keys())}")
                return None, None
            
            # Extract reference text/label
            reference = None
            label_keys = [label_key, "text", "transcript", "transcription", "sentence", "label", "target"]
            
            for key in label_keys:
                if key in sample and sample[key] is not None:
                    reference = sample[key]
                    break
            
            if reference is None:
                print(f"   ⚠️ No reference found in sample keys: {list(sample.keys())}")
                return None, None
            
            # Convert reference to string if needed
            if not isinstance(reference, str):
                reference = str(reference)
            
            reference = reference.strip()
            if not reference:
                print(f"   ⚠️ Empty reference text")
                return None, None
            
            print(f"   🎵 Processing audio sample (ref: '{reference[:50]}{'...' if len(reference) > 50 else ''}')")
            
            # Process audio data
            try:
                # Handle different audio formats
                audio_array = None
                sampling_rate = 16000  # Default sampling rate
                
                if isinstance(audio, dict):
                    # Common format: {"array": [...], "sampling_rate": 16000}
                    if "array" in audio:
                        audio_array = audio["array"]
                        sampling_rate = audio.get("sampling_rate", 16000)
                    elif "path" in audio:
                        # Load from file path
                        audio_array, sampling_rate = self._load_audio_file(audio["path"])
                    else:
                        print(f"   ❌ Unsupported audio dict format: {audio.keys()}")
                        return None, None
                elif isinstance(audio, str):
                    # Direct file path
                    audio_array, sampling_rate = self._load_audio_file(audio)
                elif hasattr(audio, 'shape') or hasattr(audio, '__len__'):
                    # Direct array/tensor
                    import numpy as np
                    import torch
                    
                    if isinstance(audio, torch.Tensor):
                        audio_array = audio.cpu().numpy()
                    elif isinstance(audio, np.ndarray):
                        audio_array = audio
                    elif isinstance(audio, list):
                        audio_array = np.array(audio)
                    else:
                        print(f"   ❌ Unsupported audio array type: {type(audio)}")
                        return None, None
                else:
                    print(f"   ❌ Unsupported audio format: {type(audio)}")
                    return None, None
                
                if audio_array is None or len(audio_array) == 0:
                    print(f"   ❌ Empty audio array")
                    return None, None
                
                # Ensure audio is 1D
                import numpy as np
                audio_array = np.array(audio_array)
                if len(audio_array.shape) > 1:
                    # Take first channel if multi-channel
                    audio_array = audio_array.flatten() if audio_array.shape[0] == 1 else audio_array[:, 0]
                
                # Normalize audio if needed
                if audio_array.dtype != np.float32:
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    elif audio_array.dtype == np.int32:
                        audio_array = audio_array.astype(np.float32) / 2147483648.0
                    else:
                        audio_array = audio_array.astype(np.float32)
                
                # Resample if needed (processor usually handles this)
                target_sr = getattr(processor, 'sampling_rate', 16000) if processor else 16000
                if sampling_rate != target_sr:
                    audio_array = self._resample_audio(audio_array, sampling_rate, target_sr)
                    sampling_rate = target_sr
                
                print(f"   🎵 Audio processed: {len(audio_array)} samples at {sampling_rate} Hz")
                
                # Process with the audio processor
                if processor is not None:
                    try:
                        # Different processors have different interfaces
                        if hasattr(processor, 'sampling_rate'):
                            # Wav2Vec2Processor, WhisperProcessor, etc.
                            inputs = processor(
                                audio_array, 
                                sampling_rate=sampling_rate,
                                return_tensors="pt",
                                padding=True
                            )
                        else:
                            # Generic processor
                            inputs = processor(
                                audio_array,
                                return_tensors="pt"
                            )
                    except Exception as proc_e:
                        print(f"   ❌ Processor error: {proc_e}")
                        return None, None
                else:
                    print(f"   ⚠️ No processor available")
                    return None, None
                
                # Move inputs to appropriate device
                if hasattr(model, 'device') and model.device.type == 'cuda':
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process outputs based on model type
                prediction = self._extract_audio_prediction(outputs, model, processor)
                
                if prediction is None:
                    print(f"   ❌ Could not extract prediction from model outputs")
                    return None, None
                
                print(f"   ✅ Audio processing complete - Predicted: '{prediction[:50]}{'...' if len(prediction) > 50 else ''}'")
                
                return prediction, reference
                
            except Exception as proc_e:
                print(f"   ❌ Audio processing error: {proc_e}")
                return None, None
        
        except Exception as e:
            print(f"   ❌ Error processing audio sample: {e}")
            return None, None


    def _load_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Load audio file from path
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sampling_rate)
        """
        try:
            import numpy as np
            
            # Try different audio loading libraries
            audio_array = None
            sampling_rate = 16000
            
            # Try librosa first (most common)
            try:
                import librosa
                audio_array, sampling_rate = librosa.load(file_path, sr=None)
                print(f"   📁 Loaded audio with librosa: {len(audio_array)} samples at {sampling_rate} Hz")
                return audio_array, sampling_rate
            except ImportError:
                pass
            except Exception as e:
                print(f"   ⚠️ Librosa loading failed: {e}")
            
            # Try soundfile
            try:
                import soundfile as sf
                audio_array, sampling_rate = sf.read(file_path)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array[:, 0]  # Take first channel
                print(f"   📁 Loaded audio with soundfile: {len(audio_array)} samples at {sampling_rate} Hz")
                return audio_array, sampling_rate
            except ImportError:
                pass
            except Exception as e:
                print(f"   ⚠️ Soundfile loading failed: {e}")
            
            # Try torchaudio
            try:
                import torchaudio
                waveform, sampling_rate = torchaudio.load(file_path)
                audio_array = waveform.numpy().flatten()
                print(f"   📁 Loaded audio with torchaudio: {len(audio_array)} samples at {sampling_rate} Hz")
                return audio_array, sampling_rate
            except ImportError:
                pass
            except Exception as e:
                print(f"   ⚠️ Torchaudio loading failed: {e}")
            
            print(f"   ❌ No audio library available to load {file_path}")
            return None, 16000
            
        except Exception as e:
            print(f"   ❌ Error loading audio file {file_path}: {e}")
            return None, 16000


    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sampling rate
        
        Args:
            audio: Audio array
            orig_sr: Original sampling rate
            target_sr: Target sampling rate
            
        Returns:
            Resampled audio array
        """
        try:
            if orig_sr == target_sr:
                return audio
            
            # Try librosa resampling
            try:
                import librosa
                resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
                print(f"   🔄 Resampled from {orig_sr} Hz to {target_sr} Hz")
                return resampled
            except ImportError:
                pass
            
            # Simple linear interpolation fallback
            import numpy as np
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)
            print(f"   🔄 Simple resampling from {orig_sr} Hz to {target_sr} Hz")
            return resampled
            
        except Exception as e:
            print(f"   ⚠️ Resampling failed: {e}, using original audio")
            return audio


    def _extract_audio_prediction(self, outputs, model: Any, processor: Any) -> Optional[str]:
        """
        Extract prediction from model outputs based on model type
        
        Args:
            outputs: Model outputs
            model: Audio model
            processor: Audio processor
            
        Returns:
            Predicted text/class or None
        """
        try:
            import torch
            
            # Determine model type and extract appropriate prediction
            model_name = str(type(model).__name__).lower()
            
            if "ctc" in model_name or "wav2vec2" in model_name:
                # Speech recognition (CTC models)
                return self._extract_ctc_prediction(outputs, processor)
            
            elif "whisper" in model_name:
                # Whisper models
                return self._extract_whisper_prediction(outputs, processor)
            
            elif "classification" in model_name:
                # Audio classification
                return self._extract_classification_prediction(outputs, model)
            
            else:
                # Generic handling
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    
                    # Try CTC decoding first
                    try:
                        prediction = self._extract_ctc_prediction(outputs, processor)
                        if prediction:
                            return prediction
                    except:
                        pass
                    
                    # Try classification
                    try:
                        prediction = self._extract_classification_prediction(outputs, model)
                        if prediction:
                            return prediction
                    except:
                        pass
                    
                    # Fallback: return top logit index as string
                    if len(logits.shape) >= 2:
                        pred_ids = torch.argmax(logits, dim=-1)
                        return str(pred_ids.flatten()[0].item())
            
            print(f"   ⚠️ Could not determine model type for prediction extraction")
            return None
            
        except Exception as e:
            print(f"   ❌ Error extracting audio prediction: {e}")
            return None


    def _extract_ctc_prediction(self, outputs, processor: Any) -> Optional[str]:
        """Extract prediction from CTC model outputs"""
        try:
            import torch
            
            if not hasattr(outputs, 'logits'):
                return None
            
            logits = outputs.logits
            
            # Get predicted token IDs
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode using processor
            if processor and hasattr(processor, 'decode'):
                try:
                    # Try batch_decode first
                    if hasattr(processor, 'batch_decode'):
                        transcription = processor.batch_decode(predicted_ids)[0]
                    else:
                        transcription = processor.decode(predicted_ids[0])
                    
                    return transcription.strip()
                except Exception as decode_e:
                    print(f"   ⚠️ Processor decode failed: {decode_e}")
            
            # Fallback: basic CTC decoding
            # Remove repeated tokens and special tokens
            pred_list = predicted_ids[0].tolist()
            
            # Simple CTC decoding: remove consecutive duplicates and zeros (blank token)
            decoded = []
            prev_token = None
            for token in pred_list:
                if token != 0 and token != prev_token:  # 0 is usually blank token
                    decoded.append(token)
                prev_token = token
            
            # Convert to string (basic approach)
            if processor and hasattr(processor, 'tokenizer'):
                try:
                    text = processor.tokenizer.decode(decoded)
                    return text.strip()
                except:
                    pass
            
            # Very basic fallback
            return " ".join([str(t) for t in decoded[:20]])  # Limit length
            
        except Exception as e:
            print(f"   ❌ CTC prediction extraction failed: {e}")
            return None


    def _extract_whisper_prediction(self, outputs, processor: Any) -> Optional[str]:
        """Extract prediction from Whisper model outputs"""
        try:
            import torch
            
            # Get logits from outputs
            if hasattr(outputs, 'prediction_scores'):
                logits = outputs.prediction_scores
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                print(f"   ⚠️ No logits found in Whisper outputs")
                return None
            
            # Get predicted token IDs - handle different tensor shapes
            if len(logits.shape) >= 2:
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                # Handle case where logits is already 1D or scalar
                predicted_ids = logits.unsqueeze(0) if logits.dim() == 0 else logits
            
            # Ensure predicted_ids is at least 2D for batch processing
            if predicted_ids.dim() == 0:  # Scalar
                predicted_ids = predicted_ids.unsqueeze(0).unsqueeze(0)
            elif predicted_ids.dim() == 1:  # 1D tensor
                predicted_ids = predicted_ids.unsqueeze(0)
            
            # Debug info
            print(f"   🔍 Whisper logits shape: {logits.shape}, predicted_ids shape: {predicted_ids.shape}")
            
            # Decode using processor
            if processor and hasattr(processor, 'decode'):
                try:
                    if hasattr(processor, 'batch_decode'):
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    else:
                        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
                    
                    return transcription.strip()
                except Exception as decode_e:
                    print(f"   ⚠️ Whisper decode failed: {decode_e}")
                    # Try alternative decoding
                    try:
                        # Direct tokenizer access if available
                        if hasattr(processor, 'tokenizer'):
                            transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                            return transcription.strip()
                    except Exception as fallback_e:
                        print(f"   ⚠️ Whisper tokenizer fallback failed: {fallback_e}")
            
            return None
            
        except Exception as e:
            print(f"   ❌ Whisper prediction extraction failed: {e}")
            return None

    def _extract_classification_prediction(self, outputs, model: Any) -> Optional[str]:
        """Extract prediction from audio classification model outputs"""
        try:
            import torch
            
            if not hasattr(outputs, 'logits'):
                return None
            
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            
            # Try to get class name from model config
            if hasattr(model, 'config'):
                config = model.config
                
                if hasattr(config, 'id2label') and config.id2label:
                    class_name = config.id2label.get(predicted_class_id, f"class_{predicted_class_id}")
                    return class_name
                elif hasattr(config, 'label2id') and config.label2id:
                    id2label = {v: k for k, v in config.label2id.items()}
                    class_name = id2label.get(predicted_class_id, f"class_{predicted_class_id}")
                    return class_name
            
            # Fallback to class ID
            return f"class_{predicted_class_id}"
            
        except Exception as e:
            print(f"   ❌ Classification prediction extraction failed: {e}")
            return None


    def _analyze_audio_features(self, audio_array: np.ndarray, sampling_rate: int) -> Dict[str, Any]:
        """
        Extract basic audio features for analysis
        
        Args:
            audio_array: Audio signal array
            sampling_rate: Sampling rate
            
        Returns:
            Dictionary with audio features
        """
        try:
            import numpy as np
            
            if audio_array is None or len(audio_array) == 0:
                return {"error": "Empty audio"}
            
            # Basic features
            features = {
                "duration_seconds": len(audio_array) / sampling_rate,
                "sampling_rate": sampling_rate,
                "num_samples": len(audio_array),
                "mean_amplitude": float(np.mean(np.abs(audio_array))),
                "max_amplitude": float(np.max(np.abs(audio_array))),
                "rms_energy": float(np.sqrt(np.mean(audio_array ** 2))),
                "zero_crossing_rate": float(np.mean(np.diff(np.sign(audio_array)) != 0))
            }
            
            # Spectral features (if audio is long enough)
            if len(audio_array) > 1024:
                try:
                    # Simple spectral features
                    fft = np.fft.fft(audio_array)
                    magnitude = np.abs(fft)[:len(fft)//2]
                    
                    features.update({
                        "spectral_centroid": float(np.mean(magnitude)),
                        "spectral_rolloff": float(np.percentile(magnitude, 95)),
                        "spectral_bandwidth": float(np.std(magnitude))
                    })
                except Exception as spectral_e:
                    print(f"   ⚠️ Spectral analysis failed: {spectral_e}")
            
            return features
            
        except Exception as e:
            return {"error": f"Feature analysis failed: {str(e)}"}


    def _validate_audio_setup(self, model: Any, processor: Any) -> bool:
        """
        Validate audio model and processor setup
        
        Args:
            model: Audio model
            processor: Audio processor
            
        Returns:
            True if setup is valid
        """
        try:
            # Check model
            if not hasattr(model, '__call__'):
                print(f"   ❌ Model is not callable")
                return False
            
            # Check processor
            if processor is None:
                print(f"   ⚠️ No processor provided - may affect quality")
                return True  # Can still work without processor
            
            if not hasattr(processor, '__call__'):
                print(f"   ❌ Processor is not callable")
                return False
            
            # Set model to eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            print(f"   ✅ Audio setup validated")
            return True
            
        except Exception as e:
            print(f"   ❌ Audio setup validation failed: {e}")
            return False
        
    def extract_text_from_complex_dataset(self, sample: Dict[str, Any], dataset_name: str, label_key: str) -> Optional[str]:
        """
        Extract usable text from complex dataset structures with enhanced parsing
        """
        dataset_lower = dataset_name.lower()

        # OpenAssistant format
        if "oasst" in dataset_lower:
            text = sample.get("text", "")
            if text and len(text.strip()) > 50:
                return text.strip()

        # Alpaca instruction format
        elif "alpaca" in dataset_lower:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output_text = sample.get("output", "")

            if instruction:
                parts = [f"Instruction: {instruction}"]
                if input_text:
                    parts.append(f"Input: {input_text}")
                if output_text:
                    parts.append(f"Response: {output_text}")

                combined = "\n".join(parts)
                if len(combined.strip()) > 50:
                    return combined

        # UltraChat conversation format
        elif "ultrachat" in dataset_lower:
            messages = sample.get("messages", [])
            if messages and isinstance(messages, list):
                conversation_parts = []

                for msg in messages[:6]:  # Limit to first 6 messages
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        if role and content:
                            role_name = "Human" if role.lower() in ["user", "human"] else "Assistant"
                            conversation_parts.append(f"{role_name}: {content}")

                if conversation_parts:
                    combined = "\n".join(conversation_parts)
                    if len(combined.strip()) > 50:
                        return combined

        # Generic text extraction
        else:
            # Try common text fields
            text_fields = [label_key, "text", "content", "input", "question", "prompt"]

            for field in text_fields:
                if field in sample:
                    text = sample[field]
                    if isinstance(text, str) and len(text.strip()) > 50:
                        return text.strip()

        return None

    def create_instruction_prompt_completion_pairs(self, text: str, dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Create prompt-completion pairs for instruction/conversation datasets
        """
        if not text or len(text.strip()) < 50:
            return None, None

        dataset_lower = dataset_name.lower()

        # OpenAssistant conversation parsing
        if "oasst" in dataset_lower:
            if "Assistant:" in text:
                parts = text.split("Assistant:", 1)
                if len(parts) == 2:
                    prompt = parts[0].strip()
                    completion = parts[1].strip()

                    if len(prompt) > 20 and len(completion) > 10:
                        return prompt, completion

        # Alpaca instruction parsing
        elif "alpaca" in dataset_lower:
            if "Response:" in text:
                parts = text.split("Response:", 1)
                if len(parts) == 2:
                    prompt = parts[0].strip()
                    completion = parts[1].strip()

                    if len(prompt) > 20 and len(completion) > 10:
                        return prompt, completion

        # UltraChat conversation parsing
        elif "ultrachat" in dataset_lower:
            lines = text.split("\n")
            human_parts = []
            assistant_parts = []

            for line in lines:
                line = line.strip()
                if line.startswith("Human:"):
                    human_parts.append(line)
                elif line.startswith("Assistant:"):
                    assistant_parts.append(line)

            if human_parts and assistant_parts:
                prompt = "\n".join(human_parts)
                completion = assistant_parts[-1].replace("Assistant:", "").strip()

                if len(prompt) > 20 and len(completion) > 10:
                    return prompt, completion

        # Fallback to generic text splitting
        return self.create_text_prompt_completion_pairs(text)

    def create_text_prompt_completion_pairs(self, text: str, max_prompt_ratio: float = 0.3) -> Tuple[Optional[str], Optional[str]]:
        """
        Split text into prompt and completion for evaluation
        """
        if len(text.strip()) < 50:
            return None, None

        # Attempt to split at sentence boundaries first
        sentences = text.split('. ')
        if len(sentences) >= 3:
            prompt_sentences = max(1, int(len(sentences) * max_prompt_ratio))
            prompt = '. '.join(sentences[:prompt_sentences])
            completion = '. '.join(sentences[prompt_sentences:])

            if len(prompt) >= 20 and len(completion) >= 10:
                return prompt.strip(), completion.strip()

        # Fallback to word-based splitting
        words = text.split()
        if len(words) < 10:
            return None, None

        prompt_len = max(10, int(len(words) * max_prompt_ratio))
        prompt = " ".join(words[:prompt_len])
        completion = " ".join(words[prompt_len:])

        return prompt.strip(), completion.strip()


class SklearnDataset:
    """
    Wrapper for sklearn datasets to provide consistent interface
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]