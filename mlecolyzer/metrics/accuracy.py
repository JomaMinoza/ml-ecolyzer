"""
Accuracy Metrics Module

This module provides task-specific accuracy computation with robust error handling
for HuggingFace, scikit-learn, and PyTorch models.
"""

import numpy as np
from typing import List, Any, Dict, Optional

try:
    from evaluate import load_metric
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False

try:
    from sacrebleu import corpus_bleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

try:
    from jiwer import wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AccuracyMetrics:
    """
    Compute appropriate metrics based on task type and framework with robust error handling
    """

    def __init__(self):
        """Initialize accuracy metrics calculator"""
        pass

    def compute_metrics(self, predictions: List[Any], references: List[Any],
                       task: str, model_type: Optional[str] = None, 
                       framework: str = "huggingface") -> Dict[str, Any]:
        """
        Compute appropriate metrics based on task type and framework with robust error handling

        Enhanced Metrics Computation:
        1. Framework-specific metric selection
        2. Task-appropriate metric evaluation
        3. Robust error handling for metric computation failures
        4. Multiple metric evaluation for comprehensive assessment
        5. Quality indicators and confidence measures
        """
        if not predictions or not references:
            return {"error": "No predictions or references available for metric computation"}

        if len(predictions) != len(references):
            print(f"   ⚠️ Prediction/reference length mismatch: {len(predictions)} vs {len(references)}")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        try:
            if framework == "sklearn":
                return self._compute_sklearn_metrics(predictions, references, task)
            elif framework == "pytorch":
                return self._compute_pytorch_metrics(predictions, references, task)
            elif framework == "huggingface":
                if task == "text":
                    return self._compute_text_metrics(predictions, references)
                elif task == "image":
                    return self._compute_classification_metrics(predictions, references)
                elif task == "audio":
                    if model_type == "asr":
                        return self._compute_asr_metrics(predictions, references)
                    else:
                        return self._compute_classification_metrics(predictions, references)
                else:
                    return self._compute_generic_metrics(predictions, references)
            else:
                return self._compute_generic_metrics(predictions, references)

        except Exception as e:
            print(f"   ❌ Metrics computation error: {e}")
            return {"error": f"Metrics computation failed: {str(e)}"}

    def _compute_sklearn_metrics(self, predictions: List[Any], references: List[Any], 
                                task: str) -> Dict[str, Any]:
        """Compute sklearn-specific metrics"""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available for sklearn metrics"}

        try:
            predictions = np.array(predictions)
            references = np.array(references)

            if task == "classification":
                return self._compute_sklearn_classification_metrics(predictions, references)
            elif task == "regression":
                return self._compute_sklearn_regression_metrics(predictions, references)
            else:
                return self._compute_sklearn_classification_metrics(predictions, references)

        except Exception as e:
            return {"error": f"sklearn metrics computation failed: {str(e)}"}

    def _compute_sklearn_classification_metrics(self, predictions: np.ndarray, 
                                              references: np.ndarray) -> Dict[str, Any]:
        """Compute classification metrics for sklearn models"""
        try:
            # Basic accuracy
            accuracy = accuracy_score(references, predictions)
            
            # Additional metrics
            metrics = {
                "accuracy": accuracy,
                "metric_type": "sklearn_classification",
                "num_samples": len(predictions)
            }

            # Multi-class metrics
            try:
                # Check if binary or multi-class
                unique_classes = len(np.unique(references))
                
                if unique_classes == 2:
                    # Binary classification
                    precision = precision_score(references, predictions, average='binary')
                    recall = recall_score(references, predictions, average='binary')
                    f1 = f1_score(references, predictions, average='binary')
                    
                    try:
                        auc = roc_auc_score(references, predictions)
                        metrics["auc_roc"] = auc
                    except:
                        pass
                else:
                    # Multi-class classification
                    precision = precision_score(references, predictions, average='weighted')
                    recall = recall_score(references, predictions, average='weighted')
                    f1 = f1_score(references, predictions, average='weighted')

                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "num_classes": unique_classes
                })

                # Classification report (as string summary)
                try:
                    report = classification_report(references, predictions, output_dict=True)
                    metrics["macro_avg_f1"] = report["macro avg"]["f1-score"]
                    metrics["weighted_avg_f1"] = report["weighted avg"]["f1-score"]
                except:
                    pass

            except Exception as e:
                print(f"   ⚠️ Extended classification metrics failed: {e}")

            return metrics

        except Exception as e:
            return {"error": f"sklearn classification metrics computation failed: {str(e)}"}

    def _compute_sklearn_regression_metrics(self, predictions: np.ndarray, 
                                          references: np.ndarray) -> Dict[str, Any]:
        """Compute regression metrics for sklearn models"""
        try:
            # Core regression metrics
            mse = mean_squared_error(references, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(references, predictions)
            r2 = r2_score(references, predictions)

            # Additional metrics
            mean_ref = np.mean(references)
            variance_ref = np.var(references)
            
            # Explained variance score
            explained_variance = 1 - np.var(references - predictions) / variance_ref if variance_ref > 0 else 0

            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((references - predictions) / references)) * 100 if np.all(references != 0) else float('inf')

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "explained_variance": explained_variance,
                "mape": mape if mape != float('inf') else None,
                "metric_type": "sklearn_regression",
                "num_samples": len(predictions),
                "mean_prediction": np.mean(predictions),
                "std_prediction": np.std(predictions),
                "mean_reference": mean_ref,
                "std_reference": np.std(references)
            }

            return metrics

        except Exception as e:
            return {"error": f"sklearn regression metrics computation failed: {str(e)}"}

    def _compute_pytorch_metrics(self, predictions: List[Any], references: List[Any], 
                                task: str) -> Dict[str, Any]:
        """Compute PyTorch-specific metrics"""
        if not HAS_TORCH:
            return {"error": "PyTorch not available for pytorch metrics"}

        try:
            # Convert to tensors if needed
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, dtype=torch.float32)
            if not isinstance(references, torch.Tensor):
                references = torch.tensor(references, dtype=torch.float32)

            if task == "classification":
                return self._compute_pytorch_classification_metrics(predictions, references)
            elif task == "regression":
                return self._compute_pytorch_regression_metrics(predictions, references)
            elif task == "image":
                return self._compute_pytorch_image_metrics(predictions, references)
            else:
                return self._compute_pytorch_classification_metrics(predictions, references)

        except Exception as e:
            return {"error": f"PyTorch metrics computation failed: {str(e)}"}

    def _compute_pytorch_classification_metrics(self, predictions: torch.Tensor, 
                                              references: torch.Tensor) -> Dict[str, Any]:
        """Compute classification metrics for PyTorch models"""
        try:
            # Ensure integer type for classification
            if predictions.dtype == torch.float32:
                predictions = predictions.long()
            if references.dtype == torch.float32:
                references = references.long()

            # Basic accuracy
            accuracy = (predictions == references).float().mean().item()

            metrics = {
                "accuracy": accuracy,
                "metric_type": "pytorch_classification",
                "num_samples": len(predictions)
            }

            # Additional metrics using sklearn if available
            if HAS_SKLEARN:
                pred_np = predictions.cpu().numpy()
                ref_np = references.cpu().numpy()
                
                sklearn_metrics = self._compute_sklearn_classification_metrics(pred_np, ref_np)
                # Merge sklearn metrics (excluding redundant accuracy)
                for key, value in sklearn_metrics.items():
                    if key not in ["accuracy", "metric_type"]:
                        metrics[key] = value

            # PyTorch-specific metrics
            unique_classes = torch.unique(references).numel()
            metrics["num_classes"] = unique_classes

            # Cross-entropy loss (if applicable)
            try:
                if unique_classes > 1:
                    # Convert predictions to probabilities if needed
                    if predictions.dim() == 1:
                        # Predictions are class indices
                        one_hot_pred = F.one_hot(predictions, num_classes=unique_classes).float()
                        one_hot_ref = F.one_hot(references, num_classes=unique_classes).float()
                        
                        # Compute cross-entropy
                        ce_loss = F.cross_entropy(one_hot_pred, references, reduction='mean').item()
                        metrics["cross_entropy_loss"] = ce_loss
            except Exception as e:
                print(f"   ⚠️ Cross-entropy computation failed: {e}")

            return metrics

        except Exception as e:
            return {"error": f"PyTorch classification metrics computation failed: {str(e)}"}

    def _compute_pytorch_regression_metrics(self, predictions: torch.Tensor, 
                                          references: torch.Tensor) -> Dict[str, Any]:
        """Compute regression metrics for PyTorch models"""
        try:
            # Ensure float type for regression
            predictions = predictions.float()
            references = references.float()

            # PyTorch-specific losses
            mse_loss = F.mse_loss(predictions, references).item()
            mae_loss = F.l1_loss(predictions, references).item()

            metrics = {
                "mse": mse_loss,
                "mae": mae_loss,
                "rmse": np.sqrt(mse_loss),
                "metric_type": "pytorch_regression",
                "num_samples": len(predictions)
            }

            # Additional metrics using sklearn if available
            if HAS_SKLEARN:
                pred_np = predictions.cpu().numpy()
                ref_np = references.cpu().numpy()
                
                sklearn_metrics = self._compute_sklearn_regression_metrics(pred_np, ref_np)
                # Merge sklearn metrics (excluding redundant MSE/MAE)
                for key, value in sklearn_metrics.items():
                    if key not in ["mse", "mae", "metric_type"]:
                        metrics[key] = value

            # PyTorch-specific statistics
            metrics.update({
                "mean_prediction": predictions.mean().item(),
                "std_prediction": predictions.std().item(),
                "mean_reference": references.mean().item(),
                "std_reference": references.std().item()
            })

            return metrics

        except Exception as e:
            return {"error": f"PyTorch regression metrics computation failed: {str(e)}"}

    def _compute_pytorch_image_metrics(self, predictions: torch.Tensor, 
                                     references: torch.Tensor) -> Dict[str, Any]:
        """Compute image-specific metrics for PyTorch models"""
        try:
            # For image tasks, often classification or generation
            if predictions.dim() == 1 and references.dim() == 1:
                # Image classification
                return self._compute_pytorch_classification_metrics(predictions, references)
            else:
                # Image generation/reconstruction metrics
                mse = F.mse_loss(predictions, references).item()
                
                # PSNR (Peak Signal-to-Noise Ratio)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(predictions, references))).item()
                
                # SSIM would require additional implementation
                metrics = {
                    "mse": mse,
                    "psnr": psnr,
                    "metric_type": "pytorch_image",
                    "num_samples": len(predictions)
                }

                return metrics

        except Exception as e:
            return {"error": f"PyTorch image metrics computation failed: {str(e)}"}

    def _compute_text_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute text generation metrics for HuggingFace models"""
        try:
            # Try BLEU score first using sacrebleu
            if HAS_SACREBLEU:
                try:
                    # Convert single references to list format for sacrebleu
                    refs_list = [[ref] for ref in references]
                    bleu_result = corpus_bleu(predictions, refs_list)
                    return {
                        "bleu_score": bleu_result.score,
                        "metric_type": "bleu",
                        "num_samples": len(predictions)
                    }
                except Exception as e:
                    print(f"   ⚠️ SACREBLEU failed: {e}")

            # Try evaluate library BLEU
            if HAS_EVALUATE:
                try:
                    metric = load_metric("sacrebleu")
                    bleu_result = metric.compute(
                        predictions=predictions,
                        references=[[r] for r in references]
                    )
                    return {
                        "bleu_score": bleu_result["score"],
                        "metric_type": "bleu",
                        "num_samples": len(predictions)
                    }
                except Exception as e:
                    print(f"   ⚠️ Evaluate BLEU failed: {e}")

            # Fallback to ROUGE if available
            if HAS_EVALUATE:
                try:
                    metric = load_metric("rouge")
                    rouge_result = metric.compute(
                        predictions=predictions,
                        references=references
                    )
                    return {
                        "rouge1": rouge_result["rouge1"],
                        "rouge2": rouge_result["rouge2"],
                        "rougeL": rouge_result["rougeL"],
                        "metric_type": "rouge",
                        "num_samples": len(predictions)
                    }
                except Exception as e:
                    print(f"   ⚠️ ROUGE failed: {e}")

            # Fallback to custom similarity metric
            similarities = []
            for pred, ref in zip(predictions, references):
                if len(ref) > 0:
                    # Simple token overlap similarity
                    pred_tokens = set(pred.lower().split())
                    ref_tokens = set(ref.lower().split())
                    if ref_tokens:
                        similarity = len(pred_tokens & ref_tokens) / len(ref_tokens)
                        similarities.append(similarity)

            if similarities:
                return {
                    "similarity_score": np.mean(similarities) * 100,
                    "metric_type": "token_overlap",
                    "num_samples": len(similarities)
                }
            else:
                return {"error": "Could not compute any text metrics"}

        except Exception as e:
            return {"error": f"Text metrics computation failed: {str(e)}"}

    def _compute_classification_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, Any]:
        """Compute classification metrics for HuggingFace models"""
        try:
            # Try evaluate library accuracy
            if HAS_EVALUATE:
                try:
                    metric = load_metric("accuracy")
                    result = metric.compute(predictions=predictions, references=references)
                    return {
                        "accuracy": result["accuracy"],
                        "metric_type": "accuracy",
                        "num_samples": len(predictions)
                    }
                except Exception as e:
                    print(f"   ⚠️ Evaluate accuracy failed: {e}")

            # Fallback to manual accuracy calculation
            correct = sum(1 for p, r in zip(predictions, references) if p == r)
            accuracy = correct / len(predictions)
            
            return {
                "accuracy": accuracy,
                "metric_type": "manual_accuracy",
                "num_samples": len(predictions),
                "correct_predictions": correct
            }

        except Exception as e:
            return {"error": f"Classification metrics computation failed: {str(e)}"}

    def _compute_asr_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute ASR-specific metrics"""
        try:
            # Try jiwer for WER computation
            if HAS_JIWER:
                try:
                    word_error_rate = wer(references, predictions)
                    return {
                        "word_error_rate": word_error_rate,
                        "metric_type": "wer",
                        "num_samples": len(predictions)
                    }
                except Exception as e:
                    print(f"   ⚠️ WER computation failed: {e}")

            # Fallback to character error rate
            char_errors = []
            for pred, ref in zip(predictions, references):
                pred_chars = list(pred.lower())
                ref_chars = list(ref.lower())
                
                # Simple character-level edit distance approximation
                if len(ref_chars) > 0:
                    # Count character differences
                    max_len = max(len(pred_chars), len(ref_chars))
                    errors = 0
                    
                    for i in range(max_len):
                        pred_char = pred_chars[i] if i < len(pred_chars) else ""
                        ref_char = ref_chars[i] if i < len(ref_chars) else ""
                        if pred_char != ref_char:
                            errors += 1
                    
                    char_error_rate = errors / len(ref_chars)
                    char_errors.append(char_error_rate)

            if char_errors:
                return {
                    "character_error_rate": np.mean(char_errors),
                    "metric_type": "cer",
                    "num_samples": len(char_errors)
                }
            else:
                return {"error": "Could not compute ASR metrics"}

        except Exception as e:
            return {"error": f"ASR metrics computation failed: {str(e)}"}

    def _compute_generic_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, Any]:
        """Compute generic metrics when specific metrics unavailable"""
        try:
            # Simple exact match accuracy
            correct = sum(1 for p, r in zip(predictions, references) if str(p) == str(r))
            accuracy = correct / len(predictions)

            # Additional generic metrics
            prediction_lengths = [len(str(p)) for p in predictions]
            reference_lengths = [len(str(r)) for r in references]

            return {
                "exact_match_accuracy": accuracy,
                "metric_type": "exact_match",
                "num_samples": len(predictions),
                "correct_predictions": correct,
                "avg_prediction_length": np.mean(prediction_lengths),
                "avg_reference_length": np.mean(reference_lengths),
                "length_ratio": np.mean(prediction_lengths) / np.mean(reference_lengths) if np.mean(reference_lengths) > 0 else 1.0
            }
        except Exception as e:
            return {"error": f"Generic metrics computation failed: {str(e)}"}

    def compute_perplexity(self, model: Any, tokenizer: Any, texts: List[str]) -> Optional[float]:
        """
        Compute perplexity for language models (HuggingFace only)
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of texts to evaluate
            
        Returns:
            Average perplexity or None if computation fails
        """
        try:
            import torch
            
            model.eval()
            total_loss = 0
            total_tokens = 0
            
            with torch.no_grad():
                for text in texts:
                    # Tokenize text
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    
                    # Move to same device as model
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Compute loss
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # Accumulate loss and token count
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = np.exp(avg_loss)
                return perplexity
            
        except Exception as e:
            print(f"   ⚠️ Perplexity computation failed: {e}")
        
        return None

    def compute_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics for generated text
        
        Args:
            predictions: List of generated texts
            
        Returns:
            Dictionary with diversity metrics
        """
        try:
            if not predictions:
                return {}
            
            # Tokenize all predictions
            all_tokens = []
            for pred in predictions:
                tokens = pred.lower().split()
                all_tokens.extend(tokens)
            
            if not all_tokens:
                return {}
            
            # Vocabulary diversity
            unique_tokens = set(all_tokens)
            vocab_diversity = len(unique_tokens) / len(all_tokens)
            
            # Average sequence length
            avg_length = np.mean([len(pred.split()) for pred in predictions])
            
            # Repetition metrics
            bigrams = []
            trigrams = []
            
            for pred in predictions:
                tokens = pred.lower().split()
                if len(tokens) >= 2:
                    bigrams.extend([f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)])
                if len(tokens) >= 3:
                    trigrams.extend([f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens)-2)])
            
            # Distinct n-gram ratios
            distinct_1 = len(unique_tokens) / len(all_tokens) if all_tokens else 0
            distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
            distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0
            
            return {
                "vocab_diversity": vocab_diversity,
                "avg_length": avg_length,
                "distinct_1": distinct_1,
                "distinct_2": distinct_2,
                "distinct_3": distinct_3,
                "total_unique_tokens": len(unique_tokens),
                "total_tokens": len(all_tokens)
            }
            
        except Exception as e:
            print(f"   ⚠️ Diversity metrics computation failed: {e}")
            return {}

    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> Optional[float]:
        """
        Compute semantic similarity using sentence embeddings (if available)
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            Average cosine similarity or None if computation fails
        """
        try:
            # Try to use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                pred_embeddings = model.encode(predictions)
                ref_embeddings = model.encode(references)
                
                # Compute cosine similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = []
                
                for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                    sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                    similarities.append(sim)
                
                return np.mean(similarities)
                
            except ImportError:
                print("   ⚠️ sentence-transformers not available for semantic similarity")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Semantic similarity computation failed: {e}")
            return None

    def compute_framework_specific_metrics(self, model: Any, predictions: List[Any], 
                                         references: List[Any], framework: str, 
                                         task: str) -> Dict[str, Any]:
        """
        Compute framework-specific advanced metrics
        
        Args:
            model: The trained model
            predictions: Model predictions
            references: Ground truth references
            framework: Framework type (sklearn, pytorch, huggingface)
            task: Task type
            
        Returns:
            Dictionary with advanced metrics
        """
        try:
            if framework == "sklearn" and HAS_SKLEARN:
                return self._compute_sklearn_advanced_metrics(model, predictions, references, task)
            elif framework == "pytorch" and HAS_TORCH:
                return self._compute_pytorch_advanced_metrics(model, predictions, references, task)
            elif framework == "huggingface":
                return self._compute_huggingface_advanced_metrics(model, predictions, references, task)
            else:
                return {}
                
        except Exception as e:
            print(f"   ⚠️ Framework-specific metrics computation failed: {e}")
            return {}

    def _compute_sklearn_advanced_metrics(self, model: Any, predictions: List[Any], 
                                        references: List[Any], task: str) -> Dict[str, Any]:
        """Compute advanced sklearn-specific metrics"""
        try:
            metrics = {}
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                metrics["feature_importance_available"] = True
                metrics["top_features"] = model.feature_importances_[:5].tolist()  # Top 5
            
            # Model complexity
            if hasattr(model, 'n_features_in_'):
                metrics["n_features"] = model.n_features_in_
            
            # Tree-based metrics
            if hasattr(model, 'n_estimators'):
                metrics["n_estimators"] = model.n_estimators
            if hasattr(model, 'max_depth'):
                metrics["max_depth"] = model.max_depth
            
            return metrics
            
        except Exception as e:
            return {"error": f"sklearn advanced metrics failed: {str(e)}"}

    def _compute_pytorch_advanced_metrics(self, model: Any, predictions: List[Any], 
                                        references: List[Any], task: str) -> Dict[str, Any]:
        """Compute advanced PyTorch-specific metrics"""
        try:
            metrics = {}
            
            # Model parameters
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                metrics.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
                })
            
            return metrics
            
        except Exception as e:
            return {"error": f"PyTorch advanced metrics failed: {str(e)}"}

    def _compute_huggingface_advanced_metrics(self, model: Any, predictions: List[Any], 
                                            references: List[Any], task: str) -> Dict[str, Any]:
        """Compute advanced HuggingFace-specific metrics"""
        try:
            metrics = {}
            
            # Model configuration
            if hasattr(model, 'config'):
                config = model.config
                if hasattr(config, 'num_parameters'):
                    metrics["num_parameters"] = config.num_parameters
                if hasattr(config, 'vocab_size'):
                    metrics["vocab_size"] = config.vocab_size
                if hasattr(config, 'hidden_size'):
                    metrics["hidden_size"] = config.hidden_size
                if hasattr(config, 'num_attention_heads'):
                    metrics["num_attention_heads"] = config.num_attention_heads
                if hasattr(config, 'num_hidden_layers'):
                    metrics["num_hidden_layers"] = config.num_hidden_layers
            
            return metrics
            
        except Exception as e:
            return {"error": f"HuggingFace advanced metrics failed: {str(e)}"}