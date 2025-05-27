"""
Accuracy Metrics Module

This module provides task-specific accuracy computation with robust error handling.
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


class AccuracyMetrics:
    """
    Compute appropriate metrics based on task type with robust error handling
    """

    def __init__(self):
        """Initialize accuracy metrics calculator"""
        pass

    def compute_metrics(self, predictions: List[Any], references: List[Any],
                       task: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute appropriate metrics based on task type with robust error handling

        Enhanced Metrics Computation:
        1. Task-appropriate metric selection
        2. Robust error handling for metric computation failures
        3. Multiple metric evaluation for comprehensive assessment
        4. Quality indicators and confidence measures
        """
        if not predictions or not references:
            return {"error": "No predictions or references available for metric computation"}

        if len(predictions) != len(references):
            print(f"   ⚠️ Prediction/reference length mismatch: {len(predictions)} vs {len(references)}")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]

        try:
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

        except Exception as e:
            print(f"   ❌ Metrics computation error: {e}")
            return {"error": f"Metrics computation failed: {str(e)}"}

    def _compute_text_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute text generation metrics"""
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
        """Compute classification metrics"""
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
        Compute perplexity for language models
        
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