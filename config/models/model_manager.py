"""
Model management and loading
"""
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import TFAutoModel, TFAutoTokenizer
from typing import Dict, Optional, Any
import numpy as np
from config.models.model_config import ModelType, ModelConfig
from utils.logging_utils import get_logger

class ModelManager:
    """Manages loading and caching of models"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models: Dict[ModelType, Any] = {}
        self.tokenizers: Dict[ModelType, Any] = {}

    def get_model(self, model_type: ModelType) -> tuple:
        """Get model and tokenizer, loading if necessary"""
        try:
            if model_type not in self.models:
                self._load_model(model_type)
            return self.models[model_type], self.tokenizers[model_type]
        except Exception as e:
            self.logger.error(f"Error getting model {model_type}: {str(e)}")
            raise

    def _load_model(self, model_type: ModelType):
        """Load model and tokenizer"""
        try:
            model_path = ModelConfig.MODELS[model_type]
            config = ModelConfig.get_model_config(model_type)
            
            # Load tokenizer
            tokenizer = TFAutoTokenizer.from_pretrained(model_path)
            
            # Load model with memory efficiency settings
            model = TFAutoModel.from_pretrained(
                model_path,
                from_pt=True,  # Convert from PyTorch if necessary
                **config
            )
            
            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_type}: {str(e)}")
            raise

    def get_embeddings(self, texts: list, model_type: ModelType) -> np.ndarray:
        """Get embeddings for texts using specified model"""
        try:
            model, tokenizer = self.get_model(model_type)
            config = ModelConfig.get_model_config(model_type)
            
            # Tokenize texts
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config['max_length'],
                return_tensors="tf"
            )
            
            # Get embeddings
            outputs = model(inputs)
            
            # Use pooled output if available, otherwise mean pooling
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
            
            return embeddings.numpy()
            
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            return np.array([])

    def analyze_sentiment(self, texts: list) -> list:
        """Analyze sentiment using FinBERT"""
        try:
            model, tokenizer = self.get_model(ModelType.FINBERT)
            config = ModelConfig.get_model_config(ModelType.FINBERT)
            
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config['max_length'],
                return_tensors="tf"
            )
            
            # Get predictions
            outputs = model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1)
            
            # Convert to sentiment labels
            sentiments = []
            for pred in predictions:
                label_idx = tf.argmax(pred).numpy()
                score = float(tf.reduce_max(pred))
                sentiments.append({
                    'label': ['negative', 'neutral', 'positive'][label_idx],
                    'score': score
                })
            
            return sentiments
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return []

    def classify_text(self, texts: list, labels: list) -> list:
        """Classify text using DistilBERT"""
        try:
            model, tokenizer = self.get_model(ModelType.DISTILBERT)
            config = ModelConfig.get_model_config(ModelType.DISTILBERT)
            
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config['max_length'],
                return_tensors="tf"
            )
            
            # Get predictions
            outputs = model(inputs)
            predictions = tf.nn.softmax(outputs.logits, axis=-1)
            
            # Convert to classification results
            results = []
            for pred in predictions:
                label_idx = tf.argmax(pred).numpy()
                score = float(tf.reduce_max(pred))
                results.append({
                    'label': labels[label_idx],
                    'score': score
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in text classification: {str(e)}")
            return []

    def extract_metrics(self, texts: list) -> list:
        """Extract metrics using RoBERTa"""
        try:
            model, tokenizer = self.get_model(ModelType.ROBERTA)
            config = ModelConfig.get_model_config(ModelType.ROBERTA)
            
            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config['max_length'],
                return_tensors="tf"
            )
            
            # Get embeddings
            outputs = model(inputs)
            embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
            
            # Process embeddings for metric extraction
            # This will be extended based on specific metric extraction logic
            return embeddings.numpy()
            
        except Exception as e:
            self.logger.error(f"Error in metric extraction: {str(e)}")
            return []