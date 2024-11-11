"""
Configuration for different models used in ESG analysis
"""
from enum import Enum
from typing import Dict, Any

class ModelType(Enum):
    ESG_BERT = "esg_bert"
    FINBERT = "finbert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"

class ModelConfig:
    """Model configurations and task assignments"""
    
    # Model paths/identifiers
    MODELS = {
        ModelType.ESG_BERT: "nbroad/ESG-BERT",
        ModelType.FINBERT: "ProsusAI/finbert",
        ModelType.ROBERTA: "roberta-base",
        ModelType.DISTILBERT: "distilbert-base-uncased"
    }

    # Task-specific model assignments
    TASK_MODELS = {
        'context_analysis': ModelType.ESG_BERT,
        'metric_extraction': ModelType.ROBERTA,
        'document_classification': ModelType.DISTILBERT,
        'sentiment_analysis': ModelType.FINBERT
    }

    # Model-specific configurations
    MODEL_CONFIGS = {
        ModelType.ESG_BERT: {
            'max_length': 512,
            'batch_size': 16,
            'use_cuda': False,
            'quantize': True  # Use quantization for memory efficiency
        },
        ModelType.FINBERT: {
            'max_length': 256,
            'batch_size': 32,
            'use_cuda': False
        },
        ModelType.ROBERTA: {
            'max_length': 256,
            'batch_size': 32,
            'use_cuda': False
        },
        ModelType.DISTILBERT: {
            'max_length': 512,
            'batch_size': 64,
            'use_cuda': False
        }
    }

    @classmethod
    def get_model_for_task(cls, task: str) -> ModelType:
        """Get appropriate model for a specific task"""
        return cls.TASK_MODELS.get(task, ModelType.ESG_BERT)

    @classmethod
    def get_model_config(cls, model_type: ModelType) -> Dict[str, Any]:
        """Get configuration for specific model"""
        return cls.MODEL_CONFIGS.get(model_type, {})