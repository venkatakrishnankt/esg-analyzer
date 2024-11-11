"""
Core ESG analysis functionality using specialized models
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import tensorflow as tf
from config.models.model_config import ModelType, ModelConfig
from core.models.model_manager import ModelManager
from utils.logging_utils import get_logger

class ESGAnalyzer:
    """
    Analyzes ESG factors using multiple specialized models
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_manager = ModelManager()
        
        # Initialize category classifiers
        self.categories = {
            'E': ['environmental', 'climate', 'emissions', 'energy', 'waste'],
            'S': ['social', 'employee', 'community', 'human rights', 'health'],
            'G': ['governance', 'board', 'ethics', 'compliance', 'risk']
        }

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive ESG analysis of document
        """
        try:
            # Break document into manageable chunks
            chunks = self._split_into_chunks(text)
            
            results = {
                'factors': self._identify_esg_factors(chunks),
                'metrics': self._extract_metrics(chunks),
                'sentiment': self._analyze_esg_sentiment(chunks),
                'categories': self._classify_esg_categories(chunks),
                'summary': {}  # Will be populated after analysis
            }
            
            # Generate summary
            results['summary'] = self._generate_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ESG analysis: {str(e)}")
            return {}

    def _split_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split text into chunks for processing
        """
        try:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return [text]

    def _identify_esg_factors(self, chunks: List[str]) -> Dict[str, List[Dict]]:
        """
        Identify ESG factors using ESG-BERT
        """
        try:
            factors = {'E': [], 'S': [], 'G': []}
            
            # Get embeddings for chunks
            embeddings = self.model_manager.get_embeddings(
                chunks, ModelType.ESG_BERT
            )
            
            for i, chunk in enumerate(chunks):
                # Classify ESG category
                category = self._classify_esg_category(embeddings[i])
                
                # Extract factors based on category
                chunk_factors = self._extract_factors_for_category(
                    chunk, category, embeddings[i]
                )
                
                factors[category].extend(chunk_factors)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error identifying ESG factors: {str(e)}")
            return {'E': [], 'S': [], 'G': []}

    def _extract_metrics(self, chunks: List[str]) -> Dict[str, List[Dict]]:
        """
        Extract ESG metrics using RoBERTa
        """
        try:
            metrics = {}
            
            for chunk in chunks:
                # Use RoBERTa for numeric understanding
                chunk_metrics = self.model_manager.extract_metrics([chunk])
                
                for metric in chunk_metrics:
                    category = metric['category']
                    if category not in metrics:
                        metrics[category] = []
                    metrics[category].append(metric)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}")
            return {}

    def _analyze_esg_sentiment(self, chunks: List[str]) -> Dict[str, float]:
        """
        Analyze ESG sentiment using FinBERT
        """
        try:
            sentiments = self.model_manager.analyze_sentiment(chunks)
            
            # Aggregate sentiments by category
            results = {'E': 0.0, 'S': 0.0, 'G': 0.0}
            counts = {'E': 0, 'S': 0, 'G': 0}
            
            for chunk, sentiment in zip(chunks, sentiments):
                category = self._classify_esg_category(chunk)
                
                # Convert sentiment to score (-1 to 1)
                score = {
                    'positive': 1.0,
                    'neutral': 0.0,
                    'negative': -1.0
                }[sentiment['label']]
                
                results[category] += score
                counts[category] += 1
            
            # Calculate averages
            for category in results:
                if counts[category] > 0:
                    results[category] /= counts[category]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing ESG sentiment: {str(e)}")
            return {'E': 0.0, 'S': 0.0, 'G': 0.0}

    def _classify_esg_categories(self, chunks: List[str]) -> Dict[str, float]:
        """
        Classify text into ESG categories using DistilBERT
        """
        try:
            labels = ['Environmental', 'Social', 'Governance']
            classifications = self.model_manager.classify_text(chunks, labels)
            
            # Aggregate classification scores
            scores = {'E': 0.0, 'S': 0.0, 'G': 0.0}
            for classification in classifications:
                category = classification['label'][0]  # First letter of category
                scores[category] += classification['score']
            
            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error classifying ESG categories: {str(e)}")
            return {'E': 0.0, 'S': 0.0, 'G': 0.0}

    def _classify_esg_category(self, text_or_embedding: Any) -> str:
        """
        Classify single text or embedding into ESG category
        """
        try:
            if isinstance(text_or_embedding, str):
                # Convert text to embedding using ESG-BERT
                embedding = self.model_manager.get_embeddings(
                    [text_or_embedding], ModelType.ESG_BERT
                )[0]
            else:
                embedding = text_or_embedding
            
            # Calculate similarity with category keywords
            max_score = -1
            best_category = 'E'  # Default to Environmental
            
            for category, keywords in self.categories.items():
                # Get embeddings for keywords
                keyword_embeddings = self.model_manager.get_embeddings(
                    keywords, ModelType.ESG_BERT
                )
                
                # Calculate average similarity
                similarities = [
                    np.dot(embedding, kw_emb) / 
                    (np.linalg.norm(embedding) * np.linalg.norm(kw_emb))
                    for kw_emb in keyword_embeddings
                ]
                score = np.mean(similarities)
                
                if score > max_score:
                    max_score = score
                    best_category = category
            
            return best_category
            
        except Exception as e:
            self.logger.error(f"Error classifying ESG category: {str(e)}")
            return 'E'  # Default to Environmental

    def _extract_factors_for_category(self, text: str, category: str, 
                                    embedding: np.ndarray) -> List[Dict]:
        """
        Extract factors for a specific ESG category
        """
        try:
            factors = []
            # Use RoBERTa for metric extraction
            metrics = self.model_manager.extract_metrics([text])
            
            # Use ESG-BERT for context understanding
            context = self._analyze_context(text, category)
            
            # Combine metrics and context
            for metric in metrics:
                factor = {
                    'text': text,
                    'metric': metric,
                    'context': context,
                    'category': category,
                    'confidence': self._calculate_confidence(metric, context)
                }
                factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error extracting factors: {str(e)}")
            return []

    def _analyze_context(self, text: str, category: str) -> Dict[str, Any]:
        """
        Analyze context using ESG-BERT
        """
        try:
            # Get embeddings for text
            embedding = self.model_manager.get_embeddings(
                [text], ModelType.ESG_BERT
            )[0]
            
            # Analyze temporal aspects
            temporal = self._extract_temporal_info(text)
            
            # Analyze relationships
            relationships = self._extract_relationships(text)
            
            return {
                'temporal': temporal,
                'relationships': relationships,
                'embedding': embedding.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing context: {str(e)}")
            return {}

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of ESG analysis
        """
        try:
            summary = {
                'category_distribution': results['categories'],
                'sentiment_overview': results['sentiment'],
                'key_factors': self._extract_key_factors(results['factors']),
                'metric_summary': self._summarize_metrics(results['metrics'])
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {}

    def _extract_key_factors(self, factors: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Extract key factors based on confidence and relevance
        """
        try:
            key_factors = []
            
            for category, category_factors in factors.items():
                # Sort factors by confidence
                sorted_factors = sorted(
                    category_factors,
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )
                
                # Take top factors
                key_factors.extend(sorted_factors[:5])
            
            return key_factors
            
        except Exception as e:
            self.logger.error(f"Error extracting key factors: {str(e)}")
            return []

    def _summarize_metrics(self, metrics: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate summary statistics for metrics
        """
        try:
            summary = {}
            
            for category, category_metrics in metrics.items():
                values = [m['value'] for m in category_metrics if 'value' in m]
                if values:
                    summary[category] = {
                        'count': len(values),
                        'average': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing metrics: {str(e)}")
            return {}