"""
Core data processing functionality for ESG analysis
"""
import time
import PyPDF2
import re
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
import nltk
from typing import Dict, List, Optional, Tuple
from config.constants import CUSTOM_STOP_WORDS, WINDOW_SIZES
from utils.logging_utils import LogManager, PerformanceLogger
from .factor_patterns import ESGFactorClassifier, MetricPatternBuilder, MetricExtractor, MetricType
from config.enums import MetricType, PatternType
from config.esg_config import ESGMetricConfig

class DataProcessor:
    """
    Handles text extraction and processing
    """
    def __init__(self):
        self.logger = LogManager()
        self.perf_logger = PerformanceLogger()
        
        # Initialize BERT
        self.logger.info("Initializing BERT model and tokenizer")
        self._tokenizer = None
        self._model = None
        
        # Initialize pattern system
        self.classifier = ESGFactorClassifier()
        self.pattern_builder = MetricPatternBuilder(self.classifier)
        self.metric_extractor = MetricExtractor(self.classifier, self.pattern_builder)
        
        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(CUSTOM_STOP_WORDS)

        # Initialize specific patterns
        self.specific_phrases = {
            'employee volunteering participation rate': ['employee volunteering participation rate'],
            'employee volunteering days': ['employee volunteering days'],
            'carbon emissions': ['carbon emissions', 'co2 emissions', 'scope 1 2 emissions'],
            'board diversity': ['board diversity', 'women holding senior leadership'],
            'sustainability': ['sustainability', 'sustainable development']
        }

    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return self._tokenizer

    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            self._model = TFBertModel.from_pretrained('bert-base-uncased')
        return self._model

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract and clean text from PDF"""
        start_time = time.time()
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ' '.join([page.extract_text() for page in reader.pages])
            print(f"\nDEBUG - Raw PDF text (first 500 chars):\n{text[:500]}")
            
            # Clean extracted text
            text = self.clean_text(text)
            print(f"\nDEBUG - Cleaned PDF text (first 500 chars):\n{text[:500]}")
            
            self.perf_logger.log_processing_time("PDF extraction", time.time() - start_time)
            return text
            
        except Exception as e:
            self.logger.error("Error extracting text from PDF", error=e)
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        try:
            # Basic cleanup
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
            text = re.sub(r'\*+', '', text)  # Remove asterisks
            text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Space between letters and numbers
            text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Space between numbers and letters
            
            # Remove stop words
            text = text.lower()
            words = text.split()
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
            
        except Exception as e:
            self.logger.error("Error cleaning text", error=e)
            return text

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        start_time = time.time()
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="tf",
                truncation=True,
                max_length=512
            )
            outputs = self.model(inputs)
            embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
            
            self.perf_logger.log_processing_time("BERT embedding", time.time() - start_time)
            return embedding
            
        except Exception as e:
            self.logger.error("Error generating BERT embedding", error=e)
            raise

    def get_context_window(self, text: str, factor: str, window_size: Optional[int] = None) -> str:
        """Get relevant context window for a factor"""
        try:
            if window_size is None:
                window_size = WINDOW_SIZES['context']
                
            text_lower = text.lower()
            factor_lower = factor.lower()
            
            # Get search phrases for the factor
            search_phrases = self.specific_phrases.get(factor_lower, [factor_lower])
            
            # Find best context
            best_context = None
            best_position = float('inf')
            
            for phrase in search_phrases:
                index = text_lower.find(phrase)
                if index != -1 and index < best_position:
                    # Find sentence boundaries
                    start = max(0, index - window_size)
                    end = min(len(text), index + len(phrase) + window_size)
                    
                    # Look for sentence boundaries
                    while start > max(0, index - window_size) and text[start] not in '.!?\n':
                        start -= 1
                    if start > 0:
                        start += 1
                        
                    while end < min(len(text), index + window_size) and text[end] not in '.!?\n':
                        end += 1
                    
                    context = text[start:end].strip()
                    context = ' '.join(context.split())  # Normalize spaces
                    
                    if index < best_position:
                        best_context = context
                        best_position = index
            
            return best_context if best_context else "Context not found"
            
        except Exception as e:
            self.logger.error(f"Error getting context for factor: {factor}", error=e)
            return "Context not found"

    def extract_metrics_from_text(self, text: str, factor: str) -> Dict[str, List[str]]:
        try:
            context = self.get_context_window(text, factor)
            print(f"\nDEBUG - Context for metric extraction ({factor}):\n{context}")
            
            config = ESGMetricConfig.get_factor_config(factor)
            if not config:
                return {}

            metrics = {}
            metric_type = config['metric_type']
            key = f"{factor}_{metric_type.value}"

            # Use configured patterns
            for pattern_str in config['patterns']:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.finditer(context)
                
                for match in matches:
                    value = next((g for g in match.groups() if g is not None), None)
                    if value:
                        if key not in metrics:
                            metrics[key] = []
                        clean_value = self._clean_value(value, config)
                        if clean_value:
                            metrics[key].append(clean_value)

            print(f"DEBUG - Extracted metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics for {factor}: {str(e)}")
            return {}
            
    def _clean_value(self, value: str, config: Dict) -> Optional[str]:
        """Clean value based on metric type"""
        try:
            value = value.strip().replace(',', '')
            metric_type = config['metric_type']
            
            if metric_type == MetricType.PERCENTAGE:
                clean_val = float(value)
                if 0 <= clean_val <= 100:
                    return str(clean_val)
            elif metric_type == MetricType.COUNT:
                clean_val = float(value)
                if clean_val >= 0:
                    return str(clean_val)
            elif metric_type == MetricType.MEASUREMENT:
                return value  # Keep original format for measurements
                
            return None
        except ValueError:
            return None