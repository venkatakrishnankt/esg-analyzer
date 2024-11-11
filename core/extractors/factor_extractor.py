"""
Core factor extraction and analysis functionality
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import time
from config.constants import SCORING_THRESHOLDS, ESG_FACTORS
from utils.logging_utils import LogManager, MetricLogger, PerformanceLogger
from config.enums import MetricType, UnitType, ESGCategory, ScoreType
from config.esg_config import ESGMetricConfig

class FactorExtractor:
    """
    Handles ESG factor extraction and scoring
    """
    def __init__(self, data_processor):
        # Initialize loggers
        self.logger = LogManager()
        self.metric_logger = MetricLogger()
        self.perf_logger = PerformanceLogger()
        
        # Initialize components
        self.data_processor = data_processor
        self.current_factor = None

    def analyze_keyword_frequency(self, text: str, factors: List[str]) -> Dict[str, float]:
        """Calculate frequency of factor mentions"""
        try:
            start_time = time.time()
            word_count = len(text.split())
            
            frequency_dict = {}
            for factor in factors:
                factor_words = factor.split()
                count = sum(text.count(word) for word in factor_words)
                frequency = (count / word_count) * 100
                frequency_dict[factor] = frequency

            self.perf_logger.log_processing_time(
                "Keyword frequency analysis", 
                time.time() - start_time
            )
            return frequency_dict
            
        except Exception as e:
            self.logger.error("Error analyzing keyword frequency", error=e)
            return {}

    def calculate_semantic_score(self, doc_embedding: np.ndarray, factor: str) -> float:
        """Calculate semantic similarity score"""
        try:
            factor_embedding = self.data_processor.get_bert_embedding(factor)
            similarity = np.dot(doc_embedding[0], factor_embedding[0]) / (
                np.linalg.norm(doc_embedding[0]) * np.linalg.norm(factor_embedding[0]))
            score = similarity * 100
            
            print(f"DEBUG - Semantic score for {factor}: {score}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic score for factor: {factor}", error=e)
            return 0.0

    def calculate_explicit_score(self, text: str, factor: str) -> float:
        try:
            metrics = self.data_processor.extract_metrics_from_text(text, factor)
            print(f"\nDEBUG - Extracted Metrics for {factor}: {metrics}")
            
            config = ESGMetricConfig.get_factor_config(factor)
            if not config:
                return 0.0
            
            base_score = config['scoring'].get('base_score', 30)
            metric_score = 0
            
            for key, values in metrics.items():
                for value in values:
                    try:
                        clean_value = float(value.replace(',', '').strip())
                        print(f"DEBUG - Processing value: {clean_value}")
                        
                        thresholds = config['scoring']['thresholds']
                        scores = config['scoring']['scores']
                        lower_is_better = config['scoring'].get('lower_is_better', False)
                        
                        # Compare against thresholds
                        for i, threshold in enumerate(thresholds):
                            if lower_is_better and clean_value <= threshold:
                                metric_score = max(metric_score, scores[i])
                                break
                            elif not lower_is_better and clean_value >= threshold:
                                metric_score = max(metric_score, scores[i])
                                break
                        else:
                            metric_score = max(metric_score, scores[-1])
                        
                        print(f"DEBUG - Metric score: {metric_score}")
                    except ValueError as e:
                        print(f"DEBUG - Error processing value: {e}")
                        continue
            
            final_score = min(base_score + metric_score, 100)
            print(f"DEBUG - Final explicit score for {factor}: {final_score}")
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating explicit score for factor: {factor}", error=e)
            return 0.0
        
    def _score_percentage(self, value: float, lower_is_better: bool = False) -> float:
        """Score percentage values"""
        if lower_is_better:
            if value <= 30:
                return 70
            elif value <= 40:
                return 60
            elif value <= 50:
                return 50
            elif value <= 60:
                return 40
            return 30
        else:
            if value > 60:
                return 70
            elif value > 50:
                return 60
            elif value > 40:
                return 50
            elif value > 30:
                return 40
            return 30

    def _score_measurement(self, value: float) -> float:
        """Score measurement values"""
        factor_lower = self.current_factor.lower() if self.current_factor else ''
        
        # Detect scale
        scale_multiplier = 1
        if 'million' in factor_lower or 'mn' in factor_lower:
            scale_multiplier = 1_000_000
        elif 'thousand' in factor_lower or 'k' in factor_lower:
            scale_multiplier = 1_000
        
        adjusted_value = value * scale_multiplier
        
        # Determine scoring direction
        lower_is_better = any(term in factor_lower for term in [
            'emissions', 'waste', 'consumption', 'pollution',
            'energy use', 'water use', 'carbon'
        ])
        
        if lower_is_better:
            if adjusted_value <= 1000:
                return 70
            elif adjusted_value <= 2500:
                return 60
            elif adjusted_value <= 5000:
                return 50
            elif adjusted_value <= 7500:
                return 40
            return 30
        else:
            if adjusted_value >= 10000:
                return 70
            elif adjusted_value >= 7500:
                return 60
            elif adjusted_value >= 5000:
                return 50
            elif adjusted_value >= 2500:
                return 40
            return 30

    def analyze_metric_trends(self, text: str, factor: str) -> Dict[str, Any]:
        """Analyze metric trends with enhanced analysis"""
        try:
            metrics = self.data_processor.extract_metrics_from_text(text, factor)
            
            analysis = {
                'has_metrics': len(metrics) > 0,
                'metric_types': [],
                'values': {}
            }
            
            for key, values in metrics.items():
                metric_type = key.split('_')[-1]
                analysis['metric_types'].append(metric_type)
                
                try:
                    numeric_values = [float(v.replace(',', '')) for v in values]
                    analysis['values'][key] = {
                        'raw': values,
                        'numeric': numeric_values,
                        'average': np.mean(numeric_values) if numeric_values else None,
                        'max': max(numeric_values) if numeric_values else None,
                        'trend': self._calculate_trend(numeric_values) if len(numeric_values) > 1 else None
                    }
                except (ValueError, TypeError):
                    analysis['values'][key] = {
                        'raw': values,
                        'numeric': None,
                        'average': None,
                        'max': None,
                        'trend': None
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing metric trends for factor: {factor}", error=e)
            return {'has_metrics': False, 'metric_types': [], 'values': {}}

    def _calculate_trend(self, values: List[float]) -> Optional[float]:
        """Calculate trend as percentage change"""
        try:
            if len(values) < 2:
                return None
            return ((values[-1] - values[0]) / values[0]) * 100
        except (IndexError, ZeroDivisionError):
            return None

    def extract_factors(self, text: str, doc_embedding: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """Extract and analyze ESG factors"""
        try:
            start_time = time.time()
            
            extracted_factors = {}
            context_windows = {}
            factor_details = {}
            
            for category, factors in ESG_FACTORS.items():
                for factor in factors:
                    self.current_factor = factor
                    print(f"\nDEBUG - Processing factor: {factor}")
                    
                    # Get and verify context
                    context = self.data_processor.get_context_window(text, factor)
                    print(f"\nDEBUG - Initial context captured:\n{context}")
                    
                    if context == "Context not found":
                        continue
                        
                    # Calculate scores
                    explicit_score = self.calculate_explicit_score(text, factor)
                    semantic_score = self.calculate_semantic_score(doc_embedding, factor)
                    
                    # Calculate weighted score
                    weights = SCORING_THRESHOLDS['score_weights']
                    combined_score = (
                        weights['explicit'] * explicit_score +
                        weights['semantic'] * semantic_score
                    )
                    
                    final_score = min(combined_score, 100)
                    
                    if final_score > SCORING_THRESHOLDS['minimum_score']:
                        extracted_factors[factor] = final_score
                        context_windows[factor] = context
                        factor_details[factor] = {
                            "explicit_score": explicit_score,
                            "semantic_score": semantic_score,
                            "combined_score": combined_score,
                            "final_score": final_score,
                            "context": context,
                            "metrics": self.analyze_metric_trends(text, factor)
                        }
            
            self.perf_logger.log_processing_time(
                "Factor extraction",
                time.time() - start_time
            )
            
            self.current_factor = None
            return extracted_factors, context_windows, factor_details
            
        except Exception as e:
            self.logger.error("Error in factor extraction", error=e)
            self.current_factor = None
            return {}, {}, {}