"""
Utilities for metric calculations and scoring
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from config.constants import SCORING_THRESHOLDS

class ScoreCalculator:
    """
    Utility class for calculating various scores
    """
    @staticmethod
    def calculate_combined_score(explicit_score: float,
                               semantic_score: float,
                               metric_score: float) -> float:
        """
        Calculate combined score using configured weights
        """
        weights = SCORING_THRESHOLDS['score_weights']
        combined_score = (
            weights['explicit'] * explicit_score +
            weights['semantic'] * semantic_score +
            weights['metric'] * metric_score
        )
        return min(combined_score, 100)

    @staticmethod
    def score_percentage(value: float, 
                        lower_is_better: bool = False) -> float:
        """
        Score percentage values
        """
        thresholds = SCORING_THRESHOLDS['percentage_thresholds']
        
        if lower_is_better:
            if value <= thresholds['fair']:
                return 70
            elif value <= thresholds['good']:
                return 60
            elif value <= thresholds['very_good']:
                return 50
            elif value <= thresholds['exceptional']:
                return 40
            return 30
        else:
            if value > thresholds['exceptional']:
                return 70
            elif value > thresholds['very_good']:
                return 60
            elif value > thresholds['good']:
                return 50
            elif value > thresholds['fair']:
                return 40
            return 30

    @staticmethod
    def score_count(value: float, 
                    lower_is_better: bool = False) -> float:
        """
        Score count/numeric values
        """
        thresholds = SCORING_THRESHOLDS['count_thresholds']
        
        if lower_is_better:
            if value <= thresholds['good']:
                return 70
            elif value <= thresholds['very_good']:
                return 50
            elif value <= thresholds['exceptional']:
                return 30
            return 20
        else:
            if value >= thresholds['exceptional']:
                return 70
            elif value >= thresholds['very_good']:
                return 50
            elif value >= thresholds['good']:
                return 30
            return 20

class MetricAnalyzer:
    """
    Utility class for analyzing metrics
    """
    @staticmethod
    def calculate_trend(values: List[float]) -> Optional[float]:
        """
        Calculate trend (percentage change) in values
        """
        if len(values) < 2:
            return None
        
        first, last = values[0], values[-1]
        if first == 0:
            return None
            
        return ((last - first) / first) * 100

    @staticmethod
    def detect_anomalies(values: List[float], 
                        threshold: float = 2.0) -> List[int]:
        """
        Detect anomalous values using z-score
        """
        if len(values) < 2:
            return []
            
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return []
            
        z_scores = [(x - mean) / std for x in values]
        return [i for i, z in enumerate(z_scores) if abs(z) > threshold]

    @staticmethod
    def summarize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generate summary statistics for metrics
        """
        summary = {}
        for key, values in metrics.items():
            if not values:
                continue
                
            numeric_values = [v for v in values if v is not None]
            if not numeric_values:
                continue
                
            summary[key] = {
                'count': len(numeric_values),
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'trend': MetricAnalyzer.calculate_trend(numeric_values)
            }
        return summary