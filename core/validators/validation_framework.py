"""
Validation framework for ESG metrics and analysis
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from config.base.enums import MetricType, ESGCategory
from utils.logging_utils import get_logger

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class ESGValidator:
    """
    Comprehensive validation for ESG metrics and analysis
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self._setup_validation_rules()

    def _setup_validation_rules(self):
        """Setup validation rules for different metric types"""
        self.validation_rules = {
            # Environmental metrics
            'emissions': {
                'type': MetricType.NUMERIC,
                'category': ESGCategory.ENVIRONMENTAL,
                'range': (0, 1e9),  # Up to 1 billion tons
                'units': ['tCO2e', 'tonnes CO2', 'MT'],
                'required_context': ['scope', 'emission', 'carbon'],
                'temporal_required': True
            },
            'energy_consumption': {
                'type': MetricType.NUMERIC,
                'category': ESGCategory.ENVIRONMENTAL,
                'range': (0, 1e9),  # Up to 1 billion kWh
                'units': ['kWh', 'MWh', 'GWh'],
                'required_context': ['energy', 'consumption', 'usage'],
                'temporal_required': True
            },
            
            # Social metrics
            'employee_volunteering_participation': {
                'type': MetricType.PERCENTAGE,
                'category': ESGCategory.SOCIAL,
                'range': (0, 100),
                'units': ['%'],
                'required_context': ['employee', 'volunteering', 'participation'],
                'temporal_required': True,
                'historical_variance_threshold': 30  # % change threshold
            },
            'employee_volunteering_days': {
                'type': MetricType.NUMERIC,
                'category': ESGCategory.SOCIAL,
                'range': (0, 1e6),  # Up to 1 million days
                'units': ['days'],
                'required_context': ['employee', 'volunteering', 'days'],
                'temporal_required': True,
                'historical_variance_threshold': 50  # % change threshold
            },
            
            # Governance metrics
            'board_independence': {
                'type': MetricType.PERCENTAGE,
                'category': ESGCategory.GOVERNANCE,
                'range': (0, 100),
                'units': ['%'],
                'required_context': ['board', 'independent', 'director'],
                'temporal_required': True
            }
        }

    def validate_metric(self, metric: Dict[str, Any], 
                       context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate a single metric
        """
        try:
            errors = []
            warnings = []
            metadata = {}
            
            # Get validation rules
            rules = self._get_validation_rules(metric['type'])
            if not rules:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=['Unknown metric type'],
                    warnings=[],
                    metadata={}
                )
            
            # Basic validation
            if not self._validate_basic_requirements(metric, rules, errors):
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    errors=errors,
                    warnings=warnings,
                    metadata=metadata
                )
            
            # Value range validation
            if not self._validate_value_range(metric, rules, errors):
                warnings.append('Value outside typical range')
            
            # Context validation
            context_score = self._validate_context(metric, rules, context, warnings)
            
            # Temporal validation
            temporal_score = self._validate_temporal_info(metric, rules, warnings)
            
            # Historical validation if context provided
            historical_score = 1.0
            if context and 'historical_data' in context:
                historical_score = self._validate_historical(
                    metric, rules, context['historical_data'], warnings
                )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                context_score, temporal_score, historical_score
            )
            
            # Add metadata
            metadata.update({
                'context_score': context_score,
                'temporal_score': temporal_score,
                'historical_score': historical_score,
                'validation_timestamp': datetime.now().isoformat()
            })
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error validating metric: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=[str(e)],
                warnings=[],
                metadata={}
            )

    def _get_validation_rules(self, metric_type: str) -> Dict[str, Any]:
        """Get validation rules for metric type"""
        return self.validation_rules.get(metric_type, {})

    def _validate_basic_requirements(self, metric: Dict[str, Any], 
                                   rules: Dict[str, Any], 
                                   errors: List[str]) -> bool:
        """Validate basic metric requirements"""
        try:
            # Check required fields
            required_fields = ['value', 'unit', 'type']
            for field in required_fields:
                if field not in metric:
                    errors.append(f"Missing required field: {field}")
                    return False
            
            # Validate unit
            if metric['unit'] not in rules.get('units', []):
                errors.append(f"Invalid unit: {metric['unit']}")
                return False
            
            # Validate metric type
            if metric['type'] != rules.get('type'):
                errors.append(f"Invalid metric type: {metric['type']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in basic validation: {str(e)}")
            return False

    def _validate_value_range(self, metric: Dict[str, Any], 
                            rules: Dict[str, Any], 
                            errors: List[str]) -> bool:
        """Validate metric value range"""
        try:
            value = float(metric['value'])
            min_val, max_val = rules.get('range', (float('-inf'), float('inf')))
            
            if not min_val <= value <= max_val:
                errors.append(
                    f"Value {value} outside valid range [{min_val}, {max_val}]"
                )
                return False
            
            return True
            
        except ValueError:
            errors.append("Invalid numeric value")
            return False
        except Exception as e:
            self.logger.error(f"Error in range validation: {str(e)}")
            return False

    def _validate_context(self, metric: Dict[str, Any], 
                         rules: Dict[str, Any],
                         context: Optional[Dict[str, Any]], 
                         warnings: List[str]) -> float:
        """Validate metric context"""
        try:
            if not context:
                warnings.append("No context provided")
                return 0.5
            
            score = 1.0
            required_context = rules.get('required_context', [])
            
            # Check for required context words
            context_text = context.get('text', '').lower()
            found_context = sum(
                1 for word in required_context if word in context_text
            )
            context_score = found_context / len(required_context) if required_context else 1.0
            
            if context_score < 0.5:
                warnings.append("Missing important context elements")
            
            score *= context_score
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in context validation: {str(e)}")
            return 0.5

    def _validate_temporal_info(self, metric: Dict[str, Any], 
                              rules: Dict[str, Any],
                              warnings: List[str]) -> float:
        """Validate temporal information"""
        try:
            if not rules.get('temporal_required', False):
                return 1.0
            
            temporal_info = metric.get('temporal_info', {})
            if not temporal_info:
                warnings.append("Missing temporal information")
                return 0.5
            
            score = 1.0
            
            # Check for required temporal elements
            if 'year' not in temporal_info:
                warnings.append("Missing year information")
                score *= 0.7
            
            if 'period_type' not in temporal_info:
                warnings.append("Missing period type")
                score *= 0.9
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in temporal validation: {str(e)}")
            return 0.5

    def _validate_historical(self, metric: Dict[str, Any], 
                           rules: Dict[str, Any],
                           historical_data: List[Dict[str, Any]], 
                           warnings: List[str]) -> float:
        """Validate against historical data"""
        try:
            if not historical_data:
                return 1.0
            
            current_value = float(metric['value'])
            historical_values = [float(h['value']) for h in historical_data]
            
            # Calculate statistics
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            if std == 0:
                return 1.0
            
            # Calculate z-score
            z_score = abs(current_value - mean) / std
            
            # Get threshold
            threshold = rules.get('historical_variance_threshold', 3)
            
            if z_score > threshold:
                warnings.append(
                    f"Value deviates significantly from historical data "
                    f"(z-score: {z_score:.2f})"
                )
                return 0.5
            
            # Calculate confidence based on z-score
            confidence = max(0.0, min(1.0, 1.0 - (z_score / threshold)))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error in historical validation: {str(e)}")
            return 0.5

    def _calculate_confidence(self, context_score: float, 
                            temporal_score: float,
                            historical_score: float) -> float:
        """Calculate overall confidence score"""
        try:
            # Weighted average of scores
            weights = {
                'context': 0.4,
                'temporal': 0.3,
                'historical': 0.3
            }
            
            confidence = (
                weights['context'] * context_score +
                weights['temporal'] * temporal_score +
                weights['historical'] * historical_score
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5