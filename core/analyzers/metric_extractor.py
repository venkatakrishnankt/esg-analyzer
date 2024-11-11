"""
Metric extraction using RoBERTa and pattern matching
"""
from typing import Dict, List, Optional, Tuple, Any
import re
import numpy as np
from dataclasses import dataclass
from config.models.model_config import ModelType
from core.models.model_manager import ModelManager
from utils.logging_utils import get_logger

@dataclass
class ExtractedMetric:
    """Container for extracted metric information"""
    value: float
    unit: str
    context: str
    confidence: float
    category: str
    metric_type: str
    temporal_info: Dict[str, Any]
    source_text: str
    position: Tuple[int, int]

class MetricExtractor:
    """
    Advanced metric extraction using RoBERTa and pattern matching
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_manager = ModelManager()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for metric extraction"""
        self.patterns = {
            'number': re.compile(
                r'(?:^|[\s:])([><]?\s*(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*'
                r'(?:million|billion|mn|bn|m|b|k)?',
                re.IGNORECASE
            ),
            'percentage': re.compile(
                r'(?:^|[\s:])([><]?\s*\d+(?:\.\d+)?)\s*%',
                re.IGNORECASE
            ),
            'currency': re.compile(
                r'(?:^|[\s:])(?:USD|€|£|\$)?\s*([><]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
                r'\s*(?:million|billion|mn|bn|m|b|k)?',
                re.IGNORECASE
            ),
            'temporal': re.compile(
                r'\b(?:in|during|for|as of|year|FY|Q[1-4]|quarter|period|'
                r'January|February|March|April|May|June|July|August|'
                r'September|October|November|December|'
                r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                r'\s*(?:20\d{2}|'\d{2}|[12][0-9]|[1-4]Q)?\b',
                re.IGNORECASE
            )
        }

        # ESG-specific metric patterns
        self.esg_patterns = {
            'emissions': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*(?:tCO2e|tonnes?\s+CO2|MT|tons?)',
                re.IGNORECASE
            ),
            'energy': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*(?:MWh|GWh|kWh)',
                re.IGNORECASE
            ),
            'water': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*(?:m3|cubic\s+meters?|gallons?)',
                re.IGNORECASE
            ),
            'waste': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*(?:tons?|tonnes?|kg)\s+(?:of\s+)?(?:waste|recycl)',
                re.IGNORECASE
            ),
            'diversity': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*%\s*(?:women|females?|gender|diversity)',
                re.IGNORECASE
            ),
            'training': re.compile(
                r'(?:\d+(?:\.\d+)?)\s*(?:hours?|days?)\s+(?:of\s+)?training',
                re.IGNORECASE
            )
        }

    def extract_metrics(self, text: str) -> List[ExtractedMetric]:
        """
        Extract metrics from text using combined approach
        """
        try:
            metrics = []
            
            # Get RoBERTa embeddings for context understanding
            embeddings = self.model_manager.get_embeddings(
                [text], ModelType.ROBERTA
            )[0]
            
            # Find all potential metrics
            metric_matches = self._find_metric_matches(text)
            
            # Process each match
            for match in metric_matches:
                metric = self._process_metric_match(match, text, embeddings)
                if metric:
                    metrics.append(metric)
            
            # Post-process to remove duplicates and low confidence matches
            metrics = self._post_process_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}")
            return []

    def _find_metric_matches(self, text: str) -> List[Dict[str, Any]]:
        """
        Find all potential metric matches in text
        """
        matches = []
        
        # Check general patterns
        for pattern_type, pattern in self.patterns.items():
            if pattern_type != 'temporal':
                for match in pattern.finditer(text):
                    matches.append({
                        'value': match.group(1),
                        'type': pattern_type,
                        'start': match.start(),
                        'end': match.end(),
                        'pattern': 'general'
                    })
        
        # Check ESG-specific patterns
        for metric_type, pattern in self.esg_patterns.items():
            for match in pattern.finditer(text):
                matches.append({
                    'value': match.group(0),
                    'type': metric_type,
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': 'esg'
                })
        
        return matches

    def _process_metric_match(self, match: Dict[str, Any], 
                            text: str, 
                            embeddings: np.ndarray) -> Optional[ExtractedMetric]:
        """
        Process a single metric match
        """
        try:
            # Extract context
            context = self._get_metric_context(text, match['start'], match['end'])
            
            # Get temporal information
            temporal_info = self._extract_temporal_info(context)
            
            # Clean and convert value
            clean_value = self._clean_metric_value(match['value'], match['type'])
            if clean_value is None:
                return None
            
            # Determine unit
            unit = self._determine_unit(match['value'], match['type'])
            
            # Calculate confidence
            confidence = self._calculate_metric_confidence(
                match, context, embeddings
            )
            
            # Determine category
            category = self._determine_category(context, match['type'])
            
            return ExtractedMetric(
                value=clean_value,
                unit=unit,
                context=context,
                confidence=confidence,
                category=category,
                metric_type=match['type'],
                temporal_info=temporal_info,
                source_text=text[match['start']:match['end']],
                position=(match['start'], match['end'])
            )
            
        except Exception as e:
            self.logger.error(f"Error processing metric match: {str(e)}")
            return None

    def _get_metric_context(self, text: str, start: int, 
                          end: int, window_size: int = 150) -> str:
        """
        Get context around metric with smart boundary detection
        """
        try:
            # Find sentence boundaries
            text_before = text[max(0, start - window_size):start]
            text_after = text[end:min(len(text), end + window_size)]
            
            sentence_start = text_before.rfind('.')
            if sentence_start == -1:
                sentence_start = 0
            else:
                sentence_start += 1
            
            sentence_end = text_after.find('.')
            if sentence_end == -1:
                sentence_end = len(text_after)
            
            context = (text_before[sentence_start:] + 
                      text[start:end] + 
                      text_after[:sentence_end]).strip()
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting metric context: {str(e)}")
            return text[max(0, start - 50):min(len(text), end + 50)]

    def _clean_metric_value(self, value: str, metric_type: str) -> Optional[float]:
        """
        Clean and convert metric value to float
        """
        try:
            # Remove special characters and whitespace
            value = value.strip('><').strip()
            
            # Handle multipliers
            multiplier = 1
            if any(x in value.lower() for x in ['billion', 'bn', 'b']):
                multiplier = 1_000_000_000
            elif any(x in value.lower() for x in ['million', 'mn', 'm']):
                multiplier = 1_000_000
            elif any(x in value.lower() for x in ['thousand', 'k']):
                multiplier = 1_000
            
            # Clean value
            value = re.sub(r'[^\d.,]', '', value)
            value = value.replace(',', '')
            
            # Convert to float and apply multiplier
            return float(value) * multiplier
            
        except ValueError:
            return None

    def _determine_unit(self, value: str, metric_type: str) -> str:
        """
        Determine unit from value and metric type
        """
        value_lower = value.lower()
        
        if metric_type == 'percentage':
            return '%'
        elif metric_type == 'currency':
            if '$' in value:
                return 'USD'
            elif '€' in value:
                return 'EUR'
            elif '£' in value:
                return 'GBP'
            return 'USD'  # Default currency
        elif metric_type == 'emissions':
            if 'tco2e' in value_lower:
                return 'tCO2e'
            return 'tonnes CO2'
        elif metric_type == 'energy':
            if 'mwh' in value_lower:
                return 'MWh'
            elif 'gwh' in value_lower:
                return 'GWh'
            return 'kWh'
        
        return ''

    def _calculate_metric_confidence(self, match: Dict[str, Any], 
                                   context: str, 
                                   embeddings: np.ndarray) -> float:
        """
        Calculate confidence score for metric
        """
        try:
            confidence = 1.0
            
            # Pattern type confidence
            if match['pattern'] == 'esg':
                confidence *= 0.9  # Higher confidence for ESG-specific patterns
            else:
                confidence *= 0.7
            
            # Context relevance
            if self._is_relevant_context(context, match['type']):
                confidence *= 0.9
            else:
                confidence *= 0.6
            
            # Value reasonability
            if self._is_reasonable_value(match['value'], match['type']):
                confidence *= 0.9
            else:
                confidence *= 0.5
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _is_relevant_context(self, context: str, metric_type: str) -> bool:
        """
        Check if context is relevant for metric type
        """
        context_lower = context.lower()
        
        # Keyword sets for different metric types
        relevance_keywords = {
            'emissions': {'carbon', 'emission', 'ghg', 'scope', 'climate'},
            'energy': {'energy', 'power', 'electricity', 'consumption'},
            'water': {'water', 'consumption', 'usage', 'withdrawal'},
            'waste': {'waste', 'recycl', 'disposal', 'landfill'},
            'diversity': {'women', 'gender', 'diversity', 'female', 'board'},
            'training': {'training', 'development', 'learning', 'hours'}
        }
        
        keywords = relevance_keywords.get(metric_type, set())
        return any(keyword in context_lower for keyword in keywords)

    def _is_reasonable_value(self, value: str, metric_type: str) -> bool:
        """
        Check if value is within reasonable range for metric type
        """
        try:
            clean_value = self._clean_metric_value(value, metric_type)
            if clean_value is None:
                return False
            
            # Reasonable ranges for different metric types
            ranges = {
                'percentage': (0, 100),
                'emissions': (0, 1_000_000_000),  # Up to 1B tonnes
                'energy': (0, 1_000_000_000),     # Up to 1B kWh
                'water': (0, 1_000_000_000),      # Up to 1B m3
                'waste': (0, 1_000_000_000),      # Up to 1B tonnes
                'diversity': (0, 100),            # Percentage
                'training': (0, 1000)             # Hours per person
            }
            
            min_val, max_val = ranges.get(metric_type, (float('-inf'), float('inf')))
            return min_val <= clean_value <= max_val
            
        except Exception:
            return False

    def _post_process_metrics(self, metrics: List[ExtractedMetric]) -> List[ExtractedMetric]:
        """
        Post-process metrics to remove duplicates and low confidence matches
        """
        if not metrics:
            return []
        
        # Sort by confidence
        metrics.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates and low confidence metrics
        seen_positions = set()
        final_metrics = []
        
        for metric in metrics:
            position = metric.position
            if position not in seen_positions and metric.confidence > 0.3:
                seen_positions.add(position)
                final_metrics.append(metric)
        
        return final_metrics

    def _extract_temporal_info(self, context: str) -> Dict[str, Any]:
        """
        Extract temporal information from context
        """
        temporal_info = {
            'year': None,
            'quarter': None,
            'period_type': None
        }
        
        matches = self.patterns['temporal'].finditer(context)
        for match in matches:
            text = match.group(0).lower()
            
            # Extract year
            year_match = re.search(r'20\d{2}', text)
            if year_match:
                temporal_info['year'] = int(year_match.group(0))
            
            # Extract quarter
            quarter_match = re.search(r'q[1-4]', text)
            if quarter_match:
                temporal_info['quarter'] = int(quarter_match.group(0)[1])
            
            # Determine period type
            if 'quarter' in text or 'q' in text:
                temporal_info['period_type'] = 'quarter'
            elif 'year' in text or 'fy' in text:
                temporal_info['period_type'] = 'year'
        
        return temporal_info

    def _determine_category(self, context: str, metric_type: str) -> str:
        """
        Determine ESG category for metric
        """
        context_lower = context.lower()
        
        # Environmental indicators
        if any(word in context_lower for word in [
            'emission', 'carbon', 'energy', 'water', 'waste', 
            'recycl', 'climate', 'environmental', 'pollution',
            'renewable', 'biodiversity'
        ]):
            return 'E'
            
        # Social indicators
        elif any(word in context_lower for word in [
            'employee', 'worker', 'staff', 'training', 'diversity',
            'gender', 'community', 'social', 'health', 'safety',
            'human rights', 'labor', 'customer'
        ]):
            return 'S'
            
        # Governance indicators
        elif any(word in context_lower for word in [
            'board', 'governance', 'compliance', 'ethics', 'risk',
            'audit', 'policy', 'disclosure', 'transparency',
            'shareholder', 'executive'
        ]):
            return 'G'
        
        # Default categorization based on metric type
        metric_categories = {
            'emissions': 'E',
            'energy': 'E',
            'water': 'E',
            'waste': 'E',
            'diversity': 'S',
            'training': 'S'
        }
        
        return metric_categories.get(metric_type, 'E')  # Default to Environmental if unsure