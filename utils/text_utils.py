
"""
Text processing and manipulation utilities
"""
from typing import List, Optional, Dict, Any, Tuple
import re
from config.patterns import CLEANING_PATTERNS, CONTEXT_PATTERNS
from config.constants import CUSTOM_STOP_WORDS, WINDOW_SIZES

class TextCleaner:
    """
    Utility class for text cleaning and preprocessing
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        """
        # Normalize whitespace
        text = CLEANING_PATTERNS['whitespace'].sub(' ', text)
        # Remove non-ASCII characters
        text = CLEANING_PATTERNS['non_ascii'].sub('', text)
        # Remove asterisks
        text = CLEANING_PATTERNS['asterisks'].sub('', text)
        # Add space between letters and numbers
        text = CLEANING_PATTERNS['letter_number'].sub(r'\1 \2', text)
        text = CLEANING_PATTERNS['number_letter'].sub(r'\1 \2', text)
        return text.strip()

    @staticmethod
    def remove_stop_words(text: str, additional_stops: Optional[set] = None) -> str:
        """
        Remove stop words from text
        """
        stop_words = CUSTOM_STOP_WORDS.copy()
        if additional_stops:
            stop_words.update(additional_stops)
        
        words = text.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

class ContextExtractor:
    """
    Utility class for extracting relevant context around phrases
    """
    @staticmethod
    def find_sentence_boundaries(text: str, position: int) -> tuple:
        """
        Find the start and end of the sentence containing the position
        """
        start = position
        while start > 0 and not CONTEXT_PATTERNS['sentence_boundary'].match(text[start-1]):
            start -= 1
        
        end = position
        while end < len(text) and not CONTEXT_PATTERNS['sentence_boundary'].match(text[end]):
            end += 1
            
        return start, end + 1

    @staticmethod
    def get_context_window(text: str, target: str, window_size: Optional[int] = None) -> Optional[str]:
        """
        Get context window around target phrase with smart boundary detection
        """
        if window_size is None:
            window_size = WINDOW_SIZES['context']
            
        text_lower = text.lower()
        target_lower = target.lower()
        
        index = text_lower.find(target_lower)
        if index == -1:
            return None
            
        # Find sentence boundaries
        sent_start, sent_end = ContextExtractor.find_sentence_boundaries(text, index)
        
        # Extend to window size if needed
        start = max(0, min(sent_start, index - window_size))
        end = min(len(text), max(sent_end, index + len(target) + window_size))
        
        context = text[start:end].strip()
        return TextCleaner.clean_text(context)

class NumericExtractor:
    """
    Utility class for extracting numeric values
    """
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """
        Extract all numbers from text
        """
        numbers = []
        matches = re.finditer(r'[-+]?\d*\.\d+|\d+', text)
        for match in matches:
            try:
                numbers.append(float(match.group()))
            except ValueError:
                continue
        return numbers

    @staticmethod
    def extract_percentages(text: str) -> List[float]:
        """
        Extract percentage values from text
        """
        percentages = []
        matches = re.finditer(r'(\d+(?:\.\d+)?)\s*%', text)
        for match in matches:
            try:
                percentages.append(float(match.group(1)))
            except ValueError:
                continue
        return percentages

    @staticmethod
    def clean_numeric_value(value: str) -> Optional[float]:
        """
        Clean and convert numeric string to float
        """
        try:
            # Remove common prefixes/suffixes and spaces
            value = value.strip('>').strip()
            value = re.sub(r'[,\s]', '', value)
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def format_metric(value: float, metric_type: str, precision: int = 2) -> str:
        """
        Format numeric value based on metric type
        """
        if metric_type == 'percentage':
            return f"{value:.{precision}f}%"
        elif metric_type == 'currency':
            if value >= 1_000_000:
                return f"${value/1_000_000:.{precision}f}M"
            elif value >= 1_000:
                return f"${value/1_000:.{precision}f}K"
            return f"${value:.{precision}f}"
        return f"{value:.{precision}f}"