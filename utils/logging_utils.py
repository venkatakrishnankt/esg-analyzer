"""
Logging utilities for the ESG analyzer
"""
import logging
import logging.config
import traceback
from typing import Optional, Any, Dict, List
from config.settings import LOGGING_CONFIG

class LogManager:
    """
    Centralized logging management
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logging()
        return cls._instance
    
    def _initialize_logging(self):
        """
        Initialize logging configuration
        """
        logging.config.dictConfig(LOGGING_CONFIG)
        self.logger = logging.getLogger(__name__)

    def info(self, message: str, **kwargs):
        """
        Log info message with optional context
        """
        self.logger.info(message, extra=kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """
        Log error message with stack trace
        """
        if error:
            tb = traceback.format_exc()
            message = f"{message}\nError: {str(error)}\nTrace:\n{tb}"
        self.logger.error(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """
        Log debug message with optional context
        """
        self.logger.debug(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """
        Log warning message with optional context
        """
        self.logger.warning(message, extra=kwargs)

class PerformanceLogger:
    """
    Utility for logging performance metrics
    """
    def __init__(self):
        self.logger = LogManager()
        
    def log_processing_time(self, operation: str, time_taken: float):
        """
        Log processing time for operations
        """
        self.logger.info(
            f"Processing time for {operation}: {time_taken:.2f} seconds",
            operation=operation,
            time_taken=time_taken
        )

    def log_memory_usage(self, operation: str, memory_used: float):
        """
        Log memory usage for operations
        """
        self.logger.info(
            f"Memory usage for {operation}: {memory_used:.2f} MB",
            operation=operation,
            memory_used=memory_used
        )

class MetricLogger:
    """
    Utility for logging metric-related information
    """
    def __init__(self):
        self.logger = LogManager()
        
    def log_extraction_result(self, factor: str, metrics: Dict[str, Any]):
        """
        Log metric extraction results
        """
        self.logger.debug(
            f"Extracted metrics for {factor}",
            factor=factor,
            metrics=metrics
        )

    def log_score_calculation(self, factor: str, scores: Dict[str, float]):
        """
        Log score calculation results
        """
        self.logger.debug(
            f"Calculated scores for {factor}",
            factor=factor,
            scores=scores
        )