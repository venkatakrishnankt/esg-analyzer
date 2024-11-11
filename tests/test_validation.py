"""
Unit tests for validation framework
"""
import unittest
from core.validators.validation_framework import ESGValidator, ValidationResult

class TestESGValidator(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.validator = ESGValidator()
        self.sample_metric = {
            'type': 'employee_volunteering_participation',
            'value': 61,
            'unit': '%',
            'temporal_info': {'year': 2023}
        }

    def test_basic_validation(self):
        """Test basic metric validation"""
        # Valid metric
        result = self.validator.validate_metric(self.sample_metric)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.confidence, 0.8)

        # Invalid value
        invalid_metric = self.sample_metric.copy()
        invalid_metric['value'] = 150  # Percentage > 100
        result = self.validator.validate_metric(invalid_metric)
        self.assertFalse(result.is_valid)

    def test_context_validation(self):
        """Test context validation"""
        context = {
            'text': 'Employee volunteering participation rate increased to 61%.',
            'historical_data': {
                '2022': {'employee_volunteering_participation': 39}
            }
        }
        
        result = self.validator.validate_metric(self.sample_metric, context)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.confidence, 0.8)

        # Test with irrelevant context
        irrelevant_context = {
            'text': 'The company reported various metrics.',
            'historical_data': {}
        }
        result = self.validator.validate_metric(self.sample_metric, irrelevant_context)
        self.assertTrue(result.is_valid)  # Still valid but lower confidence
        self.assertLess(result.confidence, 0.8)

    def test_historical_validation(self):
        """Test historical data validation"""
        context = {
            'historical_data': {
                '2022': {'employee_volunteering_participation': 39},
                '2021': {'employee_volunteering_participation': 35}
            }
        }
        
        # Normal increase
        result = self.validator.validate_metric(self.sample_metric, context)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.confidence, 0.8)

        # Suspicious jump
        suspicious_metric = self.sample_metric.copy()
        suspicious_metric['value'] = 90  # Unusual jump from historical data
        result = self.validator.validate_metric(suspicious_metric, context)
        self.assertTrue(len(result.warnings) > 0)
        self.assertLess(result.confidence, 0.8)

    def test_temporal_validation(self):
        """Test temporal information validation"""
        # Missing temporal info
        metric_no_temporal = self.sample_metric.copy()
        del metric_no_temporal['temporal_info']
        result = self.validator.validate_metric(metric_no_temporal)
        self.assertTrue(len(result.warnings) > 0)

        # Future year
        metric_future = self.sample_metric.copy()
        metric_future['temporal_info']['year'] = 2025
        result = self.validator.validate_metric(metric_future)
        self.assertTrue(len(result.warnings) > 0)

    def test_unit_validation(self):
        """Test unit validation"""
        # Wrong unit type
        invalid_unit_metric = self.sample_metric.copy()
        invalid_unit_metric['unit'] = 'days'
        result = self.validator.validate_metric(invalid_unit_metric)
        self.assertFalse(result.is_valid)

        # Missing unit
        no_unit_metric = self.sample_metric.copy()
        del no_unit_metric['unit']
        result = self.validator.validate_metric(no_unit_metric)
        self.assertFalse(result.is_valid)

if __name__ == '__main__':
    unittest.main()