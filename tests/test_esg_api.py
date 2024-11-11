"""
Unit tests for ESG API
"""
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from api.esg_api import ESGAPI, APIRequest, APIResponse

class TestESGAPI(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.api = ESGAPI()
        self.sample_text = """
        In 2023, our employee volunteering participation rate reached 61%, 
        with over 76,000 employee volunteering days contributed. We achieved 
        a 30% reduction in carbon emissions compared to the previous year, 
        reaching 1.2 million tonnes of CO2 equivalent.
        """
        self.sample_request = APIRequest(
            text=self.sample_text,
            document_type="sustainability_report",
            year=2023,
            company_id="TEST001",
            industry="financial_services",
            historical_data={
                "2022": {
                    "employee_volunteering_participation": 39,
                    "employee_volunteering_days": 49500,
                    "carbon_emissions": 1.7
                }
            }
        )

    def test_request_validation(self):
        """Test request validation"""
        # Test valid request
        errors = self.api._validate_request(self.sample_request)
        self.assertEqual(len(errors), 0, "Valid request should have no errors")

        # Test empty text
        invalid_request = APIRequest(
            text="",
            document_type="sustainability_report",
            year=2023
        )
        errors = self.api._validate_request(invalid_request)
        self.assertTrue(any("text" in e.lower() for e in errors))

        # Test invalid year
        invalid_request = APIRequest(
            text=self.sample_text,
            document_type="sustainability_report",
            year=1800
        )
        errors = self.api._validate_request(invalid_request)
        self.assertTrue(any("year" in e.lower() for e in errors))

    @patch('core.processors.esg_processor.ESGProcessor.process_document')
    def test_process_document(self, mock_process):
        """Test document processing"""
        # Mock processor response
        mock_process.return_value = Mock(
            metrics=[{
                'type': 'employee_volunteering_participation',
                'value': 61,
                'unit': '%',
                'confidence': 0.9,
                'category': 'S'
            }],
            analysis={'factors': {'S': []}},
            validations={'metrics': {}},
            summary={'key_findings': []},
            metadata={},
            confidence=0.9,
            processing_time=1.0
        )

        # Test processing
        response = self.api.process_document(self.sample_request)
        self.assertTrue(response.success)
        self.assertTrue('metrics' in response.data)
        self.assertEqual(response.errors, [])

    def test_format_response(self):
        """Test response formatting"""
        # Create mock processing result
        mock_result = Mock(
            metrics=[{
                'type': 'employee_volunteering_participation',
                'value': 61,
                'unit': '%',
                'confidence': 0.9,
                'category': 'S'
            }],
            analysis={'factors': {'S': []}},
            validations={'metrics': {}},
            summary={'key_findings': []},
            metadata={},
            confidence=0.9,
            processing_time=1.0
        )

        formatted = self.api._format_response(mock_result)
        self.assertTrue('metrics' in formatted)
        self.assertTrue('analysis' in formatted)
        self.assertTrue('summary' in formatted)
        self.assertTrue('validation' in formatted)

    def test_export_results(self):
        """Test results export"""
        # Create mock response
        response = APIResponse(
            success=True,
            data={
                'metrics': {
                    'S': {
                        'metrics': [{
                            'type': 'employee_volunteering_participation',
                            'value': 61,
                            'unit': '%',
                            'confidence': 0.9
                        }],
                        'count': 1
                    }
                }
            },
            errors=[],
            warnings=[],
            metadata={},
            processing_time=1.0,
            request_id="TEST_001"
        )

        # Test JSON export
        json_output = self.api.export_results(response, 'json')
        self.assertIsNotNone(json_output)
        self.assertIn("metrics", json_output)

        # Test CSV export
        csv_output = self.api.export_results(response, 'csv')
        self.assertIsNotNone(csv_output)
        self.assertIn("Category", csv_output)

        # Test invalid format
        with self.assertRaises(ValueError):
            self.api.export_results(response, 'invalid_format')

if __name__ == '__main__':
    unittest.main()