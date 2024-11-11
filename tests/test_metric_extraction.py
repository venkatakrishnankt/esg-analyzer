"""
Unit tests for metric extraction
"""
import unittest
from core.extractors.metric_extractor import MetricExtractor, ExtractedMetric

class TestMetricExtractor(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        self.extractor = MetricExtractor()
        self.sample_text = """
        In 2023, our employee volunteering participation rate reached 61%, 
        with over 76,000 employee volunteering days contributed. We achieved 
        a 30% reduction in carbon emissions compared to the previous year, 
        reaching 1.2 million tonnes of CO2 equivalent.
        """

    def test_extract_percentage(self):
        """Test percentage extraction"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        percentage_metrics = [m for m in metrics if '%' in m.unit]
        
        self.assertTrue(any(
            m.value == 61 and 'volunteering participation' in m.context.lower()
            for m in percentage_metrics
        ))
        self.assertTrue(any(
            m.value == 30 and 'reduction' in m.context.lower()
            for m in percentage_metrics
        ))

    def test_extract_numeric(self):
        """Test numeric extraction"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        numeric_metrics = [m for m in metrics if m.unit == 'days']
        
        self.assertTrue(any(
            m.value == 76000 and 'volunteering days' in m.context.lower()
            for m in numeric_metrics
        ))

    def test_extract_measurement(self):
        """Test measurement extraction"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        emission_metrics = [m for m in metrics if 'CO2' in m.unit]
        
        self.assertTrue(any(
            m.value == 1.2 and 'million tonnes' in m.context.lower()
            for m in emission_metrics
        ))

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        
        # High confidence cases (explicit mentions with clear context)
        high_confidence_metrics = [m for m in metrics if m.confidence > 0.8]
        self.assertTrue(len(high_confidence_metrics) > 0)

        # Lower confidence cases (ambiguous context)
        text_with_ambiguity = "The number increased to 50%."
        ambiguous_metrics = self.extractor.extract_metrics(text_with_ambiguity)
        self.assertTrue(all(m.confidence < 0.8 for m in ambiguous_metrics))

    def test_temporal_info_extraction(self):
        """Test temporal information extraction"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        
        metrics_with_year = [m for m in metrics if m.temporal_info.get('year') == 2023]
        self.assertTrue(len(metrics_with_year) > 0)

    def test_context_extraction(self):
        """Test context extraction"""
        metrics = self.extractor.extract_metrics(self.sample_text)
        
        for metric in metrics:
            self.assertTrue(len(metric.context) > 0)
            self.assertTrue(
                str(metric.value) in metric.context or 
                f"{metric.value}%" in metric.context or
                f"{metric.value} {metric.unit}" in metric.context
            )

    def test_unit_detection(self):
        """Test unit detection"""
        text_with_units = """
        We consumed 500 MWh of energy and spent $2.5 million on environmental initiatives.
        The project achieved 85% completion rate.
        """
        metrics = self.extractor.extract_metrics(text_with_units)
        
        units_found = [m.unit for m in metrics]
        self.assertTrue('MWh' in units_found)
        self.assertTrue('%' in units_found)
        self.assertTrue('USD' in units_found)

if __name__ == '__main__':
    unittest.main()