"""
Main ESG processing and coordination
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

from core.analyzers.esg_analyzer import ESGAnalyzer
from core.extractors.metric_extractor import MetricExtractor
from core.validators.validation_framework import ESGValidator
from core.models.model_manager import ModelManager
from config.models.model_config import ModelType
from utils.logging_utils import get_logger

@dataclass
class ProcessingResult:
    """Container for ESG processing results"""
    metrics: Dict[str, Any]
    analysis: Dict[str, Any]
    validations: Dict[str, Any]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float
    processing_time: float

class ESGProcessor:
    """
    Main coordinator for ESG processing
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_manager = ModelManager()
        self.esg_analyzer = ESGAnalyzer()
        self.metric_extractor = MetricExtractor()
        self.validator = ESGValidator()
        
        # Initialize processing metadata
        self.metadata = {
            'version': '1.0',
            'processing_date': None,
            'models_used': [],
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }

    def process_document(self, text: str, 
                        historical_data: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process document with comprehensive ESG analysis
        """
        try:
            start_time = datetime.now()
            self.metadata['processing_date'] = start_time.isoformat()
            
            # Step 1: Initial ESG Analysis
            self.logger.info("Starting ESG analysis")
            analysis_results = self.esg_analyzer.analyze_document(text)
            
            # Step 2: Metric Extraction
            self.logger.info("Extracting metrics")
            metrics = self.metric_extractor.extract_metrics(text)
            
            # Step 3: Validation
            self.logger.info("Validating results")
            validation_results = self._validate_results(
                metrics, analysis_results, historical_data
            )
            
            # Step 4: Generate Summary
            self.logger.info("Generating summary")
            summary = self._generate_summary(
                metrics, analysis_results, validation_results
            )
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                metrics, analysis_results, validation_results
            )
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                metrics=metrics,
                analysis=analysis_results,
                validations=validation_results,
                summary=summary,
                metadata=self.metadata,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in ESG processing: {str(e)}")
            raise

    def _validate_results(self, metrics: List[Dict], 
                         analysis_results: Dict,
                         historical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate extracted metrics and analysis results
        """
        validations = {
            'metrics': {},
            'analysis': {},
            'overall': {}
        }
        
        try:
            # Validate metrics
            for metric in metrics:
                validation = self.validator.validate_metric(
                    metric,
                    {
                        'historical_data': historical_data,
                        'text': metric.get('context', '')
                    }
                )
                
                validations['metrics'][metric['type']] = {
                    'is_valid': validation.is_valid,
                    'confidence': validation.confidence,
                    'errors': validation.errors,
                    'warnings': validation.warnings,
                    'metadata': validation.metadata
                }
            
            # Validate analysis results
            for category in ['E', 'S', 'G']:
                factors = analysis_results.get('factors', {}).get(category, [])
                for factor in factors:
                    validation = self.validator.validate_metric(
                        {
                            'type': factor['metric_type'],
                            'value': factor.get('value'),
                            'unit': factor.get('unit'),
                            'temporal_info': factor.get('temporal_info', {})
                        },
                        {'text': factor.get('context', '')}
                    )
                    
                    validations['analysis'][f"{category}_{factor['text'][:50]}"] = {
                        'is_valid': validation.is_valid,
                        'confidence': validation.confidence,
                        'errors': validation.errors,
                        'warnings': validation.warnings,
                        'metadata': validation.metadata
                    }
            
            # Calculate overall validation metrics
            valid_count = sum(
                1 for v in validations['metrics'].values() if v['is_valid']
            )
            total_count = len(validations['metrics'])
            
            validations['overall'] = {
                'validation_rate': valid_count / total_count if total_count > 0 else 0,
                'average_confidence': np.mean([
                    v['confidence'] for v in validations['metrics'].values()
                ]),
                'error_count': sum(
                    len(v['errors']) for v in validations['metrics'].values()
                ),
                'warning_count': sum(
                    len(v['warnings']) for v in validations['metrics'].values()
                )
            }
            
            return validations
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            return validations

    def _generate_summary(self, metrics: List[Dict], 
                         analysis_results: Dict,
                         validation_results: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive summary of ESG analysis
        """
        try:
            summary = {
                'metrics_summary': self._summarize_metrics(metrics),
                'category_distribution': self._summarize_categories(analysis_results),
                'key_findings': self._extract_key_findings(
                    metrics, analysis_results, validation_results
                ),
                'validation_summary': self._summarize_validations(validation_results),
                'recommendations': self._generate_recommendations(
                    metrics, analysis_results, validation_results
                )
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {}

    def _summarize_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """
        Summarize extracted metrics
        """
        summary = {
            'total_metrics': len(metrics),
            'by_category': {'E': 0, 'S': 0, 'G': 0},
            'by_type': {},
            'key_metrics': []
        }
        
        try:
            # Group metrics by category and type
            for metric in metrics:
                category = metric.get('category', 'E')
                metric_type = metric.get('type', 'unknown')
                
                summary['by_category'][category] = \
                    summary['by_category'].get(category, 0) + 1
                summary['by_type'][metric_type] = \
                    summary['by_type'].get(metric_type, 0) + 1
                
                # Identify key metrics based on confidence
                if metric.get('confidence', 0) >= 0.8:
                    summary['key_metrics'].append({
                        'type': metric_type,
                        'value': metric.get('value'),
                        'unit': metric.get('unit'),
                        'confidence': metric.get('confidence')
                    })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing metrics: {str(e)}")
            return summary

    def _summarize_categories(self, analysis_results: Dict) -> Dict[str, Any]:
        """
        Summarize ESG category distribution
        """
        try:
            factors = analysis_results.get('factors', {})
            total_factors = sum(len(factors.get(cat, [])) for cat in ['E', 'S', 'G'])
            
            distribution = {}
            for category in ['E', 'S', 'G']:
                category_factors = factors.get(category, [])
                distribution[category] = {
                    'count': len(category_factors),
                    'percentage': (len(category_factors) / total_factors * 100 
                                 if total_factors > 0 else 0),
                    'average_confidence': np.mean([
                        f.get('confidence', 0) for f in category_factors
                    ]) if category_factors else 0
                }
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error summarizing categories: {str(e)}")
            return {}

    def _extract_key_findings(self, metrics: List[Dict], 
                            analysis_results: Dict,
                            validation_results: Dict) -> List[Dict]:
        """
        Extract key findings from analysis
        """
        try:
            findings = []
            
            # High confidence metrics
            high_confidence_metrics = [
                m for m in metrics 
                if m.get('confidence', 0) >= self.metadata['confidence_thresholds']['high']
            ]
            
            for metric in high_confidence_metrics:
                findings.append({
                    'type': 'metric',
                    'category': metric.get('category', 'E'),
                    'description': f"Found {metric['type']} of {metric.get('value')} "
                                 f"{metric.get('unit', '')}",
                    'confidence': metric.get('confidence', 0)
                })
            
            # Significant factors
            factors = analysis_results.get('factors', {})
            for category in ['E', 'S', 'G']:
                category_factors = factors.get(category, [])
                significant_factors = [
                    f for f in category_factors 
                    if f.get('confidence', 0) >= self.metadata['confidence_thresholds']['high']
                ]
                
                for factor in significant_factors:
                    findings.append({
                        'type': 'factor',
                        'category': category,
                        'description': factor.get('text', '')[:200],
                        'confidence': factor.get('confidence', 0)
                    })
            
            # Validation issues
            if validation_results.get('overall', {}).get('warning_count', 0) > 0:
                findings.append({
                    'type': 'validation',
                    'category': 'all',
                    'description': f"Found {validation_results['overall']['warning_count']} "
                                 f"validation warnings",
                    'confidence': 1.0
                })
            
            return sorted(findings, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error extracting key findings: {str(e)}")
            return []

    def _summarize_validations(self, validation_results: Dict) -> Dict[str, Any]:
        """
        Summarize validation results
        """
        try:
            summary = {
                'metrics_validation': {
                    'total': len(validation_results.get('metrics', {})),
                    'valid': sum(
                        1 for v in validation_results.get('metrics', {}).values() 
                        if v.get('is_valid', False)
                    ),
                    'high_confidence': sum(
                        1 for v in validation_results.get('metrics', {}).values() 
                        if v.get('confidence', 0) >= self.metadata['confidence_thresholds']['high']
                    )
                },
                'analysis_validation': {
                    'total': len(validation_results.get('analysis', {})),
                    'valid': sum(
                        1 for v in validation_results.get('analysis', {}).values() 
                        if v.get('is_valid', False)
                    ),
                    'high_confidence': sum(
                        1 for v in validation_results.get('analysis', {}).values() 
                        if v.get('confidence', 0) >= self.metadata['confidence_thresholds']['high']
                    )
                },
                'overall': validation_results.get('overall', {})
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing validations: {str(e)}")
            return {}

    def _generate_recommendations(self, metrics: List[Dict], 
                                analysis_results: Dict,
                                validation_results: Dict) -> List[Dict]:
        """
        Generate recommendations based on analysis
        """
        try:
            recommendations = []
            
            # Check for missing metrics
            covered_metrics = set(m['type'] for m in metrics)
            expected_metrics = set(self.validator.validation_rules.keys())
            missing_metrics = expected_metrics - covered_metrics
            
            if missing_metrics:
                recommendations.append({
                    'type': 'missing_metrics',
                    'priority': 'high',
                    'description': f"Consider including these metrics: {', '.join(missing_metrics)}"
                })
            
            # Check for low confidence metrics
            low_confidence_metrics = [
                m for m in metrics 
                if m.get('confidence', 0) < self.metadata['confidence_thresholds']['medium']
            ]
            
            if low_confidence_metrics:
                recommendations.append({
                    'type': 'low_confidence',
                    'priority': 'medium',
                    'description': "Improve clarity for these metrics: " + 
                                 ', '.join(m['type'] for m in low_confidence_metrics)
                })
            
            # Check category balance
            category_distribution = analysis_results.get('category_distribution', {})
            for category, stats in category_distribution.items():
                if stats.get('percentage', 0) < 20:  # Less than 20% coverage
                    recommendations.append({
                        'type': 'category_balance',
                        'priority': 'medium',
                        'description': f"Consider increasing coverage of {category} factors"
                    })
            
            # Check validation issues
            validation_overall = validation_results.get('overall', {})
            if validation_overall.get('validation_rate', 1.0) < 0.8:
                recommendations.append({
                    'type': 'validation',
                    'priority': 'high',
                    'description': "Address validation issues to improve data quality"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _calculate_overall_confidence(self, metrics: List[Dict], 
                                   analysis_results: Dict,
                                   validation_results: Dict) -> float:
        """
        Calculate overall confidence score
        """
        try:
            scores = []
            
            # Metric confidence
            if metrics:
                scores.append(np.mean([m.get('confidence', 0) for m in metrics]))
            
            # Analysis confidence
            factors = analysis_results.get('factors', {})
            for category in ['E', 'S', 'G']:
                category_factors = factors.get(category, [])
                if category_factors:
                    scores.append(np.mean([
                        f.get('confidence', 0) for f in category_factors
                    ]))
            
            # Validation confidence
            validation_overall = validation_results.get('overall', {})
            if validation_overall:
                scores.append(validation_overall.get('average_confidence', 0))
            
            # Calculate weighted average
            if scores:
                weights = {
                    'metrics': 0.4,
                    'analysis': 0.4,
                    'validation': 0.2
                }
                
                return min(1.0, np.average(scores, weights=list(weights.values())))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {str(e)}")
            return 0.0