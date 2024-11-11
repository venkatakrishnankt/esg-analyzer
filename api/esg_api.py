"""
API interface for ESG analysis
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import pandas as pd

from core.processors.esg_processor import ESGProcessor, ProcessingResult
from utils.logging_utils import get_logger

@dataclass
class APIRequest:
    """Container for API requests"""
    text: str
    document_type: str
    year: int
    company_id: Optional[str] = None
    industry: Optional[str] = None
    historical_data: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

@dataclass
class APIResponse:
    """Container for API responses"""
    success: bool
    data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    request_id: str

class ESGAPI:
    """
    API interface for ESG analysis
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.processor = ESGProcessor()
        self.request_counter = 0

    def process_document(self, request: APIRequest) -> APIResponse:
        """
        Process document and return ESG analysis results
        """
        start_time = datetime.now()
        self.request_counter += 1
        request_id = f"REQ_{datetime.now().strftime('%Y%m%d')}_{self.request_counter}"
        
        try:
            self.logger.info(f"Processing request {request_id}")
            
            # Validate request
            validation_errors = self._validate_request(request)
            if validation_errors:
                return APIResponse(
                    success=False,
                    data={},
                    errors=validation_errors,
                    warnings=[],
                    metadata=self._get_request_metadata(request),
                    processing_time=0.0,
                    request_id=request_id
                )
            
            # Process document
            result = self.processor.process_document(
                request.text,
                request.historical_data
            )
            
            # Format response
            response_data = self._format_response(result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return APIResponse(
                success=True,
                data=response_data,
                errors=[],
                warnings=self._get_warnings(result),
                metadata=self._get_response_metadata(request, result),
                processing_time=processing_time,
                request_id=request_id
            )
            
        except Exception as e:
            self.logger.error(f"Error processing request {request_id}: {str(e)}")
            return APIResponse(
                success=False,
                data={},
                errors=[str(e)],
                warnings=[],
                metadata=self._get_request_metadata(request),
                processing_time=(datetime.now() - start_time).total_seconds(),
                request_id=request_id
            )

    def _validate_request(self, request: APIRequest) -> List[str]:
        """
        Validate API request
        """
        errors = []
        
        if not request.text:
            errors.append("No text provided for analysis")
        
        if not request.document_type:
            errors.append("Document type not specified")
        
        if request.year and (request.year < 1900 or request.year > datetime.now().year + 1):
            errors.append("Invalid year specified")
        
        if request.historical_data:
            if not isinstance(request.historical_data, dict):
                errors.append("Historical data must be a dictionary")
            else:
                # Validate historical data structure
                for year, data in request.historical_data.items():
                    if not isinstance(data, dict):
                        errors.append(f"Invalid historical data format for year {year}")
        
        return errors

    def _format_response(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Format processing result for API response
        """
        try:
            response = {
                'metrics': self._format_metrics(result.metrics),
                'analysis': {
                    'factors': result.analysis.get('factors', {}),
                    'sentiment': result.analysis.get('sentiment', {}),
                    'categories': result.analysis.get('categories', {})
                },
                'summary': {
                    'key_findings': result.summary.get('key_findings', []),
                    'category_distribution': result.summary.get('category_distribution', {}),
                    'recommendations': result.summary.get('recommendations', [])
                },
                'validation': {
                    'overall_confidence': result.confidence,
                    'validation_summary': result.validations.get('overall', {}),
                    'detailed_validations': self._format_validations(
                        result.validations.get('metrics', {})
                    )
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return {}

    def _format_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """
        Format metrics for response
        """
        try:
            formatted_metrics = {
                'E': {'metrics': [], 'count': 0},
                'S': {'metrics': [], 'count': 0},
                'G': {'metrics': [], 'count': 0}
            }
            
            for metric in metrics:
                category = metric.get('category', 'E')
                formatted_metrics[category]['metrics'].append({
                    'type': metric.get('type'),
                    'value': metric.get('value'),
                    'unit': metric.get('unit'),
                    'confidence': metric.get('confidence'),
                    'context': metric.get('context'),
                    'temporal_info': metric.get('temporal_info')
                })
                formatted_metrics[category]['count'] += 1
            
            return formatted_metrics
            
        except Exception as e:
            self.logger.error(f"Error formatting metrics: {str(e)}")
            return {}

    def _format_validations(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format validation results for response
        """
        try:
            formatted_validations = {}
            
            for metric_type, validation in validations.items():
                formatted_validations[metric_type] = {
                    'is_valid': validation.get('is_valid', False),
                    'confidence': validation.get('confidence', 0.0),
                    'errors': validation.get('errors', []),
                    'warnings': validation.get('warnings', [])
                }
            
            return formatted_validations
            
        except Exception as e:
            self.logger.error(f"Error formatting validations: {str(e)}")
            return {}

    def _get_warnings(self, result: ProcessingResult) -> List[str]:
        """
        Extract warnings from processing result
        """
        warnings = []
        
        # Add validation warnings
        for validation in result.validations.get('metrics', {}).values():
            warnings.extend(validation.get('warnings', []))
        
        # Add low confidence warnings
        if result.confidence < 0.6:
            warnings.append("Overall confidence score is low")
        
        return warnings

    def _get_request_metadata(self, request: APIRequest) -> Dict[str, Any]:
        """
        Get metadata for request
        """
        return {
            'document_type': request.document_type,
            'year': request.year,
            'company_id': request.company_id,
            'industry': request.industry,
            'has_historical_data': bool(request.historical_data),
            'options': request.options or {}
        }

    def _get_response_metadata(self, request: APIRequest, 
                             result: ProcessingResult) -> Dict[str, Any]:
        """
        Get metadata for response
        """
        return {
            **self._get_request_metadata(request),
            **result.metadata,
            'processing_time': result.processing_time
        }

    def export_results(self, response: APIResponse, 
                      format: str = 'json') -> Any:
        """
        Export results in specified format
        """
        try:
            if format.lower() == 'json':
                return json.dumps(asdict(response), indent=2)
                
            elif format.lower() == 'csv':
                # Convert metrics to DataFrame
                metrics_data = []
                for category, data in response.data['metrics'].items():
                    for metric in data['metrics']:
                        metrics_data.append({
                            'Category': category,
                            'Type': metric['type'],
                            'Value': metric['value'],
                            'Unit': metric['unit'],
                            'Confidence': metric['confidence']
                        })
                
                return pd.DataFrame(metrics_data).to_csv(index=False)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            return None