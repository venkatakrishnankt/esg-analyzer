"""
Example usage of ESG API
"""
from api.esg_api import ESGAPI, APIRequest

def main():
    # Initialize API
    api = ESGAPI()
    
    # Example document text
    text = """
    In 2023, our employee volunteering participation rate reached 61%, 
    with over 76,000 employee volunteering days contributed. We achieved 
    a 30% reduction in carbon emissions compared to the previous year, 
    reaching 1.2 million tonnes of CO2 equivalent.
    """
    
    # Create request
    request = APIRequest(
        text=text,
        document_type="sustainability_report",
        year=2023,
        company_id="COMP001",
        industry="financial_services",
        historical_data={
            "2022": {
                "employee_volunteering_participation": 39,
                "employee_volunteering_days": 49500,
                "carbon_emissions": 1.7
            }
        }
    )
    
    # Process request
    response = api.process_document(request)
    
    # Print results
    if response.success:
        print(f"Request ID: {response.request_id}")
        print(f"Processing Time: {response.processing_time:.2f} seconds")
        print("\nKey Findings:")
        for finding in response.data['summary']['key_findings']:
            print(f"- {finding['description']}")
        
        print("\nMetrics:")
        for category, data in response.data['metrics'].items():
            print(f"\n{category} Category ({data['count']} metrics):")
            for metric in data['metrics']:
                print(f"- {metric['type']}: {metric['value']} {metric['unit']} "
                      f"(confidence: {metric['confidence']:.2f})")
        
        print("\nRecommendations:")
        for rec in response.data['summary']['recommendations']:
            print(f"- [{rec['priority']}] {rec['description']}")
        
        # Export results
        json_output = api.export_results(response, 'json')
        csv_output = api.export_results(response, 'csv')
        
    else:
        print("Processing failed:")
        for error in response.errors:
            print(f"- {error}")

if __name__ == "__main__":
    main()