"""
Streamlit UI for ESG Analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Dict, Any, List

from api.esg_api import ESGAPI, APIRequest
from utils.logging_utils import get_logger

class ESGAnalyzerUI:
    """
    Streamlit UI for ESG Analysis
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api = ESGAPI()
        self.setup_page_config()

    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="ESG Analyzer",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
            <style>
            .main {
                padding: 0rem 1rem;
            }
            .stAlert {
                padding: 0.5rem;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            .confidence-high { color: #28a745; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
            </style>
            """, unsafe_allow_html=True)

    def run(self):
        """Run the Streamlit app"""
        st.title("🌍 ESG Analysis Dashboard")
        st.markdown("""
            Analyze Environmental, Social, and Governance factors from documents
            using advanced NLP and machine learning techniques.
        """)

        # Sidebar configuration
        self.setup_sidebar()

        # Main content
        uploaded_files = self.file_upload_section()
        if uploaded_files and st.button("Analyze Documents"):
            self.process_documents(uploaded_files)

    def setup_sidebar(self):
        """Setup sidebar configuration"""
        with st.sidebar:
            st.header("📊 Analysis Configuration")
            st.markdown("---")

            # Industry selection
            industry = st.selectbox(
                "Select Industry",
                ["Financial Services", "Manufacturing", "Technology", 
                 "Healthcare", "Retail"]
            )

            # Analysis options
            st.subheader("Analysis Options")
            options = {
                "extract_metrics": st.checkbox("Extract Metrics", value=True),
                "perform_sentiment": st.checkbox("Sentiment Analysis", value=True),
                "validate_data": st.checkbox("Data Validation", value=True),
                "generate_visualizations": st.checkbox("Generate Visualizations", value=True)
            }

            # Advanced settings
            with st.expander("Advanced Settings"):
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    help="Minimum confidence score for reported metrics"
                )
                
                context_window = st.slider(
                    "Context Window Size",
                    min_value=50,
                    max_value=500,
                    value=150,
                    help="Size of context window for metric extraction"
                )

            return {
                "industry": industry,
                "options": options,
                "confidence_threshold": confidence_threshold,
                "context_window": context_window
            }

    def file_upload_section(self) -> Dict[str, Any]:
        """Create file upload section"""
        st.header("📁 Document Upload")
        
        uploaded_files = {}
        col1, col2 = st.columns(2)
        
        with col1:
            num_years = st.number_input(
                "Number of years to analyze",
                min_value=1,
                max_value=5,
                value=2
            )

        # Create file upload widgets for each year
        for i in range(num_years):
            with st.expander(f"Year {i+1} Configuration", expanded=True):
                year = st.text_input(
                    "Year",
                    value=str(datetime.now().year - i),
                    key=f"year_{i}"
                )
                
                file = st.file_uploader(
                    "Upload Document (PDF)",
                    type=["pdf"],
                    key=f"file_{i}",
                    help="Upload ESG report in PDF format"
                )
                
                if file:
                    uploaded_files[year] = file
                    st.success(f"✅ File uploaded for {year}")

        return uploaded_files

    def process_documents(self, uploaded_files: Dict[str, Any]):
        """Process uploaded documents"""
        try:
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                results = {}
                
                for i, (year, file) in enumerate(uploaded_files.items()):
                    # Create API request
                    request = APIRequest(
                        text=self.extract_text(file),
                        document_type="sustainability_report",
                        year=int(year),
                        company_id=None,  # Could be added as input
                        industry=st.session_state.get('industry'),
                        historical_data=self.get_historical_data(results),
                        options=st.session_state.get('options', {})
                    )
                    
                    # Process document
                    response = self.api.process_document(request)
                    results[year] = response
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))

                # Display results
                self.display_results(results)
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            self.logger.error(f"Error in document processing: {str(e)}")

    def extract_text(self, file) -> str:
        """Extract text from uploaded file"""
        try:
            # File content is already available in memory
            return file.read()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            self.logger.error(f"Error extracting text: {str(e)}")
            return ""

    def get_historical_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format historical data from previous results"""
        historical_data = {}
        
        for year, response in results.items():
            if response.success:
                year_data = {}
                for category in response.data['metrics']:
                    for metric in response.data['metrics'][category]['metrics']:
                        year_data[metric['type']] = metric['value']
                historical_data[year] = year_data
                
        return historical_data

    def display_results(self, results: Dict[str, Any]):
        """Display analysis results"""
        st.header("📊 Analysis Results")
        
        # Create tabs for different views
        tabs = st.tabs([
            "Overview",
            "Detailed Metrics",
            "Visualizations",
            "Validation Results"
        ])
        
        with tabs[0]:
            self.display_overview(results)
            
        with tabs[1]:
            self.display_detailed_metrics(results)
            
        with tabs[2]:
            self.display_visualizations(results)
            
        with tabs[3]:
            self.display_validation_results(results)

    def display_overview(self, results: Dict[str, Any]):
        """Display overview of results"""
        st.subheader("📈 Key Metrics Overview")
        
        # Display metrics summary for each year
        cols = st.columns(len(results))
        for i, (year, response) in enumerate(results.items()):
            if response.success:
                with cols[i]:
                    st.metric(
                        f"Year {year}",
                        f"{len(response.data['metrics'])} metrics extracted",
                        f"{response.data['validation']['overall_confidence']:.2f} confidence"
                    )

        # Display key findings
        st.subheader("🔍 Key Findings")
        for year, response in results.items():
            if response.success:
                with st.expander(f"Findings for {year}", expanded=True):
                    for finding in response.data['summary']['key_findings']:
                        confidence_class = self.get_confidence_class(
                            finding['confidence']
                        )
                        st.markdown(
                            f"- {finding['description']} "
                            f"<span class='{confidence_class}'>"
                            f"({finding['confidence']:.2f})</span>",
                            unsafe_allow_html=True
                        )

    def display_detailed_metrics(self, results: Dict[str, Any]):
        """Display detailed metrics"""
        st.subheader("📊 Detailed Metrics Analysis")
        
        # Create DataFrame for metrics
        metrics_data = []
        for year, response in results.items():
            if response.success:
                for category in response.data['metrics']:
                    for metric in response.data['metrics'][category]['metrics']:
                        metrics_data.append({
                            'Year': year,
                            'Category': category,
                            'Type': metric['type'],
                            'Value': metric['value'],
                            'Unit': metric['unit'],
                            'Confidence': metric['confidence']
                        })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                selected_categories = st.multiselect(
                    "Filter by Category",
                    df['Category'].unique(),
                    default=df['Category'].unique()
                )
            with col2:
                confidence_filter = st.slider(
                    "Minimum Confidence",
                    0.0, 1.0, 0.5
                )
            
            # Filter DataFrame
            filtered_df = df[
                (df['Category'].isin(selected_categories)) & 
                (df['Confidence'] >= confidence_filter)
            ]
            
            # Display filtered metrics
            st.dataframe(
                filtered_df.style.format({
                    'Value': '{:.2f}',
                    'Confidence': '{:.2%}'
                }).background_gradient(
                    subset=['Confidence'],
                    cmap='RdYlGn'
                )
            )

    def display_visualizations(self, results: Dict[str, Any]):
        """Display visualizations"""
        st.subheader("📈 Visualizations")
        
        if not results:
            st.warning("No data available for visualization")
            return
            
        # Prepare data for visualization
        metrics_data = []
        for year, response in results.items():
            if response.success:
                for category in response.data['metrics']:
                    for metric in response.data['metrics'][category]['metrics']:
                        metrics_data.append({
                            'Year': year,
                            'Category': category,
                            'Type': metric['type'],
                            'Value': metric['value'],
                            'Confidence': metric['confidence']
                        })
        
        if not metrics_data:
            st.warning("No metrics data available for visualization")
            return
            
        df = pd.DataFrame(metrics_data)
        
        # Metric trends
        st.subheader("Metric Trends")
        selected_metric = st.selectbox(
            "Select Metric",
            df['Type'].unique()
        )
        
        metric_df = df[df['Type'] == selected_metric]
        if not metric_df.empty:
            fig = px.line(
                metric_df,
                x='Year',
                y='Value',
                title=f"Trend for {selected_metric}",
                markers=True
            )
            st.plotly_chart(fig)
        
        # Category distribution
        st.subheader("Category Distribution")
        fig = px.pie(
            df,
            names='Category',
            title="Distribution of Metrics by Category"
        )
        st.plotly_chart(fig)
        
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig = px.histogram(
            df,
            x='Confidence',
            color='Category',
            title="Distribution of Confidence Scores"
        )
        st.plotly_chart(fig)

    def display_validation_results(self, results: Dict[str, Any]):
        """Display validation results"""
        st.subheader("✅ Validation Results")
        
        for year, response in results.items():
            if response.success:
                with st.expander(f"Validation Results for {year}", expanded=True):
                    # Overall validation metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Overall Confidence",
                            f"{response.data['validation']['overall_confidence']:.2%}"
                        )
                    
                    with col2:
                        validation_summary = response.data['validation']['validation_summary']
                        st.metric(
                            "Validation Rate",
                            f"{validation_summary['validation_rate']:.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Warning Count",
                            validation_summary['warning_count']
                        )
                    
                    # Detailed validations
                    st.markdown("#### Detailed Validation Results")
                    detailed_validations = response.data['validation']['detailed_validations']
                    
                    for metric_type, validation in detailed_validations.items():
                        confidence_class = self.get_confidence_class(
                            validation['confidence']
                        )
                        
                        st.markdown(
                            f"**{metric_type}** - "
                            f"<span class='{confidence_class}'>"
                            f"Confidence: {validation['confidence']:.2%}</span>",
                            unsafe_allow_html=True
                        )
                        
                        if validation['warnings']:
                            st.warning(
                                "Warnings:\n" + "\n".join(
                                    f"- {w}" for w in validation['warnings']
                                )
                            )
                        
                        if validation['errors']:
                            st.error(
                                "Errors:\n" + "\n".join(
                                    f"- {e}" for e in validation['errors']
                                )
                            )

    @staticmethod
    def get_confidence_class(confidence: float) -> str:
        """Get CSS class based on confidence score"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"

def main():
    """Main entry point"""
    app = ESGAnalyzerUI()
    app.run()

if __name__ == "__main__":
    main()