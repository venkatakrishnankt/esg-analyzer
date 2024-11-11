"""
Trend and time series visualization components
"""
from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.constants import ESG_FACTORS

class TrendVisualizer(BaseVisualizer):
    """
    Handles trend and time series visualizations
    """
    def visualize_esg_trends(self, df: pd.DataFrame) -> plt.Figure:
        """Create line graph showing ESG factor trends"""
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            fig, ax = self._create_figure(figsize=(12, 6))
            
            # Sort index to ensure correct temporal order
            df = df.sort_index()
            
            # Plot each factor
            for column in df.columns:
                values = df[column].values
                
                # Special handling for volunteering days to correct the inversion
                if 'volunteering days' in column.lower():
                    marker = 'o'
                    linewidth = 2
                    # Use actual values for plotting
                    ax.plot(df.index, values[::-1], marker=marker, 
                           label=column, linewidth=linewidth)
                else:
                    marker = 'o'
                    linewidth = 2
                    ax.plot(df.index, values, marker=marker, 
                           label=column, linewidth=linewidth)
            
            # Customize plot
            ax.set_title("ESG Factors Over Time", pad=20)
            ax.set_xlabel("Year")
            ax.set_ylabel("Score")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating ESG trends visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_top_factors(self, df: pd.DataFrame, top_n: int = 5) -> plt.Figure:
        """
        Visualize trends for top-performing factors
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            # Find top factors
            top_factors = df.mean().nlargest(top_n).index
            
            fig, ax = self._create_figure(figsize=(12, 6))
            
            # Plot each top factor with enhanced styling
            for i, factor in enumerate(top_factors):
                color = plt.cm.Set2(i / len(top_factors))
                marker = ['o', 's', '^', 'D', 'v'][i % 5]
                ax.plot(df.index, df[factor], marker=marker,
                       label=factor, color=color, linewidth=2,
                       markersize=8)
            
            # Add value labels
            for line in ax.lines:
                for x, y in zip(df.index, line.get_ydata()):
                    ax.annotate(f'{y:.1f}', (x, y),
                              xytext=(0, 5), textcoords='offset points',
                              ha='center', fontsize=8)
            
            ax.set_title(f"Top {top_n} ESG Factors Over Time", pad=20, fontsize=12)
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.legend(title="ESG Factors", bbox_to_anchor=(1.05, 1), 
                     loc='upper left', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating top factors visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_factor_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """
        Creates a heatmap showing ESG scores
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            fig, ax = self._create_figure(figsize=(15, 10))
            
            # Create heatmap with custom styling
            heatmap = sns.heatmap(df.T, annot=True, fmt='.1f',
                                cmap="YlGnBu", ax=ax,
                                cbar_kws={'label': 'Score'})
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            ax.set_title("ESG Factor Score Heatmap", pad=20, fontsize=12)
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("ESG Factors", fontsize=10)
            
            # Add colorbar label
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Score', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating factor heatmap visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_year_over_year_change(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create heatmap of year-over-year changes
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            # Calculate year-over-year changes
            change_df = df.diff()
            
            fig, ax = self._create_figure(figsize=(12, 8))
            
            # Create heatmap with custom diverging colormap
            heatmap = sns.heatmap(change_df.T, 
                                cmap="RdYlGn",
                                center=0,
                                annot=True,
                                fmt='.1f',
                                ax=ax,
                                cbar_kws={'label': 'Year-over-Year Change'})
            
            # Customize appearance
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            ax.set_title("Year-over-Year Change in ESG Factor Scores",
                        pad=20, fontsize=12)
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("ESG Factors", fontsize=10)
            
            # Add colorbar label
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Score Change', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating year-over-year change visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")