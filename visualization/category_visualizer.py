"""
Category-based visualization components
"""
from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.constants import ESG_FACTORS
from config.settings import VIZ_SETTINGS

class CategoryVisualizer(BaseVisualizer):
    """
    Handles category-based visualizations
    """
    def visualize_category_breakdown(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create stacked bar chart of ESG categories
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            # Calculate category scores
            categories = {cat: [] for cat in ESG_FACTORS.keys()}
            for factor in df.columns:
                for category, factors in ESG_FACTORS.items():
                    if factor in factors:
                        categories[category].append(factor)
                        break
            
            category_scores = pd.DataFrame({
                category: df[factors].sum(axis=1) 
                for category, factors in categories.items()
                if factors  # Only include categories with factors
            })
            
            # Create visualization
            fig, ax = self._create_figure(figsize=(12, 6))
            
            category_scores.plot(
                kind='bar', 
                stacked=True, 
                ax=ax,
                color=[VIZ_SETTINGS['color_scheme'][cat] for cat in category_scores.columns]
            )
            
            ax.set_title("ESG Category Breakdown by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Score")
            plt.legend(title="ESG Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating category breakdown visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_radar_chart(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create radar chart for ESG categories
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            # Calculate category averages
            category_scores = {}
            for category, factors in ESG_FACTORS.items():
                category_factors = [f for f in factors if f in df.columns]
                if category_factors:
                    category_scores[category] = df[category_factors].mean(axis=1)

            if not category_scores:
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax, "No category data available")

            # Create subplot for each year
            n_years = len(df)
            fig, axs = plt.subplots(
                1, max(1, n_years),
                figsize=(20, 5),
                subplot_kw=dict(projection='polar')
            )
            
            # Handle single year case
            if n_years == 1:
                axs = [axs]
            
            # Create radar chart for each year
            for i, (year, _) in enumerate(df.iterrows()):
                values = [category_scores[cat][year] for cat in category_scores.keys()]
                angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
                
                # Close the plot
                values = np.concatenate((values, [values[0]]))
                angles = np.concatenate((angles, [angles[0]]))
                
                ax = axs[i]
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(list(category_scores.keys()), size=8)
                ax.set_title(f"ESG Categories for {year}")
                
                # Add value labels
                for angle, value in zip(angles[:-1], values[:-1]):
                    ax.text(angle, value, f'{value:.1f}', 
                           ha='center', va='center')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating radar chart visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")