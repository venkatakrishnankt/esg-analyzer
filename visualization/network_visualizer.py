"""
Network and relationship visualization components
"""
from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from config.constants import ESG_FACTORS
from config.settings import VIZ_SETTINGS

class NetworkVisualizer(BaseVisualizer):
    """
    Handles network and relationship visualizations
    """
    def visualize_semantic_similarity_network(self, df: pd.DataFrame, 
                                           threshold: float = 0.5) -> plt.Figure:
        """
        Create network diagram of ESG factor relationships
        """
        try:
            if not self._validate_data(df):
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax)

            # Create network graph
            G = nx.Graph()
            
            # Calculate factor importance and correlations
            factor_importance = df.mean()
            max_importance = factor_importance.max()
            
            # Prepare node attributes
            node_colors = []
            node_sizes = []
            
            # Add nodes with attributes
            for factor in df.columns:
                category = next((cat for cat, factors in ESG_FACTORS.items() 
                               if factor in factors), None)
                if category:
                    G.add_node(factor)
                    node_colors.append(VIZ_SETTINGS['color_scheme'][category])
                    importance_value = float(factor_importance[factor])
                    node_sizes.append(2000 * importance_value / max_importance)
            
            # Add edges with correlation weights
            correlations = []
            for i, factor1 in enumerate(df.columns):
                if factor1 in G.nodes():
                    for factor2 in df.columns[i+1:]:
                        if factor2 in G.nodes():
                            similarity = float(df[factor1].corr(df[factor2]))
                            if similarity > threshold:
                                G.add_edge(factor1, factor2, weight=similarity)
                                correlations.append((factor1, factor2, similarity))
            
            # Create visualization
            fig, ax = self._create_figure(figsize=(15, 10))
            
            # Use spring layout with adjusted parameters
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw edges with weighted thickness
            edges = G.edges()
            if edges:
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, alpha=0.5,
                                     width=[2 * w for w in weights],
                                     edge_color='gray')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                 node_size=node_sizes, alpha=0.7)
            
            # Add labels with custom font
            nx.draw_networkx_labels(G, pos, font_size=8,
                                  font_weight='bold',
                                  font_family='sans-serif')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                          label=cat, markerfacecolor=color,
                          markersize=10)
                for cat, color in VIZ_SETTINGS['color_scheme'].items()
            ]
            ax.legend(handles=legend_elements,
                     title='ESG Categories',
                     loc='upper left',
                     bbox_to_anchor=(1, 1))
            
            # Add network metrics
            plt.figtext(0.99, 0.5,
                       f"Network Statistics:\n"
                       f"Nodes: {len(G.nodes())}\n"
                       f"Edges: {len(G.edges())}\n"
                       f"Avg. Correlation: {np.mean(weights):.2f}\n"
                       f"Threshold: {threshold}",
                       fontsize=8, ha='right')
            
            plt.title("ESG Factor Similarity Network", pad=20)
            plt.axis('off')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating semantic network visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")