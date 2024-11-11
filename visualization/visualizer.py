"""
Main visualization coordinator class
"""
from .base_visualizer import BaseVisualizer
from .trend_visualizer import TrendVisualizer
from .category_visualizer import CategoryVisualizer
from .network_visualizer import NetworkVisualizer
from .model_visualizer import ModelVisualizer
from utils.logging_utils import LogManager

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional

class Visualizer:
    """
    Main visualization coordinator
    """
    def __init__(self):
        self.logger = LogManager()
        self.trend_viz = TrendVisualizer()
        self.category_viz = CategoryVisualizer()
        self.network_viz = NetworkVisualizer()
        self.model_viz = ModelVisualizer()

    def visualize_esg_trends(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Delegate to TrendVisualizer"""
        try:
            return self.trend_viz.visualize_esg_trends(df)
        except Exception as e:
            self.logger.error("Error in ESG trends visualization", error=e)
            return None

    def visualize_category_breakdown(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Delegate to CategoryVisualizer"""
        try:
            return self.category_viz.visualize_category_breakdown(df)
        except Exception as e:
            self.logger.error("Error in category breakdown visualization", error=e)
            return None

    def visualize_factor_heatmap(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Delegate to TrendVisualizer"""
        try:
            return self.trend_viz.visualize_factor_heatmap(df)
        except Exception as e:
            self.logger.error("Error in factor heatmap visualization", error=e)
            return None

    def visualize_top_factors_over_time(self, df: pd.DataFrame, 
                                      top_n: int = 5) -> Optional[plt.Figure]:
        """Delegate to TrendVisualizer"""
        try:
            return self.trend_viz.visualize_top_factors(df, top_n)
        except Exception as e:
            self.logger.error("Error in top factors visualization", error=e)
            return None

    def visualize_semantic_similarity_network(self, df: pd.DataFrame, 
                                           threshold: float = 0.5) -> Optional[plt.Figure]:
        """Delegate to NetworkVisualizer"""
        try:
            return self.network_viz.visualize_semantic_similarity_network(df, threshold)
        except Exception as e:
            self.logger.error("Error in semantic similarity network visualization", error=e)
            return None

    def visualize_year_over_year_change(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Delegate to TrendVisualizer"""
        try:
            return self.trend_viz.visualize_year_over_year_change(df)
        except Exception as e:
            self.logger.error("Error in year-over-year change visualization", error=e)
            return None

    def visualize_radar_chart(self, df: pd.DataFrame) -> Optional[plt.Figure]:
        """Delegate to CategoryVisualizer"""
        try:
            return self.category_viz.visualize_radar_chart(df)
        except Exception as e:
            self.logger.error("Error in radar chart visualization", error=e)
            return None

    def visualize_model_performance(self, y_true, y_pred, 
                                  model_name: str) -> Optional[plt.Figure]:
        """Delegate to ModelVisualizer"""
        try:
            return self.model_viz.visualize_model_performance(y_true, y_pred, model_name)
        except Exception as e:
            self.logger.error("Error in model performance visualization", error=e)
            return None

    def visualize_prediction_error_distribution(self, y_true, y_pred_xgb, 
                                             y_pred_rf) -> Optional[plt.Figure]:
        """Delegate to ModelVisualizer"""
        try:
            return self.model_viz.visualize_prediction_error_distribution(
                y_true, y_pred_xgb, y_pred_rf)
        except Exception as e:
            self.logger.error("Error in prediction error distribution visualization", error=e)
            return None

    def visualize_feature_importance(self, xgb_model, rf_model, 
                                   feature_names: list) -> Optional[plt.Figure]:
        """Delegate to ModelVisualizer"""
        try:
            return self.model_viz.visualize_feature_importance(
                xgb_model, rf_model, feature_names)
        except Exception as e:
            self.logger.error("Error in feature importance visualization", error=e)
            return None

    def visualize_model_comparison(self, xgb_metrics: Dict[str, float], 
                                 rf_metrics: Dict[str, float]) -> Optional[plt.Figure]:
        """Delegate to ModelVisualizer"""
        try:
            return self.model_viz.visualize_model_comparison(xgb_metrics, rf_metrics)
        except Exception as e:
            self.logger.error("Error in model comparison visualization", error=e)
            return None