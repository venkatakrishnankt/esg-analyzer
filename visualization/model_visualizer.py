"""
Model performance visualization components
"""
from .base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class ModelVisualizer(BaseVisualizer):
    """
    Handles model performance visualizations
    """
    def visualize_model_performance(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  model_name: str) -> plt.Figure:
        """
        Create scatter plot of predicted vs actual values
        """
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax, "No data available")

            fig, ax = self._create_figure(figsize=(10, 6))
            
            # Plot predictions vs actual
            ax.scatter(y_true, y_pred, alpha=0.6, color='blue',
                      label='Predictions')
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfect Prediction')
            
            # Add trend line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(y_true, p(y_true), 'g--', 
                   alpha=0.8, label='Trend Line')
            
            # Calculate R² for trend line
            r2 = np.corrcoef(y_true, y_pred)[0,1]**2
            
            ax.set_xlabel('Actual Values', fontsize=10)
            ax.set_ylabel('Predicted Values', fontsize=10)
            ax.set_title(f'{model_name}: Predicted vs Actual (R² = {r2:.3f})',
                        pad=20, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model performance visualization for {model_name}", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_prediction_error_distribution(self, y_true: np.ndarray, 
                                             y_pred_xgb: np.ndarray, 
                                             y_pred_rf: np.ndarray) -> plt.Figure:
        """
        Create error distribution plots for both models
        """
        try:
            if len(y_true) == 0 or len(y_pred_xgb) == 0 or len(y_pred_rf) == 0:
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax, "No data available")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Calculate errors
            errors_xgb = y_true - y_pred_xgb
            errors_rf = y_true - y_pred_rf
            
            # Plot XGBoost errors
            sns.histplot(errors_xgb, kde=True, ax=ax1, color='blue', alpha=0.6)
            ax1.axvline(x=0, color='r', linestyle='--')
            ax1.set_title("XGBoost Error Distribution\n" +
                         f"Mean Error: {np.mean(errors_xgb):.2f}\n" +
                         f"Std Dev: {np.std(errors_xgb):.2f}")
            ax1.set_xlabel("Prediction Error")
            
            # Plot Random Forest errors
            sns.histplot(errors_rf, kde=True, ax=ax2, color='green', alpha=0.6)
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_title("Random Forest Error Distribution\n" +
                         f"Mean Error: {np.mean(errors_rf):.2f}\n" +
                         f"Std Dev: {np.std(errors_rf):.2f}")
            ax2.set_xlabel("Prediction Error")
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating error distribution visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_feature_importance(self, xgb_model, rf_model, 
                                   feature_names: list) -> plt.Figure:
        """
        Create feature importance comparison plots
        """
        try:
            if xgb_model is None or rf_model is None:
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax, "No model data available")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # XGBoost importance
            xgb_importance = xgb_model.feature_importances_
            xgb_indices = np.argsort(xgb_importance)[::-1]
            
            ax1.bar(range(len(xgb_importance)), 
                   xgb_importance[xgb_indices],
                   color='blue', alpha=0.6)
            ax1.set_title("XGBoost Feature Importance")
            ax1.set_xlabel("Features")
            ax1.set_ylabel("Importance")
            ax1.set_xticks(range(len(xgb_importance)))
            ax1.set_xticklabels([feature_names[i] for i in xgb_indices], 
                               rotation=45, ha='right')
            
            # Random Forest importance
            rf_importance = rf_model.feature_importances_
            rf_indices = np.argsort(rf_importance)[::-1]
            
            ax2.bar(range(len(rf_importance)), 
                   rf_importance[rf_indices],
                   color='green', alpha=0.6)
            ax2.set_title("Random Forest Feature Importance")
            ax2.set_xlabel("Features")
            ax2.set_ylabel("Importance")
            ax2.set_xticks(range(len(rf_importance)))
            ax2.set_xticklabels([feature_names[i] for i in rf_indices], 
                               rotation=45, ha='right')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating feature importance visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")

    def visualize_model_comparison(self, xgb_metrics: dict, 
                                 rf_metrics: dict) -> plt.Figure:
        """
        Create model performance comparison plots
        """
        try:
            if not xgb_metrics or not rf_metrics:
                fig, ax = self._create_figure()
                return self._handle_empty_data(fig, ax, "No metrics available")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            models = ['XGBoost', 'Random Forest']
            mse_scores = [xgb_metrics.get('mse', 0), rf_metrics.get('mse', 0)]
            r2_scores = [xgb_metrics.get('r2', 0), rf_metrics.get('r2', 0)]
            
            # Plot MSE comparison
            bars1 = ax1.bar(models, mse_scores, color=['blue', 'green'], alpha=0.6)
            ax1.set_title("Mean Squared Error Comparison")
            ax1.set_ylabel("MSE")
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            # Plot R² comparison
            bars2 = ax2.bar(models, r2_scores, color=['blue', 'green'], alpha=0.6)
            ax2.set_title("R-squared Comparison")
            ax2.set_ylabel("R-squared")
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error("Error creating model comparison visualization", error=e)
            fig, ax = self._create_figure()
            return self._handle_empty_data(fig, ax, "Error creating visualization")