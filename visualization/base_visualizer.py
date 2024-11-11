"""
Base visualization functionality and utilities
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from config.settings import VIZ_SETTINGS
from utils.logging_utils import LogManager

class BaseVisualizer(ABC):
    """
    Base class for visualization components
    """
    def __init__(self):
        self.logger = LogManager()
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """
        Configure matplotlib/seaborn plotting style
        """
        try:
            plt.style.use(VIZ_SETTINGS['style'])
            plt.rcParams['figure.figsize'] = VIZ_SETTINGS['figure_size']
            plt.rcParams['axes.grid'] = VIZ_SETTINGS['grid']
            plt.rcParams['figure.autolayout'] = VIZ_SETTINGS['auto_layout']
            sns.set_theme(style="whitegrid")
        except Exception as e:
            self.logger.error("Error setting up plotting style", error=e)

    def _create_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create figure with error handling
        """
        try:
            if figsize is None:
                figsize = VIZ_SETTINGS['figure_size']
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax
        except Exception as e:
            self.logger.error("Error creating figure", error=e)
            return plt.subplots(figsize=VIZ_SETTINGS['figure_size'])

    def _handle_empty_data(self, fig: plt.Figure, ax: plt.Axes, message: str = "No data available"):
        """
        Handle empty data case
        """
        try:
            ax.clear()
            ax.text(0.5, 0.5, message,
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        except Exception as e:
            self.logger.error("Error handling empty data", error=e)
            return fig

    def _save_figure(self, fig: plt.Figure, filename: str):
        """
        Save figure with error handling
        """
        try:
            fig.savefig(filename, bbox_inches='tight', dpi=300)
        except Exception as e:
            self.logger.error(f"Error saving figure to {filename}", error=e)

    def _validate_data(self, df) -> bool:
        """
        Validate input data
        """
        try:
            return not (df.empty or df.shape[1] == 0)
        except Exception:
            return False