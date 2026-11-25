"""
Models package initialization.
Exports ForecastModel base contract, baseline persistence heuristics, and XGBoost implementation.
"""

from src.models.base import ForecastModel, calculate_metrics
from src.models.baseline_model import NaiveLagBaselineModel, MovingAverageBaselineModel
from src.models.xgboost_model import XGBoostModel

__all__ = [
    "ForecastModel",
    "calculate_metrics",
    "NaiveLagBaselineModel",
    "MovingAverageBaselineModel",
    "XGBoostModel",
]
