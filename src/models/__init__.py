"""
Models package initialization.
Exports ForecastModel base contract, baseline persistence heuristics, XGBoost, and Prophet implementations.
"""

from src.models.base import ForecastModel, calculate_metrics
from src.models.baseline_model import NaiveLagBaselineModel, MovingAverageBaselineModel
from src.models.xgboost_model import XGBoostModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel

__all__ = [
    "ForecastModel",
    "calculate_metrics",
    "NaiveLagBaselineModel",
    "MovingAverageBaselineModel",
    "XGBoostModel",
    "ProphetModel",
    "LSTMModel",
]
