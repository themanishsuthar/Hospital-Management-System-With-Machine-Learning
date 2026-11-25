"""
Baseline Forecasting Models
Aligned with SRS Chapter 5.4 Class Diagram (FR6)
Implements persistence heuristics (Naive Lag-7 and Moving Average baselines)
to benchmark machine learning algorithm gains.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class NaiveLagBaselineModel(ForecastModel):
    """
    Persistence baseline that forecasts patient load using an autoregressive lag
    (typically Lag-7: the same day of the week from the previous week).
    """

    def __init__(
        self,
        target_col: str,
        lag: int = 7,
        lag_col: Optional[str] = None,
    ):
        super().__init__(
            name=f"Naive Lag-{lag} Baseline",
            target_col=target_col,
            feature_cols=[lag_col or f"{target_col}_lag_{lag}"],
        )
        self.lag = lag
        self.lag_col = lag_col or f"{target_col}_lag_{lag}"
        self.fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLagBaselineModel":
        """Fit caches the target series mean as a fallback value for any missing rows."""
        self.fallback_mean = float(y.mean())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Outputs the lag feature value directly as the predicted patient load."""
        if not self.is_fitted:
            raise ValueError(f"Model '{self.name}' must be fitted before predict.")

        if self.lag_col in X.columns:
            preds = X[self.lag_col].fillna(self.fallback_mean).values
        else:
            # If explicit lag column is absent, use fallback mean
            preds = np.full(len(X), self.fallback_mean)

        return np.maximum(0.0, preds.astype(float))


class MovingAverageBaselineModel(ForecastModel):
    """
    Moving-average baseline that forecasts patient volume using the historical
    rolling mean (e.g. 7-day or 14-day moving average).
    """

    def __init__(
        self,
        target_col: str,
        window: int = 7,
        roll_mean_col: Optional[str] = None,
    ):
        super().__init__(
            name=f"Moving Average ({window}d) Baseline",
            target_col=target_col,
            feature_cols=[roll_mean_col or f"{target_col}_roll_mean_{window}"],
        )
        self.window = window
        self.roll_mean_col = roll_mean_col or f"{target_col}_roll_mean_{window}"
        self.fallback_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageBaselineModel":
        """Fit caches overall series mean for fallback handling."""
        self.fallback_mean = float(y.mean())
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Outputs the pre-computed rolling mean as the forecast."""
        if not self.is_fitted:
            raise ValueError(f"Model '{self.name}' must be fitted before predict.")

        if self.roll_mean_col in X.columns:
            preds = X[self.roll_mean_col].fillna(self.fallback_mean).values
        else:
            preds = np.full(len(X), self.fallback_mean)

        return np.maximum(0.0, preds.astype(float))
