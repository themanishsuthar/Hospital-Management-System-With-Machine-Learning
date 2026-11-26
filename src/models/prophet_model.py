"""
Prophet Forecasting Model (ProphetModel)
Aligned with SRS Chapter 5.4 Class Diagram (FR6, FR7)
Decomposes patient flow into trend, weekly/annual seasonality, holiday effects,
and external environmental/event regressors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from prophet import Prophet
import joblib

from src.models.base import ForecastModel
from src.config import DEFAULT_HORIZON_DAYS


class ProphetModel(ForecastModel):
    """
    Decomposable additive time-series forecasting model using Facebook Prophet.
    Handles non-linear trends with yearly, weekly, and holiday seasonalities plus exogenous regressors.
    """

    def __init__(
        self,
        target_col: str,
        regressor_cols: Optional[List[str]] = None,
        date_col: str = "date",
        country_holidays: str = "IN",
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        interval_width: float = 0.95,
        **prophet_kwargs: Any,
    ):
        super().__init__(
            name="Prophet Time-Series Regressor",
            target_col=target_col,
            feature_cols=regressor_cols or [],
        )
        self.date_col = date_col
        self.country_holidays = country_holidays
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.prophet_kwargs = prophet_kwargs

        # Internal estimator
        self.estimator: Optional[Prophet] = None
        self._init_estimator()

    def _init_estimator(self):
        """Initializes Prophet estimator with specified seasonality and regressors."""
        self.estimator = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            **self.prophet_kwargs,
        )
        if self.country_holidays:
            try:
                self.estimator.add_country_holidays(country_name=self.country_holidays)
            except Exception:
                pass  # Fall back gracefully if holidays country missing

        # Register external regressors
        for reg in self.feature_cols:
            self.estimator.add_regressor(reg)

    def _format_dataframe(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Converts input DataFrame into Prophet's expected 'ds' and 'y' schema."""
        df = pd.DataFrame()
        if "ds" in X.columns:
            df["ds"] = pd.to_datetime(X["ds"])
        elif self.date_col in X.columns:
            df["ds"] = pd.to_datetime(X[self.date_col], dayfirst=True)
        else:
            raise ValueError(f"Prophet requires a date column '{self.date_col}' or 'ds'.")

        if y is not None:
            df["y"] = np.asarray(y, dtype=float)

        # Append registered regressors
        for reg in self.feature_cols:
            if reg in X.columns:
                df[reg] = X[reg].values
            else:
                df[reg] = 0.0

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetModel":
        """
        Fits Prophet model on chronological time-series with holidays and regressors.
        """
        prophet_df = self._format_dataframe(X, y=y)
        self._init_estimator()
        # Suppress cmdstanpy verbose logs
        self.estimator.fit(prophet_df)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates point forecasts (clamped to non-negative counts)."""
        if not self.is_fitted or self.estimator is None:
            raise ValueError(f"Model '{self.name}' must be fitted before predict.")

        prophet_df = self._format_dataframe(X)
        forecast = self.estimator.predict(prophet_df)
        point_preds = forecast["yhat"].values
        # Non-negative clinical count constraint
        return np.maximum(0.0, np.round(point_preds))

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates full predictive distribution: point estimate, lower and upper confidence bounds.
        """
        if not self.is_fitted or self.estimator is None:
            raise ValueError(f"Model '{self.name}' must be fitted before predict.")

        prophet_df = self._format_dataframe(X)
        forecast = self.estimator.predict(prophet_df)
        res = pd.DataFrame({
            "ds": forecast["ds"],
            "yhat": np.maximum(0.0, np.round(forecast["yhat"])),
            "yhat_lower": np.maximum(0.0, np.round(forecast["yhat_lower"])),
            "yhat_upper": np.maximum(0.0, np.round(forecast["yhat_upper"])),
        })
        return res

    def forecast_horizon(
        self,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generates multi-step lookahead forecast for the configured planning horizon.
        """
        if not self.is_fitted or self.estimator is None:
            raise ValueError(f"Model '{self.name}' must be fitted before forecast.")

        future_df = self.estimator.make_future_dataframe(periods=horizon_days, freq="D")

        # Fill future regressors if provided, else forward-fill from history
        for reg in self.feature_cols:
            if future_regressors is not None and reg in future_regressors.columns:
                # Merge or align
                future_df[reg] = future_regressors[reg].values[-len(future_df):]
            else:
                future_df[reg] = 0.0

        forecast = self.estimator.predict(future_df)
        lookahead = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        lookahead["yhat"] = np.maximum(0.0, np.round(lookahead["yhat"]))
        lookahead["yhat_lower"] = np.maximum(0.0, np.round(lookahead["yhat_lower"]))
        lookahead["yhat_upper"] = np.maximum(0.0, np.round(lookahead["yhat_upper"]))
        return lookahead.reset_index(drop=True)
