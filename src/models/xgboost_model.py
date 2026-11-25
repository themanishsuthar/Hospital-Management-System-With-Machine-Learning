"""
XGBoost Forecasting Model (XGBoostModel)
Aligned with SRS Chapter 5.4 Class Diagram (FR6, FR7)
Implements gradient boosted decision trees for OPD and Emergency load prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import ForecastModel
from src.config import RANDOM_SEED


class XGBoostModel(ForecastModel):
    """
    Gradient Boosted Decision Tree Regressor specialized for patient load forecasting.
    Optimizes non-linear interactions across temporal, cyclical, and climatic features.
    """

    def __init__(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = RANDOM_SEED,
        **kwargs: Any,
    ):
        super().__init__(
            name="XGBoost Regressor",
            target_col=target_col,
            feature_cols=feature_cols or [],
        )
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1,
            **kwargs,
        }
        self.estimator = xgb.XGBRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """
        Fits XGBoost Regressor on input features and target series.
        Automatically resolves feature columns if not explicitly provided at initialization.
        """
        if not self.feature_cols:
            self.feature_cols = [c for c in X.columns if c not in ["date", self.target_col, "total_patient_load"]]

        X_train = X[self.feature_cols]
        y_train = np.asarray(y, dtype=float)

        self.estimator.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates non-negative patient count predictions."""
        if not self.is_fitted:
            raise ValueError(f"Model '{self.name}' must be fitted before predict.")

        X_pred = X[self.feature_cols]
        preds = self.estimator.predict(X_pred)
        # Clinical constraint: patient load cannot be negative
        return np.maximum(0.0, np.round(preds))

    def get_feature_importances(self) -> pd.Series:
        """Returns Gini feature importance hierarchy sorted descending."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to compute feature importances.")
        importances = pd.Series(
            self.estimator.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)
        return importances
