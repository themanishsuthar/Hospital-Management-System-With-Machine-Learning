"""
Abstract Base Forecast Model (ForecastModel)
Aligned with SRS Chapter 5.4 Class Diagram (FR6, FR7)
Defines uniform contract for all predictive time-series models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE (%): Mean Absolute Percentage Error
    - R2: Coefficient of Determination
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    mask = y_t != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE(%)": round(mape, 2),
        "R2": round(r2, 4),
    }


class ForecastModel(ABC):
    """
    Abstract base class for patient load forecasting models (FR6, FR7).
    Guarantees standard API across Baseline, XGBoost, Prophet, and LSTM implementations.
    """

    def __init__(
        self,
        name: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
    ):
        self.name = name
        self.target_col = target_col
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        """Fit the model onto training features X and target series y."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts for input features X."""
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluates model performance against true values and caches metrics."""
        y_pred = self.predict(X)
        self.metrics = calculate_metrics(y.values, y_pred)
        return self.metrics

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serializes the fitted model pipeline to disk."""
        if not self.is_fitted:
            raise ValueError(f"Cannot save unfitted model '{self.name}'. Call fit() first.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ForecastModel":
        """Deserializes a fitted model pipeline from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file '{path}' does not exist.")
        loaded = joblib.load(path)
        if not isinstance(loaded, ForecastModel):
            raise TypeError(f"Loaded object is not an instance of ForecastModel: {type(loaded)}")
        return loaded
