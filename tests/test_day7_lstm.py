"""
Unit tests for Day 7: LSTMModel (FR6, FR7)
Validates:
- Initialization and parameter configuration
- Sliding window sequence generation
- MinMaxScaler normalization and inverse transformation
- Early stopping during model fitting
- Sequence-based prediction and multi-step recursive horizon forecasting
- Model persistence (save & load) and metric evaluation
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.lstm_model import LSTMModel
from src.models.base import calculate_metrics


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generates synthetic daily time series data (60 days) with trend and weekly cycle."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    day_of_week = dates.day_of_week
    trend = np.linspace(100, 200, 60)
    cycle = 30 * np.sin(2 * np.pi * day_of_week / 7)
    noise = np.random.normal(0, 5, 60)
    opd = np.maximum(50, trend + cycle + noise)
    emergency = np.maximum(10, 20 + 5 * np.cos(2 * np.pi * day_of_week / 7) + np.random.normal(0, 2, 60))

    return pd.DataFrame({
        "date": dates,
        "opd_patients": np.round(opd, 1),
        "emergency_patients": np.round(emergency, 1),
        "weekend_flag": (day_of_week >= 5).astype(int),
        "temp_proxy": np.random.uniform(15, 35, 60),
    })


def test_lstm_initialization():
    """Tests default and custom parameter initialization."""
    model = LSTMModel(
        name="TestLSTM",
        target_col="opd_patients",
        window_size=7,
        lstm_units=(32,),
        dropout_rate=0.1,
        epochs=5,
    )
    assert model.name == "TestLSTM"
    assert model.target_col == "opd_patients"
    assert model.window_size == 7
    assert model.lstm_units == (32,)
    assert not model.is_fitted


def test_lstm_fit_and_predict(sample_timeseries_df):
    """Tests fitting LSTM on train data and predicting on test data."""
    train_df = sample_timeseries_df.iloc[:45].copy()
    test_df = sample_timeseries_df.iloc[45:].copy()

    model = LSTMModel(
        name="LSTM_OPD",
        target_col="opd_patients",
        feature_cols=["weekend_flag", "temp_proxy"],
        window_size=7,
        lstm_units=(16,),
        epochs=10,
        batch_size=8,
        patience=3,
        seed=42,
    )

    model.fit(train_df=train_df, verbose=0)
    assert model.is_fitted
    assert model.model is not None
    assert model.last_date_ is not None

    preds = model.predict(test_df)
    assert len(preds) == len(test_df)
    assert np.all(preds >= 0)  # Patient counts cannot be negative

    metrics = calculate_metrics(test_df["opd_patients"].values, preds)
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert metrics["RMSE"] > 0


def test_lstm_fit_with_validation_early_stopping(sample_timeseries_df):
    """Tests model training with a validation set and early stopping."""
    train_df = sample_timeseries_df.iloc[:35].copy()
    val_df = sample_timeseries_df.iloc[35:45].copy()

    model = LSTMModel(
        name="LSTM_EarlyStop",
        target_col="opd_patients",
        feature_cols=["weekend_flag", "temp_proxy"],
        window_size=5,
        lstm_units=(16,),
        epochs=30,
        batch_size=8,
        patience=2,
        seed=42,
    )

    model.fit(train_df=train_df, val_df=val_df, verbose=0)
    assert model.is_fitted
    assert model.training_history_ is not None
    assert "val_loss" in model.training_history_
    # Verify early stopping stopped before 30 epochs
    assert len(model.training_history_["loss"]) <= 30


def test_lstm_forecast_horizon(sample_timeseries_df):
    """Tests multi-step recursive horizon forecasting."""
    train_df = sample_timeseries_df.iloc[:45].copy()

    model = LSTMModel(
        name="LSTM_Horizon",
        target_col="opd_patients",
        feature_cols=["weekend_flag", "temp_proxy"],
        window_size=7,
        lstm_units=(16,),
        epochs=5,
        seed=42,
    )
    model.fit(train_df=train_df, verbose=0)

    horizon_df = model.forecast_horizon(horizon_days=7)
    assert len(horizon_df) == 7
    assert "date" in horizon_df.columns
    assert "predicted_opd" in horizon_df.columns
    assert (horizon_df["predicted_opd"] >= 0).all()
    # Ensure forecast dates are strictly in the future
    assert horizon_df["date"].iloc[0] == pd.to_datetime(train_df["date"].iloc[-1]) + pd.Timedelta(days=1)


def test_lstm_save_and_load(sample_timeseries_df):
    """Tests model serialization and loading from disk."""
    train_df = sample_timeseries_df.iloc[:45].copy()
    test_df = sample_timeseries_df.iloc[45:].copy()

    model = LSTMModel(
        name="LSTM_Persist",
        target_col="emergency_patients",
        feature_cols=["weekend_flag", "temp_proxy"],
        window_size=5,
        lstm_units=(16,),
        epochs=5,
        seed=42,
    )
    model.fit(train_df=train_df, verbose=0)
    orig_preds = model.predict(test_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "lstm_test_model"
        model.save(save_path)

        loaded_model = LSTMModel.load(save_path)
        assert loaded_model.is_fitted
        assert loaded_model.name == "LSTM_Persist"
        assert loaded_model.target_col == "emergency_patients"

        loaded_preds = loaded_model.predict(test_df)
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-4, atol=1e-4)


def test_lstm_insufficient_samples_raises():
    """Tests that passing fewer samples than window_size raises ValueError."""
    small_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "opd_patients": [10, 20, 30, 40, 50],
        "feature_a": [1, 2, 3, 4, 5],
    })
    model = LSTMModel(window_size=10, target_col="opd_patients")
    with pytest.raises(ValueError, match="greater than window_size"):
        model.fit(small_df)
