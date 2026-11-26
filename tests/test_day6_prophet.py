"""
Unit tests for Day 6: ProphetModel, holiday effects, regressors, and horizon forecasting (FR6, FR7).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.prophet_model import ProphetModel


@pytest.fixture
def prophet_sample_data():
    """Generates synthetic longitudinal time series for testing ProphetModel."""
    np.random.seed(42)
    n = 90
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base_opd = 6000 + 400 * np.sin(2 * np.pi * dates.day_of_week / 7.0)
    noise = np.random.normal(0, 50, size=n)
    opd = np.maximum(0.0, np.round(base_opd + noise))

    df = pd.DataFrame({
        "date": dates,
        "opd_patients": opd,
        "weekend_flag": [1.0 if d.weekday() >= 5 else 0.0 for d in dates],
        "heat_flag": [1.0 if i in [20, 21, 22] else 0.0 for i in range(n)],
    })
    return df


def test_prophet_model_fit_and_predict(prophet_sample_data):
    """Verify ProphetModel training and inference on OPD patient load."""
    train = prophet_sample_data.iloc[:75]
    test = prophet_sample_data.iloc[75:]

    model = ProphetModel(
        target_col="opd_patients",
        regressor_cols=["weekend_flag"],
        date_col="date",
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.fit(train, train["opd_patients"])
    preds = model.predict(test)

    assert len(preds) == len(test)
    assert (preds >= 0).all(), "Predictions must adhere to non-negative clinical counts."
    metrics = model.evaluate(test, test["opd_patients"])
    assert "MAE" in metrics
    assert "MAPE(%)" in metrics
    assert metrics["MAE"] < 500.0


def test_prophet_predict_interval(prophet_sample_data):
    """Verify confidence interval bounds generation."""
    train = prophet_sample_data.iloc[:70]
    test = prophet_sample_data.iloc[70:]

    model = ProphetModel(
        target_col="opd_patients",
        regressor_cols=["weekend_flag"],
        weekly_seasonality=True,
    )
    model.fit(train, train["opd_patients"])
    interval_df = model.predict_interval(test)

    assert len(interval_df) == len(test)
    for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
        assert col in interval_df.columns
    # Check lower <= point <= upper in general
    assert (interval_df["yhat_lower"] <= interval_df["yhat_upper"]).all()


def test_prophet_forecast_horizon(prophet_sample_data):
    """Verify multi-step forward horizon forecast (7-day lookahead)."""
    model = ProphetModel(
        target_col="opd_patients",
        regressor_cols=["weekend_flag"],
        weekly_seasonality=True,
    )
    model.fit(prophet_sample_data, prophet_sample_data["opd_patients"])

    horizon_7 = model.forecast_horizon(horizon_days=7)
    assert len(horizon_7) == 7
    assert (horizon_7["yhat"] >= 0).all()


def test_prophet_model_save_and_load(prophet_sample_data, tmp_path):
    """Verify serialization and deserialization of fitted ProphetModel."""
    model = ProphetModel(
        target_col="opd_patients",
        regressor_cols=["weekend_flag"],
        weekly_seasonality=True,
    )
    model.fit(prophet_sample_data.iloc[:60], prophet_sample_data.iloc[:60]["opd_patients"])

    save_path = tmp_path / "prophet_test.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = ProphetModel.load(save_path)
    assert loaded.is_fitted is True

    orig_preds = model.predict(prophet_sample_data.iloc[60:65])
    loaded_preds = loaded.predict(prophet_sample_data.iloc[60:65])
    assert np.allclose(orig_preds, loaded_preds)
