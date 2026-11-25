"""
Unit tests for Day 5: Baseline models, XGBoostModel, metrics, and persistence (FR6, FR7).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.base import ForecastModel, calculate_metrics
from src.models.baseline_model import NaiveLagBaselineModel, MovingAverageBaselineModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def small_sample_dataset():
    """Generates small sample patient flow data for testing Day 5 models."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    opd = np.random.randint(5000, 7000, size=n)
    emg = np.random.randint(350, 550, size=n)

    df = pd.DataFrame({
        "date": dates,
        "opd_patients": opd,
        "emergency_patients": emg,
        "total_patient_load": opd + emg,
        "day_of_week": dates.day_of_week,
        "month": dates.month,
        "weekend_flag": [1 if d.weekday() >= 5 else 0 for d in dates],
        "opd_patients_lag_1": pd.Series(opd).shift(1).bfill(),
        "opd_patients_lag_7": pd.Series(opd).shift(7).bfill(),
        "opd_patients_roll_mean_7": pd.Series(opd).shift(1).rolling(7, min_periods=1).mean().bfill(),
        "emergency_patients_lag_1": pd.Series(emg).shift(1).bfill(),
        "emergency_patients_lag_7": pd.Series(emg).shift(7).bfill(),
        "emergency_patients_roll_mean_7": pd.Series(emg).shift(1).rolling(7, min_periods=1).mean().bfill(),
    })
    return df


def test_forecast_model_cannot_be_instantiated():
    """Verify ForecastModel is an abstract base class enforcing fit() and predict()."""
    with pytest.raises(TypeError):
        ForecastModel("test", "opd_patients")


def test_calculate_metrics():
    """Verify regression metrics calculation."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 305.0])
    metrics = calculate_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE(%)" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] == pytest.approx(8.33, abs=0.1)


def test_naive_lag_baseline(small_sample_dataset):
    """Verify NaiveLagBaselineModel on OPD patients."""
    train = small_sample_dataset.iloc[:40]
    test = small_sample_dataset.iloc[40:]

    model = NaiveLagBaselineModel(target_col="opd_patients", lag=7)
    model.fit(train, train["opd_patients"])
    preds = model.predict(test)

    assert len(preds) == len(test)
    assert (preds >= 0).all()
    metrics = model.evaluate(test, test["opd_patients"])
    assert "MAE" in metrics
    assert metrics["MAE"] > 0


def test_moving_average_baseline(small_sample_dataset):
    """Verify MovingAverageBaselineModel on Emergency patients."""
    train = small_sample_dataset.iloc[:40]
    test = small_sample_dataset.iloc[40:]

    model = MovingAverageBaselineModel(target_col="emergency_patients", window=7)
    model.fit(train, train["emergency_patients"])
    preds = model.predict(test)

    assert len(preds) == len(test)
    assert (preds >= 0).all()
    metrics = model.evaluate(test, test["emergency_patients"])
    assert "RMSE" in metrics


def test_xgboost_model_opd(small_sample_dataset):
    """Verify XGBoostModel training and inference for OPD patients on small sample."""
    train = small_sample_dataset.iloc[:45]
    test = small_sample_dataset.iloc[45:]

    feature_cols = [
        "day_of_week", "month", "weekend_flag",
        "opd_patients_lag_1", "opd_patients_lag_7", "opd_patients_roll_mean_7"
    ]
    xgb_model = XGBoostModel(
        target_col="opd_patients",
        feature_cols=feature_cols,
        n_estimators=30,
        max_depth=3,
        random_state=42,
    )
    xgb_model.fit(train, train["opd_patients"])
    preds = xgb_model.predict(test)

    assert len(preds) == len(test)
    assert (preds >= 0).all()
    metrics = xgb_model.evaluate(test, test["opd_patients"])
    assert metrics["MAE"] < 1000.0  # reasonable error on small sample

    # Verify feature importances
    importances = xgb_model.get_feature_importances()
    assert len(importances) == len(feature_cols)
    assert importances.sum() > 0


def test_xgboost_model_emergency(small_sample_dataset):
    """Verify XGBoostModel training for Emergency patients."""
    train = small_sample_dataset.iloc[:45]
    test = small_sample_dataset.iloc[45:]

    feature_cols = [
        "day_of_week", "month", "weekend_flag",
        "emergency_patients_lag_1", "emergency_patients_lag_7", "emergency_patients_roll_mean_7"
    ]
    xgb_model = XGBoostModel(
        target_col="emergency_patients",
        feature_cols=feature_cols,
        n_estimators=30,
        max_depth=3,
        random_state=42,
    )
    xgb_model.fit(train, train["emergency_patients"])
    preds = xgb_model.predict(test)

    assert len(preds) == len(test)
    assert (preds >= 0).all()


def test_xgboost_model_save_and_load(small_sample_dataset, tmp_path):
    """Verify serialization and deserialization of fitted XGBoostModel."""
    feature_cols = ["day_of_week", "opd_patients_lag_1", "opd_patients_lag_7"]
    model = XGBoostModel(target_col="opd_patients", feature_cols=feature_cols, n_estimators=20)
    model.fit(small_sample_dataset, small_sample_dataset["opd_patients"])

    save_path = tmp_path / "xgb_test.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = XGBoostModel.load(save_path)
    assert loaded.name == model.name
    assert loaded.is_fitted is True

    orig_preds = model.predict(small_sample_dataset.iloc[:5])
    loaded_preds = loaded.predict(small_sample_dataset.iloc[:5])
    assert np.allclose(orig_preds, loaded_preds)
