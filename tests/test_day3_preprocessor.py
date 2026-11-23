"""
Unit tests for Day 3: Preprocessor, feature engineering, and chronological splits (FR3, FR5).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor


@pytest.fixture
def sample_raw_data():
    """Generates synthetic patient flow for testing."""
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%d-%m-%Y"),
        "opd_patients": np.random.randint(5000, 7000, size=100),
        "emergency_patients": np.random.randint(350, 550, size=100),
        "total_patient_load": 0,
        "mean_temp_proxy_c_simulated": np.random.uniform(20.0, 35.0, size=100),
        "weekend_flag": [1 if d.weekday() >= 5 else 0 for d in dates],
        "festival_event_proxy_flag": [1 if i in [15, 60] else 0 for i in range(100)],
    })
    df["total_patient_load"] = df["opd_patients"] + df["emergency_patients"]
    return df


def test_preprocessor_clean(sample_raw_data):
    """Verify cleaning and missing context imputation."""
    raw = sample_raw_data.copy()
    raw.loc[5, "mean_temp_proxy_c_simulated"] = np.nan
    raw.loc[10, "festival_event_proxy_flag"] = np.nan

    p = Preprocessor()
    cleaned = p.clean(raw, missing_strategy="interpolate")

    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["mean_temp_proxy_c_simulated"].isna().sum() == 0
    assert cleaned["festival_event_proxy_flag"].isna().sum() == 0
    assert cleaned.loc[10, "festival_event_proxy_flag"] == 0  # Missing flag defaults to 0


def test_preprocessor_feature_generation(sample_raw_data):
    """Verify generation of lags, rolling stats, and cyclical features."""
    p = Preprocessor(
        target_cols=["opd_patients", "emergency_patients"],
        lags=[1, 7],
        rolling_windows=[7],
    )
    cleaned = p.clean(sample_raw_data)
    feats = p.generate_features(cleaned, drop_warmup=True)

    # Check calendar & cyclical columns
    for col in ["day_of_week", "month", "is_weekend", "sin_dow", "cos_dow"]:
        assert col in feats.columns

    # Check lag columns
    assert "opd_patients_lag_1" in feats.columns
    assert "opd_patients_lag_7" in feats.columns
    assert "emergency_patients_lag_1" in feats.columns

    # Check rolling statistics
    assert "opd_patients_roll_mean_7" in feats.columns
    assert "opd_patients_roll_std_7" in feats.columns

    # Verify no NaN values remained after drop_warmup
    assert feats.isna().sum().sum() == 0


def test_zero_data_leakage(sample_raw_data):
    """
    Data Leakage Audit:
    Confirm opd_patients_lag_1 on day t equals opd_patients on day t-1.
    Confirm opd_patients_roll_mean_7 does NOT include day t target.
    """
    p = Preprocessor(
        target_cols=["opd_patients"],
        lags=[1],
        rolling_windows=[7],
    )
    cleaned = p.clean(sample_raw_data)
    feats = p.generate_features(cleaned, drop_warmup=False)

    # Day index 1: lag_1 must equal day index 0's opd_patients
    assert feats.loc[1, "opd_patients_lag_1"] == feats.loc[0, "opd_patients"]

    # Day index 7: roll_mean_7 must equal average of indices 0 through 6
    expected_mean = feats.loc[0:6, "opd_patients"].mean()
    actual_mean = feats.loc[7, "opd_patients_roll_mean_7"]
    assert np.isclose(actual_mean, expected_mean)


def test_chronological_splits():
    """Verify chronological train/val/test splits."""
    dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "val": range(len(dates))})

    train, val, test = Preprocessor.get_train_val_test_splits(df, val_year=2024, test_year=2025)

    assert train["date"].max() == pd.Timestamp("2023-12-31")
    assert val["date"].min() == pd.Timestamp("2024-01-01")
    assert val["date"].max() == pd.Timestamp("2024-12-31")
    assert test["date"].min() == pd.Timestamp("2025-01-01")
    assert test["date"].max() == pd.Timestamp("2025-12-31")
    # Zero overlap across sets
    assert len(set(train["date"]).intersection(set(val["date"]))) == 0
    assert len(set(val["date"]).intersection(set(test["date"]))) == 0
