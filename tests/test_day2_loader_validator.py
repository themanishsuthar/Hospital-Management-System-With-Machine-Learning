"""
Unit tests for Day 2: DataLoader, DataValidator, and error handling (FR1, FR2, FR12).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import DataLoader, DataLoadError
from src.validator import DataValidator, DataValidationError


@pytest.fixture
def loader():
    return DataLoader()


def test_loader_aggregate_data(loader):
    """Verify loading aggregate flow data."""
    df = loader.load_aggregate_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1826
    assert "opd_patients" in df.columns
    assert "emergency_patients" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_loader_hospital_data(loader):
    """Verify loading hospital-level flow records."""
    df = loader.load_hospital_data()
    assert len(df) == 18260
    assert df["hospital_id"].nunique() == 10
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_loader_context_data(loader):
    """Verify loading environmental and calendar event context."""
    df = loader.load_context_data()
    assert len(df) == 1826
    assert "weekend_flag" in df.columns


def test_loader_missing_file_error(loader, tmp_path):
    """Verify clear error message when file does not exist (FR12)."""
    with pytest.raises(DataLoadError) as exc_info:
        loader.load_csv("non_existent_file_123.csv")
    assert "does not exist" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_loader_missing_columns_error(loader, tmp_path):
    """Verify clear error message when required columns are missing (FR12)."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,patient_name\n01-01-2023,John\n", encoding="utf-8")

    with pytest.raises(DataLoadError) as exc_info:
        loader.load_csv(bad_csv, required_columns=["date", "opd_patients"])
    assert "missing required columns" in str(exc_info.value)


def test_validator_on_clean_aggregate(loader):
    """Verify DataValidator passes on clean aggregate patient dataset."""
    df = loader.load_aggregate_data()
    report = DataValidator.validate(
        df,
        key_cols=["date"],
        required_cols=["date", "opd_patients", "emergency_patients", "total_patient_load"],
        check_targets=True,
        raise_on_error=True,
    )
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0


def test_validator_detects_duplicates():
    """Verify DataValidator detects duplicate date/hospital records."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "opd_patients": [100, 110, 120],
        "emergency_patients": [10, 12, 14],
        "total_patient_load": [110, 122, 134],
    })
    report = DataValidator.validate(df, key_cols=["date"])
    assert report["is_valid"] is False
    assert any("duplicate" in err.lower() for err in report["errors"])


def test_validator_detects_negative_values():
    """Verify DataValidator detects negative counts."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "opd_patients": [-50, 100],
        "emergency_patients": [10, 20],
        "total_patient_load": [50, 120],
    })
    report = DataValidator.validate(df, key_cols=["date"])
    assert report["is_valid"] is False
    assert any("range violations" in err.lower() for err in report["errors"])


def test_validator_detects_target_inconsistency():
    """Verify DataValidator detects when total != opd + emergency."""
    df = pd.DataFrame({
        "date": ["2023-01-01"],
        "opd_patients": [100],
        "emergency_patients": [20],
        "total_patient_load": [150],  # Inconsistent: 100 + 20 != 150
    })
    report = DataValidator.validate(df, key_cols=["date"], check_targets=True)
    assert report["is_valid"] is False
    assert any("target inconsistency" in err.lower() for err in report["errors"])


def test_validator_detects_date_gaps():
    """Verify date gap detection flags missing days."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-04"]),  # missing Jan 2 and Jan 3
        "opd_patients": [100, 110],
    })
    gaps = DataValidator.check_date_gaps(df, date_col="date")
    assert "overall" in gaps
    assert len(gaps["overall"]) == 2
