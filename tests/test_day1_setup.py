"""
Unit tests for Day 1: Setup, configuration, and raw data integrity checks.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    AGGREGATE_FLOW_FILE,
    HOSPITAL_FLOW_FILE,
    RANDOM_SEED
)


def test_directories_exist():
    """Verify runtime directories exist."""
    assert RAW_DATA_DIR.exists(), "data/raw directory must exist."
    assert PROCESSED_DATA_DIR.exists(), "data/processed directory must exist."


def test_aggregate_data_integrity():
    """Verify aggregate patient flow dataset loads with 1,826 days and zero nulls."""
    assert AGGREGATE_FLOW_FILE.exists(), f"Missing {AGGREGATE_FLOW_FILE}"
    df = pd.read_csv(AGGREGATE_FLOW_FILE)
    assert df.shape == (1826, 14), f"Unexpected shape: {df.shape}"
    assert df.isnull().sum().sum() == 0, "Aggregate data must have zero null values."
    
    # Date parsing test
    dates = pd.to_datetime(df["date"], format="%d-%m-%Y")
    assert dates.min() == pd.Timestamp("2021-01-01")
    assert dates.max() == pd.Timestamp("2025-12-31")
    
    # Target consistency test
    assert (df["total_patient_load"] == df["opd_patients"] + df["emergency_patients"]).all(), (
        "total_patient_load must strictly equal opd_patients + emergency_patients"
    )


def test_hospital_flow_data_integrity():
    """Verify hospital-level patient flow dataset contains 10 hospitals and 18,260 rows."""
    assert HOSPITAL_FLOW_FILE.exists(), f"Missing {HOSPITAL_FLOW_FILE}"
    df = pd.read_csv(HOSPITAL_FLOW_FILE)
    assert df.shape == (18260, 21), f"Unexpected shape: {df.shape}"
    assert df["hospital_id"].nunique() == 10, "Dataset must cover exactly 10 hospitals."
    assert df.isnull().sum().sum() == 0, "Hospital flow data must have zero null values."
