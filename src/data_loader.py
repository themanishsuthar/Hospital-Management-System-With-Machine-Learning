"""
Data Loader & Validator Module
Aligned with SRS Chapter 5.4 Class Diagram Design
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


class DataLoader:
    """Loads raw structured CSV datasets from the project data folder."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Auto-detect data folder
            candidates = [
                Path("data"),
                Path("../data"),
                Path("../../data"),
                Path(__file__).resolve().parent.parent / "data",
            ]
            self.data_dir = next((p for p in candidates if (p / "ML_model_training_based.csv").exists()), None)
            if self.data_dir is None:
                raise FileNotFoundError("Could not auto-locate the 'data' directory with project CSV files.")
        else:
            self.data_dir = Path(data_dir)

    def load_aggregate_data(self) -> pd.DataFrame:
        """Load the city-wide all-hospitals aggregated daily time-series."""
        file_path = self.data_dir / "ML_model_training_based.csv"
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        return df.sort_values("date").reset_index(drop=True)

    def load_hospital_data(self) -> pd.DataFrame:
        """Load the 10-hospital per-day historical patient flow records."""
        file_path = self.data_dir / "patient_flow_from_2021_to_2025.csv"
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        return df.sort_values(["hospital_id", "date"]).reset_index(drop=True)

    def load_context_data(self) -> pd.DataFrame:
        """Load daily environmental, seasonal, and calendar event flags."""
        file_path = self.data_dir / "daily_patient_flow_based_on_events.csv"
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        return df.sort_values("date").reset_index(drop=True)

    def load_hospital_directory(self) -> pd.DataFrame:
        """Load hospital addresses, categories, and facility scale directory."""
        file_path = self.data_dir / "hospital_and_address_related.csv"
        return pd.read_csv(file_path, encoding="utf-8-sig")

    def load_service_types(self) -> pd.DataFrame:
        """Load hospital service listings and 24x7 emergency status."""
        file_path = self.data_dir / "hospital_and_service_type.csv"
        return pd.read_csv(file_path, encoding="utf-8-sig")


class DataValidator:
    """Validates data integrity, checks missing values, ranges, and cross-consistency."""

    @staticmethod
    def check_missing(df: pd.DataFrame) -> Dict[str, int]:
        """Return counts of missing values per column."""
        return df.isna().sum().to_dict()

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: list) -> int:
        """Count duplicate rows based on subset keys."""
        return int(df.duplicated(subset=subset).sum())

    @staticmethod
    def check_ranges(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Verify numeric columns do not have impossible negative values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {col: (float(df[col].min()), float(df[col].max())) for col in numeric_cols}

    @staticmethod
    def validate_dataset(df: pd.DataFrame, key_cols: list) -> bool:
        """Comprehensive validation check."""
        missing_count = sum(DataValidator.check_missing(df).values())
        dup_count = DataValidator.check_duplicates(df, key_cols)
        assert missing_count == 0, f"Validation Failed: {missing_count} missing values found."
        assert dup_count == 0, f"Validation Failed: {dup_count} duplicate keys found."
        return True

