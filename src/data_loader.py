"""
Data Ingestion Module (DataLoader)
Aligned with SRS Chapter 5.4 Class Diagram (FR1, FR2, FR12)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    DATA_DIR,
    AGGREGATE_FLOW_FILE,
    HOSPITAL_FLOW_FILE,
    EVENT_FLOW_FILE,
    HOSPITAL_METADATA_FILE,
    HOSPITAL_SERVICE_FILE,
)


class DataLoadError(Exception):
    """Custom exception raised when data ingestion fails (FR12)."""
    pass


class DataLoader:
    """Loads raw structured CSV datasets for hospital patient flow and context factors."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        elif RAW_DATA_DIR.exists() and (RAW_DATA_DIR / "ML_model_training_based.csv").exists():
            self.data_dir = RAW_DATA_DIR
        elif DATA_DIR.exists() and (DATA_DIR / "ML_model_training_based.csv").exists():
            self.data_dir = DATA_DIR
        else:
            raise DataLoadError(
                f"Data directory not found. Expected CSV files in either '{RAW_DATA_DIR}' or '{DATA_DIR}'."
            )

    def _resolve_file_path(self, filename: str) -> Path:
        """Finds the file in self.data_dir, or falls back to RAW_DATA_DIR / DATA_DIR."""
        target = self.data_dir / filename
        if target.exists():
            return target
        if (RAW_DATA_DIR / filename).exists():
            return RAW_DATA_DIR / filename
        if (DATA_DIR / filename).exists():
            return DATA_DIR / filename
        raise DataLoadError(
            f"File '{filename}' not found. Searched in '{self.data_dir}', '{RAW_DATA_DIR}', and '{DATA_DIR}'."
        )

    def load_csv(
        self,
        file_path_or_name: Union[str, Path],
        required_columns: Optional[List[str]] = None,
        date_column: Optional[str] = "date",
    ) -> pd.DataFrame:
        """
        Generic robust CSV loader with column verification and date parsing.
        Raises DataLoadError with readable descriptions if the file is missing or invalid (FR12).
        """
        path = Path(file_path_or_name)
        if not path.is_absolute() and not path.exists():
            path = self._resolve_file_path(path.name)

        if not path.exists():
            raise DataLoadError(f"Dataset file '{path}' does not exist.")

        if path.suffix.lower() != ".csv":
            raise DataLoadError(f"Invalid file format: '{path.name}'. Only CSV files are supported.")

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            raise DataLoadError(f"Failed to read CSV file '{path.name}': {str(e)}") from e

        if df.empty:
            raise DataLoadError(f"Dataset file '{path.name}' is empty (0 rows).")

        if required_columns:
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                raise DataLoadError(
                    f"Dataset '{path.name}' is missing required columns: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )

        if date_column and date_column in df.columns:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format="%d-%m-%Y")
            except Exception:
                try:
                    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
                except Exception as e:
                    raise DataLoadError(
                        f"Failed to parse dates in column '{date_column}' for file '{path.name}': {str(e)}"
                    ) from e

        return df

    def load_aggregate_data(self) -> pd.DataFrame:
        """Load the city-wide all-hospitals aggregated daily time-series (FR1)."""
        required = ["date", "opd_patients", "emergency_patients", "total_patient_load"]
        df = self.load_csv("ML_model_training_based.csv", required_columns=required)
        return df.sort_values("date").reset_index(drop=True)

    def load_hospital_data(self) -> pd.DataFrame:
        """Load the 10-hospital daily patient flow records (FR1)."""
        required = ["date", "hospital_id", "opd_patients", "emergency_patients", "total_patient_load"]
        df = self.load_csv("patient_flow_from_2021_to_2025.csv", required_columns=required)
        return df.sort_values(["hospital_id", "date"]).reset_index(drop=True)

    def load_context_data(self) -> pd.DataFrame:
        """Load daily environmental, seasonal, and calendar event flags (FR2)."""
        required = ["date", "weekend_flag"]
        df = self.load_csv("daily_patient_flow_based_on_events.csv", required_columns=required)
        return df.sort_values("date").reset_index(drop=True)

    def load_hospital_directory(self) -> pd.DataFrame:
        """Load hospital addresses, categories, and facility scale directory."""
        required = ["hospital_id", "hospital_name", "city"]
        df = self.load_csv("hospital_and_address_related.csv", required_columns=required, date_column=None)
        return df.reset_index(drop=True)

    def load_service_types(self) -> pd.DataFrame:
        """Load hospital service listings and emergency capability metadata."""
        required = ["hospital_id", "hospital_name"]
        df = self.load_csv("hospital_and_service_type.csv", required_columns=required, date_column=None)
        return df.reset_index(drop=True)
