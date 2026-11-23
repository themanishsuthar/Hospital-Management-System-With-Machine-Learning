"""
Preprocessor Module (Preprocessor)
Aligned with SRS Chapter 5.4 Class Diagram (FR3, FR5)
Handles data cleaning, missing contextual data imputation, feature generation,
and chronological train/val/test partitioning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED


class Preprocessor:
    """
    Cleans raw hospital and context datasets, imputes missing external factors,
    builds leak-free temporal features, and persists processed outputs.
    """

    def __init__(
        self,
        target_cols: List[str] = ["opd_patients", "emergency_patients"],
        lags: List[int] = [1, 2, 3, 7, 14, 21, 28],
        rolling_windows: List[int] = [7, 14, 30],
    ):
        self.target_cols = list(target_cols)
        self.lags = list(lags)
        self.rolling_windows = list(rolling_windows)

    def clean(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: Optional[str] = None,
        drop_duplicates: bool = True,
        missing_strategy: str = "interpolate",
    ) -> pd.DataFrame:
        """
        Cleans raw dataframe:
        - Parses date column
        - Handles duplicate records
        - Explicitly handles missing contextual variables (weather, holidays, epidemic flags)
        """
        cleaned = df.copy()

        # Date normalization
        if date_col in cleaned.columns:
            if not pd.api.types.is_datetime64_any_dtype(cleaned[date_col]):
                try:
                    cleaned[date_col] = pd.to_datetime(cleaned[date_col], format="%d-%m-%Y")
                except Exception:
                    cleaned[date_col] = pd.to_datetime(cleaned[date_col], dayfirst=True)

            if group_col and group_col in cleaned.columns:
                cleaned = cleaned.sort_values([group_col, date_col]).reset_index(drop=True)
            else:
                cleaned = cleaned.sort_values(date_col).reset_index(drop=True)

        # Duplicate handling
        if drop_duplicates:
            subset = [group_col, date_col] if group_col and group_col in cleaned.columns else [date_col]
            subset = [c for c in subset if c in cleaned.columns]
            if subset:
                cleaned = cleaned.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

        # Handling missing contextual/weather attributes explicitly (SRS Rule 7)
        weather_cols = [c for c in cleaned.columns if "temp" in c.lower() or "precip" in c.lower()]
        for col in weather_cols:
            if cleaned[col].isna().any():
                if missing_strategy == "interpolate":
                    cleaned[col] = cleaned[col].interpolate(method="linear").bfill().ffill()
                elif missing_strategy == "mean":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                else:
                    cleaned[col] = cleaned[col].ffill().bfill()

        # Contextual event/holiday/epidemic flags: missing means no event occurred (default to 0)
        flag_cols = [c for c in cleaned.columns if "flag" in c.lower() or "event" in c.lower() or "epidemic" in c.lower()]
        for col in flag_cols:
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(0).astype(int)

        # Targets: if missing, interpolate linearly or drop
        for target in self.target_cols:
            if target in cleaned.columns and cleaned[target].isna().any():
                if missing_strategy == "drop":
                    cleaned = cleaned.dropna(subset=[target]).reset_index(drop=True)
                else:
                    cleaned[target] = cleaned[target].interpolate(method="linear").round().bfill().ffill()

        return cleaned

    def add_calendar_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Extracts temporal calendar signals."""
        out = df.copy()
        dates = pd.to_datetime(out[date_col])
        out["day_of_week"] = dates.dt.day_of_week
        out["day_of_month"] = dates.dt.day
        out["month"] = dates.dt.month
        out["year"] = dates.dt.year
        out["day_of_year"] = dates.dt.day_of_year
        out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["is_month_start"] = dates.dt.is_month_start.astype(int)
        out["is_month_end"] = dates.dt.is_month_end.astype(int)
        return out

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trigonometric encoding for smooth seasonal boundary transitions."""
        out = df.copy()
        if "day_of_week" in out.columns:
            out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
            out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
        if "month" in out.columns:
            out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
            out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
        if "day_of_year" in out.columns:
            out["sin_doy"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
            out["cos_doy"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
        return out

    def add_lag_features(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Autoregressive lag signals (strictly k >= 1 to prevent data leakage)."""
        out = df.copy()
        for col in self.target_cols:
            if col not in out.columns:
                continue
            for lag in self.lags:
                feat_name = f"{col}_lag_{lag}"
                if group_col and group_col in out.columns:
                    out[feat_name] = out.groupby(group_col)[col].shift(lag)
                else:
                    out[feat_name] = out[col].shift(lag)
        return out

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Rolling window means and standard deviations (pre-shifted by 1 to prevent leakage)."""
        out = df.copy()
        for col in self.target_cols:
            if col not in out.columns:
                continue
            for w in self.rolling_windows:
                mean_name = f"{col}_roll_mean_{w}"
                std_name = f"{col}_roll_std_{w}"
                if group_col and group_col in out.columns:
                    out[mean_name] = out.groupby(group_col)[col].transform(
                        lambda s: s.shift(1).rolling(w).mean()
                    )
                    out[std_name] = out.groupby(group_col)[col].transform(
                        lambda s: s.shift(1).rolling(w).std()
                    )
                else:
                    out[mean_name] = out[col].shift(1).rolling(w).mean()
                    out[std_name] = out[col].shift(1).rolling(w).std()
        return out

    def add_momentum_and_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives differential momentum and weather/event interaction features."""
        out = df.copy()
        # OPD momentum
        if "opd_patients_lag_1" in out.columns and "opd_patients_lag_7" in out.columns:
            out["opd_lag_diff_1_7"] = out["opd_patients_lag_1"] - out["opd_patients_lag_7"]
            out["opd_ratio_1_7"] = out["opd_patients_lag_1"] / (out["opd_patients_lag_7"] + 1e-5)
        if "opd_patients_lag_7" in out.columns and "opd_patients_lag_14" in out.columns:
            out["opd_lag_diff_7_14"] = out["opd_patients_lag_7"] - out["opd_patients_lag_14"]
        if "opd_patients_lag_1" in out.columns and "opd_patients_roll_mean_7" in out.columns:
            out["opd_diff_from_roll_7"] = out["opd_patients_lag_1"] - out["opd_patients_roll_mean_7"]

        # Emergency momentum
        if "emergency_patients_lag_1" in out.columns and "emergency_patients_lag_7" in out.columns:
            out["emergency_lag_diff_1_7"] = out["emergency_patients_lag_1"] - out["emergency_patients_lag_7"]
        if "emergency_patients_lag_1" in out.columns and "emergency_patients_roll_mean_7" in out.columns:
            out["emergency_diff_from_roll_7"] = out["emergency_patients_lag_1"] - out["emergency_patients_roll_mean_7"]

        # Post-festival surge rebound
        if "festival_event_proxy_flag" in out.columns:
            out["post_festival_flag"] = out["festival_event_proxy_flag"].shift(1).fillna(0).astype(int)

        # Rain threshold indicator
        precip_col = next((c for c in out.columns if "precipitation" in c.lower() or "precip" in c.lower()), None)
        if precip_col:
            out["is_rainy_day"] = (out[precip_col] > 1.0).astype(int)

        return out

    def generate_features(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: Optional[str] = None,
        drop_warmup: bool = True,
    ) -> pd.DataFrame:
        """
        Runs full feature engineering pipeline:
        calendar + cyclical + lags + rolling windows + momentum.
        """
        feats = df.copy()
        feats = self.add_calendar_features(feats, date_col=date_col)
        feats = self.add_cyclical_features(feats)
        feats = self.add_lag_features(feats, group_col=group_col)
        feats = self.add_rolling_features(feats, group_col=group_col)
        feats = self.add_momentum_and_interaction_features(feats)

        if drop_warmup:
            feats = feats.dropna().reset_index(drop=True)

        return feats

    def process_and_save(
        self,
        raw_df: pd.DataFrame,
        output_filename: str = "features_aggregate_daily.csv",
        group_col: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Cleans, engineers features, and saves the matrix to data/processed/."""
        cleaned = self.clean(raw_df, group_col=group_col)
        features = self.generate_features(cleaned, group_col=group_col)

        save_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / output_filename
        features.to_csv(out_path, index=False)
        return features

    @staticmethod
    def get_train_val_test_splits(
        df: pd.DataFrame,
        date_col: str = "date",
        val_year: int = 2024,
        test_year: int = 2025,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological time-series partitioning.
        Strictly prevents look-ahead bias across time horizons.
        """
        years = pd.to_datetime(df[date_col]).dt.year
        train = df[years < val_year].copy().reset_index(drop=True)
        val = df[years == val_year].copy().reset_index(drop=True)
        test = df[years == test_year].copy().reset_index(drop=True)
        return train, val, test
