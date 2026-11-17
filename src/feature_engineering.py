"""
Feature Engineering Pipeline Module
Aligned with SRS Chapter 5.4 Class Diagram (Preprocessor)
Extracts calendar, cyclical, autoregressive lag, rolling statistics,
and contextual event signals with strict data leakage prevention.
"""

from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Extract temporal attributes from date."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df["day_of_week"] = dates.dt.day_of_week
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year
    df["day_of_year"] = dates.dt.day_of_year
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add smooth sine and cosine representations of periodic cycles."""
    df = df.copy()
    # Weekly cycle (period 7)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Annual cycle by month (period 12)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    # Annual cycle by day of year (period 365.25)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_cols: List[str] = ["opd_patients", "emergency_patients"],
    lags: List[int] = [1, 2, 3, 7, 14, 21, 28],
    group_col: str = None,
) -> pd.DataFrame:
    """
    Generate past historical lag features.
    Lags are strictly >= 1 to prevent data leakage.
    """
    df = df.copy()
    for col in target_cols:
        for lag in lags:
            feat_name = f"{col}_lag_{lag}"
            if group_col:
                df[feat_name] = df.groupby(group_col)[col].shift(lag)
            else:
                df[feat_name] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_cols: List[str] = ["opd_patients", "emergency_patients"],
    windows: List[int] = [7, 14, 30],
    group_col: str = None,
) -> pd.DataFrame:
    """
    Generate moving window summary statistics (mean, std).
    Crucial: shift(1) is applied BEFORE rolling to prevent target leakage on day t.
    """
    df = df.copy()
    for col in target_cols:
        for w in windows:
            mean_name = f"{col}_roll_mean_{w}"
            std_name = f"{col}_roll_std_{w}"
            if group_col:
                df[mean_name] = df.groupby(group_col)[col].transform(
                    lambda s: s.shift(1).rolling(w).mean()
                )
                df[std_name] = df.groupby(group_col)[col].transform(
                    lambda s: s.shift(1).rolling(w).std()
                )
            else:
                df[mean_name] = df[col].shift(1).rolling(w).mean()
                df[std_name] = df[col].shift(1).rolling(w).std()
    return df


def add_interaction_and_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct rate-of-change and seasonal delta features."""
    df = df.copy()
    if "opd_patients_lag_1" in df.columns and "opd_patients_lag_7" in df.columns:
        # Momentum: change from same weekday last week
        df["opd_lag_diff_1_7"] = df["opd_patients_lag_1"] - df["opd_patients_lag_7"]
        # Ratio of yesterday to same day last week
        df["opd_ratio_1_7"] = df["opd_patients_lag_1"] / (df["opd_patients_lag_7"] + 1e-5)

    if "opd_patients_lag_7" in df.columns and "opd_patients_lag_14" in df.columns:
        # Multi-week trend
        df["opd_lag_diff_7_14"] = df["opd_patients_lag_7"] - df["opd_patients_lag_14"]

    if "opd_patients_lag_1" in df.columns and "opd_patients_roll_mean_7" in df.columns:
        # Local baseline divergence
        df["opd_diff_from_roll_7"] = df["opd_patients_lag_1"] - df["opd_patients_roll_mean_7"]

    if "emergency_patients_lag_1" in df.columns and "emergency_patients_lag_7" in df.columns:
        df["emergency_lag_diff_1_7"] = df["emergency_patients_lag_1"] - df["emergency_patients_lag_7"]

    if "emergency_patients_lag_1" in df.columns and "emergency_patients_roll_mean_7" in df.columns:
        df["emergency_diff_from_roll_7"] = df["emergency_patients_lag_1"] - df["emergency_patients_roll_mean_7"]

    # Post-festival rebound flag (shift festival flag by 1 day)
    if "festival_event_proxy_flag" in df.columns:
        df["post_festival_flag"] = df["festival_event_proxy_flag"].shift(1).fillna(0).astype(int)

    # Rainy day indicator (precipitation > 1.0 mm)
    if "mean_precipitation_proxy_mm_simulated" in df.columns:
        df["is_rainy_day"] = (df["mean_precipitation_proxy_mm_simulated"] > 1.0).astype(int)
    elif "precipitation_proxy_mm_simulated" in df.columns:
        df["is_rainy_day"] = (df["precipitation_proxy_mm_simulated"] > 1.0).astype(int)

    return df


def build_aggregate_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature pipeline for city-wide aggregate dataset."""
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y") if df["date"].dtype == object else df["date"]
    df = df.sort_values("date").reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, target_cols=["opd_patients", "emergency_patients", "admissions"])
    df = add_rolling_features(df, target_cols=["opd_patients", "emergency_patients"])
    df = add_interaction_and_momentum_features(df)

    # Drop the initial 30 days containing NaN values caused by 30-day rolling/lag window
    clean_df = df.dropna().reset_index(drop=True)
    return clean_df


def build_hospital_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature pipeline for per-hospital dataset."""
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y") if df["date"].dtype == object else df["date"]
    df = df.sort_values(["hospital_id", "date"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(
        df,
        target_cols=["opd_patients", "emergency_patients"],
        lags=[1, 2, 7, 14, 21],
        group_col="hospital_id",
    )
    df = add_rolling_features(
        df,
        target_cols=["opd_patients", "emergency_patients"],
        windows=[7, 30],
        group_col="hospital_id",
    )
    df = add_interaction_and_momentum_features(df)

    # Hospital categorical encoding
    df["is_government"] = (df["ownership_type"] == "Government").astype(int)

    clean_df = df.dropna().reset_index(drop=True)
    return clean_df


def get_train_val_test_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    val_year: int = 2024,
    test_year: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological time-series split.
    Guarantees no future test information bleeds into training.
    """
    years = pd.to_datetime(df[date_col]).dt.year
    train = df[years < val_year].copy()
    val = df[years == val_year].copy()
    test = df[years == test_year].copy()
    return train, val, test

