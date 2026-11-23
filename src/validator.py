"""
Data Validator Module (DataValidator)
Aligned with SRS Chapter 5.4 Class Diagram (FR1, FR2, FR12)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Custom exception raised when data validation fails critical checks (FR12)."""
    pass


class DataValidator:
    """Validates schema integrity, missingness, duplicates, value ranges, and date continuity."""

    @staticmethod
    def check_missing(df: pd.DataFrame) -> Dict[str, int]:
        """Returns missing value counts per column."""
        return {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: List[str]) -> int:
        """Returns count of duplicate rows on specified key columns."""
        for col in subset:
            if col not in df.columns:
                raise DataValidationError(f"Key column '{col}' for duplicate check not found in dataset.")
        return int(df.duplicated(subset=subset).sum())

    @staticmethod
    def check_ranges(
        df: pd.DataFrame,
        non_negative_cols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validates numeric ranges: confirms non-negative patient counts and valid bounds.
        Returns report of any columns violating constraints.
        """
        violations = {}
        target_cols = non_negative_cols or [
            c for c in df.select_dtypes(include=[np.number]).columns
            if "flag" not in c and "sin" not in c and "cos" not in c
        ]

        for col in target_cols:
            if col in df.columns:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    violations[col] = {
                        "min": min_val,
                        "max": max_val,
                        "negative_count": neg_count,
                        "issue": f"Contains {neg_count} negative values.",
                    }
        return violations

    @staticmethod
    def check_date_gaps(
        df: pd.DataFrame,
        date_col: str = "date",
        group_col: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Checks for missing calendar days between min and max date.
        If group_col is specified, checks continuity per entity (e.g. hospital_id).
        """
        if date_col not in df.columns:
            raise DataValidationError(f"Date column '{date_col}' not found in DataFrame.")

        gaps_report: Dict[str, List[str]] = {}

        if group_col and group_col in df.columns:
            for entity_id, grp in df.groupby(group_col):
                dates = pd.to_datetime(grp[date_col]).sort_values()
                if len(dates) == 0:
                    continue
                expected_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
                missing_dates = expected_range.difference(dates)
                if len(missing_dates) > 0:
                    gaps_report[str(entity_id)] = [d.strftime("%Y-%m-%d") for d in missing_dates[:10]]
        else:
            dates = pd.to_datetime(df[date_col]).sort_values().drop_duplicates()
            if len(dates) > 0:
                expected_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
                missing_dates = expected_range.difference(dates)
                if len(missing_dates) > 0:
                    gaps_report["overall"] = [d.strftime("%Y-%m-%d") for d in missing_dates[:10]]

        return gaps_report

    @staticmethod
    def check_target_consistency(
        df: pd.DataFrame,
        opd_col: str = "opd_patients",
        emg_col: str = "emergency_patients",
        total_col: str = "total_patient_load",
    ) -> Dict[str, Any]:
        """Verifies total_patient_load strictly equals opd_patients + emergency_patients."""
        if not {opd_col, emg_col, total_col}.issubset(df.columns):
            return {"verified": False, "issue": "Target columns missing from DataFrame."}

        mismatches = int((df[opd_col] + df[emg_col] != df[total_col]).sum())
        return {
            "verified": mismatches == 0,
            "mismatch_count": mismatches,
        }

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        key_cols: List[str],
        required_cols: Optional[List[str]] = None,
        date_col: Optional[str] = "date",
        group_col: Optional[str] = None,
        check_targets: bool = False,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive data validation suite and returns a structured validation report.
        If raise_on_error is True, raises DataValidationError on first critical failure.
        """
        report: Dict[str, Any] = {
            "is_valid": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "errors": [],
            "warnings": [],
        }

        # 1. Required columns
        if required_cols:
            missing_required = [c for c in required_cols if c not in df.columns]
            if missing_required:
                msg = f"Missing required columns: {missing_required}"
                report["errors"].append(msg)
                report["is_valid"] = False

        # 2. Missing values
        missing_dict = cls.check_missing(df)
        if missing_dict:
            total_missing = sum(missing_dict.values())
            msg = f"Found {total_missing} missing values across columns: {missing_dict}"
            report["warnings"].append(msg)

        # 3. Duplicate records
        dup_count = cls.check_duplicates(df, subset=key_cols)
        if dup_count > 0:
            msg = f"Found {dup_count} duplicate rows for keys {key_cols}"
            report["errors"].append(msg)
            report["is_valid"] = False

        # 4. Value ranges (negative counts)
        range_violations = cls.check_ranges(df)
        if range_violations:
            msg = f"Range violations detected: {range_violations}"
            report["errors"].append(msg)
            report["is_valid"] = False

        # 5. Date continuity
        if date_col and date_col in df.columns:
            gaps = cls.check_date_gaps(df, date_col=date_col, group_col=group_col)
            if gaps:
                msg = f"Date continuity gaps detected: {gaps}"
                report["warnings"].append(msg)

        # 6. Target sum consistency
        if check_targets:
            target_check = cls.check_target_consistency(df)
            if not target_check["verified"]:
                msg = f"Target inconsistency: {target_check.get('mismatch_count', 0)} rows where total != opd + emg"
                report["errors"].append(msg)
                report["is_valid"] = False

        if raise_on_error and not report["is_valid"]:
            raise DataValidationError(f"Data validation failed: {'; '.join(report['errors'])}")

        return report
