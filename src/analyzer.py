"""
Historical Analyzer Module (HistoricalAnalyzer)
Aligned with SRS Chapter 5.4 Class Diagram (FR4)
Computes historical trends, weekly patterns, seasonal summaries,
weather/event impact analysis, and generates offline reports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/offline rendering
import matplotlib.pyplot as plt

from src.config import REPORTS_DIR


class HistoricalAnalyzer:
    """
    Analyzes historical patient inflow patterns across aggregate and facility levels (FR4).
    Produces descriptive tables and offline plots for reporting and clinical planning.
    """

    def __init__(self, target_cols: Optional[List[str]] = None):
        self.target_cols = target_cols or ["opd_patients", "emergency_patients", "total_patient_load"]

    def compute_daily_trends(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Computes overall longitudinal trends and summary statistics.
        Calculates baseline statistics, yearly averages, and 5-year growth rates.
        """
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)

        data["year"] = data[date_col].dt.year

        records = []
        for target in self.target_cols:
            if target not in data.columns:
                continue

            series = data[target]
            yearly = data.groupby("year")[target].mean().to_dict()
            first_year = min(yearly.keys())
            last_year = max(yearly.keys())
            cagr = ((yearly[last_year] / max(1.0, yearly[first_year])) ** (1.0 / max(1, last_year - first_year)) - 1.0) * 100.0

            records.append({
                "metric": target,
                "overall_mean": round(float(series.mean()), 2),
                "overall_std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "q25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.median()), 2),
                "q75": round(float(series.quantile(0.75)), 2),
                "max": round(float(series.max()), 2),
                f"mean_{first_year}": round(float(yearly[first_year]), 2),
                f"mean_{last_year}": round(float(yearly[last_year]), 2),
                "5yr_growth_pct": round(float(((yearly[last_year] - yearly[first_year]) / yearly[first_year]) * 100.0), 2),
                "cagr_pct": round(float(cagr), 2),
            })

        return pd.DataFrame(records)

    def compute_weekly_pattern(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Computes day-of-week inflow distribution and percentage deviation from baseline.
        """
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)

        data["day_of_week"] = data[date_col].dt.day_of_week
        dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        data["day_name"] = data["day_of_week"].map(dow_names)

        results = []
        for dow_idx in range(7):
            dow_name = dow_names[dow_idx]
            row_data = {"day_of_week": dow_idx, "day_name": dow_name}

            dow_subset = data[data["day_of_week"] == dow_idx]
            for target in self.target_cols:
                if target in data.columns:
                    overall_avg = data[target].mean()
                    dow_avg = dow_subset[target].mean()
                    pct_diff = ((dow_avg - overall_avg) / overall_avg) * 100.0

                    row_data[f"{target}_mean"] = round(float(dow_avg), 1)
                    row_data[f"{target}_std"] = round(float(dow_subset[target].std()), 1)
                    row_data[f"{target}_diff_pct"] = round(float(pct_diff), 2)

            results.append(row_data)

        return pd.DataFrame(results)

    def compute_seasonal_summary(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Aggregates volume by season (Monsoon, Summer, Winter) and calendar month.
        """
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)

        data["month"] = data[date_col].dt.month
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        data["month_name"] = data["month"].map(month_names)

        # Classify season if flag columns present, else by month
        if "monsoon_season_flag" in data.columns and "summer_season_flag" in data.columns:
            def assign_season(row):
                if row.get("monsoon_season_flag") == 1:
                    return "Monsoon (Jul-Sep)"
                if row.get("summer_season_flag") == 1:
                    return "Summer (Mar-Jun)"
                if row.get("winter_season_flag") == 1:
                    return "Winter (Nov-Feb)"
                return "Transition/Autumn"
            data["season"] = data.apply(assign_season, axis=1)
        else:
            def assign_month_season(m):
                if m in [7, 8, 9]:
                    return "Monsoon (Jul-Sep)"
                if m in [3, 4, 5, 6]:
                    return "Summer (Mar-Jun)"
                return "Winter (Nov-Feb)"
            data["season"] = data["month"].apply(assign_month_season)

        summary = (
            data.groupby(["season"])[self.target_cols]
            .agg(["mean", "std", "count"])
            .round(2)
        )
        return summary

    def compute_weather_and_event_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates impact of heatwaves, rainfall, weekend closures, and festival events.
        """
        impact = {}

        # 1. Weekend vs Weekday
        if "weekend_flag" in df.columns:
            wknd = df[df["weekend_flag"] == 1]
            wkday = df[df["weekend_flag"] == 0]
            impact["weekend_impact"] = {
                "opd_weekday_avg": round(float(wkday["opd_patients"].mean()), 1),
                "opd_weekend_avg": round(float(wknd["opd_patients"].mean()), 1),
                "opd_change_pct": round(float(((wknd["opd_patients"].mean() - wkday["opd_patients"].mean()) / wkday["opd_patients"].mean()) * 100.0), 2),
                "emergency_weekday_avg": round(float(wkday["emergency_patients"].mean()), 1),
                "emergency_weekend_avg": round(float(wknd["emergency_patients"].mean()), 1),
                "emergency_change_pct": round(float(((wknd["emergency_patients"].mean() - wkday["emergency_patients"].mean()) / wkday["emergency_patients"].mean()) * 100.0), 2),
            }

        # 2. Extreme Heat
        if "heat_flag" in df.columns:
            heat = df[df["heat_flag"] == 1]
            non_heat = df[df["heat_flag"] == 0]
            if len(heat) > 0 and len(non_heat) > 0:
                impact["heat_impact"] = {
                    "heat_days_count": len(heat),
                    "emergency_normal_avg": round(float(non_heat["emergency_patients"].mean()), 1),
                    "emergency_heat_avg": round(float(heat["emergency_patients"].mean()), 1),
                    "emergency_surge_pct": round(float(((heat["emergency_patients"].mean() - non_heat["emergency_patients"].mean()) / non_heat["emergency_patients"].mean()) * 100.0), 2),
                }

        # 3. Monsoon Season
        if "monsoon_season_flag" in df.columns:
            monsoon = df[df["monsoon_season_flag"] == 1]
            non_monsoon = df[df["monsoon_season_flag"] == 0]
            impact["monsoon_impact"] = {
                "emergency_monsoon_avg": round(float(monsoon["emergency_patients"].mean()), 1),
                "emergency_non_monsoon_avg": round(float(non_monsoon["emergency_patients"].mean()), 1),
                "emergency_surge_pct": round(float(((monsoon["emergency_patients"].mean() - non_monsoon["emergency_patients"].mean()) / non_monsoon["emergency_patients"].mean()) * 100.0), 2),
            }

        # 4. Festival / Public Gatherings
        if "festival_event_proxy_flag" in df.columns:
            fest = df[df["festival_event_proxy_flag"] == 1]
            non_fest = df[df["festival_event_proxy_flag"] == 0]
            impact["festival_impact"] = {
                "festival_days_count": len(fest),
                "opd_festival_avg": round(float(fest["opd_patients"].mean()), 1),
                "opd_non_festival_avg": round(float(non_fest["opd_patients"].mean()), 1),
                "opd_change_pct": round(float(((fest["opd_patients"].mean() - non_fest["opd_patients"].mean()) / non_fest["opd_patients"].mean()) * 100.0), 2),
            }

        return impact

    def compute_hospital_distribution(self, hosp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes patient load distribution across individual hospital facilities.
        """
        hosp_summary = (
            hosp_df.groupby(["hospital_id", "hospital_name", "ownership_type"])[self.target_cols]
            .mean()
            .reset_index()
        )
        total_load_sum = hosp_summary["total_patient_load"].sum()
        hosp_summary["share_pct"] = round((hosp_summary["total_patient_load"] / total_load_sum) * 100.0, 2)
        return hosp_summary.sort_values(by="total_patient_load", ascending=False).reset_index(drop=True)

    def export_reports(
        self,
        agg_df: pd.DataFrame,
        hosp_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        date_col: str = "date",
    ) -> Dict[str, Path]:
        """
        Generates and exports summary CSV tables and visualization plots to reports/ directory.
        """
        out_dir = Path(output_dir) if output_dir else REPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        exported_files: Dict[str, Path] = {}

        # 1. Daily trends summary CSV
        trends_df = self.compute_daily_trends(agg_df, date_col=date_col)
        trends_path = out_dir / "historical_trends_summary.csv"
        trends_df.to_csv(trends_path, index=False)
        exported_files["trends"] = trends_path

        # 2. Weekly patterns summary CSV
        weekly_df = self.compute_weekly_pattern(agg_df, date_col=date_col)
        weekly_path = out_dir / "weekly_patterns.csv"
        weekly_df.to_csv(weekly_path, index=False)
        exported_files["weekly"] = weekly_path

        # 3. Hospital facility distribution CSV
        if hosp_df is not None:
            dist_df = self.compute_hospital_distribution(hosp_df)
            dist_path = out_dir / "hospital_distribution.csv"
            dist_df.to_csv(dist_path, index=False)
            exported_files["hospital_distribution"] = dist_path

        # 4. Offline evaluation plot (matplotlib)
        plot_path = out_dir / "historical_patterns.png"
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        data = agg_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col], dayfirst=True)

        # Plot 1: 5-Year Time Series
        axes[0, 0].plot(data[date_col], data["opd_patients"], label="OPD Patients", color="#1f77b4", alpha=0.8, lw=0.8)
        axes[0, 0].plot(data[date_col], data["emergency_patients"], label="Emergency Patients", color="#e74c3c", alpha=0.9, lw=0.8)
        axes[0, 0].set_title("5-Year Inflow Trajectory (2021-2025)", fontweight="bold")
        axes[0, 0].set_ylabel("Daily Patients")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Day-of-Week Pattern
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[0, 1].bar(dow_names, weekly_df["opd_patients_mean"], color="#3498db", label="OPD Patients")
        axes[0, 1].set_title("Weekly OPD Pattern (Monday Surge vs Sunday Drop)", fontweight="bold")
        axes[0, 1].set_ylabel("Mean Patients")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Emergency Inflow by Day of Week
        axes[1, 0].bar(dow_names, weekly_df["emergency_patients_mean"], color="#e67e22", label="Emergency")
        axes[1, 0].set_title("Weekly Emergency Pattern (Consistent Daily Demand)", fontweight="bold")
        axes[1, 0].set_ylabel("Mean Patients")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Yearly Growth Trends
        yearly_means = data.groupby(data[date_col].dt.year)[["opd_patients", "emergency_patients"]].mean()
        yearly_means.plot(kind="bar", ax=axes[1, 1], colormap="viridis")
        axes[1, 1].set_title("Annual Mean Daily Load Progression", fontweight="bold")
        axes[1, 1].set_ylabel("Mean Daily Volume")
        axes[1, 1].set_xticklabels(yearly_means.index, rotation=0)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        exported_files["plot"] = plot_path

        return exported_files
