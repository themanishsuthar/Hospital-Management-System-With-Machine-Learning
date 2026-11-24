"""
Unit tests for Day 4: HistoricalAnalyzer and reporting capabilities (FR4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.analyzer import HistoricalAnalyzer


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def analyzer():
    return HistoricalAnalyzer()


def test_analyzer_daily_trends(loader, analyzer):
    """Verify daily trends computation."""
    agg_df = loader.load_aggregate_data()
    trends = analyzer.compute_daily_trends(agg_df)

    assert isinstance(trends, pd.DataFrame)
    assert len(trends) >= 2
    assert "overall_mean" in trends.columns
    assert "5yr_growth_pct" in trends.columns
    # OPD mean should be ~6000
    opd_row = trends[trends["metric"] == "opd_patients"].iloc[0]
    assert 5500 < opd_row["overall_mean"] < 6500


def test_analyzer_weekly_pattern(loader, analyzer):
    """Verify weekly inflow pattern computation across 7 days."""
    agg_df = loader.load_aggregate_data()
    weekly = analyzer.compute_weekly_pattern(agg_df)

    assert len(weekly) == 7
    assert list(weekly["day_of_week"]) == [0, 1, 2, 3, 4, 5, 6]
    assert "opd_patients_mean" in weekly.columns
    assert "emergency_patients_mean" in weekly.columns

    # Sunday (day 6) OPD should be lower than Monday (day 0)
    monday_opd = weekly.loc[weekly["day_of_week"] == 0, "opd_patients_mean"].values[0]
    sunday_opd = weekly.loc[weekly["day_of_week"] == 6, "opd_patients_mean"].values[0]
    assert monday_opd > sunday_opd


def test_analyzer_seasonal_summary(loader, analyzer):
    """Verify seasonal summary computation."""
    agg_df = loader.load_aggregate_data()
    seasonal = analyzer.compute_seasonal_summary(agg_df)
    assert isinstance(seasonal, pd.DataFrame)
    assert len(seasonal) >= 3


def test_analyzer_weather_and_event_impact(loader, analyzer):
    """Verify calculation of weather and event surge multipliers."""
    agg_df = loader.load_aggregate_data()
    impact = analyzer.compute_weather_and_event_impact(agg_df)

    assert "weekend_impact" in impact
    assert "heat_impact" in impact
    assert "monsoon_impact" in impact
    assert impact["monsoon_impact"]["emergency_surge_pct"] > 0


def test_analyzer_hospital_distribution(loader, analyzer):
    """Verify hospital-level patient volume breakdown."""
    hosp_df = loader.load_hospital_data()
    dist = analyzer.compute_hospital_distribution(hosp_df)

    assert len(dist) == 10
    assert "share_pct" in dist.columns
    assert np.isclose(dist["share_pct"].sum(), 100.0, atol=0.2)


def test_analyzer_export_reports(loader, analyzer, tmp_path):
    """Verify offline export of summary CSVs and evaluation plot."""
    agg_df = loader.load_aggregate_data()
    hosp_df = loader.load_hospital_data()

    exported = analyzer.export_reports(agg_df, hosp_df=hosp_df, output_dir=tmp_path)

    assert "trends" in exported and exported["trends"].exists()
    assert "weekly" in exported and exported["weekly"].exists()
    assert "hospital_distribution" in exported and exported["hospital_distribution"].exists()
    assert "plot" in exported and exported["plot"].exists()
    assert exported["plot"].stat().st_size > 1000
