# AI-Based Patient Load Prediction for Government Hospitals (Jaipur Case Study)

[![Python 3.13](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/pytest-23%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit%20Dashboard-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-In%20Active%20Development%20(Day%204%20Complete)-green.svg)]()

A predictive decision-support system designed for government and private hospitals in Jaipur, Rajasthan. The system forecasts daily Outpatient Department (OPD) and Emergency patient inflows, optimizes doctor and nursing shift rosters, tracks hospital bed occupancy under Indian Public Health Standards (IPHS), and raises automated threshold alerts to mitigate clinical overcrowding.

---

## 📌 Executive Summary & Key Highlights

* **5-Year Longitudinal Dataset (2021–2025):** 1,826 continuous daily records across 10 healthcare facilities (3 major government tertiary centers and 7 private institutions).
* **Dual-Target Inflow Modeling:**
  * **OPD Patient Load:** Highly structured, periodic outpatient attendance ($R^2 \approx 0.88$, MAPE $\approx 3.0\%$).
  * **Emergency Patient Load:** Event-driven, weather-sensitive acute arrivals ($R^2 \approx 0.60$, MAPE $\approx 3.7\%$).
* **Clinical Decision Support & Staffing (IPHS Norms):**
  * Automated duty rosters for OPD doctors (~1:45 ratio) and emergency physicians (~1:18 ratio).
  * Ward bed occupancy projections and admission estimations.
  * Consumable medical supplies and IV fluids demand buffer index.
  * 4-Tier Surge Alert Engine (**GREEN**, **YELLOW**, **ORANGE**, **RED**) with operational SOP directives.
* **Strict Zero Data Leakage:**
  * Chronological time-series partition: Train (2021–2023), Validation (2024), Hold-Out Test (2025).
  * All autoregressive lags strictly use $t-k$ ($k \ge 1$) and rolling statistics are pre-shifted with `shift(1)`.

---

## 📊 Key Historical Insights (Jaipur Patient Flow)

| Metric / Dimension | Observation & Operational Impact |
| :--- | :--- |
| **Monday Inflow Surge** | Outpatient volume peaks on Mondays (**+6.56%** above weekly average) due to weekend backlog accumulation. |
| **Sunday Curtailment** | Outpatient clinics drop by **-16.34%** on Sundays due to routine department closures. |
| **Emergency Invariance** | Emergency arrivals remain practically identical across all 7 days (~461 patients/day), proving acute emergencies occur independent of administrative calendar cycles. |
| **Monsoon Rain Impact** | Emergency admissions surge by **+12.8%** during the monsoon season (July–September) from waterborne and trauma cases. |
| **Heatwave Peaks (>40°C)** | Emergency cases rise by **+8.0%** during extreme summer heatwaves. |
| **Facility Concentration** | 3 Government institutions (SMS Hospital, Zanana Hospital, Bani Park Satellite) absorb **54.2%** of all patient visits in Jaipur. |

---

## 📂 Project Architecture

```text
Final Year Project/
│
├── data/                               # Hospital & Patient Flow Datasets
│   ├── ML_model_training_based.csv     # 1,826 aggregate daily records (all 10 hospitals summed)
│   ├── patient_flow_from_2021_to_2025.csv # 18,260 facility-level records (10 hospitals × 1,826 days)
│   ├── daily_patient_flow_based_on_events.csv # Environmental and calendar proxy flags
│   ├── hospital_and_address_related.csv # Facility metadata, geocodes, sanctioned beds
│   ├── hospital_and_service_type.csv   # Departmental service listings
│   └── README.md                       # Comprehensive data documentation and schemas
│
├── src/                                # Reusable Core Backend Modules
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Centralized paths, thresholds, and operational parameters
│   ├── data_loader.py                  # Multi-path ingestion and human-readable DataLoadError (FR1, FR2, FR12)
│   ├── validator.py                    # DataValidator checking duplicates, ranges, and date gaps
│   ├── preprocessor.py                 # Preprocessor with zero-leakage lags, rolling stats, and imputation (FR3, FR5)
│   ├── analyzer.py                     # HistoricalAnalyzer for longitudinal trends and seasonal summaries (FR4)
│   ├── resource_planner.py             # Staffing rosters & bed occupancy calculator (IPHS standards)
│   └── alert_engine.py                 # 4-tier surge alert engine with SOP directives
│
├── tests/                              # Pytest Automated Test Suite (23 passed)
│   ├── test_day1_setup.py              # Environment and raw data integrity tests
│   ├── test_day2_loader_validator.py   # Loader, error handling, and validation tests
│   ├── test_day3_preprocessor.py       # Preprocessor, feature engineering, and leakage audit tests
│   └── test_day4_analyzer.py           # Historical trends, weekly patterns, and report export tests
│
├── reports/                            # Generated Analysis Outputs & Figures
│   ├── historical_trends_summary.csv   # 5-year longitudinal growth percentiles
│   ├── weekly_patterns.csv             # Day-of-week inflow distribution
│   ├── hospital_distribution.csv       # Facility-level volume share breakdown
│   ├── historical_patterns.png         # 4-panel diagnostic trajectory plot
│   └── day4_findings.md                # Detailed analytical summary report
│
├── notebook/                           # Academic Research & Experimentation Notebooks
│   ├── EDA.ipynb                       # 16-section exploratory data analysis
│   └── 02_feature_engineering.ipynb    # Feature engineering pipeline demonstration
│
├── app.py                              # Streamlit Interactive Web Application
├── requirements.txt                    # Pinned Python package dependencies
└── Readme.md                           # Project documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup
Clone the repository and create the virtual environment:
```powershell
# Clone the repository
git clone https://github.com/themanishsuthar/Hospital-Management-System-With-Machine-Learning.git
cd "Hospital-Management-System-With-Machine-Learning"

# Create and activate .venv
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Automated Unit Tests
Verify data loaders, validators, feature engineering, and analyzers:
```powershell
pytest tests/ -v
```
*(All 23 test cases should pass).*

### 3. Running the Historical Analysis & Generating Reports
Execute the analytical suite to regenerate summary tables and plots in `reports/`:
```powershell
python -c "
from src.data_loader import DataLoader
from src.analyzer import HistoricalAnalyzer
loader = DataLoader()
agg_df = loader.load_aggregate_data()
hosp_df = loader.load_hospital_data()
analyzer = HistoricalAnalyzer()
analyzer.export_reports(agg_df, hosp_df=hosp_df)
print('Reports exported successfully!')
"
```

### 4. Launching the Interactive Web Dashboard
Run the Streamlit decision-support application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access:
- **Executive Command Center:** Real-time KPI cards, hospital bed occupancy gauges, and surge alerts.
- **Forecasting & Trends:** Interactive multi-horizon lookahead curves (7, 14, 30 days).
- **Resource Allocator:** 14-day shift rosters with downloadable CSV reports.
- **Surge Scenario Simulator ("What-If"):** Sliders for heatwaves, monsoon downpours, and staff absenteeism.

---

## ⚖️ Academic Disclosure & Research Methodology
* **Hospital Data:** Based on 5-year longitudinal observations across healthcare centers in Jaipur, Rajasthan.
* **Simulated Proxies:** In accordance with academic transparency, weather indicators (`mean_temp_proxy_c_simulated`, `mean_precipitation_proxy_mm_simulated`) and internal bed occupancy rates are simulated proxies modeled after Jaipur's regional climate averages and hospital norms.
* **Staffing Standards:** Formulas follow standard Indian Public Health Standards (IPHS) guidelines (~45 OPD patients per doctor shift, ~18 emergency cases per acute doctor shift, 1:4 general ward nursing ratio).
