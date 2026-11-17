# AI-Based Patient Load Prediction for Government Hospitals (Jaipur Case Study)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit%20Dashboard-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Production%20Ready-green.svg)]()

A comprehensive, end-to-end Machine Learning and Decision Support System for forecasting outpatient department (OPD) and emergency patient inflows, automating doctor staffing rosters, optimizing hospital bed occupancy, and providing real-time surge alerts for government and private hospitals across Jaipur, Rajasthan (2021–2025).

---

## 📌 Project Overview & Key Highlights

* **Multi-Hospital Coverage:** 10 healthcare facilities (3 Government tertiary centers including SMS Hospital, and 7 Private hospitals).
* **Dual Inflow Dynamics:**
  * **OPD Patient Flow:** Scheduled, periodic outpatient flow ($R^2 \approx 0.88$, MAPE $\approx 3.0\%$).
  * **Emergency Patient Flow:** Stochastic, unscheduled acute arrivals ($R^2 \approx 0.60$, MAPE $\approx 3.7\%$).
* **Clinical Decision Support:** Incorporates Indian Public Health Standards (IPHS) to compute:
  * Required daily doctors and nursing staff.
  * Projected inpatient bed occupancy and bed availability.
  * Essential medicines and fluids demand buffer index.
  * 4-Tier Surge Alert Engine (**GREEN**, **YELLOW**, **ORANGE**, **RED**) with operational SOP directives.
* **Leakage-Free Validation:** Strict chronological time splits:
  * **Train Set:** 2021-01-31 to 2023-12-31 (1,065 days)
  * **Validation Set:** 2024-01-01 to 2024-12-31 (366 days, Leap Year)
  * **Hold-Out Test Set:** 2025-01-01 to 2025-12-31 (365 days)

---

## 📂 Project Structure

```text
Final Year Project/
│
├── data/
│   ├── ML_model_training_based.csv          # Aggregate daily flow (all 10 hospitals summed)
│   ├── patient_flow_from_2021_to_2025.csv   # Hospital-specific daily records (18,260 rows)
│   ├── daily_patient_flow_based_on_events.csv # Environmental and calendar flags
│   ├── hospital_and_address_related.csv     # Hospital geographical coordinates & bed capacities
│   ├── hospital_and_service_type.csv        # Departmental service capabilities
│   └── processed/
│       ├── features_aggregate_daily.csv     # Engineered aggregate feature matrix (1,796 × 70)
│       ├── features_hospital_daily.csv      # Engineered hospital-level matrix (17,960 × 63)
│       └── feature_metadata.json            # Target definitions and feature column schemas
│
├── notebook/
│   ├── EDA.ipynb                            # Complete Exploratory Data Analysis (16 sections)
│   ├── 02_feature_engineering.ipynb         # Temporal, cyclical, lag & rolling feature pipeline
│   └── 03_model_experiments.ipynb           # Model benchmarking, diagnostics & export
│
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_loader.py                       # Data loading and schema integrity validation
│   ├── feature_engineering.py               # Reusable feature engineering transformers
│   ├── models.py                            # Candidate model definitions and benchmark trainer
│   ├── resource_planner.py                  # Clinical staffing and bed occupancy calculator
│   └── alert_engine.py                      # Multi-tier surge alert and SOP recommendation engine
│
├── models/
│   ├── best_opd_model.joblib                # Serialized best model pipeline for OPD load
│   ├── best_emergency_model.joblib          # Serialized best model pipeline for Emergency load
│   └── benchmark_summary.json               # Full evaluation metrics across all algorithms
│
├── app.py                                   # Streamlit interactive web dashboard
├── requirements.txt                         # Pinned Python package dependencies
└── Readme.md                                # Project documentation
```

---

## 📊 Empirical Model Benchmark Results (Hold-out Test 2025)

### Outpatient Department (OPD) Patient Load
| Model Architecture | Test MAE (pts) | Test RMSE | Test MAPE (%) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2 Scaled)** | **187.49** | **234.50** | **3.00%** | **0.8800** |
| Random Forest Regressor | 223.82 | 284.97 | 3.50% | 0.8227 |
| LightGBM Regressor | 234.72 | 297.04 | 3.67% | 0.8074 |
| XGBoost Regressor | 235.28 | 300.89 | 3.67% | 0.8024 |
| Naive Lag-7 (Persistence Baseline) | 259.56 | 323.62 | 4.14% | 0.7714 |

### Emergency Department Patient Load
| Model Architecture | Test MAE (pts) | Test RMSE | Test MAPE (%) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2 Scaled)** | **17.24** | **21.57** | **3.76%** | **0.5985** |
| Random Forest Regressor | 17.72 | 22.19 | 3.84% | 0.5751 |
| LightGBM Regressor | 18.00 | 22.45 | 3.90% | 0.5649 |
| XGBoost Regressor | 18.04 | 22.85 | 3.91% | 0.5494 |
| Naive Lag-7 (Persistence Baseline) | 25.65 | 32.47 | 5.56% | 0.0900 |

> **Viva Insight:** While OPD inflow is strongly autoregressive and predictable using weekly cycles (Lag-7), Emergency arrivals are unscheduled and stochastic. The machine learning pipeline improves Emergency $R^2$ from 0.09 up to ~0.60 by learning weather proxies (heat waves, monsoon rainfall) and rolling momentum.

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
Activate the project conda environment in terminal / PowerShell:
```powershell
# Open terminal inside the project directory
& ".\.conda\python.exe" -m pip install -r requirements.txt
```

### 2. Running Jupyter Notebooks in VS Code
Open VS Code in this directory and select the pre-configured kernel:
- **Kernel Name:** `conda-patient-load`
- **Interpreter Path:** `Final Year Project\.conda\python.exe`
- Open and run the notebooks in sequence:
  1. `notebook/EDA.ipynb`
  2. `notebook/02_feature_engineering.ipynb`
  3. `notebook/03_model_experiments.ipynb`

### 3. Launching the Interactive Web Application
Launch the Streamlit interactive dashboard:
```powershell
& ".\.conda\python.exe" -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

1. **Executive Command Center:**
   - Multi-hospital dropdown selector (Aggregate Jaipur view or individual hospital facilities).
   - High-visibility KPI metric cards (Predicted OPD, Emergency, Required Doctors, Bed Occupancy %).
   - Real-time Alert Banner with incident management directives.
2. **Longitudinal Forecasting & Trends:**
   - Interactive multi-day lookahead charts (Plotly).
   - Day-of-week seasonality analysis (Sunday dip vs. Monday rush).
3. **Clinical Staffing & Bed Allocator:**
   - 14-day shift rosters with doctor and nurse requirements based on IPHS standards.
   - Downloadable shift schedule reports (CSV).
4. **Surge Scenario Simulator ("What-If" Analysis):**
   - Interactive parameter sliders for heat waves, monsoon downpours, epidemic outbreaks, and doctor absenteeism.
   - Live stress-testing of hospital capacity.
5. **Model Evaluation & Academic Viva Portal:**
   - Full model comparison tables and research observations ready for academic viva defense.

---

## ⚖️ Academic Disclosure & Research Notes
* **Data Sources:** 5-year longitudinal patient records from Jaipur healthcare facilities (2021–2025).
* **Proxy Columns:** In accordance with academic transparency, weather indicators (`mean_temp_proxy_c_simulated`, `mean_precipitation_proxy_mm_simulated`) and internal hospital bed occupancies are simulated proxies based on regional climate norms and clinical averages.
* **Clinical Ratios:** Staffing calculations follow standard Indian Public Health Standards (IPHS) guidelines (~45 OPD patients per doctor shift, ~18 emergency cases per acute doctor shift, 1:4 general ward nursing ratio).
