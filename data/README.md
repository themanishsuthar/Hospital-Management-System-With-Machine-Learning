# Data Documentation: Hospital Patient Flow (Jaipur 2021–2025)

## 1. Overview and Source of Data
This dataset contains a 5-year longitudinal study (1,826 consecutive days from January 1, 2021 to December 31, 2025) of outpatient and emergency patient visits across 10 hospitals in Jaipur, Rajasthan (3 government tertiary hospitals and 7 private healthcare institutions).

In compliance with academic disclosures:
- Weather indicators (`mean_temp_proxy_c_simulated`, `precipitation_proxy_mm_simulated`, `heat_flag`) are simulated climatic proxies representative of Jaipur's semi-arid weather patterns.
- Internal hospital operational metrics (`doctors_on_duty_simulated`, `total_beds_capacity_simulated`, `beds_available_simulated`, `occupancy_pct_simulated`) are simulated proxies modeled on Indian Public Health Standards (IPHS).
- All patient volume counts are anonymized aggregates with zero personally identifiable patient data.

---

## 2. Inventory of Data Files

### 2.1 `ML_model_training_based.csv` (Primary Aggregate Flow)
- **Granularity:** Daily city-level aggregate (sum of all 10 hospitals).
- **Dimensions:** 1,826 rows × 14 columns.
- **Date Range:** 2021-01-01 to 2025-12-31 (continuous, 0 missing days).
- **Missing Values:** 0 nulls across all cells.
- **Columns:**
  - `date`: Daily timestamp string (format: DD-MM-YYYY).
  - `opd_patients` (Target 1): Scheduled outpatient department patient volume (Mean ~6,028; Min 4,503; Max 7,633).
  - `emergency_patients` (Target 2): Unscheduled acute emergency patient volume (Mean ~461; Min 380; Max 570).
  - `total_patient_load`: Exact sum of `opd_patients` + `emergency_patients`.
  - `admissions`: Inpatient admissions from OPD and emergency cases.
  - `mean_temp_proxy_c_simulated`: Daily mean ambient temperature in Celsius.
  - `mean_precipitation_proxy_mm_simulated`: Daily precipitation in millimeters.
  - `weekend_flag`: Binary indicator (1 = Saturday/Sunday, 0 = Weekday).
  - `monsoon_season_flag`: Binary indicator (1 = July to September, 0 = otherwise).
  - `winter_season_flag`: Binary indicator (1 = November to February, 0 = otherwise).
  - `summer_season_flag`: Binary indicator (1 = March to June, 0 = otherwise).
  - `heat_flag`: Binary indicator for extreme heat conditions (>40°C).
  - `festival_event_proxy_flag`: Binary indicator for major public gathering or regional festival.
  - `city`: Constant city name ("Jaipur").

### 2.2 `patient_flow_from_2021_to_2025.csv` (Hospital-Level Daily Flow)
- **Granularity:** Daily record per hospital (10 hospitals × 1,826 days).
- **Dimensions:** 18,260 rows × 21 columns.
- **Date Range:** 2021-01-01 to 2025-12-31.
- **Missing Values:** 0 nulls across all cells.
- **Key Entities:**
  - `hospital_id`: Unique identifier (e.g., `JPR001` SMS Hospital, `JPR002` Zanana Hospital, `JPR003` Bani Park Satellite Hospital, `JPR004`-`JPR010` Private hospitals).
  - `ownership_type`: "Government" or "Private".
  - Bed capacity, available beds, occupancy %, and doctor count columns (simulated proxies).

### 2.3 `daily_patient_flow_based_on_events.csv`
- **Granularity:** Calendar and weather factor catalog for Jaipur (1,826 rows × 10 columns).

### 2.4 `hospital_and_address_related.csv` & `hospital_and_service_type.csv`
- Hospital directory, geographical addresses, state/district, and departmental capabilities.

---

## 3. Data Integrity & Quality Audit Findings
1. **Zero Missingness in Patient Flow:** Both aggregate and facility-level CSVs contain zero missing or NaN values.
2. **Date Continuity:** Complete 1,826 calendar days present without temporal gaps or leap-year discrepancies (2024 leap day accounted for).
3. **Target Invariance:** In 100% of rows, `total_patient_load == opd_patients + emergency_patients`.
4. **Date Format Consideration:** The raw date column is encoded as `DD-MM-YYYY`. Parsers must specify `dayfirst=True` or `format='%d-%m-%Y'` to avoid ambiguous parsing.
5. **No Negative Values:** All counts and simulated metrics respect valid numerical bounds ($x \ge 0$).
