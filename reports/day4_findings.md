# Day 4 Findings: Historical Patient Flow Analysis (Jaipur 2021–2025)

**Module:** `src/analyzer.py` (`HistoricalAnalyzer`)  
**Scope:** Offline empirical evaluation of longitudinal trends, day-of-week seasonality, weather triggers, and hospital capacity shares (FR4).

---

## 1. Long-Term Trajectory & Multi-Year Growth
- **Outpatient (OPD) Progression:**
  - City aggregate OPD volume increased from **5,548 patients/day in 2021** to **6,527 patients/day in 2025** (~17.6% cumulative 5-year growth; CAGR ~4.1%).
  - Baseline distribution: Mean = 6,028 patients/day (Std = 662; IQR: 5,618 to 6,488; Range: 4,503 to 7,633).
- **Emergency Department Stability:**
  - Emergency inflows remained remarkably steady across the 5 years: Mean = **461 patients/day** (Std = 33; Range: 380 to 570).
  - Minimal annual drift (<2% variance across years), indicating emergency demand reflects acute morbidity rather than elective adoption.

---

## 2. Day-of-Week Seasonality Signatures
| Day of Week | Mean Daily OPD (pts) | OPD vs Avg (%) | Mean Daily Emergency (pts) | Emergency vs Avg (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Monday** | **6,423.2** | **+6.56%** | 461.5 | +0.11% |
| **Tuesday** | 6,285.8 | +4.28% | 460.9 | -0.02% |
| **Wednesday** | 6,242.1 | +3.55% | 461.2 | +0.04% |
| **Thursday** | 6,201.4 | +2.88% | 460.8 | -0.04% |
| **Friday** | 6,180.2 | +2.53% | 461.4 | +0.09% |
| **Saturday** | 5,821.5 | -3.42% | 462.1 | +0.24% |
| **Sunday** | **5,042.8** | **-16.34%** | 461.0 | +0.00% |

### Key Takeaways:
1. **The Monday Surge:** Mondays record the highest outpatient load (+6.56% above weekly baseline) driven by weekend backlog accumulation and elective follow-up scheduling.
2. **Sunday Outpatient Curtailment:** Outpatient services drop by ~16.3% on Sundays due to routine clinic closures.
3. **Emergency Inflow Invariance by Weekday:** Emergency arrival averages remain almost identical across all 7 days (460.8–462.1), demonstrating that acute medical emergencies occur independently of administrative weekly schedules.

---

## 3. Climate & Environmental Surge Triggers
- **Monsoon Surge Impact:**
  - During the monsoon season (July–September), acute emergency cases rise by **+12.8%** above baseline, driven by waterborne infections, respiratory exacerbations, and transit trauma.
- **Extreme Heatwave Surge:**
  - On days exceeding 40°C (`heat_flag == 1`), emergency patient volume increases by **+8.0%** due to dehydration, heat exhaustion, and cardiovascular strain.
- **Major Festivals & Public Holidays:**
  - Routine OPD appointments drop by **-18.4%** on festival dates (`festival_event_proxy_flag == 1`), followed by a noticeable rebound surge on the immediate subsequent day.

---

## 4. Facility Distribution & Healthcare Hierarchy
- **Government Healthcare Anchors:**
  - SMS Hospital (`JPR001`), Zanana Hospital (`JPR002`), and Bani Park Satellite (`JPR003`) account for **~54.2% of all daily hospital visits** in Jaipur.
  - SMS Hospital alone handles ~28.5% of total city-wide outpatient demand.
- **Private Sector Distribution:**
  - The remaining 7 private hospitals share ~45.8% of daily patient traffic.

---

## 5. Artifacts Generated in `reports/`:
- `reports/historical_trends_summary.csv` — Full 5-year statistical percentiles and growth rates.
- `reports/weekly_patterns.csv` — Day-of-week inflow distribution and percentage deviations.
- `reports/hospital_distribution.csv` — Facility-level patient shares and ownership classification.
- `reports/historical_patterns.png` — 4-panel visual trajectory of longitudinal, weekly, and yearly patterns.
