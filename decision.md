# DECISIONS.md: Architecture and Operational Decisions Log

This document records all critical design and modeling choices made with user approval.

---

## Decision 1: Column Mappings for Targets and Features (Day 1)
- **Status:** APPROVED (2026-11-21)
- **Question:** Confirm which columns map to date, OPD count, emergency count, and external factors.
- **Approved Selection: Option A (Recommended)**
  - **Date Column:** `date` (format: `DD-MM-YYYY`, parsed with `dayfirst=True`)
  - **Forecast Targets:** Dual target approach — `opd_patients` and `emergency_patients` modeled independently
  - **Total Load:** `total_patient_load` (`opd_patients + emergency_patients`)
  - **Inpatient Admissions:** `admissions`
  - **Weather / Environmental Proxies:** `mean_temp_proxy_c_simulated`, `mean_precipitation_proxy_mm_simulated`, `heat_flag`
  - **Calendar / Seasonal Flags:** `weekend_flag`, `monsoon_season_flag`, `winter_season_flag`, `summer_season_flag`, `festival_event_proxy_flag`
  - **Hospital Entity Attributes:** `hospital_id`, `ownership_type` (Government / Private)
- **Rationale:** OPD (scheduled, periodic weekly cycles) and Emergency (acute, unscheduled, weather-sensitive) have distinct statistical behaviors and require different clinical staffing formulas under IPHS.

---

## Decision 2: Day 1 Initial Commit & .gitignore Approval (Day 1)
- **Status:** APPROVED (2026-11-21)
- **Question:** Approve initial Day 1 commit and `.gitignore` setup.
- **Approved Action:** Commit Day 1 changes with message `Day 1: Setup virtual environment, project structure, config, and data audit` and push to remote `main`.
