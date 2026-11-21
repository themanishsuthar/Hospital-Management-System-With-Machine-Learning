# PROJECT_PLAN.md: Day-wise plan (backend / ML only)

The agent works through this top to bottom, one day per session. Tick `[x]` when done. Items marked **ASK** need my approval before proceeding. At the end of every day the agent asks me to approve the commit (`Day N: <summary>`), then pushes it to GitHub the same day without asking again.

## Day 1: Setup and data understanding
- [x] Create virtual environment `.venv`, activate it, install base packages, generate `requirements.txt`
- [x] Create folder structure from AGENTS.md, `config.py`, `.gitignore` (include `.venv/`, `__pycache__/`, model files)
- [x] Confirm GitHub remote is set up (**ASK** me for the repo URL if missing)
- [x] Inspect everything in `data/`: files, columns, types, date range, row counts
- [x] Write `data/README.md` (sources, columns, issues found)
- [x] **ASK:** confirm which columns map to date, OPD count, emergency count, and external factors
- [x] **ASK:** approve initial commit + `.gitignore`

## Day 2: Data loading and validation (FR1, FR2, FR12)
- [ ] `DataLoader`: load hospital and context CSVs
- [ ] `DataValidator`: missing values, duplicates, invalid ranges, date gaps
- [ ] Readable error messages for missing file / bad columns
- [ ] Unit tests
- [ ] **ASK:** missing-data handling strategy

## Day 3: Preprocessing and features (FR3, FR5)
- [ ] `Preprocessor.clean`, save to `data/processed/`
- [ ] Features: day of week, month, lag values, rolling means, holidays, weather, epidemic flag (where data exists)
- [ ] Explicit handling when contextual data is absent
- [ ] **ASK:** train/validation/test split (chronological) and forecast horizon

## Day 4: Historical analysis (FR4)
- [ ] `HistoricalAnalyzer`: daily trend, weekly pattern, seasonal summary
- [ ] Output summary tables/plots to `reports/`
- [ ] Short written findings for me to read

## Day 5: Baseline and XGBoost (FR6, FR7)
- [ ] `ForecastModel` abstract class
- [ ] Naive / moving-average baseline
- [ ] `XGBoostModel` for OPD and emergency
- [ ] Tests on a small sample

## Day 6: Prophet model
- [ ] `ProphetModel` with holidays and regressors
- [ ] Forecast for the configured horizon

## Day 7: LSTM model
- [ ] `LSTMModel` (scaling, windowing, early stopping)
- [ ] If data is too small for LSTM, report it honestly and **ASK** whether to keep it

## Day 8: Evaluation and model comparison (FR8)
- [ ] `ModelEvaluator`: RMSE, MAE, MAPE, R-squared on held-out data
- [ ] Comparison table and plots in `reports/`
- [ ] **ASK:** which model becomes "best"

## Day 9: Resource planning (FR9)
- [ ] `ResourcePlanner`: doctors, beds, staff, medicine estimates from forecast
- [ ] **ASK:** the rule values (patients per doctor, bed ratio, medicine per patient)
- [ ] Output clearly labelled as estimates

## Day 10: Alerts (FR10)
- [ ] `AlertEngine`: threshold checks, severity levels, messages
- [ ] **ASK:** threshold values and severity levels

## Day 11: API layer for teammate (FR11, FR12)
- [ ] `src/api.py` with the functions listed in AGENTS.md
- [ ] Example script `examples/usage.py` and a short `API_GUIDE.md` the UI teammate can follow
- [ ] **ASK:** review the API signatures with me before finalising

## Day 12: Testing and hardening
- [ ] Full pytest run, edge cases (empty file, missing columns, short history)
- [ ] Check no patient-identifiable data leaks into logs
- [ ] Reproducibility check (fixed seeds)

## Day 13: Documentation and handoff
- [ ] Update README with setup and run steps
- [ ] Final report: data used, method, metrics, limitations (aligned with SRS 3.7)
- [ ] Confirm API works with the teammate's dashboard

## Day 14: Buffer and final polish
- [ ] Fix issues from integration
- [ ] **ASK:** final commit and push