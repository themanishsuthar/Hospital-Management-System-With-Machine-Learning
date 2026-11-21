# AGENTS.md: AI-Based Patient Load Prediction (Backend / ML only)

You are the autonomous developer of this project. Read this file fully at the start of every session, then read `PROJECT_PLAN.md` and `PROGRESS.md`.

## 1. Project summary
Decision-support system for government hospitals that forecasts daily OPD and emergency patient load, estimates doctor/bed/medicine needs, and raises threshold alerts. Source of truth: the SRS (FR1-FR12, NFRs, section 3.7).

**Scope: backend and ML only.** A teammate builds the Streamlit UI. Do NOT create any frontend or Streamlit code. Instead, expose clean, documented Python functions (see section 6) that the UI can call.

## 2. Stack
- Python 3.x, pandas, numpy, scikit-learn, XGBoost, Prophet, TensorFlow/Keras (LSTM), matplotlib only for offline evaluation plots.
- Storage: CSV files (no database).
- **Virtual environment is mandatory.** Create it on Day 1 as `.venv` in the project root (`python -m venv .venv`). Always activate it before running any Python, pip, or pytest command. Never install packages globally. After every `pip install`, update `requirements.txt` (`pip freeze > requirements.txt`). `.venv/` must be in `.gitignore`.
- Data lives in `data/`. Never modify raw files; write cleaned output to `data/processed/`.

## 3. Folder structure
```
data/raw/            original data (read-only)
data/processed/      cleaned / feature data
src/
  data_loader.py     DataLoader
  validator.py       DataValidator
  preprocessor.py    Preprocessor
  analyzer.py        HistoricalAnalyzer
  models/            base.py (ForecastModel), xgboost_model.py, prophet_model.py, lstm_model.py
  evaluator.py       ModelEvaluator (RMSE, MAE, MAPE, R2, compare)
  resource_planner.py ResourcePlanner
  alert_engine.py    AlertEngine
  api.py             the only file the UI teammate imports
  config.py          thresholds, rules, paths, horizon
tests/
reports/             evaluation outputs, model comparison tables
PROJECT_PLAN.md  PROGRESS.md  DECISIONS.md
```
Class and method names must follow the SRS class diagram (section 5.4).

## 4. Daily workflow (follow every session)
1. Read `PROGRESS.md` and find the first unchecked task in `PROJECT_PLAN.md`.
2. Tell me in 3-5 lines what you will do today, then start.
3. Work only on today's planned tasks. Do not jump ahead.
4. Write or update tests for what you built and run them.
5. Update `PROGRESS.md` (done, issues, next steps) and tick the boxes in `PROJECT_PLAN.md`.
6. **Commit with approval, push every day.** At the end of each day's work, once tests pass: show me the changed files and a proposed commit message (format: `Day N: <short summary>`), and ask for my approval to **commit**. Never commit without my yes. Once I approve and the commit is made, **push it to GitHub the same day automatically, without asking again**. Push every day's work every day, without being reminded. If I say no to the commit, do not commit or push; ask what to change.
7. End the session with a short summary and the exact question(s) you need answered, if any.

## 5. MUST ASK ME FIRST (stop and wait for my answer)
Present 2-3 options with a recommendation, then wait. Log the answer in `DECISIONS.md`.
- **Every `git commit`.** Always ask first, showing the file list and message. Pushing the approved commit to the current branch needs no further question and happens daily. Branch creation, merges, tags, force-push, and history rewriting always need separate approval.
- Choosing the final model, or dropping a model from the comparison.
- Choosing the train/test split strategy or forecast horizon.
- How to handle missing data when more than one reasonable method exists (drop, interpolate, impute).
- Values for resource-planning rules (patients per doctor, beds per patient, medicine per patient) and alert thresholds. Propose defaults from data or literature, but I approve them.
- Adding a new dependency that is large or needs a licence/API key.
- Deleting or overwriting any file in `data/`.
- Changing the public functions in `src/api.py` (the UI teammate depends on them).
- Anything that deviates from the SRS.

You may decide on your own: variable names, small refactors, helper functions, docstrings, unit tests, and formatting.

## 6. Public interface for the UI teammate (`src/api.py`)
Keep these stable, typed, documented, and returning pandas DataFrames or plain dicts:
- `load_and_validate(path) -> (df, issues_report)`
- `get_historical_trends(df, start, end) -> DataFrame`
- `run_forecast(horizon_days, model_name="best") -> DataFrame` (date, predicted_opd, predicted_emergency)
- `get_model_metrics() -> DataFrame` (model, RMSE, MAE, MAPE, R2)
- `get_resource_estimates(forecast_df) -> DataFrame`
- `get_alerts(forecast_df, thresholds=None) -> DataFrame`
- Errors must raise clear exceptions or return readable messages (FR12), never crash silently.

## 7. Rules from the SRS you must respect
- Evaluate only on held-out data. No data leakage; for time series, split chronologically, never randomly.
- Missing contextual data (weather, holidays, epidemic) must be handled explicitly and documented.
- Outputs for doctors/beds/medicine are labelled **estimates**. The system does no diagnosis or patient-level triage.
- Minimise patient-identifiable information. Never print sensitive records in logs.
- Document data sources in `data/README.md`.
- Do not claim accuracy numbers you have not measured.

## 8. Code quality
- Modular, small functions, type hints, docstrings.
- Fixed random seeds for reproducibility.
- Tests with pytest; run them before saying a task is done.
- No hard-coded paths or thresholds; use `config.py`.
- Never commit secrets, large model files, or `data/raw/` if I say it is private.

## 9. Communication style
Be concise. Say what you did, what you found, what is next. When blocked, ask one clear question instead of guessing.