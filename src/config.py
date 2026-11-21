"""
AI-Based Patient Load Prediction
Configuration module: paths, thresholds, operational rules, and forecast horizons.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Data Files
AGGREGATE_FLOW_FILE = RAW_DATA_DIR / "ML_model_training_based.csv"
HOSPITAL_FLOW_FILE = RAW_DATA_DIR / "patient_flow_from_2021_to_2025.csv"
EVENT_FLOW_FILE = RAW_DATA_DIR / "daily_patient_flow_based_on_events.csv"
HOSPITAL_METADATA_FILE = RAW_DATA_DIR / "hospital_and_address_related.csv"
HOSPITAL_SERVICE_FILE = RAW_DATA_DIR / "hospital_and_service_type.csv"

# Global Random Seed
RANDOM_SEED = 42

# Forecasting Defaults
DEFAULT_HORIZON_DAYS = 7
SUPPORTED_HORIZONS = [7, 14, 30]

# Resource Planning Baseline Rules (IPHS Guidelines - Pending User Approval)
DEFAULT_PATIENTS_PER_OPD_DOCTOR = 45.0
DEFAULT_PATIENTS_PER_EMERGENCY_DOCTOR = 18.0
DEFAULT_NURSES_PER_OCCUPIED_BED = 0.25
DEFAULT_OPD_ADMISSION_RATE = 0.045
DEFAULT_EMERGENCY_ADMISSION_RATE = 0.22
DEFAULT_AVG_LENGTH_OF_STAY_DAYS = 3.5

# Alert Engine Thresholds (% of bed capacity / patient surge)
ALERT_THRESHOLD_YELLOW_OCCUPANCY = 80.0
ALERT_THRESHOLD_ORANGE_OCCUPANCY = 88.0
ALERT_THRESHOLD_RED_OCCUPANCY = 94.0
ALERT_THRESHOLD_OPD_SURGE = 6800.0
ALERT_THRESHOLD_EMERGENCY_SURGE = 510.0
