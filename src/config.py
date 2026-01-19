# File 1: src/config.py
"""
Configuration module for Project Vande - Aadhaar Analytics.

Contains all constants, thresholds, file paths, and hyperparameters.
All magic numbers are centralized here for easy maintenance.
"""

from pathlib import Path
from typing import Dict, List

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths - use Path for cross-platform compatibility
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Data file paths (directories containing multiple CSV files)
ENROLMENT_DIR = RAW_DATA_DIR / "api_data_aadhar_enrolment"
DEMOGRAPHIC_UPDATE_DIR = RAW_DATA_DIR / "api_data_aadhar_demographic"
BIOMETRIC_UPDATE_DIR = RAW_DATA_DIR / "api_data_aadhar_biometric"
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_data.parquet"

# Legacy single file paths (for backwards compatibility)
ENROLMENT_FILE = RAW_DATA_DIR / "enrolment.csv"
DEMOGRAPHIC_UPDATE_FILE = RAW_DATA_DIR / "demographic_update.csv"
BIOMETRIC_UPDATE_FILE = RAW_DATA_DIR / "biometric_update.csv"

# =============================================================================
# COLUMN NAMES
# =============================================================================

# Enrolment dataset columns
ENROLMENT_COLS: Dict[str, str] = {
    "date": "date",
    "state": "state",
    "district": "district",
    "pincode": "pincode",
    "age_5_17": "demo_age_5_17",
    "age_17_plus": "demo_age_17_",
}

# Demographic update columns
DEMOGRAPHIC_COLS: Dict[str, str] = {
    "date": "date",
    "state": "state",
    "district": "district",
    "pincode": "pincode",
    "age_0_5": "age_0_5",
    "age_5_17": "age_5_17",
    "age_18_plus": "age_18_greater",
}

# Biometric update columns
BIOMETRIC_COLS: Dict[str, str] = {
    "date": "date",
    "state": "state",
    "district": "district",
    "pincode": "pincode",
    "age_5": "bio_age_5",
    "age_17_plus": "bio_age_17_",
}

# Merge key columns
MERGE_KEYS: List[str] = ["date", "state", "district"]

# =============================================================================
# DERIVED METRIC NAMES
# =============================================================================

# Core derived metrics
METRIC_ENROLMENT_TOTAL = "enrolment_total"
METRIC_DEMOGRAPHIC_UPDATES_TOTAL = "demographic_updates_total"
METRIC_BIOMETRIC_UPDATES_TOTAL = "biometric_updates_total"
METRIC_TOTAL_UPDATES = "total_updates"
METRIC_UPDATE_TO_ENROLMENT_RATIO = "update_to_enrolment_ratio"
METRIC_ENROLMENT_VELOCITY = "enrolment_velocity"
METRIC_ENROLMENT_VOLATILITY = "enrolment_volatility"

# =============================================================================
# ROLLING WINDOW PARAMETERS
# =============================================================================

VELOCITY_WINDOW_DAYS: int = 7  # 7-day rolling mean for velocity
VOLATILITY_WINDOW_DAYS: int = 7  # 7-day rolling std for volatility
SATURATION_WINDOW_DAYS: int = 30  # Days to check for saturation

# =============================================================================
# ASI (AADHAAR STRESS INDEX) CONFIGURATION
# =============================================================================

ASI_WEIGHTS: Dict[str, float] = {
    "enrolment_volatility": 0.30,
    "update_to_enrolment_ratio": 0.30,
    "anomaly_frequency": 0.25,
    "forecast_residual": 0.15,
}

# ASI range
ASI_MIN: float = 0.0
ASI_MAX: float = 100.0

# Policy override thresholds
ASI_NATIONAL_THRESHOLD: float = 60.0
ASI_DISTRICT_PERCENT_THRESHOLD: float = 0.70  # 70% of districts

# =============================================================================
# ANOMALY DETECTION CONFIGURATION
# =============================================================================

# Isolation Forest parameters
ANOMALY_CONTAMINATION: float = 0.05  # 5% expected anomalies
ANOMALY_SCORE_THRESHOLD: float = -0.5  # Flag districts below this score
ANOMALY_RANDOM_STATE: int = 42
ANOMALY_N_ESTIMATORS: int = 100
ANOMALY_WINDOW_DAYS: int = 30  # Window for calculating anomaly frequency

# Features for anomaly detection
ANOMALY_FEATURES: List[str] = [
    METRIC_ENROLMENT_TOTAL,
    METRIC_TOTAL_UPDATES,
    METRIC_ENROLMENT_VOLATILITY,
    METRIC_ENROLMENT_VELOCITY,
    METRIC_UPDATE_TO_ENROLMENT_RATIO,
    METRIC_DEMOGRAPHIC_UPDATES_TOTAL,
    METRIC_BIOMETRIC_UPDATES_TOTAL,
]

# =============================================================================
# INCLUSION RISK THRESHOLDS
# =============================================================================

# Velocity thresholds
VELOCITY_PERCENTILE_THRESHOLD: float = 10.0  # 10th percentile
VELOCITY_LOW_DAYS: int = 60  # Consecutive days below threshold

# Update ratio thresholds
UPDATE_RATIO_HIGH: float = 2.0  # High update ratio
UPDATE_RATIO_LOW: float = 0.3  # Low update ratio
UPDATE_RATIO_IMBALANCE_HIGH: float = 1.5  # Imbalanced district threshold

# Enrolment growth threshold
ENROLMENT_GROWTH_LOW: float = 0.01  # 1% growth

# Zero enrolment threshold
ZERO_ENROLMENT_CONSECUTIVE_DAYS: int = 14

# =============================================================================
# SATURATION DETECTION
# =============================================================================

SATURATION_VELOCITY_PERCENT: float = 0.05  # 5% of historical average
SATURATION_DETECTION_DAYS: int = 30  # Consecutive days

# Volatility spike threshold (standard deviations)
VOLATILITY_SPIKE_STD: float = 2.0

# =============================================================================
# FORECASTING CONFIGURATION
# =============================================================================

FORECAST_HORIZON_DAYS: int = 30  # 30-day forecast
FORECAST_BACKTEST_DAYS: int = 30  # Test on last 30 days
FORECAST_TRAIN_DAYS: int = 90  # Train on T-90 days

# Prophet confidence intervals
FORECAST_CONFIDENCE_INTERVALS: List[float] = [0.80, 0.95]

# Prophet parameters
PROPHET_YEARLY_SEASONALITY: bool = True
PROPHET_WEEKLY_SEASONALITY: bool = True
PROPHET_DAILY_SEASONALITY: bool = False
PROPHET_CHANGEPOINT_PRIOR_SCALE: float = 0.05

# =============================================================================
# SERVICE LOAD RANKING
# =============================================================================

TOP_N_HIGH_PRESSURE_DISTRICTS: int = 20

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Figure sizes
FIG_WIDTH: int = 12
FIG_HEIGHT: int = 8

# Color palettes
COLOR_PRIMARY: str = "#1f77b4"
COLOR_SECONDARY: str = "#ff7f0e"
COLOR_DANGER: str = "#d62728"
COLOR_SUCCESS: str = "#2ca02c"
COLOR_WARNING: str = "#ffbb78"

# ASI color scale
ASI_COLORSCALE: str = "RdYlGn_r"  # Red = high stress, Green = low stress

# Plotly template
PLOTLY_TEMPLATE: str = "plotly_white"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: str = "INFO"

# =============================================================================
# DATA VALIDATION
# =============================================================================

# Expected date range (can be adjusted based on actual data)
DATE_FORMAT: str = "%Y-%m-%d"

# Minimum rows expected after merge
MIN_ROWS_AFTER_MERGE: int = 100

# Maximum allowed missing value percentage
MAX_MISSING_VALUE_PERCENT: float = 0.20  # 20%

# Backtest training window
BACKTEST_TRAIN_WINDOW: int = 90  # Days for training in backtesting

# =============================================================================
# CONSOLIDATED RISK CONDITIONS
# =============================================================================

RISK_CONDITIONS: Dict[str, float] = {
    'low_velocity': 10,  # percentile threshold
    'high_update_ratio': 2.0,
    'zero_enrolment_days': 14,
    'low_growth_threshold': 0.01,  # 1%
    'velocity_low_days': 60,  # consecutive days
}

# =============================================================================
# EXPECTED SCHEMAS FOR VALIDATION
# =============================================================================

# Updated schemas to match real API data
ENROLMENT_SCHEMA: Dict[str, str] = {
    'date': 'datetime64[ns]',
    'state': 'string',
    'district': 'string',
    'pincode': 'int64',
    'age_0_5': 'int64',
    'age_5_17': 'int64',
    'age_18_greater': 'int64'
}

DEMOGRAPHIC_SCHEMA: Dict[str, str] = {
    'date': 'datetime64[ns]',
    'state': 'string',
    'district': 'string',
    'pincode': 'int64',
    'demo_age_5_17': 'int64',
    'demo_age_17_': 'int64'
}

BIOMETRIC_SCHEMA: Dict[str, str] = {
    'date': 'datetime64[ns]',
    'state': 'string',
    'district': 'string',
    'pincode': 'int64',
    'bio_age_5_17': 'int64',
    'bio_age_17_': 'int64'
}
