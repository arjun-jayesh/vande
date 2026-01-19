# Project Vande - Aadhaar Analytics Hackathon Solution

A comprehensive analytics solution for UIDAI's Aadhaar data, providing enrolment dynamics analysis, update pressure monitoring, anomaly detection, and stress index calculation.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd project_vande

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place your raw data files in `data/raw/`:
- `enrolment.csv` - Enrolment data with columns: date, state, district, pincode, demo_age_5_17, demo_age_17_
- `demographic_update.csv` - Demographic updates with columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
- `biometric_update.csv` - Biometric updates with columns: date, state, district, pincode, bio_age_5, bio_age_17_

### 3. Run Analysis

```bash
# Step 1: Data Preparation
jupyter nbconvert --execute notebooks/01_data_preparation.ipynb

# Step 2: Exploratory Analysis
jupyter nbconvert --execute notebooks/02_exploratory_analysis.ipynb

# Step 3: Advanced Analytics (Anomaly Detection, ASI, Forecasting)
jupyter nbconvert --execute notebooks/03_advanced_analytics.ipynb

# Step 4: Generate Final Report
jupyter nbconvert --execute notebooks/04_final_report.ipynb

# Step 5: Launch Dashboard
streamlit run dashboard/app.py
```

## 📊 Features

### Core Analytics
- **Enrolment Dynamics**: Growth trends, velocity, volatility analysis
- **Update Pressure**: Update-to-enrolment ratio monitoring
- **Anomaly Detection**: Isolation Forest for detecting unusual patterns
- **ASI (Aadhaar Stress Index)**: Composite stress score (0-100) with policy override
- **Inclusion Risk**: Flagging districts with low enrolment/high update patterns
- **30-Day Forecasting**: Prophet-based time series forecasting

### Key Metrics
| Metric | Formula |
|--------|---------|
| `enrolment_total` | demo_age_5_17 + demo_age_17_ |
| `total_updates` | demographic_updates_total + biometric_updates_total |
| `update_to_enrolment_ratio` | total_updates / max(enrolment_total, 1) |
| `enrolment_velocity` | diff(enrolment).rolling(7).mean() |
| `enrolment_volatility` | enrolment.rolling(7).std() |

### ASI Formula
```
ASI = (volatility×0.30 + ratio×0.30 + anomaly×0.25 + forecast_error×0.15) × 100

Policy Override: If national_ASI < 60 AND 70%+ districts > 60, set national_ASI = 60
```

## 📁 Project Structure

```
project_vande/
├── data/
│   ├── raw/                    # Raw CSV files
│   └── processed/              # Merged parquet data
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants and thresholds
│   ├── preprocessing.py        # Data loading and merging
│   ├── metrics.py              # ASI, inclusion risk
│   ├── models.py               # AnomalyDetector, EnrolmentForecaster
│   └── viz.py                  # Plotting functions
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_advanced_analytics.ipynb
│   └── 04_final_report.ipynb
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── outputs/
│   ├── figures/                # Generated charts
│   └── tables/                 # Generated tables
├── requirements.txt
└── README.md
```

## 🔧 Configuration

All thresholds and constants are in `src/config.py`:
- `ANOMALY_CONTAMINATION`: 0.05 (5% expected anomalies)
- `ASI_NATIONAL_THRESHOLD`: 60
- `FORECAST_HORIZON_DAYS`: 30
- `VELOCITY_WINDOW_DAYS`: 7

## 📈 Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

Features:
- State/district filtering
- KPI cards (enrolments, updates, ASI, anomalies)
- Trend visualizations
- ASI distribution map
- Anomaly detection scatter plots
- Inclusion risk analysis
- 30-day forecasting
- CSV/PDF export

## 📝 Competition Submission Checklist

- [x] Data preprocessing pipeline
- [x] Core derived metrics
- [x] Anomaly detection (Isolation Forest)
- [x] ASI calculation with policy override
- [x] Inclusion risk detection
- [x] 30-day forecasting (Prophet)
- [x] Exploratory analysis (15+ visualizations)
- [x] Interactive dashboard
- [x] PDF report generation
- [x] Documentation

## 📄 License

This project is developed for the UIDAI Aadhaar Analytics Hackathon.

## 👥 Team

Project Vande Team
