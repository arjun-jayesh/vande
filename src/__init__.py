# src/__init__.py
"""
Project Vande - Aadhaar Analytics Hackathon Solution.

This package provides comprehensive analytics for UIDAI's Aadhaar data,
including enrolment dynamics, update pressure analysis, anomaly detection,
and stress index calculation.

Modules:
    - config: Configuration constants and thresholds
    - preprocessing: Data loading, merging, and feature engineering
    - metrics: ASI calculation, inclusion risk, derived metrics
    - models: ML models for anomaly detection and forecasting
    - viz: Visualization functions for charts and maps
"""

from . import config
from . import preprocessing
from . import metrics
from . import models
from . import viz

__version__ = "1.0.0"
__author__ = "Project Vande Team"
__all__ = ["config", "preprocessing", "metrics", "models", "viz"]
