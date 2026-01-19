# File 4: src/models.py
"""
Machine Learning models module for Project Vande.

This module provides wrapper classes for anomaly detection (Isolation Forest)
and time series forecasting (Prophet) with standardized interfaces.

Classes:
    - AnomalyDetector: Isolation Forest wrapper for detecting anomalous patterns
    - EnrolmentForecaster: Prophet wrapper for time series forecasting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest.
    
    This class wraps sklearn's IsolationForest with preprocessing,
    standardization, and methods for scoring samples.
    
    Attributes:
        contamination: Expected proportion of anomalies.
        threshold: Score threshold for flagging anomalies.
        model: Fitted IsolationForest model.
        scaler: StandardScaler for feature normalization.
        feature_names: List of feature column names.
    
    Example:
        >>> detector = AnomalyDetector(contamination=0.05)
        >>> detector.fit(df[['enrolment_total', 'total_updates', 'volatility']])
        >>> labels = detector.predict(df[features])
        >>> scores = detector.score_samples(df[features])
    """
    
    def __init__(
        self,
        contamination: float = None,
        threshold: float = None,
        n_estimators: int = None,
        random_state: int = None
    ):
        """
        Initialize AnomalyDetector.
        
        Args:
            contamination: Expected proportion of anomalies. Defaults to config value.
            threshold: Score threshold for anomalies. Defaults to config value.
            n_estimators: Number of trees. Defaults to config value.
            random_state: Random seed for reproducibility. Defaults to config value.
        """
        self.contamination = contamination or config.ANOMALY_CONTAMINATION
        self.threshold = threshold or config.ANOMALY_SCORE_THRESHOLD
        self.n_estimators = n_estimators or config.ANOMALY_N_ESTIMATORS
        self.random_state = random_state or config.ANOMALY_RANDOM_STATE
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def fit(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            features: Feature matrix (n_samples, n_features).
            feature_names: Optional list of feature names.
        
        Returns:
            self: Fitted detector instance.
            
        Raises:
            ValueError: If features contain invalid values.
            
        Example:
            >>> detector.fit(df[['enrolment_total', 'volatility']])
        """
        logger.info("Fitting anomaly detection model...")
        
        # Store feature names
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
            X = features.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
            X = features
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Validate data
        if X.shape[0] == 0:
            raise ValueError("No samples provided for fitting")
        
        if X.shape[1] == 0:
            raise ValueError("No features provided for fitting")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        
        self._is_fitted = True
        logger.info(f"Model fitted on {X.shape[0]:,} samples with {X.shape[1]} features")
        
        return self
    
    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            features: Feature matrix (n_samples, n_features).
        
        Returns:
            np.ndarray: Binary labels (1 = normal, -1 = anomaly converted to 1/0).
            
        Raises:
            RuntimeError: If model has not been fitted.
            
        Example:
            >>> labels = detector.predict(df[features])
            >>> anomaly_count = (labels == 1).sum()  # 1 = anomaly
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 for anomaly, 1 for normal)
        raw_labels = self.model.predict(X_scaled)
        
        # Convert to 0/1 (0 = normal, 1 = anomaly)
        labels = (raw_labels == -1).astype(int)
        
        logger.info(f"Predicted {labels.sum():,} anomalies out of {len(labels):,} samples")
        
        return labels
    
    def score_samples(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Lower scores indicate more anomalous samples.
        
        Args:
            features: Feature matrix (n_samples, n_features).
        
        Returns:
            np.ndarray: Anomaly scores (lower = more anomalous).
            
        Raises:
            RuntimeError: If model has not been fitted.
            
        Example:
            >>> scores = detector.score_samples(df[features])
            >>> high_risk = df[scores < -0.5]
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring. Call fit() first.")
        
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def fit_predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Fit model and return predictions in one step.
        
        Args:
            features: Feature matrix (n_samples, n_features).
        
        Returns:
            np.ndarray: Binary anomaly labels.
            
        Example:
            >>> labels = detector.fit_predict(df[features])
        """
        self.fit(features)
        return self.predict(features)
    
    def get_anomalous_indices(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        use_threshold: bool = True
    ) -> np.ndarray:
        """
        Get indices of anomalous samples.
        
        Args:
            features: Feature matrix.
            use_threshold: If True, use score threshold. Otherwise, use model prediction.
        
        Returns:
            np.ndarray: Indices of anomalous samples.
        """
        if use_threshold:
            scores = self.score_samples(features)
            return np.where(scores < self.threshold)[0]
        else:
            labels = self.predict(features)
            return np.where(labels == 1)[0]


class EnrolmentForecaster:
    """
    Time series forecasting using Facebook Prophet.
    
    This class wraps Prophet for forecasting enrolment and update metrics
    with support for backtesting and confidence intervals.
    
    Attributes:
        horizon: Forecast horizon in days.
        intervals: List of confidence interval widths.
        model: Fitted Prophet model.
        target_col: Name of target column.
    
    Example:
        >>> forecaster = EnrolmentForecaster(horizon=30)
        >>> forecaster.fit(df, target_col='enrolment_total')
        >>> forecast = forecaster.forecast()
        >>> metrics = forecaster.backtest(df, test_days=30)
    """
    
    def __init__(
        self,
        horizon: int = None,
        intervals: List[float] = None,
        yearly_seasonality: bool = None,
        weekly_seasonality: bool = None,
        daily_seasonality: bool = None,
        changepoint_prior_scale: float = None
    ):
        """
        Initialize EnrolmentForecaster.
        
        Args:
            horizon: Forecast horizon in days.
            intervals: Confidence interval widths (e.g., [0.80, 0.95]).
            yearly_seasonality: Enable yearly seasonality.
            weekly_seasonality: Enable weekly seasonality.
            daily_seasonality: Enable daily seasonality.
            changepoint_prior_scale: Flexibility of trend changes.
        """
        self.horizon = horizon or config.FORECAST_HORIZON_DAYS
        self.intervals = intervals or config.FORECAST_CONFIDENCE_INTERVALS
        self.yearly_seasonality = yearly_seasonality if yearly_seasonality is not None else config.PROPHET_YEARLY_SEASONALITY
        self.weekly_seasonality = weekly_seasonality if weekly_seasonality is not None else config.PROPHET_WEEKLY_SEASONALITY
        self.daily_seasonality = daily_seasonality if daily_seasonality is not None else config.PROPHET_DAILY_SEASONALITY
        self.changepoint_prior_scale = changepoint_prior_scale or config.PROPHET_CHANGEPOINT_PRIOR_SCALE
        
        self.model = None
        self.target_col: Optional[str] = None
        self._is_fitted = False
        self._training_data: Optional[pd.DataFrame] = None
    
    def _prepare_prophet_data(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns).
        
        Args:
            df: Input dataframe.
            target_col: Name of target column.
        
        Returns:
            pd.DataFrame: Prophet-formatted dataframe.
        """
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df['date'])
        prophet_df['y'] = df[target_col].values
        
        # Handle aggregation if multiple rows per date
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        return prophet_df
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        suppress_logging: bool = True
    ) -> 'EnrolmentForecaster':
        """
        Fit the forecasting model.
        
        Args:
            df: Input dataframe with 'date' column and target column.
            target_col: Name of column to forecast.
            suppress_logging: If True, suppress Prophet's verbose output.
        
        Returns:
            self: Fitted forecaster instance.
            
        Raises:
            ValueError: If required columns are missing.
            
        Example:
            >>> forecaster.fit(df, target_col='enrolment_total')
        """
        logger.info(f"Fitting Prophet model for '{target_col}'...")
        
        # Import Prophet here to handle optional dependency
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is required for forecasting. "
                "Install with: pip install prophet"
            )
        
        # Validate input
        if 'date' not in df.columns:
            raise ValueError("Dataframe must contain 'date' column")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        self.target_col = target_col
        
        # Prepare data
        prophet_df = self._prepare_prophet_data(df, target_col)
        self._training_data = prophet_df.copy()
        
        # Initialize Prophet model
        # Use 95% for the wider interval
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=max(self.intervals)
        )
        
        # Fit model
        if suppress_logging:
            import logging as py_logging
            py_logging.getLogger('prophet').setLevel(py_logging.WARNING)
            py_logging.getLogger('cmdstanpy').setLevel(py_logging.WARNING)
        
        self.model.fit(prophet_df)
        
        self._is_fitted = True
        logger.info(f"Prophet model fitted on {len(prophet_df):,} data points")
        
        return self
    
    def forecast(
        self,
        periods: int = None,
        include_history: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecasts for future periods.
        
        Args:
            periods: Number of days to forecast. Defaults to self.horizon.
            include_history: Whether to include historical fit.
        
        Returns:
            pd.DataFrame: Forecast with columns:
                - ds: Date
                - yhat: Point forecast
                - yhat_lower: Lower bound
                - yhat_upper: Upper bound
                - trend: Trend component
                
        Raises:
            RuntimeError: If model has not been fitted.
            
        Example:
            >>> forecast = forecaster.forecast(periods=30)
            >>> print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting. Call fit() first.")
        
        periods = periods or self.horizon
        
        logger.info(f"Generating {periods}-day forecast...")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Select key columns
        key_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        if 'weekly' in forecast.columns:
            key_cols.append('weekly')
        if 'yearly' in forecast.columns:
            key_cols.append('yearly')
        
        result = forecast[key_cols].copy()
        
        # Add actual values where available
        if self._training_data is not None:
            result = result.merge(
                self._training_data.rename(columns={'y': 'actual'}),
                on='ds',
                how='left'
            )
        
        if not include_history:
            # Return only future predictions
            last_actual_date = self._training_data['ds'].max()
            result = result[result['ds'] > last_actual_date]
        
        logger.info(f"Generated forecast with {len(result):,} data points")
        
        return result
    
    def backtest(
        self,
        df: pd.DataFrame,
        test_days: int = None,
        train_days: int = None
    ) -> Dict[str, float]:
        """
        Perform backtesting to evaluate forecast accuracy.
        
        Strategy:
            - Train on T-90 days
            - Test on T-30 to T days
            - Calculate MAPE and RMSE
        
        Args:
            df: Full dataset with 'date' column and target column.
            test_days: Number of days to use for testing.
            train_days: Number of days to use for training.
        
        Returns:
            Dict[str, float]: Dictionary with MAPE, RMSE metrics.
            
        Raises:
            RuntimeError: If model has not been fitted or target_col not set.
            
        Example:
            >>> metrics = forecaster.backtest(df, test_days=30)
            >>> print(f"MAPE: {metrics['mape']:.2f}%")
            >>> print(f"RMSE: {metrics['rmse']:.2f}")
        """
        if self.target_col is None:
            raise RuntimeError("Target column not set. Call fit() first.")
        
        test_days = test_days or config.FORECAST_BACKTEST_DAYS
        train_days = train_days or config.FORECAST_TRAIN_DAYS
        
        logger.info(f"Backtesting: train on {train_days} days, test on {test_days} days...")
        
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet is required for forecasting.")
        
        # Prepare data
        prophet_df = self._prepare_prophet_data(df, self.target_col)
        
        if len(prophet_df) < train_days + test_days:
            logger.warning(
                f"Insufficient data for full backtest. "
                f"Have {len(prophet_df)} days, need {train_days + test_days}"
            )
            # Adjust to available data
            available_days = len(prophet_df)
            test_days = min(test_days, available_days // 4)
            train_days = available_days - test_days
        
        # Split data
        cutoff_idx = len(prophet_df) - test_days
        train_df = prophet_df.iloc[:cutoff_idx]
        test_df = prophet_df.iloc[cutoff_idx:]
        
        # Train model on historical data
        bt_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=max(self.intervals)
        )
        
        import logging as py_logging
        py_logging.getLogger('prophet').setLevel(py_logging.WARNING)
        py_logging.getLogger('cmdstanpy').setLevel(py_logging.WARNING)
        
        bt_model.fit(train_df)
        
        # Generate predictions for test period
        future = bt_model.make_future_dataframe(periods=test_days)
        forecast = bt_model.predict(future)
        
        # Get predictions for test period only
        forecast_test = forecast[forecast['ds'].isin(test_df['ds'])][['ds', 'yhat']]
        
        # Merge with actuals
        comparison = test_df.merge(forecast_test, on='ds')
        
        # Calculate metrics
        actual = comparison['y'].values
        predicted = comparison['yhat'].values
        
        # MAPE (avoid division by zero)
        nonzero_mask = actual != 0
        if nonzero_mask.sum() > 0:
            mape = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
        else:
            mape = np.nan
        
        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        metrics = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'test_days': test_days,
            'train_days': train_days,
            'test_samples': len(comparison)
        }
        
        logger.info(f"Backtest results: MAPE={mape:.2f}%, RMSE={rmse:.2f}")
        
        return metrics
    
    def get_components(self) -> pd.DataFrame:
        """
        Get the decomposed time series components.
        
        Returns:
            pd.DataFrame: Dataframe with trend and seasonality components.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        forecast = self.forecast(periods=0, include_history=True)
        return forecast


def detect_anomalies_in_dataframe(
    df: pd.DataFrame,
    features: List[str] = None
) -> pd.DataFrame:
    """
    Convenience function to detect anomalies and add results to dataframe.
    
    Args:
        df: Input dataframe.
        features: List of feature columns. Defaults to config.ANOMALY_FEATURES.
    
    Returns:
        pd.DataFrame: Input dataframe with 'is_anomaly' and 'anomaly_score' columns.
        
    Example:
        >>> df = detect_anomalies_in_dataframe(df)
        >>> anomalies = df[df['is_anomaly'] == 1]
    """
    features = features or config.ANOMALY_FEATURES
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) == 0:
        logger.warning("No valid features found for anomaly detection")
        df['is_anomaly'] = 0
        df['anomaly_score'] = 0.0
        return df
    
    logger.info(f"Detecting anomalies using {len(available_features)} features")
    
    # Prepare feature matrix
    X = df[available_features].copy()
    
    # Train detector
    detector = AnomalyDetector()
    detector.fit(X)
    
    # Get predictions and scores
    df = df.copy()
    df['is_anomaly'] = detector.predict(X)
    df['anomaly_score'] = detector.score_samples(X)
    
    return df
