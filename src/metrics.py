# File 3: src/metrics.py
"""
Metrics calculation module for Project Vande.

This module contains functions for calculating the Aadhaar Stress Index (ASI),
detecting inclusion risks, determining saturation status, and other derived
analytics metrics.

Functions:
    - normalize_minmax: Min-max normalization helper
    - calculate_asi: Compute ASI with policy override
    - detect_inclusion_risk: Flag districts with inclusion risk
    - calculate_saturation_status: Classify district growth stages
    - identify_imbalanced_districts: Find update ratio imbalances
    - rank_service_load: Rank districts by service pressure
    - detect_volatility_spikes: Find unusual volatility patterns
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def normalize_minmax(
    series: pd.Series,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> pd.Series:
    """
    Apply min-max normalization to scale values to [0, 1].
    
    Args:
        series: Pandas series to normalize.
        min_val: Optional minimum value for scaling. If None, uses series min.
        max_val: Optional maximum value for scaling. If None, uses series max.
    
    Returns:
        pd.Series: Normalized series with values in [0, 1].
        
    Example:
        >>> s = pd.Series([10, 20, 30, 40, 50])
        >>> normalized = normalize_minmax(s)
        >>> print(normalized.tolist())
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    min_val = min_val if min_val is not None else series.min()
    max_val = max_val if max_val is not None else series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def calculate_anomaly_frequency_score(
    df: pd.DataFrame,
    district: Optional[str] = None
) -> float:
    """
    Calculate anomaly frequency score for ASI computation.
    
    Args:
        df: Dataframe with 'is_anomaly' column.
        district: Optional district name to filter.
    
    Returns:
        float: Normalized anomaly frequency score [0, 1].
        
    Example:
        >>> score = calculate_anomaly_frequency_score(df, district='MUMBAI')
        >>> print(f"Anomaly frequency: {score:.2f}")
    """
    if 'is_anomaly' not in df.columns:
        logger.warning("'is_anomaly' column not found, returning 0")
        return 0.0
    
    if district:
        df_subset = df[df['district'] == district]
    else:
        df_subset = df
    
    if len(df_subset) == 0:
        return 0.0
    
    # Count anomaly occurrences
    anomaly_count = df_subset['is_anomaly'].sum()
    total_count = len(df_subset)
    
    # Normalize to [0, 1]
    frequency = anomaly_count / total_count
    
    return min(1.0, frequency * 5)  # Scale up to make differences more visible


def apply_asi_policy_override(
    district_asi_df: pd.DataFrame,
    asi_col: str = 'asi_score'
) -> Dict[str, float]:
    """
    Apply ASI policy override logic.
    
    Policy Override Rule:
        If national_ASI < 60 AND 70%+ districts have ASI > 60,
        then set national_ASI = 60
    
    Args:
        district_asi_df: DataFrame with district and ASI score columns.
        asi_col: Name of ASI score column.
    
    Returns:
        Dict with 'national', 'override_applied', 'pct_above_60' keys.
        
    Example:
        >>> result = apply_asi_policy_override(asi_df)
        >>> print(f"National ASI: {result['national']:.1f}")
        >>> print(f"Override applied: {result['override_applied']}")
    """
    if asi_col not in district_asi_df.columns:
        raise ValueError(f"Column '{asi_col}' not found in DataFrame")
    
    scores = district_asi_df[asi_col].dropna()
    
    if len(scores) == 0:
        return {'national': 0.0, 'override_applied': False, 'pct_above_60': 0.0}
    
    # Calculate national average
    national_asi = scores.mean()
    
    # Calculate percentage above 60
    pct_above_60 = (scores >= config.ASI_NATIONAL_THRESHOLD).sum() / len(scores)
    
    # Apply policy override
    override_applied = False
    if national_asi < config.ASI_NATIONAL_THRESHOLD and pct_above_60 >= config.ASI_DISTRICT_PERCENT_THRESHOLD:
        logger.info(
            f"Policy override triggered: {pct_above_60:.1%} districts above 60, "
            f"adjusting national ASI from {national_asi:.1f} to {config.ASI_NATIONAL_THRESHOLD}"
        )
        national_asi = config.ASI_NATIONAL_THRESHOLD
        override_applied = True
    
    return {
        'national': national_asi,
        'override_applied': override_applied,
        'pct_above_60': pct_above_60,
        'district_count': len(scores),
        'districts_above_60': int((scores >= config.ASI_NATIONAL_THRESHOLD).sum())
    }


def calculate_forecast_residual_score(
    df: pd.DataFrame,
    district: Optional[str] = None
) -> float:
    """
    Calculate forecast residual error score for ASI computation.
    
    Args:
        df: Dataframe with 'forecast_residual' or MAPE column.
        district: Optional district name to filter.
    
    Returns:
        float: Normalized forecast residual score [0, 1].
    """
    residual_col = None
    for col in ['forecast_residual', 'mape', 'forecast_error']:
        if col in df.columns:
            residual_col = col
            break
    
    if residual_col is None:
        logger.warning("No forecast residual column found, returning 0")
        return 0.0
    
    if district:
        df_subset = df[df['district'] == district]
    else:
        df_subset = df
    
    if len(df_subset) == 0:
        return 0.0
    
    # Get mean absolute residual/error
    mean_residual = df_subset[residual_col].abs().mean()
    
    # Normalize using reasonable bounds (0-50% error range)
    normalized = min(1.0, mean_residual / 50.0)
    
    return normalized


def calculate_asi(
    df: pd.DataFrame,
    district: Optional[str] = None,
    include_national: bool = True
) -> Union[float, Dict[str, float]]:
    """
    Calculate Aadhaar Stress Index (ASI) for a district or entire dataset.
    
    ASI Formula:
        ASI = weighted_average([
            normalize(enrolment_volatility) * 0.30,
            normalize(update_to_enrolment_ratio) * 0.30,
            anomaly_frequency_score * 0.25,
            normalize(forecast_residual_error) * 0.15
        ]) * 100
    
    Policy Override:
        if (national_ASI < 60) AND (percent_districts_above_60 >= 0.70):
            national_ASI = 60
    
    Args:
        df: Dataframe with required metric columns.
        district: Optional district name. If None, calculates for all districts.
        include_national: Whether to include national aggregate.
    
    Returns:
        float or Dict[str, float]: ASI score(s) in range [0, 100].
        
    Example:
        >>> asi = calculate_asi(df, district='MUMBAI')
        >>> print(f"Mumbai ASI: {asi:.1f}")
        Mumbai ASI: 45.3
    """
    logger.info("Calculating ASI scores...")
    
    weights = config.ASI_WEIGHTS
    
    if district:
        # Calculate for single district
        df_district = df[df['district'] == district].copy()
        
        if len(df_district) == 0:
            logger.warning(f"No data found for district: {district}")
            return 0.0
        
        # Normalize components
        volatility_norm = normalize_minmax(
            df[config.METRIC_ENROLMENT_VOLATILITY]
        ).loc[df_district.index].mean()
        
        ratio_norm = normalize_minmax(
            df[config.METRIC_UPDATE_TO_ENROLMENT_RATIO]
        ).loc[df_district.index].mean()
        
        anomaly_score = calculate_anomaly_frequency_score(df, district)
        forecast_score = calculate_forecast_residual_score(df, district)
        
        # Weighted average
        asi = (
            volatility_norm * weights['enrolment_volatility'] +
            ratio_norm * weights['update_to_enrolment_ratio'] +
            anomaly_score * weights['anomaly_frequency'] +
            forecast_score * weights['forecast_residual']
        ) * 100
        
        return min(config.ASI_MAX, max(config.ASI_MIN, asi))
    
    else:
        # Calculate for all districts
        district_scores = {}
        
        for dist in df['district'].unique():
            district_scores[dist] = calculate_asi(df, district=dist, include_national=False)
        
        if include_national:
            # Apply policy override for national ASI
            all_scores = list(district_scores.values())
            national_asi = np.mean(all_scores)
            
            # Count districts above 60
            districts_above_60 = sum(1 for s in all_scores if s >= 60)
            percent_above_60 = districts_above_60 / len(all_scores) if all_scores else 0
            
            # Policy override check
            if national_asi < config.ASI_NATIONAL_THRESHOLD and \
               percent_above_60 >= config.ASI_DISTRICT_PERCENT_THRESHOLD:
                logger.info(
                    f"Policy override triggered: "
                    f"{percent_above_60:.1%} districts above 60, "
                    f"adjusting national ASI from {national_asi:.1f} to 60"
                )
                national_asi = config.ASI_NATIONAL_THRESHOLD
            
            district_scores['NATIONAL'] = national_asi
        
        return district_scores


def detect_inclusion_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag districts with inclusion risk based on multiple criteria.
    
    Risk Conditions:
        1. Enrolment velocity < 10th percentile for 60+ days
        2. Update ratio > 2.0 AND enrolment growth < 1%
        3. Zero enrolments for 14+ consecutive days
    
    Args:
        df: Dataframe with required metric columns.
    
    Returns:
        pd.DataFrame: Input dataframe with additional risk flag columns.
        
    Example:
        >>> df_risk = detect_inclusion_risk(df)
        >>> risk_count = df_risk['inclusion_risk'].sum()
        >>> print(f"Districts at risk: {risk_count}")
    """
    logger.info("Detecting inclusion risk factors...")
    
    df = df.copy()
    
    # Initialize risk columns
    df['risk_low_velocity'] = False
    df['risk_high_ratio_low_growth'] = False
    df['risk_zero_enrolments'] = False
    df['inclusion_risk'] = False
    
    # Calculate 10th percentile of velocity
    velocity_threshold = df[config.METRIC_ENROLMENT_VELOCITY].quantile(
        config.VELOCITY_PERCENTILE_THRESHOLD / 100
    )
    
    # Process each district
    for (state, district), group in df.groupby(['state', 'district']):
        group = group.sort_values('date')
        idx = group.index
        
        # Condition 1: Low velocity for extended period
        low_velocity_mask = group[config.METRIC_ENROLMENT_VELOCITY] < velocity_threshold
        consecutive_low = _count_consecutive(low_velocity_mask.values)
        risk_1 = consecutive_low >= config.VELOCITY_LOW_DAYS
        df.loc[idx[risk_1], 'risk_low_velocity'] = True
        
        # Condition 2: High update ratio with low growth
        high_ratio = group[config.METRIC_UPDATE_TO_ENROLMENT_RATIO] > config.UPDATE_RATIO_HIGH
        low_growth = group['monthly_pct_change'].fillna(0) < config.ENROLMENT_GROWTH_LOW * 100
        risk_2 = high_ratio & low_growth
        df.loc[idx[risk_2], 'risk_high_ratio_low_growth'] = True
        
        # Condition 3: Zero enrolments for consecutive days
        zero_enrol = group[config.METRIC_ENROLMENT_TOTAL] == 0
        consecutive_zero = _count_consecutive(zero_enrol.values)
        risk_3 = consecutive_zero >= config.ZERO_ENROLMENT_CONSECUTIVE_DAYS
        df.loc[idx[risk_3], 'risk_zero_enrolments'] = True
    
    # Combine all risk factors
    df['inclusion_risk'] = (
        df['risk_low_velocity'] |
        df['risk_high_ratio_low_growth'] |
        df['risk_zero_enrolments']
    )
    
    # Count risk districts
    risk_districts = df[df['inclusion_risk']]['district'].nunique()
    logger.info(f"Identified {risk_districts} districts with inclusion risk")
    
    return df


def _count_consecutive(arr: np.ndarray) -> np.ndarray:
    """
    Count consecutive True values in array, return count at each position.
    
    Args:
        arr: Boolean array.
    
    Returns:
        np.ndarray: Array of consecutive counts.
    """
    result = np.zeros_like(arr, dtype=int)
    count = 0
    
    for i in range(len(arr)):
        if arr[i]:
            count += 1
        else:
            count = 0
        result[i] = count
    
    return result


def calculate_saturation_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify districts into growth stages based on enrolment velocity.
    
    Categories:
        - SATURATED: Velocity < 5% of historical avg for 30+ days
        - STAGNANT: Velocity between 5-20% of historical avg
        - GROWING: Velocity between 20-80% of historical avg
        - HIGH_GROWTH: Velocity > 80% of historical avg
    
    Args:
        df: Dataframe with enrolment velocity column.
    
    Returns:
        pd.DataFrame: Input dataframe with 'saturation_status' column.
        
    Example:
        >>> df_status = calculate_saturation_status(df)
        >>> print(df_status['saturation_status'].value_counts())
        GROWING       1500
        STAGNANT       800
        HIGH_GROWTH    300
        SATURATED      100
    """
    logger.info("Calculating saturation status...")
    
    df = df.copy()
    df['saturation_status'] = 'UNKNOWN'
    
    for (state, district), group in df.groupby(['state', 'district']):
        idx = group.index
        velocity = group[config.METRIC_ENROLMENT_VELOCITY]
        
        # Calculate historical average velocity
        hist_avg = velocity.abs().mean()
        
        if hist_avg == 0:
            df.loc[idx, 'saturation_status'] = 'SATURATED'
            continue
        
        # Calculate relative velocity
        rel_velocity = velocity.abs() / hist_avg
        
        # Classify each day
        conditions = [
            rel_velocity < config.SATURATION_VELOCITY_PERCENT,  # < 5%
            rel_velocity < 0.20,  # 5-20%
            rel_velocity < 0.80,  # 20-80%
            rel_velocity >= 0.80  # > 80%
        ]
        choices = ['SATURATED', 'STAGNANT', 'GROWING', 'HIGH_GROWTH']
        
        status = np.select(conditions, choices, default='UNKNOWN')
        df.loc[idx, 'saturation_status'] = status
    
    # Log summary
    status_counts = df['saturation_status'].value_counts()
    logger.info(f"Saturation status distribution:\n{status_counts}")
    
    return df


def identify_imbalanced_districts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify districts with imbalanced update-to-enrolment ratios.
    
    Imbalanced Conditions:
        - HIGH: ratio > 1.5
        - LOW: ratio < 0.3
        - BALANCED: 0.3 <= ratio <= 1.5
    
    Args:
        df: Dataframe with update_to_enrolment_ratio column.
    
    Returns:
        pd.DataFrame: Input dataframe with 'balance_status' column.
        
    Example:
        >>> df_balance = identify_imbalanced_districts(df)
        >>> imbalanced = df_balance[df_balance['balance_status'] != 'BALANCED']
        >>> print(f"Imbalanced records: {len(imbalanced)}")
    """
    logger.info("Identifying imbalanced districts...")
    
    df = df.copy()
    
    ratio = df[config.METRIC_UPDATE_TO_ENROLMENT_RATIO]
    
    conditions = [
        ratio > config.UPDATE_RATIO_IMBALANCE_HIGH,
        ratio < config.UPDATE_RATIO_LOW,
    ]
    choices = ['HIGH_UPDATES', 'LOW_UPDATES']
    
    df['balance_status'] = np.select(conditions, choices, default='BALANCED')
    
    # Log summary
    balance_counts = df['balance_status'].value_counts()
    logger.info(f"Balance status distribution:\n{balance_counts}")
    
    return df


def rank_service_load(
    df: pd.DataFrame,
    top_n: int = None
) -> pd.DataFrame:
    """
    Rank districts by service load stress (update pressure).
    
    Service load is calculated as the mean update-to-enrolment ratio
    combined with total update volume.
    
    Args:
        df: Dataframe with required columns.
        top_n: Number of top districts to return. Defaults to config value.
    
    Returns:
        pd.DataFrame: Ranked districts with service load metrics.
        
    Example:
        >>> top_districts = rank_service_load(df, top_n=20)
        >>> print(top_districts[['district', 'service_load_rank']])
    """
    top_n = top_n or config.TOP_N_HIGH_PRESSURE_DISTRICTS
    
    logger.info(f"Ranking top {top_n} high-pressure districts...")
    
    # Aggregate by district
    district_stats = df.groupby(['state', 'district']).agg({
        config.METRIC_UPDATE_TO_ENROLMENT_RATIO: 'mean',
        config.METRIC_TOTAL_UPDATES: 'sum',
        config.METRIC_ENROLMENT_TOTAL: 'sum',
        config.METRIC_ENROLMENT_VOLATILITY: 'mean'
    }).reset_index()
    
    # Calculate composite service load score
    district_stats['service_load_score'] = (
        normalize_minmax(district_stats[config.METRIC_UPDATE_TO_ENROLMENT_RATIO]) * 0.4 +
        normalize_minmax(district_stats[config.METRIC_TOTAL_UPDATES]) * 0.4 +
        normalize_minmax(district_stats[config.METRIC_ENROLMENT_VOLATILITY]) * 0.2
    )
    
    # Rank districts
    district_stats = district_stats.sort_values(
        'service_load_score', ascending=False
    ).reset_index(drop=True)
    
    district_stats['service_load_rank'] = range(1, len(district_stats) + 1)
    
    return district_stats.head(top_n)


def detect_volatility_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect volatility spikes (> 2 std dev from district mean).
    
    Args:
        df: Dataframe with enrolment volatility column.
    
    Returns:
        pd.DataFrame: Input dataframe with 'volatility_spike' column.
        
    Example:
        >>> df_spikes = detect_volatility_spikes(df)
        >>> spike_count = df_spikes['volatility_spike'].sum()
        >>> print(f"Volatility spikes detected: {spike_count}")
    """
    logger.info("Detecting volatility spikes...")
    
    df = df.copy()
    df['volatility_spike'] = False
    
    for (state, district), group in df.groupby(['state', 'district']):
        idx = group.index
        volatility = group[config.METRIC_ENROLMENT_VOLATILITY]
        
        mean_vol = volatility.mean()
        std_vol = volatility.std()
        
        if std_vol > 0:
            threshold = mean_vol + (config.VOLATILITY_SPIKE_STD * std_vol)
            spikes = volatility > threshold
            df.loc[idx[spikes], 'volatility_spike'] = True
    
    spike_count = df['volatility_spike'].sum()
    logger.info(f"Detected {spike_count} volatility spikes")
    
    return df


def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all analytics metrics for the dataset.
    
    This function applies all metric calculations and risk detection
    in the correct order.
    
    Args:
        df: Dataframe with core derived metrics already calculated.
    
    Returns:
        pd.DataFrame: Enriched dataframe with all analytics metrics.
        
    Example:
        >>> df_full = calculate_all_metrics(df)
        >>> print(df_full.columns.tolist())
    """
    logger.info("=" * 60)
    logger.info("Calculating all analytics metrics")
    logger.info("=" * 60)
    
    # Calculate saturation status
    df = calculate_saturation_status(df)
    
    # Identify imbalanced districts
    df = identify_imbalanced_districts(df)
    
    # Detect volatility spikes
    df = detect_volatility_spikes(df)
    
    # Detect inclusion risk
    df = detect_inclusion_risk(df)
    
    logger.info("All metrics calculation complete")
    
    return df
