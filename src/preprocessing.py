# File 2: src/preprocessing.py
"""
Data preprocessing module for Project Vande.

This module handles all data loading, merging, feature engineering,
and data quality operations for the Aadhaar analytics pipeline.

Functions:
    - validate_schema: Validate dataframe against expected schema
    - check_date_continuity: Check for missing dates in time series
    - load_enrolment: Load enrolment data with derived totals
    - load_demographic: Load demographic update data with totals
    - load_biometric: Load biometric update data with totals
    - merge_datasets: Outer join all datasets on (date, state, district)
    - calculate_core_metrics: Add all 5 derived metrics
    - handle_missing_values: Forward fill and drop invalid rows
    - save_processed: Save to parquet format
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config

# Configure logging
logging.basicConfig(format=config.LOG_FORMAT, level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, str],
    dataset_name: str = "dataset"
) -> Tuple[bool, List[str]]:
    """
    Validate dataframe against expected schema.
    
    Checks that all required columns exist and have compatible types.
    
    Args:
        df: Dataframe to validate.
        expected_schema: Dictionary mapping column names to expected dtypes.
        dataset_name: Name of dataset for error messages.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of error messages)
        
    Example:
        >>> schema = {'date': 'datetime64[ns]', 'state': 'string', 'count': 'int64'}
        >>> is_valid, errors = validate_schema(df, schema, 'enrolment')
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """
    errors = []
    
    # Check for missing columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns in {dataset_name}: {missing_cols}")
    
    # Check column types (for columns that exist)
    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            continue
            
        actual_type = str(df[col].dtype)
        
        # Flexible type matching
        type_compatible = False
        if expected_type == 'datetime64[ns]':
            type_compatible = 'datetime' in actual_type or df[col].dtype == 'object'
        elif expected_type == 'string':
            type_compatible = df[col].dtype == 'object' or str(df[col].dtype) == 'string'
        elif expected_type == 'int64':
            type_compatible = 'int' in actual_type or 'float' in actual_type
        else:
            type_compatible = actual_type == expected_type
        
        if not type_compatible:
            logger.warning(
                f"Column '{col}' in {dataset_name} has type '{actual_type}', "
                f"expected '{expected_type}' (will attempt conversion)"
            )
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"Schema validation passed for {dataset_name}")
    else:
        logger.error(f"Schema validation failed for {dataset_name}: {errors}")
    
    return is_valid, errors


def check_date_continuity(
    df: pd.DataFrame,
    date_col: str = 'date',
    group_cols: Optional[List[str]] = None,
    max_gap_days: int = 7
) -> Tuple[bool, pd.DataFrame]:
    """
    Check for missing dates or gaps in time series data.
    
    Args:
        df: Dataframe with date column.
        date_col: Name of date column.
        group_cols: Columns to group by (e.g., ['state', 'district']).
        max_gap_days: Maximum allowed gap in days before flagging.
    
    Returns:
        Tuple[bool, pd.DataFrame]: (is_continuous, dataframe of gaps)
        
    Example:
        >>> is_continuous, gaps = check_date_continuity(df, group_cols=['district'])
        >>> if not is_continuous:
        ...     print(f"Found {len(gaps)} date gaps")
    """
    logger.info("Checking date continuity...")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    gaps_list = []
    
    if group_cols:
        for group_key, group_df in df.groupby(group_cols):
            group_df = group_df.sort_values(date_col)
            dates = group_df[date_col]
            
            # Check for gaps
            date_diffs = dates.diff().dt.days
            gap_mask = date_diffs > max_gap_days
            
            if gap_mask.any():
                gap_indices = gap_mask[gap_mask].index
                dates_list = dates.tolist()
                dates_index_list = dates.index.tolist()
                for idx in gap_indices:
                    idx_pos = dates_index_list.index(idx)
                    gap_info = {
                        'gap_start': dates_list[idx_pos - 1] if idx_pos > 0 else None,
                        'gap_end': dates.loc[idx],
                        'gap_days': date_diffs.loc[idx]
                    }
                    if isinstance(group_key, tuple):
                        for i, col in enumerate(group_cols):
                            gap_info[col] = group_key[i]
                    else:
                        gap_info[group_cols[0]] = group_key
                    gaps_list.append(gap_info)
    else:
        # Check overall continuity
        df_sorted = df.sort_values(date_col)
        dates = df_sorted[date_col].drop_duplicates()
        date_diffs = dates.diff().dt.days
        gap_mask = date_diffs > max_gap_days
        
        if gap_mask.any():
            gap_indices = date_diffs[gap_mask].index
            for idx in gap_indices:
                idx_pos = dates.index.get_loc(idx)
                gaps_list.append({
                    'gap_start': dates.iloc[idx_pos - 1] if idx_pos > 0 else None,
                    'gap_end': dates.loc[idx],
                    'gap_days': date_diffs.loc[idx]
                })
    
    gaps_df = pd.DataFrame(gaps_list)
    is_continuous = len(gaps_df) == 0
    
    if is_continuous:
        logger.info("Date continuity check passed - no significant gaps found")
    else:
        logger.warning(f"Found {len(gaps_df)} date gaps exceeding {max_gap_days} days")
    
    return is_continuous, gaps_df


def load_enrolment(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load enrolment dataset and calculate enrolment_total.
    
    Args:
        filepath: Path to the enrolment CSV file or directory containing CSVs.
                  Defaults to config.ENROLMENT_DIR.
    
    Returns:
        pd.DataFrame: Loaded dataframe with enrolment_total calculated.
        
    Raises:
        FileNotFoundError: If the file/directory does not exist.
        ValueError: If required columns are missing.
        
    Example:
        >>> df = load_enrolment('data/raw/api_data_aadhar_enrolment')
        >>> print(df.columns.tolist())
        ['date', 'state', 'district', 'pincode', 'age_0_5', 
         'age_5_17', 'age_18_greater', 'enrolment_total']
    """
    filepath = Path(filepath) if filepath else config.ENROLMENT_DIR
    
    logger.info(f"Loading enrolment data from {filepath}")
    
    if not filepath.exists():
        # Try legacy single file path
        filepath = config.ENROLMENT_FILE
        if not filepath.exists():
            raise FileNotFoundError(f"Enrolment file/directory not found: {filepath}")
    
    try:
        if filepath.is_dir():
            # Load all CSV files from directory
            csv_files = sorted(filepath.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"No CSV files found in {filepath}")
            logger.info(f"Found {len(csv_files)} CSV files in {filepath}")
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading enrolment file: {e}")
    
    # Validate required columns (using real API data schema)
    required_cols = ['date', 'state', 'district', 'pincode', 
                     'age_0_5', 'age_5_17', 'age_18_greater']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in enrolment data: {missing_cols}")
    
    # Validate schema
    validate_schema(df, config.ENROLMENT_SCHEMA, 'enrolment')
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # Ensure numeric columns
    for col in ['age_0_5', 'age_5_17', 'age_18_greater', 'pincode']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Calculate enrolment_total (sum of all age groups)
    df[config.METRIC_ENROLMENT_TOTAL] = (
        df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    )
    
    # Clean string columns
    df['state'] = df['state'].astype(str).str.strip().str.upper()
    df['district'] = df['district'].astype(str).str.strip().str.upper()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'])
    
    # Check date continuity
    check_date_continuity(df, group_cols=['state', 'district'])
    
    logger.info(f"Loaded {len(df):,} enrolment records")
    
    return df


def load_demographic(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load demographic update dataset and calculate demographic_updates_total.
    
    Args:
        filepath: Path to the demographic update CSV file or directory.
                  Defaults to config.DEMOGRAPHIC_UPDATE_DIR.
    
    Returns:
        pd.DataFrame: Loaded dataframe with demographic_updates_total calculated.
        
    Raises:
        FileNotFoundError: If the file/directory does not exist.
        ValueError: If required columns are missing.
        
    Example:
        >>> df = load_demographic('data/raw/api_data_aadhar_demographic')
        >>> print(df['demographic_updates_total'].sum())
        1234567
    """
    filepath = Path(filepath) if filepath else config.DEMOGRAPHIC_UPDATE_DIR
    
    logger.info(f"Loading demographic update data from {filepath}")
    
    if not filepath.exists():
        # Try legacy single file path
        filepath = config.DEMOGRAPHIC_UPDATE_FILE
        if not filepath.exists():
            raise FileNotFoundError(f"Demographic update file/directory not found: {filepath}")
    
    try:
        if filepath.is_dir():
            # Load all CSV files from directory
            csv_files = sorted(filepath.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"No CSV files found in {filepath}")
            logger.info(f"Found {len(csv_files)} CSV files in {filepath}")
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading demographic file: {e}")
    
    # Validate required columns (using real API data schema)
    required_cols = ['date', 'state', 'district', 'pincode',
                     'demo_age_5_17', 'demo_age_17_']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in demographic data: {missing_cols}")
    
    # Validate schema
    validate_schema(df, config.DEMOGRAPHIC_SCHEMA, 'demographic')
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # Ensure numeric columns
    for col in ['demo_age_5_17', 'demo_age_17_', 'pincode']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Calculate demographic_updates_total
    df[config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL] = (
        df['demo_age_5_17'] + df['demo_age_17_']
    )
    
    # Clean string columns
    df['state'] = df['state'].astype(str).str.strip().str.upper()
    df['district'] = df['district'].astype(str).str.strip().str.upper()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'])
    
    # Check date continuity
    check_date_continuity(df, group_cols=['state', 'district'])
    
    logger.info(f"Loaded {len(df):,} demographic update records")
    
    return df


def load_biometric(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load biometric update dataset and calculate biometric_updates_total.
    
    Args:
        filepath: Path to the biometric update CSV file or directory.
                  Defaults to config.BIOMETRIC_UPDATE_DIR.
    
    Returns:
        pd.DataFrame: Loaded dataframe with biometric_updates_total calculated.
        
    Raises:
        FileNotFoundError: If the file/directory does not exist.
        ValueError: If required columns are missing.
        
    Example:
        >>> df = load_biometric('data/raw/api_data_aadhar_biometric')
        >>> print(df['biometric_updates_total'].sum())
        987654
    """
    filepath = Path(filepath) if filepath else config.BIOMETRIC_UPDATE_DIR
    
    logger.info(f"Loading biometric update data from {filepath}")
    
    if not filepath.exists():
        # Try legacy single file path
        filepath = config.BIOMETRIC_UPDATE_FILE
        if not filepath.exists():
            raise FileNotFoundError(f"Biometric update file/directory not found: {filepath}")
    
    try:
        if filepath.is_dir():
            # Load all CSV files from directory
            csv_files = sorted(filepath.glob('*.csv'))
            if not csv_files:
                raise ValueError(f"No CSV files found in {filepath}")
            logger.info(f"Found {len(csv_files)} CSV files in {filepath}")
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading biometric file: {e}")
    
    # Validate required columns (using real API data schema)
    required_cols = ['date', 'state', 'district', 'pincode',
                     'bio_age_5_17', 'bio_age_17_']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in biometric data: {missing_cols}")
    
    # Validate schema
    validate_schema(df, config.BIOMETRIC_SCHEMA, 'biometric')
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    # Ensure numeric columns
    for col in ['bio_age_5_17', 'bio_age_17_', 'pincode']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Calculate biometric_updates_total
    df[config.METRIC_BIOMETRIC_UPDATES_TOTAL] = (
        df['bio_age_5_17'] + df['bio_age_17_']
    )
    
    # Clean string columns
    df['state'] = df['state'].astype(str).str.strip().str.upper()
    df['district'] = df['district'].astype(str).str.strip().str.upper()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'state', 'district', 'pincode'])
    
    # Check date continuity
    check_date_continuity(df, group_cols=['state', 'district'])
    
    logger.info(f"Loaded {len(df):,} biometric update records")
    
    return df


def merge_datasets(
    enrol_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    bio_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all three datasets using OUTER JOIN on (date, state, district).
    
    The merge aggregates pincode-level data to district level before joining,
    ensuring we have one row per (date, state, district) combination.
    
    Args:
        enrol_df: Enrolment dataframe with enrolment_total.
        demo_df: Demographic update dataframe with demographic_updates_total.
        bio_df: Biometric update dataframe with biometric_updates_total.
    
    Returns:
        pd.DataFrame: Merged dataframe with all columns from all sources.
        
    Raises:
        ValueError: If merge results in unexpected data loss.
        
    Example:
        >>> merged = merge_datasets(enrol_df, demo_df, bio_df)
        >>> print(merged.columns.tolist())
        ['date', 'state', 'district', 'demo_age_5_17', 'demo_age_17_', 
         'enrolment_total', 'age_0_5', 'age_5_17', 'age_18_greater',
         'demographic_updates_total', 'bio_age_5', 'bio_age_17_',
         'biometric_updates_total']
    """
    logger.info("Merging datasets on (date, state, district)")
    
    merge_keys = config.MERGE_KEYS
    
    # Aggregate to district level (sum across pincodes)
    logger.info("Aggregating enrolment data to district level...")
    enrol_agg = enrol_df.groupby(merge_keys).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        config.METRIC_ENROLMENT_TOTAL: 'sum'
    }).reset_index()
    
    logger.info("Aggregating demographic data to district level...")
    demo_agg = demo_df.groupby(merge_keys).agg({
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum',
        config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL: 'sum'
    }).reset_index()
    
    logger.info("Aggregating biometric data to district level...")
    bio_agg = bio_df.groupby(merge_keys).agg({
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum',
        config.METRIC_BIOMETRIC_UPDATES_TOTAL: 'sum'
    }).reset_index()
    
    # Perform outer joins
    logger.info("Performing outer merge...")
    merged = enrol_agg.merge(
        demo_agg, 
        on=merge_keys, 
        how='outer',
        suffixes=('', '_demo')
    )
    
    merged = merged.merge(
        bio_agg,
        on=merge_keys,
        how='outer',
        suffixes=('', '_bio')
    )
    
    # Sort by date, state, district
    merged = merged.sort_values(merge_keys).reset_index(drop=True)
    
    logger.info(f"Merged dataset has {len(merged):,} rows and "
                f"{merged['district'].nunique()} unique districts")
    
    # Validate merge
    if len(merged) < config.MIN_ROWS_AFTER_MERGE:
        logger.warning(f"Merged dataset has fewer than {config.MIN_ROWS_AFTER_MERGE} rows")
    
    return merged


def calculate_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 5 core derived metrics.
    
    Metrics calculated:
        1. total_updates = demographic_updates_total + biometric_updates_total
        2. update_to_enrolment_ratio = total_updates / max(enrolment_total, 1)
        3. enrolment_velocity = diff(enrolment_total).rolling(7).mean()
        4. enrolment_volatility = enrolment_total.rolling(7).std()
        5. Monthly percentage change (for growth analysis)
    
    Args:
        df: Merged dataframe with enrolment_total, demographic_updates_total,
            and biometric_updates_total already calculated.
    
    Returns:
        pd.DataFrame: Input dataframe with new metric columns added.
        
    Example:
        >>> df = calculate_core_metrics(merged_df)
        >>> print(df['update_to_enrolment_ratio'].mean())
        0.85
    """
    logger.info("Calculating core metrics...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Fill NaN values in totals with 0 for calculations
    for col in [config.METRIC_ENROLMENT_TOTAL, 
                config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL,
                config.METRIC_BIOMETRIC_UPDATES_TOTAL]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 1. Total updates (AFTER merge)
    df[config.METRIC_TOTAL_UPDATES] = (
        df.get(config.METRIC_DEMOGRAPHIC_UPDATES_TOTAL, 0) +
        df.get(config.METRIC_BIOMETRIC_UPDATES_TOTAL, 0)
    )
    
    # 2. Update to enrolment ratio (avoid division by zero)
    enrolment_safe = df[config.METRIC_ENROLMENT_TOTAL].replace(0, 1)
    df[config.METRIC_UPDATE_TO_ENROLMENT_RATIO] = (
        df[config.METRIC_TOTAL_UPDATES] / enrolment_safe
    )
    
    # 3 & 4. Calculate velocity and volatility per district
    logger.info("Computing rolling metrics per district...")
    
    # Sort for proper rolling calculations
    df = df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
    
    # Initialize columns
    df[config.METRIC_ENROLMENT_VELOCITY] = np.nan
    df[config.METRIC_ENROLMENT_VOLATILITY] = np.nan
    
    # Group by district and calculate rolling metrics
    for (state, district), group in df.groupby(['state', 'district']):
        idx = group.index
        
        # Enrolment velocity: 7-day rolling mean of daily difference
        diff_values = group[config.METRIC_ENROLMENT_TOTAL].diff()
        velocity = diff_values.rolling(
            window=config.VELOCITY_WINDOW_DAYS, 
            min_periods=1
        ).mean()
        df.loc[idx, config.METRIC_ENROLMENT_VELOCITY] = velocity.values
        
        # Enrolment volatility: 7-day rolling standard deviation
        volatility = group[config.METRIC_ENROLMENT_TOTAL].rolling(
            window=config.VOLATILITY_WINDOW_DAYS,
            min_periods=1
        ).std()
        df.loc[idx, config.METRIC_ENROLMENT_VOLATILITY] = volatility.values
    
    # Add monthly percentage change
    df['month'] = df['date'].dt.to_period('M')
    monthly_enrol = df.groupby(['state', 'district', 'month'])[
        config.METRIC_ENROLMENT_TOTAL
    ].sum().reset_index()
    monthly_enrol['monthly_pct_change'] = monthly_enrol.groupby(
        ['state', 'district']
    )[config.METRIC_ENROLMENT_TOTAL].pct_change() * 100
    
    # Merge back monthly change
    df = df.merge(
        monthly_enrol[['state', 'district', 'month', 'monthly_pct_change']],
        on=['state', 'district', 'month'],
        how='left'
    )
    
    logger.info("Core metrics calculation complete")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill and drop invalid rows.
    
    Strategy:
        1. Forward fill within each district for numeric columns
        2. Drop rows with critical missing values (date, state, district)
        3. Fill remaining NaN with 0 for numeric columns
        4. Log warning if missing value percentage exceeds threshold
    
    Args:
        df: Dataframe with potential missing values.
    
    Returns:
        pd.DataFrame: Cleaned dataframe with missing values handled.
        
    Raises:
        ValueError: If too many critical values are missing.
        
    Example:
        >>> df_clean = handle_missing_values(df)
        >>> print(df_clean.isnull().sum().sum())
        0
    """
    logger.info("Handling missing values...")
    
    original_len = len(df)
    
    # Drop rows with missing critical columns
    critical_cols = ['date', 'state', 'district']
    df = df.dropna(subset=critical_cols)
    
    dropped_critical = original_len - len(df)
    if dropped_critical > 0:
        logger.warning(f"Dropped {dropped_critical} rows with missing critical values")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Forward fill within each district group
    df = df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
    
    for col in numeric_cols:
        df[col] = df.groupby(['state', 'district'])[col].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )
    
    # Check missing value percentage
    total_values = df.size
    missing_values = df.isnull().sum().sum()
    missing_pct = missing_values / total_values if total_values > 0 else 0
    
    if missing_pct > config.MAX_MISSING_VALUE_PERCENT:
        logger.warning(
            f"Missing value percentage ({missing_pct:.1%}) exceeds threshold "
            f"({config.MAX_MISSING_VALUE_PERCENT:.1%})"
        )
    
    # Final fill for any remaining NaN
    df = df.fillna(0)
    
    logger.info(f"Missing value handling complete. Final row count: {len(df):,}")
    
    return df


def save_processed(
    df: pd.DataFrame, 
    filepath: Optional[Union[str, Path]] = None
) -> Path:
    """
    Save processed dataframe to parquet format.
    
    Args:
        df: Processed dataframe to save.
        filepath: Output path. Defaults to config.MERGED_DATA_FILE.
    
    Returns:
        Path: Path where file was saved.
        
    Raises:
        IOError: If file cannot be written.
        
    Example:
        >>> save_path = save_processed(df)
        >>> print(f"Saved to: {save_path}")
        Saved to: data/processed/merged_data.parquet
    """
    filepath = Path(filepath) if filepath else config.MERGED_DATA_FILE
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {filepath}")
    
    try:
        df.to_parquet(filepath, index=False, engine='pyarrow')
    except Exception as e:
        raise IOError(f"Failed to save parquet file: {e}")
    
    logger.info(f"✓ Saved {len(df):,} rows to {filepath}")
    
    return filepath


def load_processed(
    filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load previously processed data from parquet.
    
    Args:
        filepath: Path to parquet file. Defaults to config.MERGED_DATA_FILE.
    
    Returns:
        pd.DataFrame: Loaded processed dataframe.
        
    Raises:
        FileNotFoundError: If file does not exist.
        
    Example:
        >>> df = load_processed()
        >>> print(f"Loaded {len(df)} rows")
    """
    filepath = Path(filepath) if filepath else config.MERGED_DATA_FILE
    
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    logger.info(f"Loading processed data from {filepath}")
    
    df = pd.read_parquet(filepath)
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df):,} rows")
    
    return df


def run_preprocessing_pipeline(
    enrol_path: Optional[Union[str, Path]] = None,
    demo_path: Optional[Union[str, Path]] = None,
    bio_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.
    
    This function orchestrates the entire data preprocessing workflow:
    1. Load all three datasets
    2. Merge on (date, state, district)
    3. Calculate all derived metrics
    4. Handle missing values
    5. Save processed output
    
    Args:
        enrol_path: Path to enrolment CSV.
        demo_path: Path to demographic update CSV.
        bio_path: Path to biometric update CSV.
        output_path: Path for output parquet file.
    
    Returns:
        pd.DataFrame: Final processed dataframe.
        
    Example:
        >>> df = run_preprocessing_pipeline()
        >>> print(f"Pipeline complete: {len(df)} rows, {df['district'].nunique()} districts")
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)
    
    # Load datasets
    enrol_df = load_enrolment(enrol_path)
    demo_df = load_demographic(demo_path)
    bio_df = load_biometric(bio_path)
    
    # Merge datasets
    merged_df = merge_datasets(enrol_df, demo_df, bio_df)
    
    # Calculate derived metrics
    metrics_df = calculate_core_metrics(merged_df)
    
    # Handle missing values
    clean_df = handle_missing_values(metrics_df)
    
    # Save processed data
    save_processed(clean_df, output_path)
    
    logger.info("=" * 60)
    logger.info("Preprocessing pipeline complete!")
    logger.info(f"  - Total rows: {len(clean_df):,}")
    logger.info(f"  - Unique districts: {clean_df['district'].nunique()}")
    logger.info(f"  - Unique states: {clean_df['state'].nunique()}")
    logger.info(f"  - Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    logger.info("=" * 60)
    
    return clean_df
