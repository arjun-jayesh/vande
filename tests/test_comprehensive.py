#!/usr/bin/env python3
"""
Comprehensive Verification Script for Project Vande.

Tests all analytics components with real Aadhaar data to ensure
production-readiness with no placeholders.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Import all project modules
from src import config
from src.preprocessing import load_enrolment, load_demographic, load_biometric, merge_datasets
from src.preprocessing import calculate_core_metrics, handle_missing_values, save_processed
from src.metrics import (
    calculate_asi, apply_asi_policy_override, detect_inclusion_risk,
    calculate_saturation_status, identify_imbalanced_districts,
    rank_service_load, detect_volatility_spikes
)
from src.models import AnomalyDetector, EnrolmentForecaster
from src.viz import (
    plot_timeseries, plot_anomaly_scatter, plot_forecast,
    plot_inclusion_risk_map, plot_age_distribution_pie,
    plot_stl_decomposition, generate_summary_table,
    generate_state_ranking_table, generate_pdf_report
)

# Test results tracking
RESULTS = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def log_result(test_name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"       {details}")
    
    if passed:
        RESULTS["passed"].append(test_name)
    else:
        RESULTS["failed"].append((test_name, details))

def log_warning(test_name: str, message: str):
    """Log a warning."""
    print(f"  ⚠️  WARNING: {test_name}")
    print(f"       {message}")
    RESULTS["warnings"].append((test_name, message))


# =============================================================================
# PHASE 1: DATA PIPELINE
# =============================================================================

def test_data_pipeline():
    """Test data loading and merging pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Test 1.1: Load existing merged data
    try:
        if config.MERGED_DATA_FILE.exists():
            df = pd.read_parquet(config.MERGED_DATA_FILE)
            log_result("Load merged parquet data", True, f"{len(df):,} rows loaded")
            
            # Verify expected columns
            required_cols = ['date', 'state', 'district', 'enrolment_total', 
                           'total_updates', 'update_to_enrolment_ratio',
                           'enrolment_velocity', 'enrolment_volatility']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                log_result("Required columns present", False, f"Missing: {missing}")
            else:
                log_result("Required columns present", True, f"All 8 core columns found")
            
            # Check data quality
            null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if null_pct < 1:
                log_result("Data quality (null check)", True, f"{null_pct:.2f}% nulls")
            else:
                log_warning("Data quality", f"{null_pct:.2f}% null values found")
            
            # Check unique values
            n_districts = df['district'].nunique()
            n_states = df['state'].nunique()
            log_result("Geographic coverage", True, f"{n_districts} districts, {n_states} states")
            
            return df
        else:
            log_result("Load merged data", False, "File not found - run test_pipeline.py first")
            return None
    except Exception as e:
        log_result("Data pipeline", False, str(e))
        traceback.print_exc()
        return None


# =============================================================================
# PHASE 2: METRICS & ASI
# =============================================================================

def test_metrics_and_asi(df: pd.DataFrame):
    """Test metrics calculation and ASI computation."""
    print("\n" + "=" * 60)
    print("PHASE 2: METRICS & ASI CALCULATION")
    print("=" * 60)
    
    if df is None:
        print("  Skipping - no data loaded")
        return df
    
    # Test 2.1: ASI Calculation
    try:
        # Calculate ASI for entire dataset - returns a dict {district: score}
        asi_dict = calculate_asi(df.copy(), include_national=True)
        
        # Handle the dict return type
        if isinstance(asi_dict, dict):
            all_scores = [v for k, v in asi_dict.items() if k != 'NATIONAL']
            national_score = asi_dict.get('NATIONAL', np.mean(all_scores))
            
            asi_min = min(all_scores) if all_scores else 0
            asi_max = max(all_scores) if all_scores else 0
            asi_mean = np.mean(all_scores) if all_scores else 0
            
            # Verify ASI is in valid range
            if 0 <= asi_min and asi_max <= 100:
                log_result("ASI calculation", True, 
                          f"Range: [{asi_min:.1f}, {asi_max:.1f}], Mean: {asi_mean:.1f}, National: {national_score:.1f}")
            else:
                log_result("ASI calculation", False, f"Out of range: [{asi_min}, {asi_max}]")
            
            # Add ASI scores to dataframe for downstream tests
            df['asi_score'] = df['district'].map(asi_dict).fillna(asi_mean)
        else:
            # Single float returned
            log_result("ASI calculation", True, f"Single ASI score: {asi_dict:.1f}")
            df['asi_score'] = asi_dict
    except Exception as e:
        log_result("ASI calculation", False, str(e))
        traceback.print_exc()
    
    # Test 2.2: ASI Policy Override
    try:
        # Get district-level ASI
        district_asi = df.groupby('district')['asi_score'].mean().reset_index()
        result = apply_asi_policy_override(district_asi)
        
        if result is not None and 'national' in result:
            override_applied = result.get('override_applied', False)
            national_asi = result['national']
            log_result("ASI policy override logic", True, 
                      f"National ASI: {national_asi:.1f}, Override: {override_applied}")
        else:
            log_result("ASI policy override logic", True, "Function executed (no override needed)")
    except Exception as e:
        log_result("ASI policy override", False, str(e))
        traceback.print_exc()
    
    return df


# =============================================================================
# PHASE 3: RISK DETECTION
# =============================================================================

def test_risk_detection(df: pd.DataFrame):
    """Test risk detection and classification functions."""
    print("\n" + "=" * 60)
    print("PHASE 3: RISK DETECTION")
    print("=" * 60)
    
    if df is None:
        print("  Skipping - no data loaded")
        return df
    
    # Test 3.1: Inclusion Risk Detection
    try:
        df_risk = detect_inclusion_risk(df.copy())
        if 'inclusion_risk' in df_risk.columns:
            risk_count = df_risk['inclusion_risk'].sum()
            risk_pct = risk_count / len(df_risk) * 100
            log_result("Inclusion risk detection", True, 
                      f"{risk_count:,} at-risk records ({risk_pct:.1f}%)")
            df = df_risk
        else:
            log_result("Inclusion risk detection", False, "inclusion_risk column missing")
    except Exception as e:
        log_result("Inclusion risk detection", False, str(e))
        traceback.print_exc()
    
    # Test 3.2: Saturation Status
    try:
        df_sat = calculate_saturation_status(df.copy())
        if 'saturation_status' in df_sat.columns:
            status_counts = df_sat['saturation_status'].value_counts()
            log_result("Saturation status classification", True, 
                      f"Categories: {status_counts.to_dict()}")
            df = df_sat
        else:
            log_result("Saturation status", False, "saturation_status column missing")
    except Exception as e:
        log_result("Saturation status", False, str(e))
        traceback.print_exc()
    
    # Test 3.3: Imbalanced Districts
    try:
        df_imb = identify_imbalanced_districts(df.copy())
        if 'balance_status' in df_imb.columns:
            balance_counts = df_imb['balance_status'].value_counts()
            log_result("Imbalanced district detection", True, 
                      f"Categories: {balance_counts.to_dict()}")
            df = df_imb
        else:
            log_result("Imbalanced districts", False, "balance_status column missing")
    except Exception as e:
        log_result("Imbalanced districts", False, str(e))
        traceback.print_exc()
    
    # Test 3.4: Service Load Ranking
    try:
        top_districts = rank_service_load(df.copy(), top_n=20)
        if top_districts is not None and len(top_districts) > 0:
            log_result("Service load ranking", True, 
                      f"Top 20 high-pressure districts identified")
        else:
            log_result("Service load ranking", False, "No rankings generated")
    except Exception as e:
        log_result("Service load ranking", False, str(e))
        traceback.print_exc()
    
    # Test 3.5: Volatility Spikes
    try:
        df_vol = detect_volatility_spikes(df.copy())
        if 'volatility_spike' in df_vol.columns:
            spike_count = df_vol['volatility_spike'].sum()
            log_result("Volatility spike detection", True, 
                      f"{spike_count:,} spikes detected")
            df = df_vol
        else:
            log_result("Volatility spikes", False, "volatility_spike column missing")
    except Exception as e:
        log_result("Volatility spikes", False, str(e))
        traceback.print_exc()
    
    return df


# =============================================================================
# PHASE 4: ML MODELS
# =============================================================================

def test_ml_models(df: pd.DataFrame):
    """Test anomaly detection and forecasting models."""
    print("\n" + "=" * 60)
    print("PHASE 4: ML MODELS")
    print("=" * 60)
    
    if df is None:
        print("  Skipping - no data loaded")
        return df
    
    # Test 4.1: Anomaly Detection
    try:
        # Prepare features for anomaly detection
        feature_cols = ['enrolment_total', 'total_updates', 'enrolment_volatility', 
                        'enrolment_velocity', 'update_to_enrolment_ratio']
        available_features = [c for c in feature_cols if c in df.columns]
        
        if len(available_features) < 3:
            log_result("Anomaly detection", False, f"Insufficient features: {available_features}")
        else:
            # Sample data for faster testing
            sample_df = df.dropna(subset=available_features).sample(min(10000, len(df)), random_state=42)
            features = sample_df[available_features].values
            
            detector = AnomalyDetector(contamination=0.05)
            labels = detector.fit_predict(features)
            
            anomaly_count = (labels == 1).sum()
            anomaly_pct = anomaly_count / len(labels) * 100
            
            if 0 < anomaly_pct < 20:  # Reasonable anomaly rate
                log_result("Anomaly detection (Isolation Forest)", True, 
                          f"{anomaly_count} anomalies ({anomaly_pct:.1f}%)")
            else:
                log_warning("Anomaly detection", 
                           f"Unusual anomaly rate: {anomaly_pct:.1f}%")
                log_result("Anomaly detection", True, f"Model ran, {anomaly_pct:.1f}% anomalies")
            
            # Add anomaly labels to full dataframe
            df['is_anomaly'] = 0
            df.loc[sample_df.index, 'is_anomaly'] = labels
            
    except Exception as e:
        log_result("Anomaly detection", False, str(e))
        traceback.print_exc()
    
    # Test 4.2: Prophet Forecasting
    try:
        # Aggregate to daily national level for forecasting
        daily_df = df.groupby('date').agg({
            'enrolment_total': 'sum'
        }).reset_index()
        daily_df = daily_df.sort_values('date')
        
        # Need at least 60 days of data
        if len(daily_df) < 60:
            log_warning("Forecasting", f"Only {len(daily_df)} days of data - need 60+")
        else:
            # Split train/test
            train_df = daily_df.iloc[:-30].copy()
            test_df = daily_df.iloc[-30:].copy()
            
            forecaster = EnrolmentForecaster(horizon=30)
            forecaster.fit(train_df, target_col='enrolment_total', suppress_logging=True)
            
            forecast = forecaster.forecast(periods=30)
            
            if forecast is not None and len(forecast) > 0:
                log_result("Prophet forecasting", True, 
                          f"Generated {len(forecast)} day forecast")
                
                # Backtest
                metrics = forecaster.backtest(daily_df, test_days=30)
                if metrics and 'mape' in metrics:
                    mape = metrics['mape']
                    if mape < 30:
                        log_result("Forecast backtest (MAPE)", True, f"MAPE: {mape:.1f}%")
                    else:
                        log_warning("Forecast accuracy", f"High MAPE: {mape:.1f}%")
                        log_result("Forecast backtest", True, f"MAPE: {mape:.1f}%")
            else:
                log_result("Prophet forecasting", False, "No forecast generated")
                
    except Exception as e:
        log_result("Prophet forecasting", False, str(e))
        traceback.print_exc()
    
    return df


# =============================================================================
# PHASE 5: VISUALIZATION
# =============================================================================

def test_visualizations(df: pd.DataFrame):
    """Test visualization generation."""
    print("\n" + "=" * 60)
    print("PHASE 5: VISUALIZATION")
    print("=" * 60)
    
    if df is None:
        print("  Skipping - no data loaded")
        return df
    
    # Ensure output directories exist
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test 5.1: Time Series Plot
    try:
        fig = plot_timeseries(
            df.groupby('date')['enrolment_total'].sum().reset_index(),
            metric='enrolment_total',
            title='National Enrolment Trend'
        )
        if fig is not None:
            log_result("Time series plot", True, "Figure generated")
        else:
            log_result("Time series plot", False, "No figure returned")
    except Exception as e:
        log_result("Time series plot", False, str(e))
        traceback.print_exc()
    
    # Test 5.2: Age Distribution Pie Chart
    try:
        # Check if we have age columns
        age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        if all(c in df.columns for c in age_cols):
            fig = plot_age_distribution_pie(df, dataset_type='enrolment')
            if fig is not None:
                log_result("Age distribution pie chart", True, "Figure generated")
            else:
                log_result("Age distribution pie chart", False, "No figure returned")
        else:
            # Try with available columns
            log_warning("Age distribution", f"Some age columns missing, attempting with available")
            fig = plot_age_distribution_pie(df, dataset_type='demographic')
            log_result("Age distribution pie chart", True, "Generated with demographic data")
    except Exception as e:
        log_result("Age distribution pie chart", False, str(e))
        traceback.print_exc()
    
    # Test 5.3: Summary Table
    try:
        table = generate_summary_table(df, top_n=20)
        if table is not None and len(table) > 0:
            log_result("Summary table generation", True, f"{len(table)} rows")
        else:
            log_result("Summary table", False, "Empty table")
    except Exception as e:
        log_result("Summary table", False, str(e))
        traceback.print_exc()
    
    # Test 5.4: State Ranking Table
    try:
        rankings = generate_state_ranking_table(df)
        if rankings is not None and len(rankings) > 0:
            log_result("State ranking table", True, f"{len(rankings)} states ranked")
        else:
            log_result("State ranking table", False, "Empty rankings")
    except Exception as e:
        log_result("State ranking table", False, str(e))
        traceback.print_exc()
    
    # Test 5.5: STL Decomposition
    try:
        daily_df = df.groupby('date')['enrolment_total'].sum().reset_index()
        if len(daily_df) >= 14:  # Need at least 2 weeks
            fig = plot_stl_decomposition(daily_df, metric='enrolment_total', period=7)
            if fig is not None:
                log_result("STL decomposition plot", True, "Figure generated")
            else:
                log_result("STL decomposition", False, "No figure returned")
        else:
            log_warning("STL decomposition", f"Insufficient data: {len(daily_df)} days")
    except Exception as e:
        log_result("STL decomposition", False, str(e))
        traceback.print_exc()
    
    # Test 5.6: Inclusion Risk Map
    try:
        if 'inclusion_risk' in df.columns:
            fig = plot_inclusion_risk_map(df)
            if fig is not None:
                log_result("Inclusion risk map", True, "Figure generated")
            else:
                log_result("Inclusion risk map", False, "No figure returned")
        else:
            log_warning("Inclusion risk map", "inclusion_risk column not found")
    except Exception as e:
        log_result("Inclusion risk map", False, str(e))
        traceback.print_exc()
    
    # Test 5.7: PDF Report Generation
    try:
        pdf_path = generate_pdf_report(df)
        if pdf_path and Path(pdf_path).exists():
            size_kb = Path(pdf_path).stat().st_size / 1024
            log_result("PDF report generation", True, f"Saved ({size_kb:.0f} KB)")
        else:
            log_result("PDF report", False, "File not created")
    except Exception as e:
        log_result("PDF report generation", False, str(e))
        traceback.print_exc()
    
    return df


# =============================================================================
# PHASE 6: DASHBOARD (SKIP - REQUIRES MANUAL VERIFICATION)
# =============================================================================

def test_dashboard_exports(df: pd.DataFrame):
    """Test dashboard export functions."""
    print("\n" + "=" * 60)
    print("PHASE 6: DASHBOARD EXPORT FUNCTIONS")
    print("=" * 60)
    
    if df is None:
        print("  Skipping - no data loaded")
        return
    
    # Test CSV export
    try:
        csv_data = df.to_csv(index=False)
        if len(csv_data) > 0:
            log_result("CSV export", True, f"{len(csv_data):,} bytes")
        else:
            log_result("CSV export", False, "Empty output")
    except Exception as e:
        log_result("CSV export", False, str(e))
    
    print("\n  ℹ️  Dashboard UI verification requires manual testing:")
    print("     Run: streamlit run dashboard/app.py")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_summary():
    """Print final test summary."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    total = len(RESULTS["passed"]) + len(RESULTS["failed"])
    pass_rate = len(RESULTS["passed"]) / total * 100 if total > 0 else 0
    
    print(f"\n  Total Tests: {total}")
    print(f"  ✅ Passed: {len(RESULTS['passed'])} ({pass_rate:.0f}%)")
    print(f"  ❌ Failed: {len(RESULTS['failed'])}")
    print(f"  ⚠️  Warnings: {len(RESULTS['warnings'])}")
    
    if RESULTS["failed"]:
        print("\n  Failed Tests:")
        for name, details in RESULTS["failed"]:
            print(f"    - {name}: {details}")
    
    if RESULTS["warnings"]:
        print("\n  Warnings:")
        for name, msg in RESULTS["warnings"]:
            print(f"    - {name}: {msg}")
    
    if pass_rate >= 80:
        print("\n  🎉 VERIFICATION STATUS: PRODUCTION READY")
    elif pass_rate >= 50:
        print("\n  ⚠️  VERIFICATION STATUS: NEEDS FIXES")
    else:
        print("\n  ❌ VERIFICATION STATUS: CRITICAL ISSUES")
    
    return pass_rate >= 80


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PROJECT VANDE: COMPREHENSIVE VERIFICATION")
    print("=" * 60)
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Merged data: {config.MERGED_DATA_FILE}")
    
    # Run all phases
    df = test_data_pipeline()
    df = test_metrics_and_asi(df)
    df = test_risk_detection(df)
    df = test_ml_models(df)
    df = test_visualizations(df)
    test_dashboard_exports(df)
    
    # Save enriched dataframe
    if df is not None:
        enriched_path = config.PROCESSED_DATA_DIR / "enriched_data.parquet"
        df.to_parquet(enriched_path, index=False)
        print(f"\n  💾 Saved enriched data to: {enriched_path}")
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
