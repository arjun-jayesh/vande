#!/usr/bin/env python3
"""Data Pipeline Test Script"""

import sys
sys.path.insert(0, '.')

from src.preprocessing import (
    load_enrolment, load_demographic, load_biometric, 
    merge_datasets, calculate_core_metrics, handle_missing_values, save_processed
)

print('Test 1: Loading enrolment data...')
enrol_df = load_enrolment()
print(f'  Loaded {len(enrol_df):,} enrolment records')
print(f'  Columns: {enrol_df.columns.tolist()}')

print('Test 2: Loading demographic data...')
demo_df = load_demographic()
print(f'  Loaded {len(demo_df):,} demographic records')
print(f'  Columns: {demo_df.columns.tolist()}')

print('Test 3: Loading biometric data...')
bio_df = load_biometric()
print(f'  Loaded {len(bio_df):,} biometric records')
print(f'  Columns: {bio_df.columns.tolist()}')

print('Test 4: Merging datasets...')
merged = merge_datasets(enrol_df, demo_df, bio_df)
print(f'  Merged: {len(merged):,} rows')
print(f'  Columns: {merged.columns.tolist()[:10]}...')

print('Test 5: Calculating core metrics...')
metrics_df = calculate_core_metrics(merged)
print('  Metrics added - checking 5 required columns:')
required = ['enrolment_total', 'total_updates', 'update_to_enrolment_ratio', 'enrolment_velocity', 'enrolment_volatility']
for col in required:
    if col in metrics_df.columns:
        print(f'    - {col} present')
    else:
        print(f'    X {col} MISSING')

print('Test 6: Handling missing values...')
clean_df = handle_missing_values(metrics_df)
print(f'  Clean data: {len(clean_df):,} rows, nulls: {clean_df.isnull().sum().sum()}')

print('Test 7: Saving processed data...')
save_processed(clean_df)
print('  Saved to data/processed/merged_data.parquet')

print('')
print('=== DATA PIPELINE TEST COMPLETE ===')
print(f'Final shape: {clean_df.shape}')
print(f'Unique districts: {clean_df["district"].nunique()}')
print(f'Unique states: {clean_df["state"].nunique()}')
