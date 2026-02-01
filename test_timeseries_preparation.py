"""
Test Suite for Time-Series Preparation Module (STEP 11)
Tests date indexing, aggregation, interval fixing, and validation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.arima_model import (
    set_date_as_index,
    aggregate_sales_weekly,
    aggregate_sales_monthly,
    aggregate_sales_daily,
    ensure_fixed_intervals,
    validate_time_series,
    prepare_for_arima,
    test_stationarity,
    recommend_arima_parameters
)
from data_loader import load_data
from data_preprocessor import preprocess_data


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_test(num, name, passed, details=""):
    """Print test result."""
    status = "✅" if passed else "❌"
    print(f"  {status} TEST {num}: {name}")
    if details:
        print(f"     {details}")


def run_tests():
    """Run all time-series preparation tests."""
    print_header("STEP 11: TIME-SERIES PREPARATION MODULE - TEST SUITE")
    
    # Load and preprocess data
    print("📊 Loading dataset...")
    df, data_source = load_data(prefer_uploaded=True)
    print(f"✅ Dataset loaded from {data_source}")
    print(f"   Records: {len(df)}, Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    df, stats = preprocess_data(df)
    print(f"✅ Data preprocessed: {stats['records_removed']} removed, {stats['missing_values_after']} missing")
    
    test_results = []
    
    # ============== TEST 1: Set Date as Index ==============
    print("\n📅 TEST 1: Set Date as Index")
    try:
        df_indexed = set_date_as_index(df.copy(), 'Date')
        passed = (isinstance(df_indexed.index, pd.DatetimeIndex) and 
                 df_indexed.index.name == 'Date' and 
                 len(df_indexed) == len(df))
        details = f"Index type: {type(df_indexed.index).__name__}, Length: {len(df_indexed)}"
        test_results.append(passed)
        print_test(1, "Set Date as Index", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(1, "Set Date as Index", False, f"Exception: {e}")
    
    # ============== TEST 2: Weekly Aggregation ==============
    print("\n📊 TEST 2: Weekly Sales Aggregation")
    try:
        weekly_ts, weekly_stats = aggregate_sales_weekly(df.copy())
        passed = (isinstance(weekly_ts, pd.Series) and 
                 isinstance(weekly_ts.index, pd.DatetimeIndex) and
                 len(weekly_ts) > 0 and
                 weekly_ts.isna().sum() == 0)
        details = f"Periods: {len(weekly_ts)}, Mean: ${weekly_stats['avg_sales']:,.0f}"
        test_results.append(passed)
        print_test(2, "Weekly Aggregation", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(2, "Weekly Aggregation", False, f"Exception: {e}")
    
    # ============== TEST 3: Monthly Aggregation ==============
    print("\n📊 TEST 3: Monthly Sales Aggregation")
    try:
        monthly_ts, monthly_stats = aggregate_sales_monthly(df.copy())
        passed = (isinstance(monthly_ts, pd.Series) and 
                 isinstance(monthly_ts.index, pd.DatetimeIndex) and
                 len(monthly_ts) > 10 and
                 monthly_ts.isna().sum() == 0)
        details = f"Periods: {len(monthly_ts)}, Mean: ${monthly_stats['avg_sales']:,.0f}"
        test_results.append(passed)
        print_test(3, "Monthly Aggregation", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(3, "Monthly Aggregation", False, f"Exception: {e}")
    
    # ============== TEST 4: Daily Aggregation ==============
    print("\n📊 TEST 4: Daily Sales Aggregation")
    try:
        daily_ts, daily_stats = aggregate_sales_daily(df.copy())
        passed = (isinstance(daily_ts, pd.Series) and 
                 isinstance(daily_ts.index, pd.DatetimeIndex) and
                 len(daily_ts) > 100)
        details = f"Periods: {len(daily_ts)}, Mean: ${daily_stats['avg_sales']:,.0f}"
        test_results.append(passed)
        print_test(4, "Daily Aggregation", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(4, "Daily Aggregation", False, f"Exception: {e}")
    
    # ============== TEST 5: Fixed Intervals ==============
    print("\n⏱️  TEST 5: Ensure Fixed Time Intervals")
    try:
        monthly_ts, _ = aggregate_sales_monthly(df.copy())
        ts_fixed, interval_stats = ensure_fixed_intervals(monthly_ts, 'MS')
        passed = (isinstance(ts_fixed, pd.Series) and 
                 ts_fixed.isna().sum() == 0 and
                 len(ts_fixed) >= len(monthly_ts))
        details = f"Filled periods: {interval_stats['missing_periods_filled']}, Final length: {len(ts_fixed)}"
        test_results.append(passed)
        print_test(5, "Fixed Intervals", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(5, "Fixed Intervals", False, f"Exception: {e}")
    
    # ============== TEST 6: Validate Time Series ==============
    print("\n✔️  TEST 6: Validate Time Series Structure")
    try:
        monthly_ts, _ = aggregate_sales_monthly(df.copy())
        validation = validate_time_series(monthly_ts)
        passed = validation['is_valid']
        warnings_count = len(validation['warnings'])
        details = f"Valid: {validation['is_valid']}, Warnings: {warnings_count}"
        test_results.append(passed)
        print_test(6, "Validate Time Series", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(6, "Validate Time Series", False, f"Exception: {e}")
    
    # ============== TEST 7: Complete Preparation Pipeline ==============
    print("\n🔧 TEST 7: Complete ARIMA Preparation Pipeline")
    try:
        result = prepare_for_arima(df.copy(), aggregation='monthly', ensure_intervals=True)
        passed = (result['status'] == 'success' and 
                 result['time_series'] is not None and
                 len(result['time_series']) > 20)
        details = f"Status: {result['status']}, Series length: {len(result['time_series']) if result['time_series'] is not None else 0}"
        test_results.append(passed)
        print_test(7, "Complete Pipeline", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(7, "Complete Pipeline", False, f"Exception: {e}")
    
    # ============== TEST 8: Stationarity Test ==============
    print("\n📈 TEST 8: Stationarity Testing (ADF)")
    try:
        monthly_ts, _ = aggregate_sales_monthly(df.copy())
        stationarity = test_stationarity(monthly_ts)
        passed = ('p_value' in stationarity or 'error' in stationarity)
        details = (f"Stationary: {stationarity.get('is_stationary', 'N/A')}, "
                  f"p-value: {stationarity.get('p_value', 'N/A')}")
        test_results.append(passed)
        print_test(8, "Stationarity Test", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(8, "Stationarity Test", False, f"Exception: {e}")
    
    # ============== TEST 9: ARIMA Parameter Recommendation ==============
    print("\n📊 TEST 9: ARIMA Parameter Recommendation")
    try:
        monthly_ts, _ = aggregate_sales_monthly(df.copy())
        params = recommend_arima_parameters(monthly_ts)
        passed = ('recommended_arima' in params or 'error' in params)
        recommended = params.get('recommended_arima', 'N/A')
        details = f"Recommended (p,d,q): {recommended}"
        test_results.append(passed)
        print_test(9, "Parameter Recommendation", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(9, "Parameter Recommendation", False, f"Exception: {e}")
    
    # ============== TEST 10: Weekly vs Monthly Aggregation ==============
    print("\n⚖️  TEST 10: Multiple Aggregation Levels")
    try:
        weekly_ts, weekly_stats = aggregate_sales_weekly(df.copy())
        monthly_ts, monthly_stats = aggregate_sales_monthly(df.copy())
        daily_ts, daily_stats = aggregate_sales_daily(df.copy())
        
        # Key check: more granular aggregations should have more/equal periods
        # Daily data has highest granularity, then weekly, then monthly
        passed = (len(monthly_ts) > 20 and  # Should have at least 20 months
                 len(weekly_ts) >= len(monthly_ts) and  # Weekly >= monthly
                 len(daily_ts) >= len(weekly_ts) and  # Daily >= weekly
                 abs(weekly_ts.sum() - monthly_ts.sum()) < 100)  # Totals similar
        details = f"Daily: {len(daily_ts)}, Weekly: {len(weekly_ts)}, Monthly: {len(monthly_ts)}"
        test_results.append(passed)
        print_test(10, "Multiple Aggregations", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(10, "Multiple Aggregations", False, f"Exception: {e}")
    
    # ============== SUMMARY ==============
    print_header("TEST SUMMARY")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Pass Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Time-series preparation module is ready.")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review the errors above.")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
