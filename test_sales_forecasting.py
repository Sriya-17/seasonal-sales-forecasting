"""
Test Suite for STEP 14: Sales Forecasting Module
Tests all forecasting functions with real Walmart data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, '/Users/sriyakaadhuluri/Documents/B.Tech/3rd_year/3-2/DEA/DEA-Project-1/seasonal_sales_forecasting')

from data_loader import load_data
from data_preprocessor import preprocess_data
from models.arima_model import (
    set_date_as_index,
    aggregate_sales_monthly,
    fit_arima,
    generate_forecast,
    forecast_n_months,
    forecast_n_weeks,
    forecast_custom_horizon,
    forecast_with_scenario,
    generate_forecast_summary
)


def setup_test_data():
    """Load and prepare test data"""
    print("📊 Loading test data...")
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    print(f"✅ Test data ready: {len(ts)} months")
    return ts


# ═════════════════════════════════════════════════════════════════════

def test_forecast_n_months():
    """TEST 1: Forecast N months ahead"""
    print("\n" + "="*70)
    print("TEST 1: Forecast N Months")
    print("="*70)
    
    try:
        ts = setup_test_data()
        
        # Fit ARIMA model
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        # Forecast 12 months
        result = forecast_n_months(ts, model, n_months=12, confidence=0.95)
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ PASSED: Forecast N Months")
        print(f"   Periods: {result['n_months']} months")
        print(f"   Mean forecast: ${result['mean_forecast']:,.0f}")
        print(f"   Range: ${result['forecast_range'][0]:,.0f} - ${result['forecast_range'][1]:,.0f}")
        print(f"   Confidence: {result['confidence']*100:.0f}%")
        print(f"   Forecast count: {len(result['forecast_values'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_forecast_n_weeks():
    """TEST 2: Forecast N weeks ahead"""
    print("\n" + "="*70)
    print("TEST 2: Forecast N Weeks")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        # Forecast 26 weeks
        result = forecast_n_weeks(ts, model, n_weeks=26, confidence=0.95)
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ PASSED: Forecast N Weeks")
        print(f"   Periods: {result['n_weeks']} weeks")
        print(f"   Mean forecast: ${result['mean_forecast']:,.0f}")
        print(f"   Range: ${result['forecast_range'][0]:,.0f} - ${result['forecast_range'][1]:,.0f}")
        print(f"   Std dev: ${result['std_forecast']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_forecast_custom_horizon():
    """TEST 3: Forecast with custom horizon"""
    print("\n" + "="*70)
    print("TEST 3: Forecast Custom Horizon")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        # Test different horizon types
        result_months = forecast_custom_horizon(
            ts, model, steps=12, horizon_type='months'
        )
        result_weeks = forecast_custom_horizon(
            ts, model, steps=26, horizon_type='weeks'
        )
        result_periods = forecast_custom_horizon(
            ts, model, steps=12, horizon_type='periods'
        )
        
        if not all([r.get('success') for r in [result_months, result_weeks, result_periods]]):
            print(f"❌ FAILED: One or more forecasts failed")
            return False
        
        print(f"✅ PASSED: Forecast Custom Horizon")
        print(f"   Months - Mean: ${result_months['statistics']['mean']:,.0f}")
        print(f"   Weeks - Mean: ${result_weeks['statistics']['mean']:,.0f}")
        print(f"   Periods - Mean: ${result_periods['statistics']['mean']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_forecast_summary_format():
    """TEST 4: Forecast summary vs full format"""
    print("\n" + "="*70)
    print("TEST 4: Forecast Summary and Full Format")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        # Get summary format
        summary = forecast_custom_horizon(
            ts, model, steps=12, return_format='summary'
        )
        
        # Get full format
        full = forecast_custom_horizon(
            ts, model, steps=12, return_format='full'
        )
        
        if not (summary.get('success') and full.get('success')):
            print(f"❌ FAILED: Format conversion failed")
            return False
        
        # Verify summary has statistics
        has_summary = 'summary' in summary
        has_full = 'forecast_df' in full
        
        if not (has_summary and has_full):
            print(f"❌ FAILED: Missing expected keys")
            return False
        
        print(f"✅ PASSED: Forecast Summary and Full Format")
        print(f"   Summary keys: {list(summary.get('summary', {}).keys())}")
        print(f"   Full format has {len(full['forecast_values'])} forecast values")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_scenario_baseline():
    """TEST 5: Baseline scenario forecast"""
    print("\n" + "="*70)
    print("TEST 5: Scenario - Baseline")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        result = forecast_with_scenario(
            ts, model, n_periods=12, scenario='baseline'
        )
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ PASSED: Scenario - Baseline")
        print(f"   Scenario: {result['scenario']}")
        print(f"   Mean forecast: ${result['mean_forecast']:,.0f}")
        print(f"   Periods: {result['n_periods']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_scenario_pessimistic():
    """TEST 6: Pessimistic scenario forecast"""
    print("\n" + "="*70)
    print("TEST 6: Scenario - Pessimistic")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        result = forecast_with_scenario(
            ts, model, n_periods=12, scenario='pessimistic', growth_rate=-0.05
        )
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ PASSED: Scenario - Pessimistic")
        print(f"   Scenario: {result['scenario']}")
        print(f"   Growth rate: {result['growth_rate']*100:.1f}%")
        print(f"   Mean forecast: ${result['mean_forecast']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_scenario_optimistic():
    """TEST 7: Optimistic scenario forecast"""
    print("\n" + "="*70)
    print("TEST 7: Scenario - Optimistic")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        result = forecast_with_scenario(
            ts, model, n_periods=12, scenario='optimistic', growth_rate=0.05
        )
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ PASSED: Scenario - Optimistic")
        print(f"   Scenario: {result['scenario']}")
        print(f"   Growth rate: {result['growth_rate']*100:.1f}%")
        print(f"   Mean forecast: ${result['mean_forecast']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_complete_forecast_summary():
    """TEST 8: Complete forecast summary with scenarios"""
    print("\n" + "="*70)
    print("TEST 8: Complete Forecast Summary with Scenarios")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        result = generate_forecast_summary(
            ts, model, n_periods=12, include_scenarios=True
        )
        
        if not result.get('success'):
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return False
        
        # Check structure
        has_baseline = 'baseline' in result
        has_scenarios = 'scenarios' in result
        
        if not (has_baseline and has_scenarios):
            print(f"❌ FAILED: Missing structure")
            return False
        
        scenarios = result.get('scenarios', {})
        scenario_count = len(scenarios)
        
        print(f"✅ PASSED: Complete Forecast Summary")
        print(f"   Baseline mean: ${result['baseline']['statistics']['mean']:,.0f}")
        print(f"   Scenarios included: {scenario_count}")
        if 'pessimistic' in scenarios:
            print(f"   Pessimistic mean: ${scenarios['pessimistic']['mean_forecast']:,.0f}")
        if 'optimistic' in scenarios:
            print(f"   Optimistic mean: ${scenarios['optimistic']['mean_forecast']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_confidence_interval_validation():
    """TEST 9: Validate confidence intervals across forecasts"""
    print("\n" + "="*70)
    print("TEST 9: Confidence Interval Validation")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        result = forecast_n_months(ts, model, n_months=12)
        
        if not result.get('success'):
            print(f"❌ FAILED: Forecast failed")
            return False
        
        # Validate CI bounds
        forecasts = np.array(result['forecast_values'])
        lower_ci = np.array(result['lower_ci'])
        upper_ci = np.array(result['upper_ci'])
        
        # Check: lower_ci < forecast < upper_ci
        bounds_valid = np.all((lower_ci <= forecasts) & (forecasts <= upper_ci))
        
        # Check: CI widths increase over time
        ci_widths = upper_ci - lower_ci
        widths_increasing = all(ci_widths[i] <= ci_widths[i+1] for i in range(len(ci_widths)-1))
        
        if not (bounds_valid and widths_increasing):
            print(f"❌ FAILED: CI validation failed")
            return False
        
        print(f"✅ PASSED: Confidence Interval Validation")
        print(f"   Bounds valid: {bounds_valid}")
        print(f"   Widths increasing: {widths_increasing}")
        print(f"   CI width range: ${ci_widths[0]:,.0f} - ${ci_widths[-1]:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def test_forecast_forecast_consistency():
    """TEST 10: Consistency between different forecast functions"""
    print("\n" + "="*70)
    print("TEST 10: Forecast Functions Consistency")
    print("="*70)
    
    try:
        ts = setup_test_data()
        model = fit_arima(ts, order=(1, 0, 1))
        
        if not model.get('success'):
            print(f"❌ FAILED: Model fitting failed")
            return False
        
        # Get forecasts from different functions
        months_result = forecast_n_months(ts, model, n_months=12)
        custom_result = forecast_custom_horizon(
            ts, model, steps=12, horizon_type='months'
        )
        summary_result = generate_forecast_summary(ts, model, n_periods=12)
        
        if not all([months_result.get('success'), custom_result.get('success'), 
                   summary_result.get('success')]):
            print(f"❌ FAILED: One or more forecasts failed")
            return False
        
        # Compare means (should be very similar)
        months_mean = months_result['mean_forecast']
        custom_mean = custom_result['statistics']['mean']
        baseline_mean = summary_result['baseline']['statistics']['mean']
        
        # Allow small floating point differences
        tolerance = 100  # $100 difference acceptable
        consistent = (
            abs(months_mean - custom_mean) < tolerance and
            abs(months_mean - baseline_mean) < tolerance
        )
        
        if not consistent:
            print(f"❌ FAILED: Forecast inconsistency detected")
            return False
        
        print(f"✅ PASSED: Forecast Functions Consistency")
        print(f"   forecast_n_months mean: ${months_mean:,.0f}")
        print(f"   forecast_custom_horizon mean: ${custom_mean:,.0f}")
        print(f"   generate_forecast_summary mean: ${baseline_mean:,.0f}")
        print(f"   Difference: ${abs(months_mean - custom_mean):,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


# ═════════════════════════════════════════════════════════════════════

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STEP 14: SALES FORECASTING MODULE - TEST SUITE")
    print("="*70)
    
    tests = [
        test_forecast_n_months,
        test_forecast_n_weeks,
        test_forecast_custom_horizon,
        test_forecast_summary_format,
        test_scenario_baseline,
        test_scenario_pessimistic,
        test_scenario_optimistic,
        test_complete_forecast_summary,
        test_confidence_interval_validation,
        test_forecast_forecast_consistency
    ]
    
    results = []
    for i, test in enumerate(tests, 1):
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"❌ TEST ERROR: {str(e)}")
            results.append((test.__name__, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print(f"Pass Rate: {(passed_count/total_count)*100:.1f}%")
    print("="*70)
    
    return passed_count == total_count


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
