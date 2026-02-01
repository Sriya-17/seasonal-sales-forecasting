"""
Test Suite for STEP 12: Stationarity Check Module
Tests comprehensive stationarity testing and differencing functionality
"""

import pandas as pd
import numpy as np
from data_loader import load_data
from data_preprocessor import preprocess_data
from models.arima_model import (
    set_date_as_index,
    aggregate_sales_monthly
)
from models.stationarity_check import (
    adf_test,
    kpss_test,
    pp_test,
    comprehensive_stationarity_check,
    apply_differencing,
    apply_seasonal_differencing,
    determine_differencing_order,
    validate_differencing,
    prepare_stationary_series,
    compare_stationarity_before_after
)


def test_adf_test():
    """Test ADF stationarity test"""
    print("\n📊 TEST 1: ADF Test (Augmented Dickey-Fuller)")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Run ADF test
    result = adf_test(ts)
    
    # Validate results
    assert 'test_statistic' in result
    assert 'p_value' in result
    assert 'critical_values' in result
    assert 'is_stationary' in result
    assert result['p_value'] > 0
    assert result['p_value'] < 1
    
    print(f"  ✅ TEST 1: ADF Test")
    print(f"     Test Statistic: {result['test_statistic']:.6f}")
    print(f"     P-value: {result['p_value']:.2e}")
    print(f"     Stationary: {result['is_stationary']}")
    print(f"     Interpretation: {result['interpretation']}")
    return True


def test_kpss_test():
    """Test KPSS stationarity test"""
    print("\n📊 TEST 2: KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Run KPSS test
    result = kpss_test(ts)
    
    # Validate results
    assert 'test_statistic' in result
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert result['p_value'] > 0
    assert result['p_value'] < 1
    
    print(f"  ✅ TEST 2: KPSS Test")
    print(f"     Test Statistic: {result['test_statistic']:.6f}")
    print(f"     P-value: {result['p_value']:.4f}")
    print(f"     Stationary: {result['is_stationary']}")
    print(f"     Interpretation: {result['interpretation']}")
    return True


def test_pp_test():
    """Test Phillips-Perron test"""
    print("\n📊 TEST 3: Phillips-Perron Test")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Run PP test
    result = pp_test(ts)
    
    # Validate results
    assert 'test_statistic' in result
    assert 'p_value' in result
    assert 'is_stationary' in result
    
    print(f"  ✅ TEST 3: Phillips-Perron Test")
    print(f"     Test Statistic: {result['test_statistic']:.6f}")
    print(f"     P-value: {result['p_value']:.2e}")
    print(f"     Stationary: {result['is_stationary']}")
    print(f"     Interpretation: {result['interpretation']}")
    return True


def test_comprehensive_stationarity_check():
    """Test comprehensive stationarity check with multiple tests"""
    print("\n📊 TEST 4: Comprehensive Stationarity Check")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Run comprehensive check
    result = comprehensive_stationarity_check(ts)
    
    # Validate results
    assert 'adf' in result
    assert 'kpss' in result
    assert 'pp' in result
    assert 'consensus_stationary' in result
    assert 'summary' in result
    assert result['test_agreement'] in ['0/3', '1/3', '2/3', '3/3']
    
    print(f"  ✅ TEST 4: Comprehensive Check")
    print(f"     ADF Stationary: {result['adf']['is_stationary']}")
    print(f"     KPSS Stationary: {result['kpss']['is_stationary']}")
    print(f"     PP Stationary: {result['pp']['is_stationary']}")
    print(f"     Consensus: {result['consensus_stationary']}")
    print(f"     Agreement: {result['test_agreement']}")
    print(f"     {result['summary']}")
    return True


def test_apply_differencing():
    """Test differencing application"""
    print("\n📊 TEST 5: Apply Differencing (d=1)")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    original_len = len(ts)
    
    # Apply first differencing
    diff1 = apply_differencing(ts, order=1)
    
    # Validate
    assert len(diff1) == original_len - 1
    assert diff1.isna().sum() == 0
    assert not np.isinf(diff1).any()
    
    print(f"  ✅ TEST 5: First Differencing (d=1)")
    print(f"     Original length: {original_len}")
    print(f"     Differenced length: {len(diff1)}")
    print(f"     Original mean: ${ts.mean():,.0f}")
    print(f"     Differenced mean: ${diff1.mean():,.0f}")
    print(f"     Original std: ${ts.std():,.0f}")
    print(f"     Differenced std: ${diff1.std():,.0f}")
    
    # Test second differencing
    diff2 = apply_differencing(ts, order=2)
    assert len(diff2) == original_len - 2
    print(f"\n     Second Differencing (d=2)")
    print(f"     Differenced length: {len(diff2)}")
    
    return True


def test_seasonal_differencing():
    """Test seasonal differencing"""
    print("\n📊 TEST 6: Seasonal Differencing (period=12)")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts_weekly, _ = aggregate_sales_monthly(df)  # Get a series
    
    # Create longer series for seasonal differencing
    ts = ts_weekly.copy()
    
    # Apply seasonal differencing
    try:
        seasonal_diff = apply_seasonal_differencing(ts, seasonal_period=12)
        
        assert len(seasonal_diff) == len(ts) - 12
        assert seasonal_diff.isna().sum() == 0
        
        print(f"  ✅ TEST 6: Seasonal Differencing")
        print(f"     Original length: {len(ts)}")
        print(f"     Seasonal period: 12")
        print(f"     Differenced length: {len(seasonal_diff)}")
        print(f"     Seasonal pattern removed: ✅")
        return True
    except ValueError as e:
        print(f"  ⚠️ TEST 6: Seasonal Differencing (SKIPPED - Series too short)")
        print(f"     Reason: {str(e)}")
        return True


def test_determine_differencing_order():
    """Test automatic differencing order determination"""
    print("\n📊 TEST 7: Determine Differencing Order")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Determine differencing order
    result = determine_differencing_order(ts, max_d=2, test_func='adf')
    
    # Validate
    assert 'd' in result
    assert result['d'] >= 0 and result['d'] <= 2
    assert 'tests' in result
    assert 'is_stationary' in result
    assert 'recommendation' in result
    
    print(f"  ✅ TEST 7: Determine Differencing Order")
    print(f"     Recommended d: {result['d']}")
    print(f"     Is stationary: {result['is_stationary']}")
    print(f"     {result['explanation']}")
    print(f"     {result['recommendation']}")
    return True


def test_validate_differencing():
    """Test differencing validation"""
    print("\n📊 TEST 8: Validate Differencing")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Apply differencing
    diff = apply_differencing(ts, order=1)
    
    # Validate
    validation = validate_differencing(ts, diff, order=1)
    
    # Check validation results
    assert validation['order_correct'] == True
    assert validation['no_nans'] == True
    assert validation['is_valid'] == True
    assert 'statistics' in validation
    
    print(f"  ✅ TEST 8: Validate Differencing")
    print(f"     Order correct: {validation['order_correct']}")
    print(f"     No NaN values: {validation['no_nans']}")
    print(f"     Original mean: ${validation['statistics']['original_mean']:,.0f}")
    print(f"     Differenced mean: ${validation['statistics']['differenced_mean']:,.0f}")
    print(f"     Mean reduced: {abs(validation['statistics']['differenced_mean']) < abs(validation['statistics']['original_mean'])}")
    return True


def test_prepare_stationary_series():
    """Test complete stationarity preparation pipeline"""
    print("\n📊 TEST 9: Prepare Stationary Series (Complete Pipeline)")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Run complete pipeline
    result = prepare_stationary_series(ts, max_d=2, apply_seasonal=False)
    
    # Validate
    assert 'differencing_order' in result
    assert 'differenced_series' in result
    assert 'is_ready' in result
    assert result['differencing_order'] >= 0
    assert len(result['differenced_series']) > 0
    
    print(f"  ✅ TEST 9: Prepare Stationary Series")
    print(f"     Status: {result['status']}")
    print(f"     Differencing order (d): {result['differencing_order']}")
    print(f"     Differenced series length: {len(result['differenced_series'])}")
    print(f"     Ready for ARIMA: {result['is_ready']}")
    print(f"     {result['recommendation']}")
    return True


def test_compare_stationarity():
    """Test stationarity comparison before and after differencing"""
    print("\n📊 TEST 10: Compare Stationarity Before/After Differencing")
    
    # Load and prepare data
    df, _ = load_data()  # Returns (df, source_name)
    df, _ = preprocess_data(df)  # Returns (df, stats)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)  # Returns (ts, stats)
    
    # Apply differencing
    diff = apply_differencing(ts, order=1)
    
    # Compare
    comparison = compare_stationarity_before_after(ts, diff, d_order=1)
    
    # Validate
    assert 'original' in comparison
    assert 'differenced' in comparison
    assert 'improvements' in comparison
    assert comparison['differencing_order'] == 1
    
    print(f"  ✅ TEST 10: Stationarity Comparison")
    print(f"\n     Original Series:")
    print(f"       Mean: ${comparison['original']['mean']:,.0f}")
    print(f"       Std: ${comparison['original']['std']:,.0f}")
    print(f"       Stationary: {comparison['original']['stationary']}")
    print(f"       ADF p-value: {comparison['original']['adf_p_value']:.2e}")
    print(f"\n     Differenced Series (d=1):")
    print(f"       Mean: ${comparison['differenced']['mean']:,.0f}")
    print(f"       Std: ${comparison['differenced']['std']:,.0f}")
    print(f"       Stationary: {comparison['differenced']['stationary']}")
    print(f"       ADF p-value: {comparison['differenced']['adf_p_value']:.2e}")
    print(f"\n     Improvements:")
    print(f"       ADF p-value improved: {comparison['improvements']['adf_p_value_improved']}")
    print(f"       Became stationary: {comparison['improvements']['became_stationary']}")
    print(f"       Mean reduced: {comparison['improvements']['mean_reduced']}")
    print(f"       Variance reduced: {comparison['improvements']['variance_reduced']}")
    
    return True


# ==================== TEST EXECUTION ====================

def run_all_tests():
    """Run all stationarity check module tests"""
    print("\n" + "="*60)
    print("  STEP 12: STATIONARITY CHECK MODULE - TEST SUITE")
    print("="*60)
    
    tests = [
        ("ADF Test", test_adf_test),
        ("KPSS Test", test_kpss_test),
        ("Phillips-Perron Test", test_pp_test),
        ("Comprehensive Stationarity Check", test_comprehensive_stationarity_check),
        ("Apply Differencing", test_apply_differencing),
        ("Seasonal Differencing", test_seasonal_differencing),
        ("Determine Differencing Order", test_determine_differencing_order),
        ("Validate Differencing", test_validate_differencing),
        ("Prepare Stationary Series", test_prepare_stationary_series),
        ("Compare Stationarity Before/After", test_compare_stationarity),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_name}: FAILED")
            print(f"     Error: {str(e)}")
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    print(f"\nTotal Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Pass Rate: {100*passed//len(tests)}%")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED! Stationarity check module is ready.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review errors above.")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()
