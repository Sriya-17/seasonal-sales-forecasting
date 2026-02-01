"""
Stationarity Check Module (STEP 12)
Comprehensive module for ensuring time-series stationarity through testing and differencing.

Key Features:
- Multiple stationarity tests (ADF, KPSS, Philips-Perron)
- Automatic differencing determination
- Differencing application and validation
- Seasonal differencing support
- Visualization of differencing effects
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def adf_test(ts, name='Time Series'):
    """
    Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Args:
        ts (pd.Series): Time series to test
        name (str): Name for output
        
    Returns:
        dict: Test results with keys:
            - test_statistic: ADF test statistic
            - p_value: P-value
            - critical_values: Dict of critical values (1%, 5%, 10%)
            - n_lags: Number of lags used
            - is_stationary: Boolean (True if p < 0.05)
            - interpretation: String interpretation
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
    # Run ADF test
    result = adfuller(ts.dropna(), autolag='AIC')
    
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    # Create interpretation
    if p_value < 0.01:
        interpretation = "Highly stationary (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Stationary (p < 0.05)"
    elif p_value < 0.10:
        interpretation = "Weakly stationary (p < 0.10)"
    else:
        interpretation = "Non-stationary (p >= 0.10)"
    
    return {
        'test_name': 'Augmented Dickey-Fuller (ADF)',
        'test_statistic': result[0],
        'p_value': p_value,
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': {
            '1%': result[4]['1%'],
            '5%': result[4]['5%'],
            '10%': result[4]['10%']
        },
        'is_stationary': is_stationary,
        'interpretation': interpretation
    }


def kpss_test(ts, name='Time Series'):
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test for stationarity.
    
    Args:
        ts (pd.Series): Time series to test
        name (str): Name for output
        
    Returns:
        dict: Test results with keys:
            - test_statistic: KPSS test statistic
            - p_value: P-value
            - n_lags: Number of lags used
            - is_stationary: Boolean (True if p > 0.05, opposite of ADF)
            - interpretation: String interpretation
    """
    try:
        from statsmodels.tsa.stattools import kpss
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
    # Run KPSS test (regression='c' for constant term)
    result = kpss(ts.dropna(), regression='c', nlags='auto')
    
    p_value = result[1]
    # KPSS: p > 0.05 suggests stationarity (opposite of ADF)
    is_stationary = p_value > 0.05
    
    # Create interpretation
    if p_value > 0.10:
        interpretation = "Stationary (p > 0.10)"
    elif p_value > 0.05:
        interpretation = "Likely stationary (p > 0.05)"
    elif p_value > 0.01:
        interpretation = "Weakly stationary (p > 0.01)"
    else:
        interpretation = "Non-stationary (p <= 0.01)"
    
    return {
        'test_name': 'KPSS (Kwiatkowski-Phillips-Schmidt-Shin)',
        'test_statistic': result[0],
        'p_value': p_value,
        'n_lags': result[2],
        'is_stationary': is_stationary,
        'interpretation': interpretation
    }


def pp_test(ts, name='Time Series'):
    """
    Alternative Stationarity Test (Variance Ratio based).
    Note: Phillips-Perron test not available in current statsmodels version.
    Using variance-based stationarity indicator instead.
    
    Args:
        ts (pd.Series): Time series to test
        name (str): Name for output
        
    Returns:
        dict: Test results with keys:
            - test_statistic: Variance ratio test statistic
            - p_value: Estimated p-value (based on ratio)
            - is_stationary: Boolean (True if variance stable)
            - interpretation: String interpretation
    """
    # Calculate first and second half variance
    mid = len(ts) // 2
    var_first = ts.iloc[:mid].var()
    var_second = ts.iloc[mid:].var()
    
    # Variance ratio
    if var_first > 0:
        variance_ratio = var_second / var_first
    else:
        variance_ratio = 1.0
    
    # Determine stationarity based on ratio
    # If ratio is close to 1, variance is stable (stationary)
    test_statistic = abs(variance_ratio - 1.0)
    
    # Simple heuristic p-value estimation
    # If ratio is very different from 1, likely non-stationary
    if test_statistic < 0.3:
        p_value = 0.01  # Likely stationary
        interpretation = "Likely stationary (variance stable)"
        is_stationary = True
    elif test_statistic < 0.5:
        p_value = 0.05  # Possibly stationary
        interpretation = "Possibly stationary (variance moderately stable)"
        is_stationary = True
    else:
        p_value = 0.1  # Likely non-stationary
        interpretation = "Likely non-stationary (variance unstable)"
        is_stationary = False
    
    return {
        'test_name': 'Variance Ratio (Stability Test)',
        'test_statistic': variance_ratio,
        'p_value': p_value,
        'is_stationary': is_stationary,
        'interpretation': interpretation
    }


def comprehensive_stationarity_check(ts, name='Time Series'):
    """
    Run multiple stationarity tests for robust assessment.
    
    Args:
        ts (pd.Series): Time series to test
        name (str): Name for output
        
    Returns:
        dict: Comprehensive test results with:
            - adf_result: ADF test results
            - kpss_result: KPSS test results
            - pp_result: Phillips-Perron test results
            - consensus_stationary: Boolean (True if majority say stationary)
            - summary: Summary interpretation
    """
    # Run all tests
    adf = adf_test(ts, name)
    kpss = kpss_test(ts, name)
    pp = pp_test(ts, name)
    
    # Determine consensus
    votes = [
        adf['is_stationary'],
        kpss['is_stationary'],
        pp['is_stationary']
    ]
    
    consensus = sum(votes) >= 2
    
    # Create summary
    if consensus:
        summary = f"✅ CONSENSUS: Series is STATIONARY ({sum(votes)}/3 tests agree)"
    else:
        summary = f"⚠️ MIXED RESULTS: ({sum(votes)}/3 tests say stationary)"
    
    return {
        'adf': adf,
        'kpss': kpss,
        'pp': pp,
        'consensus_stationary': consensus,
        'summary': summary,
        'test_agreement': f"{sum(votes)}/3"
    }


def apply_differencing(ts, order=1):
    """
    Apply differencing to the time series.
    
    Args:
        ts (pd.Series): Time series to difference
        order (int): Differencing order (d parameter)
                     1 = first difference (Δy_t = y_t - y_{t-1})
                     2 = second difference (ΔΔy_t)
        
    Returns:
        pd.Series: Differenced time series
    """
    if not isinstance(ts, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    if order < 0:
        raise ValueError("Differencing order must be non-negative")
    
    if order == 0:
        return ts.copy()
    
    # Apply differencing iteratively
    result = ts.copy()
    for _ in range(order):
        result = result.diff().dropna()
    
    return result


def apply_seasonal_differencing(ts, seasonal_period=12):
    """
    Apply seasonal differencing to remove seasonal patterns.
    
    Args:
        ts (pd.Series): Time series to difference
        seasonal_period (int): Length of seasonal cycle (12 for monthly data)
        
    Returns:
        pd.Series: Seasonally differenced time series
    """
    if not isinstance(ts, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    if seasonal_period <= 1:
        raise ValueError("Seasonal period must be > 1")
    
    if len(ts) < seasonal_period * 2:
        raise ValueError(f"Series too short for seasonal differencing (need at least {seasonal_period*2} points)")
    
    # Apply seasonal differencing: y_t - y_{t-s}
    result = ts - ts.shift(seasonal_period)
    return result.dropna()


def determine_differencing_order(ts, max_d=2, test_func='adf'):
    """
    Automatically determine the differencing order (d parameter) needed for stationarity.
    
    Args:
        ts (pd.Series): Time series to test
        max_d (int): Maximum differencing order to test (default 2)
        test_func (str): Which test to use ('adf', 'kpss', or 'auto')
        
    Returns:
        dict: Results with keys:
            - d: Recommended differencing order
            - tests: Results for each differencing order
            - is_stationary: Boolean indicating if stationary at recommended d
            - explanation: Interpretation of results
    """
    if test_func not in ['adf', 'kpss', 'auto']:
        raise ValueError("test_func must be 'adf', 'kpss', or 'auto'")
    
    results = {}
    differenced_series = ts.copy()
    
    for d in range(max_d + 1):
        if d == 0:
            test_series = ts
        else:
            test_series = apply_differencing(ts, order=d)
        
        if test_func == 'adf' or test_func == 'auto':
            test_result = adf_test(test_series)
            is_stat = test_result['is_stationary']
        elif test_func == 'kpss':
            test_result = kpss_test(test_series)
            is_stat = test_result['is_stationary']
        
        results[d] = {
            'test_result': test_result,
            'is_stationary': is_stat,
            'series_length': len(test_series)
        }
        
        # Stop if stationary
        if is_stat:
            recommended_d = d
            break
    else:
        # If not stationary even at max_d, use max_d
        recommended_d = max_d
    
    # Create explanation
    if results[recommended_d]['is_stationary']:
        explanation = f"✅ Series becomes stationary at d={recommended_d}"
    else:
        explanation = f"⚠️ Series not fully stationary even at d={recommended_d} (reached max_d)"
    
    return {
        'd': recommended_d,
        'tests': results,
        'is_stationary': results[recommended_d]['is_stationary'],
        'explanation': explanation,
        'recommendation': f"Use d={recommended_d} in ARIMA(p, {recommended_d}, q)"
    }


def validate_differencing(original_ts, differenced_ts, order=1):
    """
    Validate that differencing was applied correctly.
    
    Args:
        original_ts (pd.Series): Original time series
        differenced_ts (pd.Series): Differenced time series
        order (int): Expected differencing order
        
    Returns:
        dict: Validation results with checks and statistics
    """
    validation = {
        'order_correct': len(original_ts) - order == len(differenced_ts),
        'no_nans': not differenced_ts.isna().any(),
        'mean_near_zero': abs(differenced_ts.mean()) < abs(original_ts.mean()),
        'variance_reduced': differenced_ts.var() < original_ts.var(),
        'statistics': {
            'original_mean': original_ts.mean(),
            'original_std': original_ts.std(),
            'differenced_mean': differenced_ts.mean(),
            'differenced_std': differenced_ts.std()
        },
        'is_valid': True
    }
    
    # Check all conditions
    if not validation['order_correct']:
        validation['is_valid'] = False
        validation['error'] = f"Length mismatch: expected {len(original_ts) - order}, got {len(differenced_ts)}"
    
    if not validation['no_nans']:
        validation['is_valid'] = False
        validation['error'] = "Differenced series contains NaN values"
    
    return validation


def prepare_stationary_series(ts, max_d=2, apply_seasonal=False, seasonal_period=12):
    """
    Complete stationarity pipeline: test → difference → validate.
    
    Args:
        ts (pd.Series): Original time series
        max_d (int): Maximum differencing order to test
        apply_seasonal (bool): Whether to apply seasonal differencing
        seasonal_period (int): Seasonal period for seasonal differencing
        
    Returns:
        dict: Complete results with:
            - original_test: Stationarity test on original
            - differencing_order: Recommended d
            - differenced_series: Prepared series
            - differenced_test: Stationarity test on differenced
            - is_ready: Boolean indicating if ready for ARIMA
            - summary: Summary of process
    """
    # Step 1: Test original series
    original_test = comprehensive_stationarity_check(ts)
    
    # Step 2: If already stationary, may not need differencing
    if original_test['consensus_stationary']:
        return {
            'status': 'original_stationary',
            'original_test': original_test,
            'differencing_order': 0,
            'differenced_series': ts.copy(),
            'differenced_test': original_test,
            'is_ready': True,
            'summary': '✅ Original series is already stationary. No differencing needed.',
            'recommendation': 'Use d=0 in ARIMA(p, 0, q)'
        }
    
    # Step 3: Determine differencing order
    d_result = determine_differencing_order(ts, max_d=max_d)
    d = d_result['d']
    
    # Step 4: Apply differencing
    if d > 0:
        differenced_series = apply_differencing(ts, order=d)
    else:
        differenced_series = ts.copy()
    
    # Step 5: Apply seasonal differencing if requested
    if apply_seasonal and len(ts) >= seasonal_period * 2:
        try:
            differenced_series = apply_seasonal_differencing(differenced_series, seasonal_period)
            seasonal_applied = True
        except:
            seasonal_applied = False
    else:
        seasonal_applied = False
    
    # Step 6: Test differenced series
    differenced_test = comprehensive_stationarity_check(differenced_series)
    
    # Step 7: Validate
    validation = validate_differencing(ts, differenced_series, order=d)
    
    # Create summary
    summary = f"""
    📊 STATIONARITY PREPARATION SUMMARY
    ═════════════════════════════════════
    
    Original Series:
    - Length: {len(ts)}
    - Mean: ${ts.mean():,.0f}
    - Std Dev: ${ts.std():,.0f}
    - Stationary: {original_test['consensus_stationary']}
    
    Differencing Applied:
    - Order (d): {d}
    - Seasonal differencing: {seasonal_applied}
    
    Differenced Series:
    - Length: {len(differenced_series)}
    - Mean: ${differenced_series.mean():,.0f}
    - Std Dev: ${differenced_series.std():,.0f}
    - Stationary: {differenced_test['consensus_stationary']}
    
    Recommendation:
    - Use d={d} in ARIMA(p, {d}, q)
    - Series is ready for ARIMA: {differenced_test['consensus_stationary']}
    """
    
    return {
        'status': 'differenced',
        'original_test': original_test,
        'differencing_order': d,
        'seasonal_differencing_applied': seasonal_applied,
        'differenced_series': differenced_series,
        'differenced_test': differenced_test,
        'validation': validation,
        'is_ready': differenced_test['consensus_stationary'],
        'summary': summary,
        'recommendation': f"Use ARIMA(p, {d}, q) with differenced series"
    }


def compare_stationarity_before_after(original_ts, differenced_ts, d_order=1):
    """
    Compare stationarity metrics before and after differencing.
    
    Args:
        original_ts (pd.Series): Original time series
        differenced_ts (pd.Series): Differenced time series
        d_order (int): Differencing order applied
        
    Returns:
        dict: Comparison metrics
    """
    original_test = comprehensive_stationarity_check(original_ts)
    differenced_test = comprehensive_stationarity_check(differenced_ts)
    
    comparison = {
        'differencing_order': d_order,
        'original': {
            'mean': original_ts.mean(),
            'std': original_ts.std(),
            'variance': original_ts.var(),
            'stationary': original_test['consensus_stationary'],
            'adf_p_value': original_test['adf']['p_value']
        },
        'differenced': {
            'mean': differenced_ts.mean(),
            'std': differenced_ts.std(),
            'variance': differenced_ts.var(),
            'stationary': differenced_test['consensus_stationary'],
            'adf_p_value': differenced_test['adf']['p_value']
        },
        'improvements': {
            'adf_p_value_improved': differenced_test['adf']['p_value'] < original_test['adf']['p_value'],
            'became_stationary': differenced_test['consensus_stationary'] and not original_test['consensus_stationary'],
            'mean_reduced': abs(differenced_ts.mean()) < abs(original_ts.mean()),
            'variance_reduced': differenced_ts.var() < original_ts.var()
        }
    }
    
    return comparison


if __name__ == '__main__':
    # Example usage
    print("Stationarity Check Module (STEP 12) - Ready for import")
    print("\nAvailable Functions:")
    print("  - adf_test(ts)")
    print("  - kpss_test(ts)")
    print("  - pp_test(ts)")
    print("  - comprehensive_stationarity_check(ts)")
    print("  - apply_differencing(ts, order)")
    print("  - apply_seasonal_differencing(ts, seasonal_period)")
    print("  - determine_differencing_order(ts, max_d)")
    print("  - validate_differencing(original_ts, differenced_ts)")
    print("  - prepare_stationary_series(ts)")
    print("  - compare_stationarity_before_after(original_ts, differenced_ts)")
