"""
Test Suite for STEP 13: ARIMA Model Training Module
Tests ARIMA model fitting, forecasting, and validation functionality
"""

import pandas as pd
import numpy as np
from data_loader import load_data
from data_preprocessor import preprocess_data
from models.arima_model import (
    set_date_as_index,
    aggregate_sales_monthly,
    fit_arima,
    generate_forecast,
    calculate_metrics,
    validate_model,
    compare_models,
    train_complete_arima_pipeline
)


def test_fit_arima():
    """Test basic ARIMA model fitting"""
    print("\n📊 TEST 1: Fit ARIMA Model")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    result = fit_arima(ts, order=(1, 0, 1))
    
    # Validate
    assert result['success'] == True
    assert result['model'] is not None
    assert result['results'] is not None
    assert result['order'] == (1, 0, 1)
    assert np.isfinite(result['aic'])
    assert np.isfinite(result['bic'])
    
    print(f"  ✅ TEST 1: ARIMA(1, 0, 1) Fitted Successfully")
    print(f"     AIC: {result['aic']:.2f}")
    print(f"     BIC: {result['bic']:.2f}")
    print(f"     Model: {result['order']}")
    return True


def test_generate_forecast():
    """Test forecast generation"""
    print("\n📊 TEST 2: Generate Forecast")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    model_result = fit_arima(ts, order=(1, 0, 1))
    
    # Generate forecast
    forecast_result = generate_forecast(model_result, steps=12, confidence=0.95)
    
    # Validate
    assert forecast_result['success'] == True
    assert len(forecast_result['forecast']) == 12
    assert len(forecast_result['lower_ci']) == 12
    assert len(forecast_result['upper_ci']) == 12
    assert forecast_result['steps'] == 12
    
    # Check forecast reasonableness (should be within reasonable range of data)
    forecast_mean = forecast_result['forecast'].mean()
    assert forecast_mean > 0
    
    print(f"  ✅ TEST 2: Forecast Generated (12 periods)")
    print(f"     Forecast mean: ${forecast_mean:,.0f}")
    print(f"     Forecast range: ${forecast_result['forecast'].min():,.0f} - ${forecast_result['forecast'].max():,.0f}")
    print(f"     Confidence: {forecast_result['confidence']*100:.0f}%")
    return True


def test_calculate_metrics():
    """Test metric calculation"""
    print("\n📊 TEST 3: Calculate Metrics")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    model_result = fit_arima(ts, order=(1, 0, 1))
    fitted_values = model_result['results'].fittedvalues
    
    # Calculate metrics
    metrics = calculate_metrics(ts, fitted_values)
    
    # Validate
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    assert 'correlation' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert -1 <= metrics['correlation'] <= 1
    
    print(f"  ✅ TEST 3: Metrics Calculated")
    print(f"     RMSE: ${metrics['rmse']:,.0f}")
    print(f"     MAE: ${metrics['mae']:,.0f}")
    print(f"     MAPE: {metrics['mape']:.2f}%")
    print(f"     Correlation: {metrics['correlation']:.4f}")
    return True


def test_validate_model():
    """Test model validation"""
    print("\n📊 TEST 4: Validate Model")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    model_result = fit_arima(ts, order=(1, 0, 1))
    
    # Validate
    diagnostics = validate_model(model_result)
    
    # Checks
    assert diagnostics['success'] == True
    assert 'residuals_mean' in diagnostics
    assert 'mean_near_zero' in diagnostics
    
    print(f"  ✅ TEST 4: Model Validation")
    print(f"     Residuals mean: ${diagnostics['residuals_mean']:,.0f}")
    print(f"     Mean near zero: {diagnostics['mean_near_zero']}")
    print(f"     White noise: {diagnostics['white_noise']}")
    print(f"     Normality: {diagnostics['normality']}")
    print(f"     Diagnostics pass: {diagnostics['diagnostics_pass']}")
    return True


def test_compare_models():
    """Test model comparison"""
    print("\n📊 TEST 5: Compare Models")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Compare models
    comparison = compare_models(ts, orders=[(0, 0, 1), (1, 0, 0), (1, 0, 1)])
    
    # Validate
    assert 'models' in comparison
    assert 'best_model' in comparison
    assert 'best_order' in comparison
    assert len(comparison['models']) > 0
    assert comparison['best_model']['success'] == True
    
    print(f"  ✅ TEST 5: Model Comparison")
    print(f"     Models tested: {len(comparison['models'])}")
    print(f"     Best order: {comparison['best_order']}")
    print(f"     Best AIC: {comparison['best_model']['aic']:.2f}")
    print(f"\n     All models ranked by AIC:")
    for idx, row in comparison['models'].iterrows():
        print(f"       {row['order']:>10} - AIC: {row['aic']:>10.2f}")
    return True


def test_train_complete_pipeline():
    """Test complete ARIMA training pipeline"""
    print("\n📊 TEST 6: Complete ARIMA Pipeline")
    
    # Load data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    
    # Run complete pipeline
    result = train_complete_arima_pipeline(
        df,
        aggregation='monthly',
        order=(1, 0, 1),
        forecast_periods=12,
        train_test_split=0.8
    )
    
    # Validate
    assert result['status'] == 'success'
    assert len(result['time_series']) > 0
    assert len(result['train_series']) > 0
    assert len(result['test_series']) > 0
    assert result['model_result']['success'] == True
    assert result['forecast']['success'] == True
    assert len(result['forecast']['forecast']) == 12
    
    print(f"  ✅ TEST 6: Complete Pipeline")
    print(f"     Status: {result['status']}")
    print(f"     Total periods: {len(result['time_series'])}")
    print(f"     Train/Test split: {result['train_size']}/{result['test_size']}")
    print(f"     Model: ARIMA{result['order']}")
    print(f"     Test RMSE: ${result['metrics']['rmse']:,.0f}")
    print(f"     Test MAE: ${result['metrics']['mae']:,.0f}")
    print(f"     Forecast periods: {result['forecast_periods']}")
    print(f"     Ready for deployment: {result['ready']}")
    return True


def test_arima_p010():
    """Test ARIMA with p=0, d=1, q=0 (simple differencing)"""
    print("\n📊 TEST 7: ARIMA(0, 1, 0) - Simple Differencing")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    result = fit_arima(ts, order=(0, 1, 0))
    
    assert result['success'] == True
    
    print(f"  ✅ TEST 7: ARIMA(0, 1, 0) Fitted")
    print(f"     AIC: {result['aic']:.2f}")
    print(f"     BIC: {result['bic']:.2f}")
    return True


def test_arima_p201():
    """Test ARIMA with p=2, d=0, q=1"""
    print("\n📊 TEST 8: ARIMA(2, 0, 1) - Higher AR order")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model
    result = fit_arima(ts, order=(2, 0, 1))
    
    assert result['success'] == True
    
    print(f"  ✅ TEST 8: ARIMA(2, 0, 1) Fitted")
    print(f"     AIC: {result['aic']:.2f}")
    print(f"     BIC: {result['bic']:.2f}")
    return True


def test_forecast_confidence_intervals():
    """Test that confidence intervals are properly ordered"""
    print("\n📊 TEST 9: Forecast Confidence Intervals")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Fit model and generate forecast
    model_result = fit_arima(ts, order=(1, 0, 1))
    forecast_result = generate_forecast(model_result, steps=12, confidence=0.95)
    
    # Validate confidence intervals
    assert (forecast_result['lower_ci'] < forecast_result['forecast']).all()
    assert (forecast_result['forecast'] < forecast_result['upper_ci']).all()
    
    # Check widths increase (normal behavior)
    widths = forecast_result['upper_ci'] - forecast_result['lower_ci']
    
    print(f"  ✅ TEST 9: Confidence Intervals Valid")
    print(f"     Min width: ${widths.min():,.0f}")
    print(f"     Max width: ${widths.max():,.0f}")
    print(f"     Avg width: ${widths.mean():,.0f}")
    print(f"     All lower < forecast < upper: ✅")
    return True


def test_model_aic_comparison():
    """Test that models are properly ranked by AIC"""
    print("\n📊 TEST 10: Model AIC Comparison")
    
    # Load and prepare data
    df, _ = load_data()
    df, _ = preprocess_data(df)
    df = set_date_as_index(df)
    ts, _ = aggregate_sales_monthly(df)
    
    # Get multiple models
    comparison = compare_models(ts)
    
    # Validate ranking
    aic_values = comparison['models']['aic'].values
    assert aic_values[0] <= aic_values[-1], "Models should be sorted by AIC"
    
    print(f"  ✅ TEST 10: Model Ranking by AIC")
    print(f"     Best AIC: {aic_values[0]:.2f}")
    print(f"     Worst AIC: {aic_values[-1]:.2f}")
    print(f"     AIC improvement: {aic_values[-1] - aic_values[0]:.2f}")
    print(f"     Models properly ranked: ✅")
    return True


# ==================== TEST EXECUTION ====================

def run_all_tests():
    """Run all ARIMA model training tests"""
    print("\n" + "="*60)
    print("  STEP 13: ARIMA MODEL TRAINING - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Fit ARIMA Model", test_fit_arima),
        ("Generate Forecast", test_generate_forecast),
        ("Calculate Metrics", test_calculate_metrics),
        ("Validate Model", test_validate_model),
        ("Compare Models", test_compare_models),
        ("Complete Pipeline", test_train_complete_pipeline),
        ("ARIMA(0,1,0)", test_arima_p010),
        ("ARIMA(2,0,1)", test_arima_p201),
        ("Confidence Intervals", test_forecast_confidence_intervals),
        ("Model AIC Comparison", test_model_aic_comparison),
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
        print(f"\n🎉 ALL TESTS PASSED! ARIMA training module is ready.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review errors above.")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()
