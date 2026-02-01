"""
Test Suite for Data Visualization Module (STEP 10)
Tests all visualization functions with real Walmart dataset
"""

import sys
import os
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_visualization import (
    create_sales_over_time_plot,
    create_seasonality_plot,
    create_seasonal_breakdown_plot,
    create_store_performance_plot,
    create_yearly_comparison_plot,
    create_distribution_plot,
    create_all_visualizations
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


def validate_base64_image(data):
    """Validate base64 encoded image."""
    if data is None:
        return False, "Image is None"
    
    if not isinstance(data, str):
        return False, "Image is not a string"
    
    # Try to decode base64
    try:
        # Check if valid base64
        if len(data) % 4 != 0:
            return False, "Invalid base64 length"
        
        decoded = base64.b64decode(data)
        
        # Check if it's valid PNG (starts with PNG header)
        if not decoded.startswith(b'\x89PNG'):
            return False, "Not a valid PNG image"
        
        return True, f"Valid PNG ({len(decoded)} bytes)"
    except Exception as e:
        return False, f"Base64 decode error: {e}"


def run_tests():
    """Run all visualization tests."""
    print_header("STEP 10: DATA VISUALIZATION MODULE - TEST SUITE")
    
    # Load and preprocess data
    print("📊 Loading dataset...")
    df, data_source = load_data(prefer_uploaded=True)
    print(f"✅ Dataset loaded from {data_source}")
    print(f"   Records: {len(df)}, Stores: {df['Store'].nunique()}")
    
    df, stats = preprocess_data(df)
    print(f"✅ Data preprocessed: {stats['records_removed']} removed, {stats['missing_values_after']} missing")
    
    test_results = []
    
    # ============== TEST 1: Sales Over Time Plot ==============
    print("\n📈 TEST 1: Sales Over Time Plot")
    try:
        plot = create_sales_over_time_plot(df)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(1, "Sales Over Time Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(1, "Sales Over Time Plot", False, f"Exception: {e}")
    
    # ============== TEST 2: Monthly Seasonality Plot ==============
    print("\n📊 TEST 2: Monthly Seasonality Plot")
    try:
        plot = create_seasonality_plot(df)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(2, "Monthly Seasonality Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(2, "Monthly Seasonality Plot", False, f"Exception: {e}")
    
    # ============== TEST 3: Seasonal Breakdown Plot ==============
    print("\n🌍 TEST 3: Seasonal Breakdown Plot")
    try:
        plot = create_seasonal_breakdown_plot(df)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(3, "Seasonal Breakdown Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(3, "Seasonal Breakdown Plot", False, f"Exception: {e}")
    
    # ============== TEST 4: Store Performance Plot ==============
    print("\n🏪 TEST 4: Store Performance Plot")
    try:
        plot = create_store_performance_plot(df, top_n=10)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(4, "Store Performance Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(4, "Store Performance Plot", False, f"Exception: {e}")
    
    # ============== TEST 5: Yearly Comparison Plot ==============
    print("\n📅 TEST 5: Yearly Comparison Plot")
    try:
        plot = create_yearly_comparison_plot(df)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(5, "Yearly Comparison Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(5, "Yearly Comparison Plot", False, f"Exception: {e}")
    
    # ============== TEST 6: Distribution Plot ==============
    print("\n📉 TEST 6: Sales Distribution Plot")
    try:
        plot = create_distribution_plot(df)
        valid, msg = validate_base64_image(plot)
        passed = valid
        details = msg
        test_results.append(passed)
        print_test(6, "Sales Distribution Plot", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(6, "Sales Distribution Plot", False, f"Exception: {e}")
    
    # ============== TEST 7: Create All Visualizations ==============
    print("\n🎨 TEST 7: Create All Visualizations at Once")
    try:
        visualizations = create_all_visualizations(df)
        
        # Check that all visualizations were generated
        all_keys = ['sales_over_time', 'seasonality', 'seasonal_breakdown', 
                   'store_performance', 'yearly_comparison', 'distribution']
        generated = sum(1 for k in all_keys if visualizations.get(k) is not None)
        
        passed = generated == len(all_keys)
        details = f"Generated {generated}/{len(all_keys)} visualizations"
        test_results.append(passed)
        print_test(7, "Create All Visualizations", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(7, "Create All Visualizations", False, f"Exception: {e}")
    
    # ============== TEST 8: Empty DataFrame Handling ==============
    print("\n⚠️  TEST 8: Empty DataFrame Handling")
    try:
        empty_df = pd.DataFrame()
        
        plot1 = create_sales_over_time_plot(empty_df)
        plot2 = create_seasonality_plot(empty_df)
        plot3 = create_seasonal_breakdown_plot(empty_df)
        
        # Should return None for empty dataframes
        passed = plot1 is None and plot2 is None and plot3 is None
        details = "Correctly returns None for empty dataframes"
        test_results.append(passed)
        print_test(8, "Empty DataFrame Handling", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(8, "Empty DataFrame Handling", False, f"Exception: {e}")
    
    # ============== TEST 9: Data Integrity Check ==============
    print("\n✔️  TEST 9: Data Integrity Check")
    try:
        # Verify that original df is not modified by visualization functions
        original_len = len(df)
        original_cols = set(df.columns)
        
        create_all_visualizations(df)
        
        passed = len(df) == original_len and set(df.columns) == original_cols
        details = f"DataFrame integrity maintained ({original_len} records, {len(original_cols)} columns)"
        test_results.append(passed)
        print_test(9, "Data Integrity Check", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(9, "Data Integrity Check", False, f"Exception: {e}")
    
    # ============== TEST 10: API Route Availability ==============
    print("\n🌐 TEST 10: API Route Availability Check")
    try:
        # This test verifies the routes can be imported from app.py
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import (
            api_visualizations,
            plot_sales_over_time,
            plot_seasonality,
            plot_seasonal_breakdown,
            plot_store_perf,
            plot_yearly_comp,
            plot_dist
        )
        
        passed = all([
            api_visualizations,
            plot_sales_over_time,
            plot_seasonality,
            plot_seasonal_breakdown,
            plot_store_perf,
            plot_yearly_comp,
            plot_dist
        ])
        
        details = "All 7 visualization API routes available"
        test_results.append(passed)
        print_test(10, "API Route Availability", passed, details)
    except Exception as e:
        test_results.append(False)
        print_test(10, "API Route Availability", False, f"Exception: {e}")
    
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
        print("\n🎉 ALL TESTS PASSED! Data visualization module is ready for production.")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review the errors above.")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
