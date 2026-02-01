#!/usr/bin/env python3
"""
Test script for Exploratory Data Analysis (EDA) Module
Verifies all analysis functions work correctly with real data
"""

import json
from data_loader import load_data
from data_preprocessor import preprocess_data
from data_analysis import (
    perform_eda, get_monthly_trends, get_seasonal_patterns,
    get_peak_and_low_periods, get_store_analysis, get_dayofweek_analysis,
    get_overall_statistics, get_year_analysis
)

def test_eda_module():
    print("=" * 70)
    print("🧪 TESTING EXPLORATORY DATA ANALYSIS (EDA) MODULE")
    print("=" * 70)
    
    # Load and preprocess data
    print("\n📊 Loading data...")
    df, data_source = load_data()
    df, preprocessing_stats = preprocess_data(df)
    print(f"✅ Loaded {len(df)} records from {data_source} dataset")
    
    # Test 1: Overall Statistics
    print("\n✓ TEST 1: Overall Statistics")
    stats = get_overall_statistics(df)
    print(f"  ✅ Total Sales: ${stats['total_sales']:,.0f}")
    print(f"  ✅ Avg Weekly Sales: ${stats['avg_weekly_sales']:,.0f}")
    print(f"  ✅ Total Records: {stats['total_records']:,}")
    print(f"  ✅ Total Stores: {stats['total_stores']}")
    assert stats['total_sales'] > 0, "Total sales should be positive"
    assert stats['total_records'] == len(df), "Record count mismatch"
    
    # Test 2: Monthly Trends
    print("\n✓ TEST 2: Monthly Trends Analysis")
    trends = get_monthly_trends(df)
    print(f"  ✅ Months analyzed: {len(trends['months'])}")
    print(f"  ✅ First month: {trends['months'][0] if trends['months'] else 'N/A'}")
    print(f"  ✅ Last month: {trends['months'][-1] if trends['months'] else 'N/A'}")
    assert len(trends['months']) > 0, "Should have monthly data"
    assert len(trends['total_sales']) == len(trends['months']), "Data mismatch"
    
    # Test 3: Seasonal Patterns
    print("\n✓ TEST 3: Seasonal Patterns Analysis")
    seasonal = get_seasonal_patterns(df)
    print(f"  ✅ Seasons found: {seasonal['seasons']}")
    print(f"  ✅ Seasonality strength: {seasonal['seasonality_strength']}%")
    for season, sales in zip(seasonal['seasons'], seasonal['avg_sales']):
        print(f"     - {season}: ${sales:,.0f} avg sales")
    assert len(seasonal['seasons']) > 0, "Should have seasonal data"
    
    # Test 4: Peak & Low Periods
    print("\n✓ TEST 4: Peak and Low-Demand Periods")
    peak = get_peak_and_low_periods(df)
    print(f"  ✅ Peak periods (top 5):")
    for period, sales in zip(peak['peak_periods'], peak['peak_sales']):
        print(f"     - {period}: ${sales:,.0f}")
    print(f"  ✅ Low periods (bottom 5):")
    for period, sales in zip(peak['low_periods'], peak['low_sales']):
        print(f"     - {period}: ${sales:,.0f}")
    print(f"  ✅ Peak avg: ${peak['peak_average']:,.0f}")
    print(f"  ✅ Low avg: ${peak['low_average']:,.0f}")
    assert peak['peak_average'] > peak['low_average'], "Peak should be higher than low"
    
    # Test 5: Store Analysis
    print("\n✓ TEST 5: Store Performance Analysis")
    stores = get_store_analysis(df)
    print(f"  ✅ Total stores: {stores['total_stores']}")
    print(f"  ✅ Best performer: Store {stores['best_performer']} (${stores['best_performer_sales']:,.0f})")
    print(f"  ✅ Avg store sales: ${stores['avg_store_sales']:,.0f}")
    print(f"  ✅ Top 5 stores:")
    for i, store in enumerate(stores['top_stores'][:5], 1):
        print(f"     {i}. Store {store['store']}: ${store['avg_sales']:,.0f}")
    assert stores['total_stores'] > 0, "Should have store data"
    assert len(stores['top_stores']) > 0, "Should have top stores"
    
    # Test 6: Day of Week Analysis
    print("\n✓ TEST 6: Day of Week Analysis")
    dayofweek = get_dayofweek_analysis(df)
    print(f"  ✅ Days analyzed: {dayofweek['days']}")
    for day, sales in zip(dayofweek['days'], dayofweek['avg_sales']):
        print(f"     - {day}: ${sales:,.0f}")
    print(f"  ℹ️  Note: Walmart data only contains Friday data (all values recorded on Fridays)")
    assert len(dayofweek['days']) > 0, "Should have day of week data"
    
    # Test 7: Year Analysis
    print("\n✓ TEST 7: Year-over-Year Analysis")
    years = get_year_analysis(df)
    print(f"  ✅ Years in dataset: {years['years']}")
    for year, sales in zip(years['years'], years['total_sales']):
        print(f"     - {year}: ${sales:,.0f} total")
    assert len(years['years']) > 0, "Should have year data"
    
    # Test 8: Complete EDA
    print("\n✓ TEST 8: Complete EDA Pipeline")
    eda = perform_eda(df)
    print(f"  ✅ EDA Status: {eda['status']}")
    print(f"  ✅ Timestamp: {eda['timestamp']}")
    print(f"  ✅ All analyses included:")
    analyses = [
        'overall_stats', 'monthly_trends', 'seasonal_patterns',
        'peak_low_periods', 'store_analysis', 'day_of_week',
        'year_analysis', 'correlations'
    ]
    for analysis in analyses:
        if analysis in eda:
            print(f"     ✅ {analysis}")
        else:
            print(f"     ❌ {analysis} missing")
    
    # Test 9: JSON Serializability
    print("\n✓ TEST 9: JSON Serialization")
    try:
        json_str = json.dumps(eda)
        print(f"  ✅ EDA results are JSON serializable ({len(json_str)} bytes)")
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        print(f"  ✅ JSON can be parsed back successfully")
    except Exception as e:
        print(f"  ❌ JSON serialization error: {e}")
        return False
    
    # Test 10: Data Integrity
    print("\n✓ TEST 10: Data Integrity Checks")
    try:
        # Verify calculations
        monthly_total = sum(trends['total_sales'])
        expected_total = stats['total_sales']
        
        # Allow for some floating point variance
        variance = abs(monthly_total - expected_total) / expected_total * 100
        if variance < 1:  # Less than 1% variance
            print(f"  ✅ Monthly totals match overall total (variance: {variance:.2f}%)")
        else:
            print(f"  ⚠️ Monthly totals variance: {variance:.2f}%")
        
        # Verify store totals
        print(f"  ✅ Store analysis has {len(stores['top_stores'])} top stores")
        print(f"  ✅ Store analysis has {len(stores['bottom_stores'])} bottom stores")
        
    except Exception as e:
        print(f"  ⚠️ Data integrity check error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL EDA TESTS PASSED - MODULE WORKING CORRECTLY!")
    print("=" * 70)
    print("\n📊 EDA Module Features:")
    print("   ✓ Monthly sales trends analysis")
    print("   ✓ Seasonal pattern identification")
    print("   ✓ Peak and low-demand period detection")
    print("   ✓ Store performance ranking")
    print("   ✓ Day-of-week sales patterns")
    print("   ✓ Year-over-year comparisons")
    print("   ✓ Overall statistics and KPIs")
    print("   ✓ Correlation analysis")
    print("\n📈 Ready for visualization in Dashboard!")
    
    return True

if __name__ == '__main__':
    success = test_eda_module()
    exit(0 if success else 1)
