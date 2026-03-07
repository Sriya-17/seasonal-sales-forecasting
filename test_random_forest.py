#!/usr/bin/env python3
"""
Simple test script to verify Random Forest integration
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.arima_model import random_forest_forecast
import pandas as pd
import numpy as np

def test_random_forest():
    """Test Random Forest forecasting function"""
    print("🧪 Testing Random Forest Integration...")

    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    sales = np.random.randn(24) * 100 + 1000  # Random sales around 1000
    ts = pd.Series(sales, index=dates)

    try:
        # Test Random Forest forecast
        result = random_forest_forecast(ts, steps=12)

        # Verify structure
        assert 'forecast' in result, "Missing forecast in result"
        assert 'conf_int' in result, "Missing confidence intervals in result"
        assert 'model' in result, "Missing model in result"
        assert len(result['forecast']) == 12, f"Expected 12 forecasts, got {len(result['forecast'])}"
        assert result['conf_int'].shape[0] == 12, f"Expected 12 confidence intervals, got {result['conf_int'].shape[0]}"

        print("✅ Random Forest forecast successful")
        print(f"   📊 Forecast length: {len(result['forecast'])}")
        print(f"   📈 Sample forecast values: {result['forecast'].head(3).values}")
        print(f"   📉 Confidence interval range: [{result['conf_int'].iloc[0]['lower']:.2f}, {result['conf_int'].iloc[0]['upper']:.2f}]")

        return True

    except Exception as e:
        print(f"❌ Random Forest test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_random_forest()
    sys.exit(0 if success else 1)