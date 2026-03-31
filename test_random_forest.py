#!/usr/bin/env python3
"""
Test script for Random Forest Sales Forecasting Model
"""

import pandas as pd
import sys
import os

# Add the models directory to path
sys.path.append(os.path.dirname(__file__))

from models.random_forest_model import clean_data, feature_engineering, train_model, generate_future_dates, predict_future, get_insights

def test_model():
    """Test the Random Forest model with sample data"""
    print("🧪 Testing Random Forest Sales Forecasting Model")

    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'Sales': [100 + i*2 + (i%7)*10 for i in range(100)]  # Some seasonality
    })

    print(f"📊 Sample data shape: {sample_data.shape}")
    print(sample_data.head())

    # Clean data
    cleaned_data = clean_data(sample_data)
    print(f"🧹 Cleaned data shape: {cleaned_data.shape}")

    # Feature engineering
    engineered_data = feature_engineering(cleaned_data)
    print(f"🔧 Engineered data shape: {engineered_data.shape}")
    print(f"   Features: {list(engineered_data.columns)}")

    # Train model
    trained_model = train_model(engineered_data)
    print("🚀 Model trained successfully!")
    print(f"   MAE: ${trained_model['metrics']['mae']:.2f}")
    print(f"   RMSE: ${trained_model['metrics']['rmse']:.2f}")
    print(f"   R²: {trained_model['metrics']['r2']:.3f}")

    # Generate future predictions
    last_date = pd.to_datetime(cleaned_data['Date'].max())
    future_dates = generate_future_dates(30, last_date)
    predictions = predict_future(trained_model, future_dates, last_date)

    print(f"🔮 Generated {len(predictions)} future predictions")
    print(predictions[['Date', 'Predicted_Sales']].head())

    # Get insights
    insights = get_insights(predictions)
    print("📈 Insights:")
    print(f"   Peak month: {insights['peak_month']}")
    print(".2f")
    print(".2f")

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_model()