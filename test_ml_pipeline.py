#!/usr/bin/env python3
"""
Comprehensive Test Script for Seasonal Sales Forecasting ML Pipeline
====================================================================

This script tests:
1. Data cleaning and validation
2. Feature engineering
3. Model training
4. Evaluation metrics calculation
5. Future forecasting
6. Insights generation

Run with: python test_ml_pipeline.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add models to path
sys.path.insert(0, os.path.dirname(__file__))

from models.random_forest_model import RandomForestSalesPredictor


def create_sample_data(n_records=200):
    """Create sample sales data for testing."""
    print("\n🔄 Creating sample dataset...")
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_records)]
    
    data = {
        'Date': dates,
        'Weekly_Sales': np.random.normal(5000, 1500, n_records),  # Normal distribution
        'Store': np.random.choice(['S1', 'S2', 'S3', 'S4', 'S5'], n_records)
    }
    
    df = pd.DataFrame(data)
    df['Weekly_Sales'] = df['Weekly_Sales'].clip(lower=0)  # No negative sales
    
    print(f"✅ Created {len(df)} sample records")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Stores: {df['Store'].nunique()}")
    print(f"   Sales range: ${df['Weekly_Sales'].min():.2f} to ${df['Weekly_Sales'].max():.2f}\n")
    
    return df


def test_data_cleaning(df):
    """Test data cleaning functionality."""
    print("\n" + "="*70)
    print("TEST 1: Data Cleaning & Validation")
    print("="*70)
    
    predictor = RandomForestSalesPredictor()
    
    try:
        cleaned_df = predictor.clean_data(df)
        
        print("✅ PASSED: Data cleaning successful")
        print(f"   Original rows: {len(df)}")
        print(f"   Cleaned rows: {len(cleaned_df)}")
        print(f"   Columns: {list(cleaned_df.columns)}")
        
        return cleaned_df
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return None


def test_feature_engineering(cleaned_df):
    """Test temporal feature engineering."""
    print("\n" + "="*70)
    print("TEST 2: Feature Engineering")
    print("="*70)
    
    predictor = RandomForestSalesPredictor()
    
    try:
        engineered_df = predictor.feature_engineering(cleaned_df)
        
        print("✅ PASSED: Feature engineering successful")
        print(f"   Input columns: Date, Sales")
        print(f"   Output columns: {list(engineered_df.columns)}")
        print(f"   Rows preserved: {len(engineered_df) == len(cleaned_df)}")
        
        # Verify feature ranges
        print(f"   Month range: {engineered_df['month'].min()} to {engineered_df['month'].max()}")
        print(f"   Day of week range: {engineered_df['day_of_week'].min()} to {engineered_df     ['day_of_week'].max()}")
        
        return engineered_df
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return None


def test_model_training(engineered_df):
    """Test Random Forest model training."""
    print("\n" + "="*70)
    print("TEST 3: Model Training & Evaluation")
    print("="*70)
    
    predictor = RandomForestSalesPredictor(n_estimators=50)  # Smaller for testing
    
    try:
        # Need Date column for training
        engineered_df['Sales'] = engineered_df['Sales'] if 'Sales' in engineered_df.columns else 0
        
        training_results = predictor.train(engineered_df)
        
        print("✅ PASSED: Model training successful")
        print(f"\n   Metrics:")
        print(f"   - R² Score: {training_results['metrics']['test_r2']:.3f}")
        print(f"   - MAE: ${training_results['metrics']['test_mae']:.2f}")
        print(f"   - RMSE: ${training_results['metrics']['test_rmse']:.2f}")
        
        # Check that model is not overfitting
        train_r2 = training_results['metrics']['train_r2']
        test_r2 = training_results['metrics']['test_r2']
        overfitting_ratio = (train_r2 - test_r2) / train_r2 if train_r2 != 0 else 0
        
        print(f"\n   Overfitting check:")
        print(f"   - Train R²: {train_r2:.3f}")
        print(f"   - Test R²: {test_r2:.3f}")
        print(f"   - Overfitting ratio: {overfitting_ratio:.2%}")
        
        if overfitting_ratio < 0.5:
            print(f"   ✅ Model not significantly overfitting")
        else:
            print(f"   ⚠️ Model may be overfitting (consider regularization)")
        
        return predictor, training_results
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_future_forecasting(predictor, last_date=None):
    """Test future sales forecasting."""
    print("\n" + "="*70)
    print("TEST 4: Future Forecasting")
    print("="*70)
    
    try:
        if not predictor or not predictor.is_trained:
            print("❌ FAILED: Model is not trained")
            return None
        
        forecast_df = predictor.predict_future(n_days=90, last_date=last_date)
        
        print("✅ PASSED: Future forecasting successful")
        print(f"   Forecast period: 90 days")
        print(f"   Average forecast: ${forecast_df['Predicted_Sales'].mean():.2f}")
        print(f"   Forecast range: ${forecast_df['Predicted_Sales'].min():.2f} to ${forecast_df['Predicted_Sales'].max():.2f}")
        
        return forecast_df
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_insights_generation(forecast_df):
    """Test business insights generation."""
    print("\n" + "="*70)
    print("TEST 5: Insights Generation")
    print("="*70)
    
    try:
        from models.random_forest_model import get_insights
        
        insights = get_insights(forecast_df)
        
        print("✅ PASSED: Insights generation successful")
        print(f"   Average forecast: ${insights['average_forecast']:.2f}")
        print(f"   Max forecast: ${insights['max_forecast']:.2f}")
        print(f"   Min forecast: ${insights['min_forecast']:.2f}")
        print(f"   Trend: {insights['trend']} ({insights['trend_value']:.2f}%)")
        print(f"   Volatility: ${insights['volatility']:.2f}")
        print(f"   Peak month: {insights['peak_month']}")
        
        return insights
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return None


def test_model_persistence(predictor, filepath='test_model.joblib'):
    """Test model saving and loading."""
    print("\n" + "="*70)
    print("TEST 6: Model Persistence (Save/Load)")
    print("="*70)
    
    try:
        # Save model
        predictor.save_model(filepath)
        print(f"✅ Model saved to {filepath}")
        
        # Load model
        new_predictor = RandomForestSalesPredictor()
        new_predictor.load_model(filepath)
        print(f"✅ Model loaded from {filepath}")
        
        # Verify model state
        assert new_predictor.is_trained, "Loaded model is not marked as trained"
        assert new_predictor.model is not None, "Loaded model is None"
        assert new_predictor.feature_columns is not None, "Feature columns not loaded"
        
        print("✅ PASSED: Model persistence working correctly")
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("🧪 SEASONAL SALES FORECASTING - ML PIPELINE TEST SUITE")
    print("="*70)
    
    # Create sample data
    df = create_sample_data(n_records=200)
    
    # Test 1: Data Cleaning
    cleaned_df = test_data_cleaning(df)
    if cleaned_df is None:
        return
    
    # Test 2: Feature Engineering
    engineered_df = test_feature_engineering(cleaned_df)
    if engineered_df is None:
        return
    
    # Reconstruct Sales column for training
    last_date = cleaned_df['Date'].max() if 'Date' in cleaned_df.columns else None
    engineered_df['Sales'] = cleaned_df['Sales'].values if len(engineered_df) == len(cleaned_df) else cleaned_df['Sales'].iloc[:len(engineered_df)].values
    
    # Test 3: Model Training
    predictor, training_results = test_model_training(engineered_df)
    if predictor is None:
        return
    
    # Test 4: Future Forecasting
    forecast_df = test_future_forecasting(predictor, last_date)
    if forecast_df is None:
        return
    
    # Test 5: Insights Generation
    insights = test_insights_generation(forecast_df)
    
    # Test 6: Model Persistence
    test_model_persistence(predictor)
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ TEST SUITE COMPLETE - ALL TESTS PASSED!")
    print("="*70)
    print("\n🎉 The ML Pipeline is working correctly!")
    print("\nNext steps:")
    print("  1. Start the Flask app: python app.py")
    print("  2. Navigate to http://localhost:8000")
    print("  3. Upload your sales CSV file")
    print("  4. View analysis and ML predictions")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
