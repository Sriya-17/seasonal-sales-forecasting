#!/usr/bin/env python3
"""
Test Suite for PySpark Integration
================================================================================
Comprehensive tests for PySpark data processing and ML models.

Test Coverage:
  - Data loading and validation
  - Missing value handling
  - Feature engineering
  - Model training
  - Predictions
  - Data type conversions

Run with: python test_pyspark_integration.py
================================================================================
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
from spark_config import SparkConfig, initialize_spark
from pyspark_data_processor import PySparkDataProcessor
from pyspark_ml_model import PySparkMLModel, SklearnMLModel, HybridMLPipeline


def create_sample_csv(filename='test_data.csv', num_rows=365):
    """
    Create a sample sales CSV for testing.
    
    Args:
        filename (str): Output CSV filename
        num_rows (int): Number of rows to generate
        
    Returns:
        str: Path to created file
    """
    dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='D')
    sales = np.random.uniform(1000, 5000, num_rows)  # Random sales 1K-5K
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales
    })
    
    df.to_csv(filename, index=False)
    logger.info(f"✅ Created sample CSV: {filename} ({num_rows} rows)")
    return filename


def test_spark_session():
    """Test Spark session creation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Spark Session Creation")
    logger.info("="*60)
    
    try:
        spark = SparkConfig.get_spark_session()
        assert spark is not None, "Spark session is None"
        
        # Test system info
        system_info = SparkConfig.get_system_info()
        logger.info(f"✅ Spark Version: {system_info['spark_version']}")
        logger.info(f"✅ Master: {system_info['master']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_csv_loading():
    """Test CSV loading with PySpark"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: CSV Loading with PySpark")
    logger.info("="*60)
    
    try:
        # Create sample data
        csv_file = create_sample_csv('test_sales.csv', num_rows=100)
        
        # Initialize processor
        spark = SparkConfig.get_spark_session()
        processor = PySparkDataProcessor(spark)
        
        # Load with Spark
        df_spark = processor.load_csv(csv_file)
        
        # Verify
        row_count = df_spark.count()
        assert row_count == 100, f"Expected 100 rows, got {row_count}"
        
        columns = df_spark.columns
        expected_cols = ['Date', 'Sales']
        assert all(col in columns for col in expected_cols), "Missing required columns"
        
        logger.info(f"✅ Loaded {row_count} rows")
        logger.info(f"✅ Columns: {columns}")
        
        # Cleanup
        os.remove(csv_file)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_missing_values():
    """Test missing value handling"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Missing Value Handling")
    logger.info("="*60)
    
    try:
        # Create data with missing values
        spark = SparkConfig.get_spark_session()
        
        # Create Spark DataFrame with NaN
        data = [
            ('2023-01-01', 1000.0),
            ('2023-01-02', None),  # Missing
            ('2023-01-03', 2000.0),
            ('2023-01-04', None),  # Missing
            ('2023-01-05', 3000.0),
        ]
        
        df_with_nulls = spark.createDataFrame(data, ['Date', 'Sales'])
        
        logger.info(f"Rows with NaN: {df_with_nulls.count()}")
        
        # Handle missing
        processor = PySparkDataProcessor(spark)
        df_clean = processor.handle_missing_values(df_with_nulls, strategy='drop')
        
        clean_count = df_clean.count()
        assert clean_count == 3, f"Expected 3 rows after drop, got {clean_count}"
        
        logger.info(f"✅ After dropping NaN: {clean_count} rows")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_feature_engineering():
    """Test temporal feature engineering"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Feature Engineering")
    logger.info("="*60)
    
    try:
        # Create sample data
        csv_file = create_sample_csv('test_features.csv', num_rows=30)
        
        spark = SparkConfig.get_spark_session()
        processor = PySparkDataProcessor(spark)
        
        # Load and engineer
        df_spark = processor.load_csv(csv_file)
        df_features = processor.engineer_features(df_spark)
        
        # Verify features created
        expected_features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
        df_columns = df_features.columns
        
        for feat in expected_features:
            assert feat in df_columns, f"Missing feature: {feat}"
        
        # Show first row
        df_features.show(1)
        
        logger.info(f"✅ All {len(expected_features)} features created")
        logger.info(f"✅ Columns: {df_columns}")
        
        # Cleanup
        os.remove(csv_file)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_full_pipeline():
    """Test complete data processing pipeline"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Full Data Processing Pipeline")
    logger.info("="*60)
    
    try:
        # Create sample data
        csv_file = create_sample_csv('test_pipeline.csv', num_rows=365)
        
        spark = SparkConfig.get_spark_session()
        processor = PySparkDataProcessor(spark)
        
        # Run full pipeline
        df_spark, df_pandas = processor.full_pipeline(
            csv_file,
            date_column='Date',
            sales_column='Sales',
            missing_strategy='drop',
            convert_to_pandas_flag=True
        )
        
        # Verify Spark DataFrame
        assert df_spark is not None, "Spark DataFrame is None"
        spark_rows = df_spark.count()
        logger.info(f"✅ Spark DataFrame: {spark_rows} rows")
        
        # Verify Pandas DataFrame
        assert df_pandas is not None, "Pandas DataFrame is None"
        assert isinstance(df_pandas, pd.DataFrame), "Not a Pandas DataFrame"
        assert df_pandas.shape[0] == spark_rows, "Row count mismatch"
        
        # Check for required features
        required_cols = ['Date', 'Sales', 'year', 'month', 'day', 'week', 
                        'day_of_week', 'quarter', 'day_of_year']
        for col in required_cols:
            assert col in df_pandas.columns, f"Missing column: {col}"
        
        logger.info(f"✅ Pandas DataFrame: {df_pandas.shape}")
        logger.info(f"✅ Columns: {df_pandas.columns.tolist()}")
        
        # Cleanup
        os.remove(csv_file)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_sklearn_training():
    """Test scikit-learn model training"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Scikit-Learn Model Training")
    logger.info("="*60)
    
    try:
        # Create sample data
        csv_file = create_sample_csv('test_training.csv', num_rows=365)
        
        spark = SparkConfig.get_spark_session()
        processor = PySparkDataProcessor(spark)
        
        # Process data
        df_spark, df_pandas = processor.full_pipeline(csv_file)
        
        # Train model
        model = SklearnMLModel(n_estimators=50, max_depth=15)
        features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
        
        metrics = model.train(df_pandas, features, label_column='Sales')
        
        # Verify metrics
        assert 'test_r2' in metrics, "Missing R² score"
        assert 'test_mae' in metrics, "Missing MAE"
        assert 'test_rmse' in metrics, "Missing RMSE"
        
        assert 0 <= metrics['test_r2'] <= 1, "Invalid R² score"
        assert metrics['test_mae'] > 0, "Invalid MAE"
        assert metrics['test_rmse'] > 0, "Invalid RMSE"
        
        logger.info(f"✅ R² Score: {metrics['test_r2']:.4f}")
        logger.info(f"✅ MAE: ${metrics['test_mae']:.2f}")
        logger.info(f"✅ RMSE: ${metrics['test_rmse']:.2f}")
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) > 0, "No feature importance"
        logger.info(f"✅ Feature importance: {importance}")
        
        # Cleanup
        os.remove(csv_file)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def test_predictions():
    """Test future predictions"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: Future Predictions")
    logger.info("="*60)
    
    try:
        # Create sample data
        csv_file = create_sample_csv('test_predictions.csv', num_rows=365)
        
        spark = SparkConfig.get_spark_session()
        processor = PySparkDataProcessor(spark)
        
        # Process data
        df_spark, df_pandas = processor.full_pipeline(csv_file)
        
        # Train model
        model = SklearnMLModel(n_estimators=50)
        features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
        model.train(df_pandas, features)
        
        # Generate future dates
        last_date = pd.to_datetime(df_pandas['Date'].max())
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        # Create future DataFrame with features
        future_df = pd.DataFrame({'Date': future_dates})
        for feat in features:
            if feat == 'year':
                future_df[feat] = future_df['Date'].dt.year
            elif feat == 'month':
                future_df[feat] = future_df['Date'].dt.month
            elif feat == 'day':
                future_df[feat] = future_df['Date'].dt.day
            elif feat == 'week':
                future_df[feat] = future_df['Date'].dt.isocalendar().week
            elif feat == 'day_of_week':
                future_df[feat] = future_df['Date'].dt.dayofweek
            elif feat == 'quarter':
                future_df[feat] = future_df['Date'].dt.quarter
            elif feat == 'day_of_year':
                future_df[feat] = future_df['Date'].dt.dayofyear
        
        # Predict
        predictions = model.predict_future(future_df)
        
        # Verify
        assert len(predictions) == 30, "Wrong number of predictions"
        assert 'Predicted_Sales' in predictions.columns, "Missing predictions column"
        assert predictions['Predicted_Sales'].notna().all(), "NaN in predictions"
        
        logger.info(f"✅ Generated {len(predictions)} predictions")
        logger.info(f"✅ Sample predictions:\n{predictions[['Date', 'Predicted_Sales']].head()}")
        
        # Cleanup
        os.remove(csv_file)
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*15 + "PYSPARK INTEGRATION TEST SUITE" + " "*13 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    tests = [
        ("Spark Session", test_spark_session),
        ("CSV Loading", test_csv_loading),
        ("Missing Values", test_missing_values),
        ("Feature Engineering", test_feature_engineering),
        ("Full Pipeline", test_full_pipeline),
        ("Model Training", test_sklearn_training),
        ("Predictions", test_predictions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*60)
    
    # Stop Spark
    SparkConfig.stop_spark_session()
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
