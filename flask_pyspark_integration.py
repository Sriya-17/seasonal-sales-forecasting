#!/usr/bin/env python3
"""
PySpark Integration Example for Flask App
================================================================================
Complete example showing how to integrate PySpark with Flask application.

This module demonstrates:
  - Spark session initialization
  - Data processing with PySpark
  - Model training with scikit-learn
  - Integration with Flask routes

Author: Seasonal Sales Forecasting App
License: MIT
================================================================================
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark_data_processor import PySparkDataProcessor
from pyspark_ml_model import SklearnMLModel
from spark_config import SparkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlaskPySparkIntegration:
    """
    Integration class for PySpark with Flask application.
    
    This class provides methods that Flask routes can use to process
    large CSV files and train ML models efficiently.
    """
    
    def __init__(self):
        """Initialize Spark and ML components"""
        self.spark = SparkConfig.get_spark_session()
        self.processor = PySparkDataProcessor(self.spark)
        self.model = SklearnMLModel()
        self.feature_columns = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
        
        logger.info("✅ FlaskPySparkIntegration initialized")
    
    def process_and_train(self, csv_file_path):
        """
        Complete workflow: Load → Process → Train → Predict
        
        This method handles the entire ML pipeline from CSV upload to predictions.
        
        Args:
            csv_file_path (str): Path to uploaded CSV file
            
        Returns:
            dict: Results containing:
                - status (str): 'success' or 'error'
                - metrics (dict): Training metrics (R², MAE, RMSE)
                - predictions (list): Future predictions
                - feature_importance (dict): Feature importance scores
                - data_summary (dict): Summary statistics
                - error (str): Error message if failed
                
        Example:
            >>> integration = FlaskPySparkIntegration()
            >>> result = integration.process_and_train('uploads/sales_data.csv')
            >>> if result['status'] == 'success':
            ...     print(result['metrics'])
        """
        try:
            logger.info(f"Starting complete ML pipeline for {csv_file_path}")
            
            # Step 1: Process data with PySpark
            logger.info("Step 1: Data processing with PySpark...")
            df_spark, df_pandas = self.processor.full_pipeline(
                csv_file_path,
                date_column='Date',
                sales_column='Sales',
                missing_strategy='drop',
                convert_to_pandas_flag=True
            )
            
            if df_pandas is None or df_pandas.empty:
                return {
                    'status': 'error',
                    'error': 'No valid data after processing'
                }
            
            # Step 2: Get data summary
            logger.info("Step 2: Calculating data summary...")
            data_summary = self._get_data_summary(df_pandas)
            
            # Step 3: Train model
            logger.info("Step 3: Training ML model...")
            metrics = self.model.train(df_pandas, self.feature_columns, label_column='Sales')
            
            # Step 4: Generate predictions
            logger.info("Step 4: Generating future predictions...")
            predictions, prediction_summary = self._generate_predictions(df_pandas)
            
            # Step 5: Get feature importance
            logger.info("Step 5: Extracting feature importance...")
            feature_importance = self.model.get_feature_importance()
            
            logger.info("✅ ML pipeline completed successfully")
            
            return {
                'status': 'success',
                'metrics': metrics,
                'predictions': predictions,
                'feature_importance': feature_importance,
                'data_summary': data_summary,
                'prediction_summary': prediction_summary
            }
        
        except Exception as e:
            logger.error(f"❌ Error in ML pipeline: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_data_summary(self, df_pandas):
        """
        Calculate summary statistics from data.
        
        Args:
            df_pandas (pd.DataFrame): Processed data
            
        Returns:
            dict: Summary statistics
        """
        return {
            'total_records': len(df_pandas),
            'date_range': {
                'start': df_pandas['Date'].min().strftime('%Y-%m-%d'),
                'end': df_pandas['Date'].max().strftime('%Y-%m-%d'),
            },
            'total_sales': float(df_pandas['Sales'].sum()),
            'average_sales': float(df_pandas['Sales'].mean()),
            'min_sales': float(df_pandas['Sales'].min()),
            'max_sales': float(df_pandas['Sales'].max()),
            'std_sales': float(df_pandas['Sales'].std()),
        }
    
    def _generate_predictions(self, df_pandas):
        """
        Generate 365-day future predictions.
        
        Args:
            df_pandas (pd.DataFrame): Training data with engineered features
            
        Returns:
            tuple: (predictions_list, summary_dict)
                - predictions_list: List of dicts with date and predicted sales
                - summary_dict: Min, max, avg of forecast
        """
        # Get last date
        last_date = pd.to_datetime(df_pandas['Date'].max())
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365, freq='D')
        
        # Create future DataFrame with features
        future_df = pd.DataFrame({'Date': future_dates})
        
        # Add engineered features
        future_df['year'] = future_df['Date'].dt.year
        future_df['month'] = future_df['Date'].dt.month
        future_df['day'] = future_df['Date'].dt.day
        future_df['week'] = future_df['Date'].dt.isocalendar().week
        future_df['day_of_week'] = future_df['Date'].dt.dayofweek
        future_df['quarter'] = future_df['Date'].dt.quarter
        future_df['day_of_year'] = future_df['Date'].dt.dayofyear
        
        # Make predictions
        predictions_df = self.model.predict_future(future_df)
        
        # Convert to list format for JSON serialization
        predictions_list = []
        for _, row in predictions_df.iterrows():
            predictions_list.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'predicted_sales': float(row['Predicted_Sales']),
                'month': row['Date'].strftime('%B %Y'),
            })
        
        # Calculate summary
        pred_values = [p['predicted_sales'] for p in predictions_list]
        summary = {
            'average_forecast': float(np.mean(pred_values)),
            'min_forecast': float(np.min(pred_values)),
            'max_forecast': float(np.max(pred_values)),
            'total_predictions': len(predictions_list)
        }
        
        return predictions_list, summary
    
    def cleanup(self):
        """Stop Spark session and cleanup resources"""
        SparkConfig.stop_spark_session()
        logger.info("✅ Cleaned up Spark resources")


# ============================================================================
# FLASK ROUTE EXAMPLE
# ============================================================================

def get_flask_integration_example():
    """
    Returns example Flask route code for using PySpark integration.
    
    This shows how to use FlaskPySparkIntegration in your Flask app.
    """
    
    example_code = '''
# In your Flask app (app.py)

from flask import Flask, request, session, jsonify
from flask_pyspark_integration import FlaskPySparkIntegration

app = Flask(__name__)
pyspark_integration = FlaskPySparkIntegration()

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV upload and trigger ML pipeline"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    import os
    from werkzeug.utils import secure_filename
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    # Process with PySpark
    result = pyspark_integration.process_and_train(filepath)
    
    if result['status'] == 'error':
        return jsonify(result), 500
    
    # Store results in session
    session['upload_completed'] = True
    session['training_results'] = result['metrics']
    session['future_predictions'] = result['predictions']
    session['feature_importance'] = result['feature_importance']
    session['data_summary'] = result['data_summary']
    
    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'message': 'File processed successfully',
        'metrics': result['metrics'],
        'predictions_count': result['prediction_summary']['total_predictions']
    }), 200

@app.teardown_appcontext
def shutdown_spark(exception=None):
    """Clean up Spark on app shutdown"""
    pyspark_integration.cleanup()

if __name__ == '__main__':
    app.run(debug=True, port=8000)
    '''
    
    return example_code


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Example: Basic usage"""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize
    integration = FlaskPySparkIntegration()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    sales = np.random.uniform(2000, 5000, 365)
    df_sample = pd.DataFrame({'Date': dates, 'Sales': sales})
    df_sample.to_csv('sample_sales.csv', index=False)
    
    # Process and train
    result = integration.process_and_train('sample_sales.csv')
    
    print(f"\nStatus: {result['status']}")
    if result['status'] == 'success':
        print(f"R² Score: {result['metrics']['test_r2']:.4f}")
        print(f"MAE: ${result['metrics']['test_mae']:.2f}")
        print(f"RMSE: ${result['metrics']['test_rmse']:.2f}")
        print(f"Predictions: {result['prediction_summary']['total_predictions']}")
        print(f"\nFeature Importance:")
        for feat, importance in result['feature_importance'].items():
            print(f"  {feat}: {importance:.4f}")
    else:
        print(f"Error: {result['error']}")
    
    # Cleanup
    import os
    os.remove('sample_sales.csv')
    integration.cleanup()


def example_session_integration():
    """Example: Integration with Flask session"""
    print("="*60)
    print("EXAMPLE 2: Flask Session Integration")
    print("="*60)
    
    # Simulate Flask session
    session_data = {}
    
    integration = FlaskPySparkIntegration()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    sales = np.random.uniform(3000, 6000, 365)
    df_sample = pd.DataFrame({'Date': dates, 'Sales': sales})
    df_sample.to_csv('sample_sales2.csv', index=False)
    
    # Process
    result = integration.process_and_train('sample_sales2.csv')
    
    # Store in session
    session_data['upload_completed'] = True
    session_data['training_results'] = result['metrics']
    session_data['future_predictions'] = result['predictions'][:7]  # First week
    session_data['data_summary'] = result['data_summary']
    
    print(f"\nSession Data:")
    print(f"  Upload Completed: {session_data['upload_completed']}")
    print(f"  Training Metrics:")
    for key, val in session_data['training_results'].items():
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
        else:
            print(f"    {key}: {val}")
    print(f"  Data Summary:")
    for key, val in session_data['data_summary'].items():
        if isinstance(val, dict):
            print(f"    {key}: {val}")
        else:
            print(f"    {key}: {val}")
    print(f"  First prediction: {session_data['future_predictions'][0]}")
    
    # Cleanup
    import os
    os.remove('sample_sales2.csv')
    integration.cleanup()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PYSPARK FLASK INTEGRATION EXAMPLES")
    print("="*60 + "\n")
    
    # Show route example
    print(get_flask_integration_example())
    
    print("\n" + "="*60)
    print("\n")
    
    # Run examples
    example_basic_usage()
    print("\n")
    example_session_integration()
    
    print("\n" + "="*60)
    print("✅ All examples completed")
    print("="*60)
