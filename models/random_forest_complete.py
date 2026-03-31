#!/usr/bin/env python3
"""
Complete Random Forest Model for Seasonal Sales Forecasting
================================================================================
Comprehensive implementation for sales prediction using scikit-learn's Random Forest.

Features:
  - Data cleaning and validation
  - Automatic temporal feature engineering
  - Model training with evaluation metrics
  - Future sales forecasting
  - Feature importance analysis
  - Visualization and insights generation

Author: Seasonal Sales Forecasting App
License: MIT
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
import warnings

warnings.filterwarnings('ignore')


class RandomForestSalesPredictor:
    """
    Random Forest-based sales forecasting model.
    
    This class encapsulates the complete ML pipeline for sales prediction:
    - Data loading and cleaning
    - Feature engineering
    - Model training and evaluation
    - Future forecasting
    - Performance analysis
    """
    
    def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Initialize the Random Forest predictor.
        
        Args:
            n_estimators (int): Number of trees in the random forest. Default: 100
            test_size (float): Proportion of data for testing. Default: 0.2 (80/20 split)
            random_state (int): Random seed for reproducibility. Default: 42
        """
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.training_metrics = None
        self.feature_importance = None
        self.is_trained = False
        
    def clean_data(self, df):
        """
        Clean and prepare the dataset for modeling.
        
        Steps:
        1. Handle missing values (drop rows with NaN)
        2. Remove duplicate rows
        3. Map alternative column names (e.g., Weekly_Sales → Sales)
        4. Aggregate multiple sales per date if needed (for multi-store data)
        
        Args:
            df (pd.DataFrame): Raw input dataset
            
        Returns:
            pd.DataFrame: Cleaned data with Date and Sales columns
            
        Raises:
            ValueError: If Date and Sales columns cannot be found
        """
        print("🧹 Cleaning data...")
        data = df.copy()
        
        # Drop rows with missing values
        initial_rows = len(data)
        data = data.dropna()
        removed_missing = initial_rows - len(data)
        if removed_missing > 0:
            print(f"  ✓ Removed {removed_missing} rows with missing values")
        
        # Remove duplicate rows
        initial_rows = len(data)
        data = data.drop_duplicates()
        removed_dupes = initial_rows - len(data)
        if removed_dupes > 0:
            print(f"  ✓ Removed {removed_dupes} duplicate rows")
        
        # Check for required columns and map if necessary
        if 'Date' not in data.columns or 'Sales' not in data.columns:
            print("  ℹ Mapping column names...")
            col_mapping = {}
            
            for col in data.columns:
                col_lower = col.lower()
                if 'date' in col_lower and 'Date' not in col_mapping:
                    col_mapping['Date'] = col
                    print(f"    → {col} → Date")
                elif ('sales' in col_lower or 'weekly' in col_lower or 'amount' in col_lower) and 'Sales' not in col_mapping:
                    col_mapping['Sales'] = col
                    print(f"    → {col} → Sales")
            
            if 'Date' in col_mapping and 'Sales' in col_mapping:
                data = data.rename(columns=col_mapping)
            else:
                raise ValueError(
                    "❌ Cannot find Date and Sales columns. "
                    "Please ensure your CSV has Date and Sales (or Weekly_Sales) columns."
                )
        
        # Select only necessary columns
        data = data[['Date', 'Sales']]
        
        # Aggregate if multiple sales per date (e.g., multiple stores per date)
        if data['Date'].duplicated().any():
            unique_dates_before = data['Date'].nunique()
            print(f"  ℹ Detected {len(data)} rows with {unique_dates_before} unique dates")
            print(f"  ⚙ Aggregating sales by date...")
            data = data.groupby('Date')['Sales'].sum().reset_index()
            print(f"  ✓ Aggregated to {len(data)} unique date records")
        
        print(f"✅ Data cleaned: {len(data)} records ready for modeling\n")
        return data
    
    def feature_engineering(self, df):
        """
        Generate temporal features from the Date column.
        
        Created features:
        - year: Year of the transaction
        - month: Month (1-12)
        - day: Day of month (1-31)
        - week: Week number (1-53)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - quarter: Quarter (1-4)
        
        Args:
            df (pd.DataFrame): Data with Date and Sales columns
            
        Returns:
            pd.DataFrame: Data with temporal features (Date column removed)
        """
        print("🔧 Engineering temporal features...")
        data = df.copy()
        
        # Convert to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Remove any rows with invalid dates
        initial_rows = len(data)
        data = data.dropna(subset=['Date'])
        if initial_rows > len(data):
            print(f"  ✓ Removed {initial_rows - len(data)} rows with invalid dates")
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        # Extract temporal features
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        data['day'] = data['Date'].dt.day
        data['week'] = data['Date'].dt.isocalendar().week
        data['day_of_week'] = data['Date'].dt.dayofweek
        data['quarter'] = data['Date'].dt.quarter
        
        # Create day-of-year for seasonal patterns
        data['day_of_year'] = data['Date'].dt.dayofyear
        
        # Drop the original Date column
        data = data.drop('Date', axis=1)
        
        print(f"✅ Features engineered: {len(data.columns)} features created")
        print(f"   Features: {', '.join(data.columns.tolist())}\n")
        
        return data
    
    def train(self, df):
        """
        Train the Random Forest model on historical data.
        
        Process:
        1. Clean and validate data
        2. Engineer temporal features
        3. Split into training (80%) and testing (20%) sets
        4. Train Random Forest Regressor
        5. Evaluate on test set
        6. Calculate feature importance
        
        Args:
            df (pd.DataFrame): Raw historical sales data with Date and Sales columns
            
        Returns:
            dict: Training results including metrics and feature importance
            
        Raises:
            ValueError: If insufficient data for training
        """
        print("\n" + "="*70)
        print("🚀 STARTING MODEL TRAINING")
        print("="*70 + "\n")
        
        # Step 1: Clean data
        cleaned_df = self.clean_data(df)
        
        # Minimum data validation
        if len(cleaned_df) < 20:
            raise ValueError(
                f"❌ Insufficient data. Need at least 20 records, got {len(cleaned_df)}"
            )
        
        # Step 2: Engineer features
        engineered_df = self.feature_engineering(cleaned_df)
        
        # Step 3: Prepare features and target
        feature_cols = [col for col in engineered_df.columns if col != 'Sales']
        X = engineered_df[feature_cols].values
        y = engineered_df['Sales'].values
        
        print("📊 Splitting data...")
        print(f"  Train size: {int(len(X) * (1 - self.test_size))} records ({(1-self.test_size)*100}%)")
        print(f"  Test size: {int(len(X) * self.test_size)} records ({self.test_size*100}%)\n")
        
        # Step 4: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Step 5: Train model
        print("🎯 Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.model.fit(X_train, y_train)
        print("✅ Model trained\n")
        
        # Step 6: Evaluate model
        print("📈 Evaluating model on test set...")
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"  Training Metrics:")
        print(f"    - MAE (Mean Absolute Error):  ${train_mae:,.2f}")
        print(f"    - RMSE (Root Mean Squared Error): ${train_rmse:,.2f}")
        print(f"    - R² Score: {train_r2:.3f}\n")
        
        print(f"  Testing Metrics:")
        print(f"    - MAE (Mean Absolute Error):  ${test_mae:,.2f}")
        print(f"    - RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
        print(f"    - R² Score: {test_r2:.3f}\n")
        
        # Step 7: Feature importance
        print("🔍 Feature Importance Analysis...")
        self.feature_columns = feature_cols
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance[:5]:
            print(f"  {feature}: {importance:.4f} ({importance*100:.2f}%)")
        print()
        
        # Store metrics
        self.training_metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
        
        self.is_trained = True
        
        print("="*70)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*70 + "\n")
        
        return {
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance
        }
    
    def generate_future_dates(self, n_days=365, last_date=None):
        """
        Generate future dates for forecasting.
        
        Args:
            n_days (int): Number of days to forecast. Default: 365
            last_date (datetime): Last date in historical data
            
        Returns:
            pd.DataFrame: Future dates with engineered features
        """
        if last_date is None:
            last_date = datetime.now()
        else:
            last_date = pd.to_datetime(last_date)
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]
        future_df = pd.DataFrame({'Date': future_dates})
        
        # Engineer features for future dates
        future_df['year'] = future_df['Date'].dt.year
        future_df['month'] = future_df['Date'].dt.month
        future_df['day'] = future_df['Date'].dt.day
        future_df['week'] = future_df['Date'].dt.isocalendar().week
        future_df['day_of_week'] = future_df['Date'].dt.dayofweek
        future_df['quarter'] = future_df['Date'].dt.quarter
        future_df['day_of_year'] = future_df['Date'].dt.dayofyear
        
        return future_df
    
    def predict_future(self, n_days=365, last_date=None):
        """
        Generate sales predictions for future dates.
        
        Args:
            n_days (int): Number of days to forecast
            last_date (datetime): Last date in historical data
            
        Returns:
            pd.DataFrame: Forecast with date and predictions
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ValueError("❌ Model must be trained before making predictions")
        
        print(f"\n🔮 Generating {n_days}-day forecast...\n")
        
        # Generate future dates
        future_df = self.generate_future_dates(n_days, last_date)
        
        # Prepare features in same order as training
        X_future = future_df[self.feature_columns].values
        
        # Make predictions
        predictions = self.model.predict(X_future)
        
        # Add predictions to dataframe
        future_df['Predicted_Sales'] = predictions
        
        # Calculate confidence intervals (simplified)
        future_df['Confidence_Lower'] = predictions * 0.9
        future_df['Confidence_Upper'] = predictions * 1.1
        
        print(f"✅ Forecast generated: {len(future_df)} predictions")
        print(f"   Average predicted sales: ${predictions.mean():,.2f}")
        print(f"   Min predicted sales: ${predictions.min():,.2f}")
        print(f"   Max predicted sales: ${predictions.max():,.2f}\n")
        
        return future_df
    
    def save_model(self, filepath):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model file (.joblib)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to model file
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.training_metrics = data['training_metrics']
        self.feature_importance = data['feature_importance']
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")


# Standalone functions for compatibility with existing code
def clean_data(df):
    """Standalone function for data cleaning."""
    predictor = RandomForestSalesPredictor()
    return predictor.clean_data(df)


def feature_engineering(df):
    """Standalone function for feature engineering."""
    predictor = RandomForestSalesPredictor()
    return predictor.feature_engineering(df)


def train_model(data):
    """Standalone function for model training."""
    predictor = RandomForestSalesPredictor()
    result = predictor.train(data)
    return {
        'model': predictor.model,
        'metrics': result['metrics'],
        'feature_importance': result['feature_importance'],
        'feature_columns': predictor.feature_columns
    }


def generate_future_dates(n_days=365, last_date=None):
    """Standalone function for generating future dates."""
    predictor = RandomForestSalesPredictor()
    return predictor.generate_future_dates(n_days, last_date)


def predict_future(trained_model_dict, future_dates_df, last_date=None):
    """Standalone function for predictions."""
    predictor = RandomForestSalesPredictor()
    predictor.model = trained_model_dict['model']
    predictor.feature_columns = trained_model_dict['feature_columns']
    predictor.is_trained = True
    return predictor.predict_future(len(future_dates_df), last_date)


def create_forecast_plot(predictions_df):
    """
    Create visualization of forecast.
    
    Args:
        predictions_df (pd.DataFrame): Dataframe with predictions
        
    Returns:
        str: Path to saved plot image
    """
    try:
        plt.figure(figsize=(14, 6))
        
        dates = pd.to_datetime(predictions_df['Date'])
        predictions = predictions_df['Predicted_Sales']
        lower = predictions_df['Confidence_Lower']
        upper = predictions_df['Confidence_Upper']
        
        plt.plot(dates, predictions, linewidth=2, label='Forecast', color='#667eea')
        plt.fill_between(dates, lower, upper, alpha=0.2, color='#667eea', label='Confidence Interval')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Predicted Sales ($)', fontsize=12)
        plt.title('365-Day Sales Forecast', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join('static', 'plots', 'forecast_plot.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None


def get_insights(predictions_df):
    """
    Generate business insights from predictions.
    
    Args:
        predictions_df (pd.DataFrame): Dataframe with predictions
        
    Returns:
        dict: Dictionary of insights
    """
    predictions = predictions_df['Predicted_Sales']
    
    insights = {
        'average_forecast': float(predictions.mean()),
        'max_forecast': float(predictions.max()),
        'min_forecast': float(predictions.min()),
        'trend': 'increasing' if predictions.iloc[-1] > predictions.iloc[0] else 'decreasing',
        'trend_value': float(((predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0] * 100)),
        'volatility': float(predictions.std()),
        'peak_month': predictions_df.loc[predictions.idxmax(), 'Date'].strftime('%B'),
    }
    
    return insights
