#!/usr/bin/env python3
"""
Train the ML model for sales prediction
This script trains a Random Forest model on sample sales data
and saves it for use in the web application
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from models.sales_predictor import SalesPredictor, train_sales_model_from_csv

def create_training_data(n_samples=2000):
    """
    Create sample sales training data in the expected format

    Args:
        n_samples (int): Number of samples to generate

    Returns:
        pd.DataFrame: Training data
    """
    np.random.seed(42)

    # Generate dates over 2 years
    start_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start_date, periods=n_samples, freq='D')

    # Products and their base prices
    products = {
        'Laptop': 1200,
        'Phone': 800,
        'Tablet': 500,
        'Headphones': 150,
        'Mouse': 50,
        'Keyboard': 100,
        'Monitor': 300,
        'Printer': 200
    }

    data = []
    for date in dates:
        # Select random product
        product = np.random.choice(list(products.keys()))

        # Generate quantity (1-10)
        quantity = np.random.randint(1, 11)

        # Base sales calculation
        base_sales = products[product] * quantity

        # Add seasonal effects
        month = date.month
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)  # Peak in summer
        if month in [11, 12]:  # Holiday season boost
            seasonal_factor *= 1.5

        # Day of week effect (weekends higher)
        day_factor = 1.2 if date.dayofweek >= 5 else 1.0

        # Calculate final sales with noise
        sales = base_sales * seasonal_factor * day_factor
        sales += np.random.normal(0, sales * 0.1)  # Add 10% noise
        sales = max(0, round(sales, 2))

        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Product': product,
            'Quantity': quantity,
            'Sales': sales
        })

    return pd.DataFrame(data)

def main():
    """Main training function"""
    print("🚀 Starting ML Model Training for Sales Prediction")
    print("=" * 60)

    # Create training data
    print("📊 Creating training dataset...")
    train_df = create_training_data(2000)
    print(f"✅ Created {len(train_df)} training samples")
    print(f"   Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"   Products: {train_df['Product'].nunique()}")
    print(f"   Total sales: ${train_df['Sales'].sum():,.2f}")
    print()

    # Save training data
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    train_csv_path = os.path.join(data_dir, 'training_sales_data.csv')
    train_df.to_csv(train_csv_path, index=False)
    print(f"💾 Training data saved to: {train_csv_path}")
    print()

    # Train the model
    print("🤖 Training Random Forest model...")
    model_save_path = os.path.join('models', 'trained_sales_predictor.pkl')
    os.makedirs('models', exist_ok=True)

    result = train_sales_model_from_csv(train_csv_path, model_save_path)

    if result['success']:
        print("✅ Model training completed successfully!")
        metrics = result['results']
        print(f"📈 Training R²: {metrics['train_r2']:.3f}")
        print(f"📈 Testing R²: {metrics['test_r2']:.3f}")
        print(f"📊 Testing RMSE: ${metrics['test_rmse']:.2f}")
        print(f"📊 Testing MAE: ${metrics['test_mae']:.2f}")
        print(f"📈 Feature importance: {list(metrics['feature_importance'].keys())[:3]}...")
        print()
        print(f"💾 Model saved to: {model_save_path}")
    else:
        print(f"❌ Training failed: {result['error']}")
        return False

    # Test the model with sample predictions
    print("🧪 Testing model predictions...")
    predictor = result['predictor']

    # Create test data
    test_dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_data = []
    for date in test_dates:
        test_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Product': 'Laptop',
            'Quantity': 2,
            'Sales': 0  # Will be predicted
        })

    test_df = pd.DataFrame(test_data)
    predictions = predictor.predict(test_df.drop('Sales', axis=1))

    print("✅ Sample predictions generated")
    print(f"   Average predicted sales for Laptop (qty=2): ${predictions.mean():.2f}")
    print()

    print("🎉 ML Model Training Complete!")
    print("The model is now ready for use in the web application.")
    print("Users can upload their sales data and get predictions based on this trained model.")

    return True

if __name__ == "__main__":
    main()