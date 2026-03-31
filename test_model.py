#!/usr/bin/env python3
"""
Test the pre-trained ML model predictions
"""

import pandas as pd
from models.sales_predictor import SalesPredictor

def test_model_predictions():
    """Test the pre-trained model with sample data"""
    print("🧪 Testing Pre-trained ML Model Predictions")
    print("=" * 50)

    # Load the pre-trained model
    predictor = SalesPredictor()
    try:
        predictor.load_model('models/trained_sales_predictor.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create test data similar to what users might upload
    test_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Product': ['Laptop', 'Phone', 'Tablet'],
        'Quantity': [2, 1, 3],
        'Sales': [0, 0, 0]  # Will be predicted
    })

    print("📊 Test Data:")
    print(test_data)
    print()

    # Make predictions
    try:
        predictions = predictor.predict(test_data.drop('Sales', axis=1))
        print("🎯 Predictions:")
        for i, pred in enumerate(predictions):
            print(".2f")
        print()

        print("✅ Model prediction test completed successfully!")
        print("The pre-trained model is ready for use in the web application.")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    test_model_predictions()