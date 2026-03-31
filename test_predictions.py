#!/usr/bin/env python3
"""
Test script for generate_future_predictions function
"""
import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import generate_future_predictions

def test_generate_predictions():
    """Test the generate_future_predictions function"""
    try:
        # Load the uploaded data
        df = pd.read_csv('data/uploaded_sales.csv')
        print(f"Loaded data: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Test the function
        predictions, model, insights = generate_future_predictions(df)
        print(f"Success! Generated {len(predictions)} predictions")

        if predictions:
            print("First prediction:", predictions[0])
            print("Last prediction:", predictions[-1])

        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generate_predictions()
    sys.exit(0 if success else 1)