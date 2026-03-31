#!/usr/bin/env python3
"""
Demo script for Random Forest Sales Predictor
Shows complete workflow: data creation, training, and prediction
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.sales_predictor import SalesPredictor, create_sample_data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    print("🎯 Random Forest Sales Predictor Demo")
    print("=" * 50)

    # Step 1: Create sample data
    print("\n1. 📊 Creating sample sales data...")
    df = create_sample_data(2000)
    print(f"   Generated {len(df)} sales records")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Products: {df['Product'].unique().tolist()}")
    print(f"   Sample data:")
    print(df.head())

    # Step 2: Initialize predictor
    print("\n2. 🤖 Initializing Random Forest predictor...")
    predictor = SalesPredictor(n_estimators=100, random_state=42)

    # Step 3: Train the model
    print("\n3. 🎓 Training the model...")
    training_results = predictor.train(df)

    print("\n📈 Training Results:")
    print(f"   Training R²: {training_results['train_r2']:.3f}")
    print(f"   Training RMSE: ${training_results['train_rmse']:.2f}")
    print(f"   Training MAE: ${training_results['train_mae']:.2f}")
    print(f"   Testing R²: {training_results['test_r2']:.3f}")
    print(f"   Testing RMSE: ${training_results['test_rmse']:.2f}")
    print(f"   Testing MAE: ${training_results['test_mae']:.2f}")
    # Step 4: Make predictions on test data
    print("\n4. 🔮 Making predictions...")
    # Create some test data (without sales column)
    test_data = df.head(10).drop('Sales', axis=1)
    predictions = predictor.predict(test_data)

    print("   Test predictions:")
    for i, pred in enumerate(predictions[:5]):
        actual = df.iloc[i]['Sales']
        print(f"     Record {i+1}: Predicted=${pred:.2f}, Actual=${actual:.2f}")

    # Step 5: Future predictions
    print("\n5. 🔮 Future sales predictions...")
    future_dates = pd.date_range('2024-01-01', periods=30, freq='D')
    future_predictions = predictor.predict_future_sales(
        future_dates,
        product_info={'product': 'Widget_A', 'quantity': 15}
    )

    print("   Future predictions (first 5 days):")
    for _, row in future_predictions.head().iterrows():
        print(f"     {row['date'].strftime('%Y-%m-%d')}: ${row['predicted_sales']:.2f}")
    # Step 6: Visualize results
    print("\n6. 📊 Creating visualizations...")

    # Get some actual vs predicted data for visualization
    processed_data = predictor.preprocess_data(df)
    X, y = predictor.prepare_features_and_target(processed_data)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = predictor.scaler.transform(X_test)
    y_pred = predictor.model.predict(X_test_scaled)

    # Create plots directory
    os.makedirs('static/plots', exist_ok=True)

    # Visualize results
    predictor.visualize_results(y_test, y_pred, save_path='static/plots/sales_demo_results.png')
    predictor.plot_feature_importance(save_path='static/plots/sales_demo_features.png')

    print("   📊 Plot saved to: static/plots/sales_demo_results.png")
    print("   📊 Feature importance saved to: static/plots/sales_demo_features.png")

    # Step 7: Save the model
    print("\n7. 💾 Saving the trained model...")
    predictor.save_model('models/sales_predictor_demo.joblib')

    print("\n✅ Demo completed successfully!")
    print("   📁 Check static/plots/ for visualizations")
    print("   💾 Model saved as models/sales_predictor_demo.joblib")
    print("\n🚀 Ready for Flask integration!")
    print("   Visit /sales-predictor in your browser to use the web interface")

if __name__ == "__main__":
    main()