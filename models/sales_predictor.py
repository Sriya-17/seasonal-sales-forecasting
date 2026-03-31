#!/usr/bin/env python3
"""
Random Forest Sales Prediction Model
Comprehensive implementation for sales forecasting using machine learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SalesPredictor:
    """
    Random Forest-based sales prediction model
    Handles CSV data with Date, Product, Quantity, and Sales columns
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the sales predictor

        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'Weekly_Sales'
        self.is_trained = False

    def preprocess_data(self, df):
        """
        Preprocess the CSV data for training/prediction

        Args:
            df (pd.DataFrame): Raw CSV data with Date, Product, Quantity, Sales columns

        Returns:
            pd.DataFrame: Preprocessed data ready for modeling
        """
        print("🔄 Starting data preprocessing...")

        # Create a copy to avoid modifying original data
        data = df.copy()

        # Convert Date column to datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])  # Remove rows with invalid dates

            # Extract date features
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day'] = data['Date'].dt.day
            data['day_of_week'] = data['Date'].dt.dayofweek
            data['quarter'] = data['Date'].dt.quarter
            data['day_of_year'] = data['Date'].dt.dayofyear

            # Drop original Date column
            data = data.drop('Date', axis=1)

        # Handle categorical columns (like Product)
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle new categories not seen during training
                try:
                    data[col] = self.label_encoders[col].transform(data[col])
                except ValueError:
                    # For unseen categories, assign a default value
                    known_categories = set(self.label_encoders[col].classes_)
                    data[col] = data[col].apply(lambda x: x if x in known_categories else list(known_categories)[0])
                    data[col] = self.label_encoders[col].transform(data[col])

        # Ensure numeric columns are properly typed
        numeric_columns = ['quantity_sold', 'Weekly_Sales', 'price', 'inventory_level', 'temperature', 'rainfall', 'year', 'month', 'day', 'day_of_week', 'quarter', 'day_of_year']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values
        data = data.dropna()

        # Remove any duplicate rows
        data = data.drop_duplicates()

        print(f"✅ Preprocessing complete. Shape: {data.shape}")
        print(f"   Features: {list(data.columns)}")

        return data

    def prepare_features_and_target(self, data):
        """
        Prepare features (X) and target (y) for modeling

        Args:
            data (pd.DataFrame): Preprocessed data

        Returns:
            tuple: (X, y) features and target
        """
        # Features are all columns except the target
        feature_cols = [col for col in data.columns if col != self.target_column]

        X = data[feature_cols]
        y = data[self.target_column]

        # Store feature columns for later use
        self.feature_columns = feature_cols

        return X, y

    def train(self, df, test_size=0.2):
        """
        Train the Random Forest model

        Args:
            df (pd.DataFrame): Raw CSV data
            test_size (float): Proportion of data to use for testing

        Returns:
            dict: Training results and metrics
        """
        print("🚀 Starting model training...")

        # Preprocess data
        processed_data = self.preprocess_data(df)

        if len(processed_data) < 10:
            raise ValueError("Insufficient data for training. Need at least 10 samples.")

        # Prepare features and target
        X, y = self.prepare_features_and_target(processed_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))

        self.is_trained = True

        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'n_samples': len(processed_data),
            'n_features': len(self.feature_columns),
            'model_params': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state
            }
        }

        print("✅ Training complete!")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Testing R²: {test_r2:.3f}")
        print(f"   Testing RMSE: ${test_rmse:.2f}")
        print(f"   Testing MAE: ${test_mae:.2f}")

        return results

    def predict(self, df):
        """
        Make predictions on new data

        Args:
            df (pd.DataFrame): New data for prediction

        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Preprocess data
        processed_data = self.preprocess_data(df)

        # Prepare features (without target)
        feature_cols = [col for col in processed_data.columns if col != self.target_column]
        X = processed_data[feature_cols]

        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0  # Default value

        # Reorder columns to match training data
        X = X[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        return predictions

    def predict_future_sales(self, future_dates, product_info=None):
        """
        Predict future sales for given dates and products

        Args:
            future_dates (list): List of future dates
            product_info (dict): Product information for predictions

        Returns:
            pd.DataFrame: Future predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create future data points
        future_data = []

        for date in future_dates:
            if isinstance(date, str):
                date = pd.to_datetime(date)

            data_point = {
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.dayofweek,
                'quarter': date.quarter,
                'day_of_year': date.dayofyear,
                'Quantity': product_info.get('quantity', 1) if product_info else 1
            }

            # Add product encoding if available
            if product_info and 'product' in product_info:
                if 'Product' in self.label_encoders:
                    try:
                        data_point['Product'] = self.label_encoders['Product'].transform([product_info['product']])[0]
                    except ValueError:
                        # Use most common product encoding for unknown products
                        data_point['Product'] = 0

            future_data.append(data_point)

        future_df = pd.DataFrame(future_data)

        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0  # Default value

        # Reorder columns to match training data
        future_df = future_df[self.feature_columns]

        # Scale and predict
        future_scaled = self.scaler.transform(future_df)
        predictions = self.model.predict(future_scaled)

        # Create results DataFrame
        results = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions
        })

        return results

    def visualize_results(self, y_true, y_pred, save_path=None):
        """
        Visualize actual vs predicted sales

        Args:
            y_true (array-like): Actual sales values
            y_pred (array-like): Predicted sales values
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        plt.grid(True, alpha=0.3)

        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Sales')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")

        plt.show()

    def plot_feature_importance(self, save_path=None):
        """
        Plot feature importance

        Args:
            save_path (str): Path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to: {save_path}")

        plt.show()

    def save_model(self, filepath):
        """
        Save the trained model to disk

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }

        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from disk

        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.is_trained = model_data['is_trained']
        self.n_estimators = model_data['n_estimators']
        self.random_state = model_data['random_state']

        print(f"📂 Model loaded from: {filepath}")


# =====================================================
# FLASK INTEGRATION FUNCTIONS
# =====================================================

def train_sales_model_from_csv(csv_path, model_save_path=None):
    """
    Train a sales prediction model from CSV file

    Args:
        csv_path (str): Path to CSV file
        model_save_path (str): Path to save trained model

    Returns:
        dict: Training results
    """
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📄 Loaded data from: {csv_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        # Initialize and train model
        predictor = SalesPredictor()
        results = predictor.train(df)

        # Save model if path provided
        if model_save_path:
            predictor.save_model(model_save_path)

        return {
            'success': True,
            'results': results,
            'predictor': predictor
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def predict_sales_from_csv(model_path, csv_path):
    """
    Make predictions using a trained model

    Args:
        model_path (str): Path to trained model
        csv_path (str): Path to CSV with data to predict

    Returns:
        dict: Prediction results
    """
    try:
        # Load model
        predictor = SalesPredictor()
        predictor.load_model(model_path)

        # Load prediction data
        df = pd.read_csv(csv_path)

        # Make predictions
        predictions = predictor.predict(df)

        return {
            'success': True,
            'predictions': predictions.tolist(),
            'n_predictions': len(predictions)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# =====================================================
# DEMO AND TESTING FUNCTIONS
# =====================================================

def create_sample_data(n_samples=1000):
    """
    Create sample sales data for testing

    Args:
        n_samples (int): Number of samples to generate

    Returns:
        pd.DataFrame: Sample sales data
    """
    np.random.seed(42)

    # Generate dates
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start_date, periods=n_samples, freq='D')

    # Generate products
    products = ['Widget_A', 'Widget_B', 'Widget_C', 'Gadget_X', 'Gadget_Y']
    product_probs = [0.3, 0.25, 0.2, 0.15, 0.1]

    # Generate data
    data = []
    for date in dates:
        product = np.random.choice(products, p=product_probs)
        quantity = np.random.randint(1, 50)

        # Sales depends on product, quantity, month, and some randomness
        base_sales = {
            'Widget_A': 100,
            'Widget_B': 150,
            'Widget_C': 200,
            'Gadget_X': 300,
            'Gadget_Y': 400
        }

        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
        sales = base_sales[product] * quantity * seasonal_factor
        sales += np.random.normal(0, sales * 0.1)  # Add noise
        sales = max(0, sales)  # Ensure non-negative

        data.append({
            'Date': date,
            'Product': product,
            'Quantity': quantity,
            'Sales': round(sales, 2)
        })

    return pd.DataFrame(data)


def demo_sales_prediction():
    """
    Complete demo of the sales prediction system
    """
    print("🎯 Sales Prediction Demo")
    print("=" * 50)

    # Create sample data
    print("\n1. Creating sample data...")
    df = create_sample_data(1000)
    print(f"   Generated {len(df)} sales records")

    # Initialize predictor
    print("\n2. Training Random Forest model...")
    predictor = SalesPredictor(n_estimators=100)

    # Train model
    results = predictor.train(df)

    print("\n📊 Training Results:")
    print(f"   Training R²: {results['train_r2']:.3f}")
    print(f"   Training RMSE: ${results['train_rmse']:.2f}")
    print(f"   Training MAE: ${results['train_mae']:.2f}")
    print(f"   Testing R²: {results['test_r2']:.3f}")
    print(f"   Testing RMSE: ${results['test_rmse']:.2f}")
    print(f"   Testing MAE: ${results['test_mae']:.2f}")
    # Visualize results
    print("\n3. Creating visualizations...")

    # Get test data for visualization
    processed_data = predictor.preprocess_data(df)
    X, y = predictor.prepare_features_and_target(processed_data)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = predictor.scaler.transform(X_test)
    y_pred = predictor.model.predict(X_test_scaled)

    # Create plots directory if it doesn't exist
    os.makedirs('static/plots', exist_ok=True)

    predictor.visualize_results(y_test, y_pred, save_path='static/plots/sales_prediction_results.png')
    predictor.plot_feature_importance(save_path='static/plots/feature_importance.png')

    # Future predictions
    print("\n4. Making future predictions...")
    future_dates = pd.date_range('2024-01-01', periods=30, freq='D')
    future_predictions = predictor.predict_future_sales(future_dates, {'quantity': 10})

    print("   Future sales predictions (first 5):")
    for i, row in future_predictions.head().iterrows():
        print(f"     {row['date']}: ${row['predicted_sales']:.2f}")
    # Save model
    print("\n5. Saving model...")
    predictor.save_model('models/sales_predictor.joblib')

    print("\n✅ Demo completed successfully!")
    print("   📁 Check static/plots/ for visualizations")
    print("   💾 Model saved as models/sales_predictor.joblib")


if __name__ == "__main__":
    demo_sales_prediction()