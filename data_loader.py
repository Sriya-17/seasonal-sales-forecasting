"""Data loading and preprocessing module for seasonal sales forecasting."""
import os
import pandas as pd
from config import Config
from csv_validator import validate_csv_structure


def load_walmart_data():
    """Load and preprocess the Walmart sales dataset."""
    data_path = os.path.join(Config.BASE_DIR, 'data', 'Walmart-dataset.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Use the flexible validator to load and validate the walmart dataset
    is_valid, message, df = validate_csv_structure(data_path)
    if not is_valid or df is None:
        raise ValueError(f"Walmart dataset failed validation: {message}")
    # Ensure required standardized columns exist
    required_cols = ['Store', 'Date', 'Weekly_Sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after validation: {missing_cols}")
    return df


def load_uploaded_data():
    """
    Load and preprocess user-uploaded sales dataset.
    Returns None if no uploaded data exists.
    """
    upload_path = os.path.join(Config.BASE_DIR, 'data', 'uploaded_sales.csv')
    
    if not os.path.exists(upload_path):
        return None
    
    try:
        # Delegate to our CSV validator which returns standardized DataFrame
        is_valid, message, df = validate_csv_structure(upload_path)
        if not is_valid or df is None:
            # Return None so caller falls back to Walmart dataset
            print(f"Uploaded data validation failed: {message}")
            return None
        return df
    except Exception as e:
        print(f"Error loading uploaded data: {e}")
        return None


def load_data(prefer_uploaded=True):
    """
    Load dataset with fallback strategy.
    
    Args:
        prefer_uploaded: If True, try to load uploaded data first,
                        then fall back to Walmart dataset.
                        
    Returns:
        tuple: (df, source_name) where source_name is 'uploaded' or 'walmart'
    """
    if prefer_uploaded:
        uploaded_df = load_uploaded_data()
        if uploaded_df is not None:
            return uploaded_df, 'uploaded'
    
    try:
        walmart_df = load_walmart_data()
        return walmart_df, 'walmart'
    except Exception as e:
        print(f"Error loading Walmart data: {e}")
        return None, None


def get_store_summary(df):
    """Get summary statistics by store."""
    if df is None or df.empty:
        return None
    
    summary = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'min', 'max', 'std']).reset_index()
    summary.columns = ['Store', 'Avg_Sales', 'Min_Sales', 'Max_Sales', 'Std_Sales']
    return summary.sort_values('Avg_Sales', ascending=False)


def get_store_data(df, store_id):
    """Get data for a specific store."""
    if df is None:
        return pd.DataFrame()
    return df[df['Store'] == store_id].copy()


def get_data_stats(df):
    """Get comprehensive statistics about the dataset."""
    if df is None or df.empty:
        return {}
    
    return {
        'total_stores': int(df['Store'].nunique()),
        'total_records': len(df),
        'avg_sales': float(df['Weekly_Sales'].mean()),
        'max_sales': float(df['Weekly_Sales'].max()),
        'min_sales': float(df['Weekly_Sales'].min()),
        'std_sales': float(df['Weekly_Sales'].std()),
        'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        'total_weeks': (df['Date'].max() - df['Date'].min()).days // 7
    }

