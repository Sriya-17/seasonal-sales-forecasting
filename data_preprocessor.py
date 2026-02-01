"""Data preprocessing module for cleaning and preparing sales data."""
import pandas as pd
import numpy as np


def handle_missing_values(df, method='forward_fill'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: 'forward_fill', 'backward_fill', 'drop', or 'mean'
        
    Returns:
        DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Count missing values before
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        return df
    
    if method == 'forward_fill':
        # Forward fill for time series data
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        df['Weekly_Sales'] = df.groupby('Store')['Weekly_Sales'].fillna(method='ffill')
        df['Weekly_Sales'] = df['Weekly_Sales'].fillna(method='bfill')
    
    elif method == 'backward_fill':
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        df['Weekly_Sales'] = df.groupby('Store')['Weekly_Sales'].fillna(method='bfill')
        df['Weekly_Sales'] = df['Weekly_Sales'].fillna(method='ffill')
    
    elif method == 'drop':
        # Drop rows with missing critical values
        df = df.dropna(subset=['Date', 'Weekly_Sales', 'Store'])
    
    elif method == 'mean':
        # Fill with store-specific mean
        df['Weekly_Sales'] = df.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df.isnull().sum().sum()
    
    if missing_after > 0:
        print(f"Warning: {missing_after} missing values remain after preprocessing")
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate records.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    df = df.copy()
    duplicates_count = df.duplicated(subset=['Store', 'Date']).sum()
    
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=['Store', 'Date'], keep='first')
        print(f"Removed {duplicates_count} duplicate records")
    
    return df


def convert_dates(df):
    """
    Convert date column to datetime format and ensure proper sorting.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with proper datetime conversion
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"Error converting dates: {e}")
            return df
    
    # Sort by Store and Date
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    return df


def extract_temporal_features(df):
    """
    Extract temporal features for seasonal analysis.
    
    Args:
        df: Input DataFrame with datetime Date column
        
    Returns:
        DataFrame with additional temporal features
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Create season labels
    def get_season(month):
        """Map month to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    
    return df


def get_preprocessing_stats(df_before, df_after):
    """
    Generate preprocessing statistics.
    
    Args:
        df_before: Original DataFrame
        df_after: Preprocessed DataFrame
        
    Returns:
        Dictionary with preprocessing statistics
    """
    stats = {
        'records_before': len(df_before) if df_before is not None else 0,
        'records_after': len(df_after) if df_after is not None else 0,
        'records_removed': (len(df_before) - len(df_after)) if (df_before is not None and df_after is not None) else 0,
        'missing_values_before': df_before.isnull().sum().sum() if df_before is not None else 0,
        'missing_values_after': df_after.isnull().sum().sum() if df_after is not None else 0,
        'date_range': f"{df_after['Date'].min().date()} to {df_after['Date'].max().date()}" if df_after is not None and not df_after.empty else "N/A"
    }
    return stats


def preprocess_data(df, handle_missing=True, remove_dupes=True, extract_features=True):
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        handle_missing: Whether to handle missing values
        remove_dupes: Whether to remove duplicates
        extract_features: Whether to extract temporal features
        
    Returns:
        tuple: (preprocessed_df, stats)
    """
    if df is None or df.empty:
        return None, {}
    
    df_original = df.copy()
    
    # Step 1: Remove duplicates
    if remove_dupes:
        df = remove_duplicates(df)
    
    # Step 2: Handle missing values
    if handle_missing:
        df = handle_missing_values(df, method='forward_fill')
    
    # Step 3: Convert dates
    df = convert_dates(df)
    
    # Step 4: Extract temporal features
    if extract_features:
        df = extract_temporal_features(df)
    
    # Generate statistics
    stats = get_preprocessing_stats(df_original, df)
    
    return df, stats
