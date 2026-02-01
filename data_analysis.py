"""
Exploratory Data Analysis (EDA) Module
Performs comprehensive analysis on sales data including trends, seasonal patterns, 
peak/low periods, and store performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


def get_monthly_trends(df):
    """
    Calculate monthly sales trends over time.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        
    Returns:
        dict: Monthly trends with dates, total sales, avg sales, and counts
    """
    if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'months': [], 'sales': [], 'avg_sales': [], 'count': []}
    
    try:
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by year-month
        monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Weekly_Sales': ['sum', 'mean', 'count']
        }).reset_index()
        
        monthly.columns = ['Month', 'Total_Sales', 'Avg_Sales', 'Records']
        monthly['Month'] = monthly['Month'].astype(str)
        
        return {
            'months': monthly['Month'].tolist(),
            'total_sales': [float(x) for x in monthly['Total_Sales'].tolist()],
            'avg_sales': [float(x) for x in monthly['Avg_Sales'].tolist()],
            'record_count': monthly['Records'].tolist()
        }
    except Exception as e:
        print(f"⚠️ Error calculating monthly trends: {e}")
        return {'months': [], 'sales': [], 'avg_sales': [], 'count': []}


def get_seasonal_patterns(df):
    """
    Identify seasonal patterns in sales data by analyzing Season column.
    
    Args:
        df (pd.DataFrame): Sales data with Season column
        
    Returns:
        dict: Seasonal analysis with avg sales, trends, and strength by season
    """
    if df.empty or 'Season' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'seasons': [], 'avg_sales': [], 'total_sales': [], 'records': [], 'strength': 0}
    
    try:
        seasonal = df.groupby('Season').agg({
            'Weekly_Sales': ['mean', 'sum', 'count', 'std']
        }).reset_index()
        
        seasonal.columns = ['Season', 'Avg_Sales', 'Total_Sales', 'Records', 'Std_Dev']
        
        # Calculate seasonality strength (coefficient of variation)
        overall_mean = df['Weekly_Sales'].mean()
        seasonal_mean_std = seasonal['Avg_Sales'].std()
        seasonality_strength = (seasonal_mean_std / overall_mean * 100) if overall_mean > 0 else 0
        
        # Order seasons naturally
        season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
        seasonal['order'] = seasonal['Season'].map(season_order)
        seasonal = seasonal.sort_values('order').drop('order', axis=1)
        
        return {
            'seasons': seasonal['Season'].tolist(),
            'avg_sales': [float(x) for x in seasonal['Avg_Sales'].tolist()],
            'total_sales': [float(x) for x in seasonal['Total_Sales'].tolist()],
            'records': seasonal['Records'].tolist(),
            'std_dev': [float(x) for x in seasonal['Std_Dev'].tolist()],
            'seasonality_strength': round(seasonality_strength, 2)
        }
    except Exception as e:
        print(f"⚠️ Error analyzing seasonal patterns: {e}")
        return {'seasons': [], 'avg_sales': [], 'total_sales': [], 'records': [], 'strength': 0}


def get_peak_and_low_periods(df):
    """
    Identify peak and low-demand periods using monthly analysis.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        
    Returns:
        dict: Peak and low periods with dates and sales values
    """
    if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'peak_periods': [], 'low_periods': [], 'peak_sales': [], 'low_sales': []}
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Monthly analysis
        monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Weekly_Sales': 'sum'
        }).reset_index()
        
        monthly.columns = ['Month', 'Total_Sales']
        monthly = monthly.sort_values('Total_Sales', ascending=False)
        
        # Top 5 peak periods
        peak_periods = monthly.head(5)
        
        # Bottom 5 low periods
        low_periods = monthly.tail(5)
        
        return {
            'peak_periods': peak_periods['Month'].astype(str).tolist(),
            'peak_sales': [float(x) for x in peak_periods['Total_Sales'].tolist()],
            'low_periods': low_periods['Month'].astype(str).tolist(),
            'low_sales': [float(x) for x in low_periods['Total_Sales'].tolist()],
            'peak_average': float(peak_periods['Total_Sales'].mean()),
            'low_average': float(low_periods['Total_Sales'].mean())
        }
    except Exception as e:
        print(f"⚠️ Error identifying peak/low periods: {e}")
        return {'peak_periods': [], 'low_periods': [], 'peak_sales': [], 'low_sales': []}


def get_store_analysis(df):
    """
    Analyze performance metrics for each store.
    
    Args:
        df (pd.DataFrame): Sales data with Store and Weekly_Sales columns
        
    Returns:
        dict: Top/bottom stores with performance metrics
    """
    if df.empty or 'Store' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'top_stores': [], 'bottom_stores': [], 'metrics': {}}
    
    try:
        store_metrics = df.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        store_metrics.columns = ['Store', 'Avg_Sales', 'Total_Sales', 'Std_Dev', 'Min_Sales', 'Max_Sales', 'Records']
        
        # Calculate consistency score (inverse of coefficient of variation)
        store_metrics['Consistency'] = 100 - (store_metrics['Std_Dev'] / store_metrics['Avg_Sales'] * 100).clip(0, 100)
        
        # Top and bottom performers
        top_stores = store_metrics.nlargest(10, 'Avg_Sales')
        bottom_stores = store_metrics.nsmallest(10, 'Avg_Sales')
        
        return {
            'top_stores': [
                {
                    'store': int(s),
                    'avg_sales': float(a),
                    'total_sales': float(t),
                    'consistency': float(c)
                }
                for s, a, t, c in zip(
                    top_stores['Store'], 
                    top_stores['Avg_Sales'], 
                    top_stores['Total_Sales'],
                    top_stores['Consistency']
                )
            ],
            'bottom_stores': [
                {
                    'store': int(s),
                    'avg_sales': float(a),
                    'total_sales': float(t),
                    'consistency': float(c)
                }
                for s, a, t, c in zip(
                    bottom_stores['Store'],
                    bottom_stores['Avg_Sales'],
                    bottom_stores['Total_Sales'],
                    bottom_stores['Consistency']
                )
            ],
            'total_stores': len(store_metrics),
            'avg_store_sales': float(store_metrics['Avg_Sales'].mean()),
            'best_performer': int(store_metrics.loc[store_metrics['Avg_Sales'].idxmax(), 'Store']),
            'best_performer_sales': float(store_metrics['Avg_Sales'].max())
        }
    except Exception as e:
        print(f"⚠️ Error analyzing stores: {e}")
        return {'top_stores': [], 'bottom_stores': [], 'metrics': {}}


def get_dayofweek_analysis(df):
    """
    Analyze sales patterns by day of week.
    
    Args:
        df (pd.DataFrame): Sales data with DayOfWeek column
        
    Returns:
        dict: Day of week sales analysis
    """
    if df.empty or 'DayOfWeek' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'days': [], 'avg_sales': [], 'total_sales': []}
    
    try:
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        
        day_analysis = df.groupby('DayOfWeek').agg({
            'Weekly_Sales': ['mean', 'sum', 'count']
        }).reset_index()
        
        day_analysis.columns = ['DayOfWeek', 'Avg_Sales', 'Total_Sales', 'Records']
        day_analysis['DayName'] = day_analysis['DayOfWeek'].map(day_names)
        day_analysis = day_analysis.sort_values('DayOfWeek')
        
        return {
            'days': day_analysis['DayName'].tolist(),
            'avg_sales': [float(x) for x in day_analysis['Avg_Sales'].tolist()],
            'total_sales': [float(x) for x in day_analysis['Total_Sales'].tolist()],
            'record_count': day_analysis['Records'].tolist()
        }
    except Exception as e:
        print(f"⚠️ Error analyzing day of week: {e}")
        return {'days': [], 'avg_sales': [], 'total_sales': []}


def get_overall_statistics(df):
    """
    Calculate overall sales statistics.
    
    Args:
        df (pd.DataFrame): Sales data
        
    Returns:
        dict: Overall statistics
    """
    if df.empty or 'Weekly_Sales' not in df.columns:
        return {
            'total_sales': 0,
            'avg_weekly_sales': 0,
            'median_sales': 0,
            'std_dev': 0,
            'min_sales': 0,
            'max_sales': 0,
            'total_records': 0
        }
    
    try:
        sales = df['Weekly_Sales']
        
        return {
            'total_sales': float(sales.sum()),
            'avg_weekly_sales': float(sales.mean()),
            'median_sales': float(sales.median()),
            'std_dev': float(sales.std()),
            'min_sales': float(sales.min()),
            'max_sales': float(sales.max()),
            'total_records': len(df),
            'total_stores': int(df['Store'].nunique()) if 'Store' in df.columns else 0,
            'date_range_start': str(df['Date'].min()) if 'Date' in df.columns else 'N/A',
            'date_range_end': str(df['Date'].max()) if 'Date' in df.columns else 'N/A'
        }
    except Exception as e:
        print(f"⚠️ Error calculating overall statistics: {e}")
        return {}


def get_correlation_analysis(df):
    """
    Analyze correlations between numerical variables.
    
    Args:
        df (pd.DataFrame): Sales data
        
    Returns:
        dict: Correlation matrix and insights
    """
    if df.empty or 'Weekly_Sales' not in df.columns:
        return {'correlations': {}, 'insights': []}
    
    try:
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return {'correlations': {}, 'insights': []}
        
        # Calculate correlations with Weekly_Sales
        correlations = {}
        for col in numerical_cols:
            if col != 'Weekly_Sales':
                corr = df['Weekly_Sales'].corr(df[col])
                if not np.isnan(corr):
                    correlations[col] = round(corr, 3)
        
        # Sort by absolute value
        correlations = dict(sorted(correlations.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True))
        
        # Generate insights
        insights = []
        for var, corr in list(correlations.items())[:3]:
            if abs(corr) > 0.5:
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"Strong {direction} correlation between Weekly_Sales and {var}")
        
        return {
            'correlations': correlations,
            'insights': insights
        }
    except Exception as e:
        print(f"⚠️ Error analyzing correlations: {e}")
        return {'correlations': {}, 'insights': []}


def get_year_analysis(df):
    """
    Analyze sales trends by year.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        
    Returns:
        dict: Year-over-year analysis
    """
    if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
        return {'years': [], 'total_sales': [], 'avg_sales': []}
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        
        yearly = df.groupby(df['Date'].dt.year).agg({
            'Weekly_Sales': ['sum', 'mean', 'count']
        }).reset_index()
        
        yearly.columns = ['Year', 'Total_Sales', 'Avg_Sales', 'Records']
        yearly['Year'] = yearly['Year'].astype(int)
        
        return {
            'years': yearly['Year'].tolist(),
            'total_sales': [float(x) for x in yearly['Total_Sales'].tolist()],
            'avg_sales': [float(x) for x in yearly['Avg_Sales'].tolist()],
            'record_count': yearly['Records'].tolist()
        }
    except Exception as e:
        print(f"⚠️ Error analyzing years: {e}")
        return {'years': [], 'total_sales': [], 'avg_sales': []}


def perform_eda(df):
    """
    Execute complete EDA on the dataframe.
    
    Args:
        df (pd.DataFrame): Sales data
        
    Returns:
        dict: Complete EDA results
    """
    if df.empty:
        return {
            'status': 'error',
            'message': 'No data available for analysis'
        }
    
    print(f"📊 Performing EDA on {len(df)} records...")
    
    analysis_results = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'overall_stats': get_overall_statistics(df),
        'monthly_trends': get_monthly_trends(df),
        'seasonal_patterns': get_seasonal_patterns(df),
        'peak_low_periods': get_peak_and_low_periods(df),
        'store_analysis': get_store_analysis(df),
        'day_of_week': get_dayofweek_analysis(df),
        'year_analysis': get_year_analysis(df),
        'correlations': get_correlation_analysis(df)
    }
    
    print("✅ EDA completed successfully")
    return analysis_results
