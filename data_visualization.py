"""
Data Visualization Module
Generates interactive and static visualizations for sales analysis
using matplotlib and plotly for web-based interactive charts.
"""

import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend to avoid macOS NSWindow errors when running in server threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import json
from datetime import datetime


def create_sales_over_time_plot(df, figsize=(14, 6)):
    """
    Generate line chart showing sales trends over time.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
            return None
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by date and sum sales
        daily_sales = df.groupby('Date')['Weekly_Sales'].sum().sort_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(daily_sales.index, daily_sales.values, 
                linewidth=2, color='#667eea', alpha=0.8)
        ax.fill_between(daily_sales.index, daily_sales.values, 
                        alpha=0.3, color='#667eea')
        
        # Formatting
        ax.set_title('Sales Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weekly Sales ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating sales over time plot: {e}")
        return None


def create_seasonality_plot(df, figsize=(14, 6)):
    """
    Generate bar chart showing monthly seasonality patterns.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by month
        monthly_sales = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Weekly_Sales': 'sum'
        }).reset_index()
        
        monthly_sales['Date'] = monthly_sales['Date'].astype(str)
        
        # Create color gradient based on sales
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(monthly_sales)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(range(len(monthly_sales)), monthly_sales['Weekly_Sales'].values,
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height/1e6:.1f}M',
                   ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_title('Monthly Seasonality Patterns', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, len(monthly_sales), max(1, len(monthly_sales)//12)))
        ax.set_xticklabels([monthly_sales['Date'].values[i] 
                           for i in range(0, len(monthly_sales), max(1, len(monthly_sales)//12))],
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating seasonality plot: {e}")
        return None


def create_seasonal_breakdown_plot(df, figsize=(12, 7)):
    """
    Generate visualization of seasonal patterns by season.
    
    Args:
        df (pd.DataFrame): Sales data with Season and Weekly_Sales columns
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Season' not in df.columns or 'Weekly_Sales' not in df.columns:
            return None
        
        # Calculate seasonal statistics
        season_stats = df.groupby('Season').agg({
            'Weekly_Sales': ['mean', 'sum', 'count', 'std']
        }).reset_index()
        
        season_stats.columns = ['Season', 'Avg_Sales', 'Total_Sales', 'Count', 'Std_Dev']
        
        # Order seasons naturally
        season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
        season_stats['order'] = season_stats['Season'].map(season_order)
        season_stats = season_stats.sort_values('order').drop('order', axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Average sales by season
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars1 = ax1.bar(season_stats['Season'], season_stats['Avg_Sales'],
                       color=colors, edgecolor='black', linewidth=2)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1e6:.2f}M',
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Average Sales by Season', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Avg Weekly Sales ($)', fontsize=11, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Standard deviation (volatility)
        bars2 = ax2.bar(season_stats['Season'], season_stats['Std_Dev'],
                       color=colors, edgecolor='black', linewidth=2)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1e6:.2f}M',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Sales Volatility by Season', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Std Dev ($)', fontsize=11, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.suptitle('Seasonal Breakdown Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating seasonal breakdown plot: {e}")
        return None


def create_store_performance_plot(df, top_n=10, figsize=(14, 8)):
    """
    Generate bar chart showing top and bottom store performance.
    
    Args:
        df (pd.DataFrame): Sales data with Store and Weekly_Sales columns
        top_n (int): Number of top/bottom stores to show
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Store' not in df.columns or 'Weekly_Sales' not in df.columns:
            return None
        
        # Calculate store metrics
        store_metrics = df.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'count']
        }).reset_index()
        
        store_metrics.columns = ['Store', 'Avg_Sales', 'Total_Sales', 'Records']
        
        top_stores = store_metrics.nlargest(top_n, 'Avg_Sales')
        bottom_stores = store_metrics.nsmallest(top_n, 'Avg_Sales')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top stores
        ax1.barh(range(len(top_stores)), top_stores['Avg_Sales'].values,
                color='#2ecc71', edgecolor='black', linewidth=1.5)
        ax1.set_yticks(range(len(top_stores)))
        ax1.set_yticklabels([f"Store {int(s)}" for s in top_stores['Store'].values])
        ax1.set_xlabel('Avg Weekly Sales ($)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top {top_n} Performing Stores', fontsize=13, fontweight='bold')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels
        for i, v in enumerate(top_stores['Avg_Sales'].values):
            ax1.text(v, i, f' ${v/1e6:.2f}M', va='center', fontweight='bold')
        
        # Bottom stores
        ax2.barh(range(len(bottom_stores)), bottom_stores['Avg_Sales'].values,
                color='#e74c3c', edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(bottom_stores)))
        ax2.set_yticklabels([f"Store {int(s)}" for s in bottom_stores['Store'].values])
        ax2.set_xlabel('Avg Weekly Sales ($)', fontsize=11, fontweight='bold')
        ax2.set_title(f'Bottom {top_n} Stores', fontsize=13, fontweight='bold')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels
        for i, v in enumerate(bottom_stores['Avg_Sales'].values):
            ax2.text(v, i, f' ${v/1e6:.2f}M', va='center', fontweight='bold')
        
        plt.suptitle('Store Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating store performance plot: {e}")
        return None


def create_yearly_comparison_plot(df, figsize=(14, 6)):
    """
    Generate visualization comparing sales across years.
    
    Args:
        df (pd.DataFrame): Sales data with Date and Weekly_Sales columns
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Date' not in df.columns or 'Weekly_Sales' not in df.columns:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        yearly_sales = df.groupby('Year').agg({
            'Weekly_Sales': 'sum'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars with gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(yearly_sales)))
        bars = ax.bar(yearly_sales['Year'].astype(str), yearly_sales['Weekly_Sales'].values,
                     color=colors, edgecolor='black', linewidth=2, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height/1e9:.2f}B',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title('Year-over-Year Sales Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating yearly comparison plot: {e}")
        return None


def create_distribution_plot(df, figsize=(14, 5)):
    """
    Generate histogram showing distribution of weekly sales.
    
    Args:
        df (pd.DataFrame): Sales data with Weekly_Sales column
        figsize (tuple): Figure size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        if df.empty or 'Weekly_Sales' not in df.columns:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(df['Weekly_Sales'], bins=50, color='#667eea', 
                edgecolor='black', alpha=0.7)
        ax1.axvline(df['Weekly_Sales'].mean(), color='#e74c3c', 
                   linestyle='--', linewidth=2, label=f'Mean: ${df["Weekly_Sales"].mean()/1e6:.2f}M')
        ax1.axvline(df['Weekly_Sales'].median(), color='#2ecc71',
                   linestyle='--', linewidth=2, label=f'Median: ${df["Weekly_Sales"].median()/1e6:.2f}M')
        ax1.set_xlabel('Weekly Sales ($)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Weekly Sales', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
        
        # Box plot
        box = ax2.boxplot([df['Weekly_Sales']], labels=['All Sales'],
                         patch_artist=True, widths=0.5)
        box['boxes'][0].set_facecolor('#667eea')
        box['boxes'][0].set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color='black', linewidth=1.5)
        
        ax2.set_ylabel('Weekly Sales ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Sales Distribution (Box Plot)', fontsize=13, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.suptitle('Weekly Sales Distribution Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"⚠️ Error creating distribution plot: {e}")
        return None


def create_all_visualizations(df):
    """
    Generate all standard visualizations.
    
    Args:
        df (pd.DataFrame): Sales data
        
    Returns:
        dict: Dictionary of all visualization plots
    """
    print("📊 Generating visualizations...")
    
    visualizations = {
        'sales_over_time': create_sales_over_time_plot(df),
        'seasonality': create_seasonality_plot(df),
        'seasonal_breakdown': create_seasonal_breakdown_plot(df),
        'store_performance': create_store_performance_plot(df),
        'yearly_comparison': create_yearly_comparison_plot(df),
        'distribution': create_distribution_plot(df)
    }
    
    # Count successful visualizations
    success_count = sum(1 for v in visualizations.values() if v is not None)
    print(f"✅ Generated {success_count}/6 visualizations")
    
    return visualizations
