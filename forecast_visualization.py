"""
STEP 15: Forecast Visualization Module
Creates interactive visualizations for sales forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class ForecastVisualizer:
    """Create visualizations for sales forecasts"""
    
    def __init__(self, output_dir='static/plots'):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_historical_vs_forecast(self, ts, forecast_result, title='Sales: Historical vs Forecast'):
        """
        Plot historical sales data vs forecast with confidence intervals.
        
        Args:
            ts (pd.Series): Historical time series with DatetimeIndex
            forecast_result (dict): Forecast result from forecast functions
            title (str): Plot title
            
        Returns:
            dict: {'success': bool, 'filepath': str, 'chart_data': dict}
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Extract forecast data
            forecast_df = forecast_result.get('forecast_df')
            if forecast_df is None:
                return {'success': False, 'error': 'No forecast_df in result'}
            
            # Get forecast dates - handle both 'date' column and DatetimeIndex
            if 'date' in forecast_df.columns:
                forecast_dates = pd.to_datetime(forecast_df['date'])
            elif isinstance(forecast_df.index, pd.DatetimeIndex):
                forecast_dates = forecast_df.index
            else:
                return {'success': False, 'error': 'Cannot determine forecast dates'}
            
            # Plot historical data
            ax.plot(ts.index, ts.values, 'b-', linewidth=2, label='Historical Sales', marker='o', markersize=4)
            
            # Plot forecast
            ax.plot(forecast_dates, forecast_df['forecast'], 'r-', 
                   linewidth=2.5, label='Forecast', marker='s', markersize=5)
            
            # Plot confidence intervals
            ax.fill_between(forecast_dates, 
                            forecast_df['lower_ci'], 
                            forecast_df['upper_ci'],
                            alpha=0.2, color='red', label='95% Confidence Interval')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('Sales ($)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
            
            # Format x-axis dates
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save figure
            filepath = os.path.join(self.output_dir, 'historical_vs_forecast.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Create chart data for JSON response - handle both date formats
            if 'date' in forecast_df.columns:
                forecast_dates_list = pd.to_datetime(forecast_df['date']).dt.strftime('%Y-%m-%d').tolist()
            else:
                forecast_dates_list = forecast_dates.strftime('%Y-%m-%d').tolist()
            
            chart_data = {
                'historical_dates': ts.index.strftime('%Y-%m-%d').tolist(),
                'historical_values': ts.values.tolist(),
                'forecast_dates': forecast_dates_list,
                'forecast_values': forecast_df['forecast'].tolist(),
                'lower_ci': forecast_df['lower_ci'].tolist(),
                'upper_ci': forecast_df['upper_ci'].tolist()
            }
            
            return {
                'success': True,
                'filepath': filepath,
                'filename': 'historical_vs_forecast.png',
                'chart_data': chart_data,
                'mean_forecast': float(forecast_df['forecast'].mean()),
                'forecast_range': (float(forecast_df['forecast'].min()), 
                                  float(forecast_df['forecast'].max()))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    def plot_scenario_comparison(self, forecast_summary, title='Sales Forecast: Scenario Analysis'):
        """
        Plot baseline, pessimistic, and optimistic scenarios.
        
        Args:
            forecast_summary (dict): Complete forecast summary with scenarios
            title (str): Plot title
            
        Returns:
            dict: {'success': bool, 'filepath': str}
        """
        try:
            if 'baseline' not in forecast_summary:
                return {'success': False, 'error': 'Missing baseline forecast'}
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Extract baseline
            baseline_df = forecast_summary['baseline'].get('forecast_df')
            if baseline_df is None:
                return {'success': False, 'error': 'No baseline forecast_df'}
            
            # Get baseline dates - handle both 'date' column and DatetimeIndex
            if 'date' in baseline_df.columns:
                baseline_dates = pd.to_datetime(baseline_df['date'])
            elif isinstance(baseline_df.index, pd.DatetimeIndex):
                baseline_dates = baseline_df.index
            else:
                return {'success': False, 'error': 'Cannot determine baseline forecast dates'}
            
            # Plot baseline
            ax.plot(baseline_dates, baseline_df['forecast'], 'g-', 
                   linewidth=2.5, label='Baseline', marker='o', markersize=6)
            
            # Plot scenarios if available
            scenarios = forecast_summary.get('scenarios', {})
            
            if 'pessimistic' in scenarios:
                pess_values = scenarios['pessimistic'].get('forecast_values', [])
                if pess_values:
                    ax.plot(baseline_dates, pess_values, 'r--', 
                           linewidth=2, label='Pessimistic (-5%)', marker='v', markersize=5)
            
            if 'optimistic' in scenarios:
                opt_values = scenarios['optimistic'].get('forecast_values', [])
                if opt_values:
                    ax.plot(baseline_dates, opt_values, 'b--', 
                           linewidth=2, label='Optimistic (+5%)', marker='^', markersize=5)
            
            # Fill baseline CI
            ax.fill_between(baseline_dates,
                            baseline_df['lower_ci'],
                            baseline_df['upper_ci'],
                            alpha=0.15, color='green', label='Baseline CI')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('Sales ($)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format axes
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save figure
            filepath = os.path.join(self.output_dir, 'scenario_comparison.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'filepath': filepath,
                'filename': 'scenario_comparison.png'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    def plot_confidence_intervals(self, forecast_result, title='Forecast with Confidence Intervals'):
        """
        Plot forecast with confidence intervals visualization.
        
        Args:
            forecast_result (dict): Forecast result
            title (str): Plot title
            
        Returns:
            dict: {'success': bool, 'filepath': str}
        """
        try:
            forecast_df = forecast_result.get('forecast_df')
            if forecast_df is None:
                return {'success': False, 'error': 'No forecast_df in result'}
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot forecast
            ax.plot(forecast_df['date'], forecast_df['forecast'], 'b-', 
                   linewidth=2.5, label='Forecast', marker='o', markersize=6)
            
            # Plot multiple CI bands for visual effect
            ax.fill_between(forecast_df['date'],
                            forecast_df['lower_ci'],
                            forecast_df['upper_ci'],
                            alpha=0.3, color='blue', label='95% Confidence Interval')
            
            # Add 68% interval (1 std)
            forecast_std = forecast_df['forecast'].std()
            ax.fill_between(forecast_df['date'],
                            forecast_df['forecast'] - forecast_std,
                            forecast_df['forecast'] + forecast_std,
                            alpha=0.15, color='blue', label='68% Confidence (1σ)')
            
            # Formatting
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax.set_ylabel('Sales ($)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format axes
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save figure
            filepath = os.path.join(self.output_dir, 'confidence_intervals.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'filepath': filepath,
                'filename': 'confidence_intervals.png'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    def plot_forecast_statistics(self, forecast_result, title='Forecast Statistics'):
        """
        Create a summary statistics plot for the forecast.
        
        Args:
            forecast_result (dict): Forecast result with statistics
            title (str): Plot title
            
        Returns:
            dict: {'success': bool, 'filepath': str}
        """
        try:
            forecast_df = forecast_result.get('forecast_df')
            if forecast_df is None:
                return {'success': False, 'error': 'No forecast_df in result'}
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Forecast values over time
            ax1.plot(forecast_df['date'], forecast_df['forecast'], 'b-', linewidth=2, marker='o')
            ax1.fill_between(forecast_df['date'], forecast_df['forecast'] - forecast_df['forecast'].std(),
                            forecast_df['forecast'] + forecast_df['forecast'].std(), alpha=0.2)
            ax1.set_title('Forecast Values Over Time', fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Sales ($)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. CI Width over time
            ci_width = forecast_df['upper_ci'] - forecast_df['lower_ci']
            ax2.bar(range(len(ci_width)), ci_width.values, color='orange', alpha=0.7)
            ax2.set_title('Confidence Interval Width (Growing Uncertainty)', fontweight='bold')
            ax2.set_xlabel('Forecast Period')
            ax2.set_ylabel('CI Width ($)')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. Distribution histogram
            ax3.hist(forecast_df['forecast'], bins=6, color='green', alpha=0.7, edgecolor='black')
            ax3.axvline(forecast_df['forecast'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax3.set_title('Forecast Distribution', fontweight='bold')
            ax3.set_xlabel('Sales ($)')
            ax3.set_ylabel('Frequency')
            ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Summary statistics table
            ax4.axis('off')
            stats = forecast_result.get('statistics', {})
            summary_text = f"""
FORECAST SUMMARY STATISTICS

Mean:     ${stats.get('mean', 0):,.0f}
Median:   ${stats.get('median', 0):,.0f}
Std Dev:  ${stats.get('std', 0):,.0f}

Min:      ${stats.get('min', 0):,.0f}
Max:      ${stats.get('max', 0):,.0f}
Range:    ${stats.get('range', (0,0))[1] - stats.get('range', (0,0))[0]:,.0f}

Forecast Periods: {len(forecast_df)}
Confidence Level: 95%
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # Save figure
            filepath = os.path.join(self.output_dir, 'forecast_statistics.png')
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'filepath': filepath,
                'filename': 'forecast_statistics.png'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    
    def generate_all_visualizations(self, ts, forecast_result, forecast_summary=None):
        """
        Generate all forecast visualizations at once.
        
        Args:
            ts (pd.Series): Historical time series
            forecast_result (dict): Forecast result
            forecast_summary (dict): Optional complete forecast summary with scenarios
            
        Returns:
            dict: {'success': bool, 'visualizations': [...], 'errors': [...]}
        """
        try:
            visualizations = []
            errors = []
            
            # 1. Historical vs Forecast
            result1 = self.plot_historical_vs_forecast(ts, forecast_result)
            if result1['success']:
                visualizations.append({
                    'name': 'historical_vs_forecast',
                    'title': 'Sales: Historical vs Forecast',
                    'filename': result1['filename']
                })
            else:
                errors.append(f"Historical vs Forecast: {result1.get('error', 'Unknown error')}")
            
            # 2. Confidence Intervals
            result2 = self.plot_confidence_intervals(forecast_result)
            if result2['success']:
                visualizations.append({
                    'name': 'confidence_intervals',
                    'title': 'Forecast with Confidence Intervals',
                    'filename': result2['filename']
                })
            else:
                errors.append(f"Confidence Intervals: {result2.get('error', 'Unknown error')}")
            
            # 3. Forecast Statistics
            result3 = self.plot_forecast_statistics(forecast_result)
            if result3['success']:
                visualizations.append({
                    'name': 'forecast_statistics',
                    'title': 'Forecast Statistics',
                    'filename': result3['filename']
                })
            else:
                errors.append(f"Forecast Statistics: {result3.get('error', 'Unknown error')}")
            
            # 4. Scenario Comparison (if available)
            if forecast_summary:
                result4 = self.plot_scenario_comparison(forecast_summary)
                if result4['success']:
                    visualizations.append({
                        'name': 'scenario_comparison',
                        'title': 'Sales Forecast: Scenario Analysis',
                        'filename': result4['filename']
                    })
                else:
                    errors.append(f"Scenario Comparison: {result4.get('error', 'Unknown error')}")
            
            return {
                'success': len(visualizations) > 0,
                'visualizations': visualizations,
                'errors': errors,
                'total_generated': len(visualizations),
                'total_errors': len(errors)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'visualizations': [],
                'errors': [str(e)]
            }


def create_forecast_json_chart(ts, forecast_result):
    """
    Create JSON-compatible chart data for frontend charting libraries.
    
    Args:
        ts (pd.Series): Historical time series
        forecast_result (dict): Forecast result
        
    Returns:
        dict: Chart data in JSON-serializable format
    """
    try:
        forecast_df = forecast_result.get('forecast_df')
        if forecast_df is None:
            return {'success': False, 'error': 'No forecast_df'}
        
        # Ensure dates are datetime and convert to strings
        forecast_dates = pd.to_datetime(forecast_df['date'])
        forecast_dates_str = forecast_dates.dt.strftime('%Y-%m-%d').tolist()
        historical_dates_str = ts.index.strftime('%Y-%m-%d').tolist()
        
        # Convert to JSON-serializable format
        chart_data = {
            'success': True,
            'historical': {
                'dates': historical_dates_str,
                'values': ts.values.tolist()
            },
            'forecast': {
                'dates': forecast_dates_str,
                'values': forecast_df['forecast'].tolist(),
                'lower_ci': forecast_df['lower_ci'].tolist(),
                'upper_ci': forecast_df['upper_ci'].tolist()
            },
            'statistics': {
                'forecast_mean': float(forecast_df['forecast'].mean()),
                'forecast_std': float(forecast_df['forecast'].std()),
                'forecast_min': float(forecast_df['forecast'].min()),
                'forecast_max': float(forecast_df['forecast'].max()),
                'historical_mean': float(ts.mean()),
                'ci_avg_width': float((forecast_df['upper_ci'] - forecast_df['lower_ci']).mean())
            }
        }
        
        return chart_data
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    print("Forecast Visualization Module - Ready for import")
    print("\nAvailable Classes:")
    print("  - ForecastVisualizer")
    print("\nAvailable Functions:")
    print("  - create_forecast_json_chart()")
