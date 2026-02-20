import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product

warnings.filterwarnings("ignore")


# =====================================================
# ARIMA FORECAST
# =====================================================

def arima_forecast(ts, order=(0, 0, 1), steps=12):
    model = ARIMA(ts, order=order)
    results = model.fit()

    forecast = results.get_forecast(steps=steps)

    return {
        "model": results,
        "forecast": forecast.predicted_mean,
        "conf_int": forecast.conf_int()
    }


# =====================================================
# SARIMA FORECAST
# =====================================================

def sarima_forecast(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), steps=12):
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order
    )

    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)

    return {
        "model": results,
        "forecast": forecast.predicted_mean,
        "conf_int": forecast.conf_int()
    }


# =====================================================
# MODEL EVALUATION
# =====================================================

def evaluate_model(ts, model_result):
    fitted_values = model_result.fittedvalues

    rmse = np.sqrt(mean_squared_error(ts, fitted_values))
    mae = mean_absolute_error(ts, fitted_values)

    return {
        "rmse": rmse,
        "mae": mae
    }
    

def find_optimal_order(ts, max_p=5, max_d=2, max_q=5):
    """
    Find optimal ARIMA order using grid search with auto.arima approach
    Tests combinations and returns best order by AIC score
    """
    best_aic = np.inf
    best_order = (1, 0, 1)
    
    try:
        # Test a subset of combinations for performance
        p_range = range(0, min(max_p + 1, 4))
        d_range = range(0, min(max_d + 1, 2))
        q_range = range(0, min(max_q + 1, 4))
        
        for p, d, q in product(p_range, d_range, q_range):
            try:
                model = ARIMA(ts, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue
    except Exception as e:
        print(f"Warning: Could not find optimal order, using default: {e}")
    
    return best_order


def check_stationarity(ts):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    Returns d value needed for differencing
    """
    try:
        d = 0
        temp_ts = ts.copy()
        
        for i in range(3):
            result = adfuller(temp_ts.dropna(), autolag='AIC')
            if result[1] < 0.05:  # p-value < 0.05 means stationary
                return d
            temp_ts = temp_ts.diff().dropna()
            d += 1
        
        return min(d, 2)  # Cap at 2 differences
    except:
        return 1  # Default to 1 difference if test fails


def fit_arima(ts_train, order=None, auto_optimize=True):
    """
    Fit ARIMA model with optional auto-optimization
    If order is None and auto_optimize=True, finds optimal order
    """
    if order is None and auto_optimize:
        # Auto-detect stationarity for d
        d = check_stationarity(ts_train)
        # Find optimal p and q
        remaining_ts = ts_train.diff(d).dropna() if d > 0 else ts_train
        best_order = find_optimal_order(remaining_ts, max_p=3, max_d=0, max_q=3)
        order = (best_order[0], d, best_order[2])
    elif order is None:
        order = (1, 1, 1)  # Default fallback
    
    try:
        model = ARIMA(ts_train, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error fitting ARIMA{order}: {e}. Trying default order...")
        # Fallback to simple model
        model = ARIMA(ts_train, order=(1, 1, 1))
        model_fit = model.fit()
        return model_fit


def forecast_n_months(ts_train, model_result, n_months=12):
    """
    Forecast future months using fitted model with confidence intervals
    Returns DataFrame with proper date index as a column for easier access
    """
    forecast_obj = model_result.get_forecast(steps=n_months)
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)  # 95% confidence interval

    forecast_df = pd.DataFrame({
        "forecast": forecast_mean,
        "lower_ci": conf_int.iloc[:, 0],
        "upper_ci": conf_int.iloc[:, 1]
    })
    
    # Add date as a column for easier access in visualization
    forecast_df['date'] = forecast_df.index
    
    return {
        "forecast_df": forecast_df
    }


def generate_forecast_summary(ts_train, model_result, n_periods=12, include_scenarios=True):
    """
    Generate forecast summary including baseline and optional scenarios
    """
    base_forecast = forecast_n_months(ts_train, model_result, n_months=n_periods)

    summary = {
        "baseline": base_forecast
    }

    if include_scenarios:
        forecast_values = base_forecast["forecast_df"]["forecast"].values

        summary["scenarios"] = {
            "pessimistic": {
                "forecast_values": forecast_values * 0.95
            },
            "optimistic": {
                "forecast_values": forecast_values * 1.05
            }
        }

    return summary



# """
# Time-Series Preparation Module for ARIMA Forecasting (STEP 11)
# Prepares dataset for ARIMA analysis by:
# - Setting date as index
# - Aggregating sales (weekly/monthly options)
# - Ensuring fixed time intervals
# - Validating time-series structure
# """

# from pyexpat import model
# from pyexpat import model
# import pandas as pd
# import numpy as np
# # from seasonal_sales_forecasting.app import forecast
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# from datetime import datetime, timedelta
# import warnings

# warnings.filterwarnings('ignore')


# def set_date_as_index(df, date_column='Date'):
#     """
#     Set the date column as the DataFrame index.
    
#     Args:
#         df (pd.DataFrame): Sales data with date column
#         date_column (str): Name of the date column
        
#     Returns:
#         pd.DataFrame: DataFrame with date as index
        
#     Raises:
#         ValueError: If date column doesn't exist
#     """
#     if date_column not in df.columns:
#         raise ValueError(f"Column '{date_column}' not found in DataFrame")
    
#     df = df.copy()
    
#     # Ensure date column is datetime
#     if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
#         df[date_column] = pd.to_datetime(df[date_column])
    
#     # Set as index
#     df.set_index(date_column, inplace=True)
    
#     # Sort by date
#     df = df.sort_index()
    
#     return df


# def aggregate_sales_weekly(df, value_column='Weekly_Sales'):
#     """
#     Aggregate sales data to weekly frequency.
    
#     Args:
#         df (pd.DataFrame): Sales data with date index (or date column)
#         value_column (str): Name of the sales value column
        
#     Returns:
#         pd.Series: Weekly aggregated sales time series
#         dict: Aggregation statistics
#     """
#     # Ensure date is index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         if 'Date' in df.columns:
#             df = set_date_as_index(df, 'Date')
#         else:
#             raise ValueError("DataFrame must have date as index or 'Date' column")
    
#     # Aggregate to weekly (sum of sales per week)
#     weekly_sales = df[value_column].resample('W').sum()
    
#     # Remove zero values (weeks with no data)
#     weekly_sales = weekly_sales[weekly_sales > 0]
    
#     stats = {
#         'original_records': len(df),
#         'weekly_periods': len(weekly_sales),
#         'date_range_start': weekly_sales.index[0],
#         'date_range_end': weekly_sales.index[-1],
#         'total_weeks': (weekly_sales.index[-1] - weekly_sales.index[0]).days / 7,
#         'avg_sales': weekly_sales.mean(),
#         'std_sales': weekly_sales.std(),
#         'min_sales': weekly_sales.min(),
#         'max_sales': weekly_sales.max()
#     }
    
#     return weekly_sales, stats


# def aggregate_sales_monthly(df, value_column='Weekly_Sales'):
#     """
#     Aggregate sales data to monthly frequency.
    
#     Args:
#         df (pd.DataFrame): Sales data with date index (or date column)
#         value_column (str): Name of the sales value column
        
#     Returns:
#         pd.Series: Monthly aggregated sales time series
#         dict: Aggregation statistics
#     """
#     # Ensure date is index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         if 'Date' in df.columns:
#             df = set_date_as_index(df, 'Date')
#         else:
#             raise ValueError("DataFrame must have date as index or 'Date' column")
    
#     # Aggregate to monthly (sum of sales per month)
#     monthly_sales = df[value_column].resample('MS').sum()
    
#     # Remove zero values (months with no data)
#     monthly_sales = monthly_sales[monthly_sales > 0]
    
#     stats = {
#         'original_records': len(df),
#         'monthly_periods': len(monthly_sales),
#         'date_range_start': monthly_sales.index[0],
#         'date_range_end': monthly_sales.index[-1],
#         'total_months': len(monthly_sales),
#         'avg_sales': monthly_sales.mean(),
#         'std_sales': monthly_sales.std(),
#         'min_sales': monthly_sales.min(),
#         'max_sales': monthly_sales.max()
#     }
    
#     return monthly_sales, stats


# def aggregate_sales_daily(df, value_column='Weekly_Sales'):
#     """
#     Aggregate sales data to daily frequency.
    
#     Args:
#         df (pd.DataFrame): Sales data with date index (or date column)
#         value_column (str): Name of the sales value column
        
#     Returns:
#         pd.Series: Daily aggregated sales time series
#         dict: Aggregation statistics
#     """
#     # Ensure date is index
#     if not isinstance(df.index, pd.DatetimeIndex):
#         if 'Date' in df.columns:
#             df = set_date_as_index(df, 'Date')
#         else:
#             raise ValueError("DataFrame must have date as index or 'Date' column")
    
#     # Aggregate to daily (sum of sales per day)
#     daily_sales = df[value_column].resample('D').sum()
    
#     # Remove zero values (days with no data)
#     daily_sales = daily_sales[daily_sales > 0]
    
#     stats = {
#         'original_records': len(df),
#         'daily_periods': len(daily_sales),
#         'date_range_start': daily_sales.index[0],
#         'date_range_end': daily_sales.index[-1],
#         'total_days': (daily_sales.index[-1] - daily_sales.index[0]).days,
#         'avg_sales': daily_sales.mean(),
#         'std_sales': daily_sales.std(),
#         'min_sales': daily_sales.min(),
#         'max_sales': daily_sales.max()
#     }
    
#     return daily_sales, stats


# def ensure_fixed_intervals(ts, frequency='D'):
#     """
#     Ensure time series has fixed intervals by filling missing dates.
    
#     Args:
#         ts (pd.Series): Time series with datetime index
#         frequency (str): Frequency ('D'=daily, 'W'=weekly, 'MS'=monthly)
        
#     Returns:
#         pd.Series: Time series with fixed intervals
#         dict: Filling statistics
#     """
#     if not isinstance(ts.index, pd.DatetimeIndex):
#         raise ValueError("Series must have datetime index")
    
#     ts = ts.copy()
#     original_length = len(ts)
    
#     # Create complete date range with fixed intervals
#     date_range = pd.date_range(
#         start=ts.index.min(),
#         end=ts.index.max(),
#         freq=frequency
#     )
    
#     # Reindex to include all dates
#     ts_fixed = ts.reindex(date_range)
    
#     # Fill missing values using forward fill
#     ts_fixed = ts_fixed.ffill()
    
#     # Backward fill for any remaining NaNs at the start
#     ts_fixed = ts_fixed.bfill()
    
#     # If still NaN, use interpolation
#     if ts_fixed.isna().any():
#         ts_fixed = ts_fixed.interpolate(method='linear')
    
#     stats = {
#         'original_length': original_length,
#         'final_length': len(ts_fixed),
#         'missing_periods_filled': len(ts_fixed) - original_length,
#         'frequency': frequency,
#         'date_range_start': str(ts_fixed.index[0]),
#         'date_range_end': str(ts_fixed.index[-1])
#     }
    
#     return ts_fixed, stats


# def validate_time_series(ts, min_periods=10):
#     """
#     Validate time series structure for ARIMA.
    
#     Args:
#         ts (pd.Series): Time series to validate
#         min_periods (int): Minimum number of periods required
        
#     Returns:
#         dict: Validation results with all checks
#     """
#     results = {
#         'is_valid': True,
#         'issues': [],
#         'warnings': [],
#         'metadata': {}
#     }
    
#     # Check if Series
#     if not isinstance(ts, pd.Series):
#         results['is_valid'] = False
#         results['issues'].append("Input must be a pandas Series")
#         return results
    
#     # Check datetime index
#     if not isinstance(ts.index, pd.DatetimeIndex):
#         results['is_valid'] = False
#         results['issues'].append("Series must have datetime index")
#         return results
    
#     # Check length
#     if len(ts) < min_periods:
#         results['is_valid'] = False
#         results['issues'].append(f"Series length ({len(ts)}) is less than minimum ({min_periods})")
    
#     # Check for NaN values
#     nan_count = ts.isna().sum()
#     if nan_count > 0:
#         nan_percent = (nan_count / len(ts)) * 100
#         if nan_percent > 20:
#             results['is_valid'] = False
#             results['issues'].append(f"Too many NaN values ({nan_percent:.1f}%)")
#         elif nan_percent > 0:
#             results['warnings'].append(f"Series has {nan_count} NaN values ({nan_percent:.1f}%)")
    
#     # Check for constant values (no variation)
#     if ts.std() == 0:
#         results['is_valid'] = False
#         results['issues'].append("Series has no variation (constant values)")
    
#     # Check for extreme outliers (>5 std devs)
#     mean = ts.mean()
#     std = ts.std()
#     outliers = ((ts - mean).abs() > 5 * std).sum()
#     if outliers > 0:
#         outlier_percent = (outliers / len(ts)) * 100
#         results['warnings'].append(f"Found {outliers} potential outliers ({outlier_percent:.1f}%)")
    
#     # Check for consistent intervals
#     if len(ts) > 1:
#         intervals = ts.index.to_series().diff()
#         unique_intervals = intervals.nunique()
#         if unique_intervals > 1:
#             results['warnings'].append(f"Series has {unique_intervals} different time intervals")
    
#     # Metadata
#     results['metadata'] = {
#         'length': len(ts),
#         'start_date': str(ts.index[0]),
#         'end_date': str(ts.index[-1]),
#         'mean': float(ts.mean()),
#         'std': float(ts.std()),
#         'min': float(ts.min()),
#         'max': float(ts.max()),
#         'nan_count': int(nan_count),
#         'outliers': int(outliers)
#     }
    
#     return results


# def prepare_for_arima(df, aggregation='monthly', ensure_intervals=True, 
#                       value_column='Weekly_Sales', date_column='Date'):
#     """
#     Complete time-series preparation pipeline for ARIMA.
    
#     Args:
#         df (pd.DataFrame): Raw sales data
#         aggregation (str): 'daily', 'weekly', or 'monthly'
#         ensure_intervals (bool): Whether to fill missing intervals
#         value_column (str): Name of sales value column
#         date_column (str): Name of date column
        
#     Returns:
#         dict: Complete preparation results with time series and metadata
#     """
#     print(f"📊 Starting ARIMA preparation pipeline...")
#     print(f"   - Aggregation: {aggregation}")
#     print(f"   - Ensure intervals: {ensure_intervals}")
    
#     results = {
#         'status': 'success',
#         'time_series': None,
#         'aggregation': aggregation,
#         'preparation_steps': {}
#     }
    
#     try:
#         # Step 1: Set date as index
#         print(f"\n1️⃣ Setting date as index...")
#         df_indexed = set_date_as_index(df, date_column)
#         results['preparation_steps']['date_indexing'] = {
#             'status': 'complete',
#             'records': len(df_indexed)
#         }
#         print(f"   ✅ Date set as index ({len(df_indexed)} records)")
        
#         # Step 2: Aggregate sales
#         print(f"\n2️⃣ Aggregating sales data ({aggregation})...")
#         if aggregation == 'weekly':
#             ts, agg_stats = aggregate_sales_weekly(df_indexed, value_column)
#         elif aggregation == 'monthly':
#             ts, agg_stats = aggregate_sales_monthly(df_indexed, value_column)
#         elif aggregation == 'daily':
#             ts, agg_stats = aggregate_sales_daily(df_indexed, value_column)
#         else:
#             raise ValueError(f"Unknown aggregation: {aggregation}")
        
#         results['preparation_steps']['aggregation'] = agg_stats
#         print(f"   ✅ Aggregated to {agg_stats[f'{aggregation}_periods']} periods")
        
#         # Step 3: Ensure fixed intervals
#         if ensure_intervals:
#             print(f"\n3️⃣ Ensuring fixed time intervals...")
#             freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS'}
#             ts, interval_stats = ensure_fixed_intervals(ts, freq_map[aggregation])
#             results['preparation_steps']['interval_fixing'] = interval_stats
#             print(f"   ✅ Fixed intervals ({interval_stats['missing_periods_filled']} periods filled)")
        
#         # Step 4: Validate time series
#         print(f"\n4️⃣ Validating time-series structure...")
#         validation = validate_time_series(ts)
#         results['preparation_steps']['validation'] = validation
        
#         if validation['is_valid']:
#             print(f"   ✅ Validation passed")
#         else:
#             print(f"   ⚠️  Validation warnings:")
#             for issue in validation['issues']:
#                 print(f"      ❌ {issue}")
#             for warning in validation['warnings']:
#                 print(f"      ⚠️  {warning}")
        
#         # Return prepared series
#         results['time_series'] = ts
#         results['metadata'] = validation['metadata']
        
#         print(f"\n✅ Preparation complete!")
#         print(f"   - Final length: {len(ts)} periods")
#         print(f"   - Mean sales: ${ts.mean():,.0f}")
#         print(f"   - Std dev: ${ts.std():,.0f}")
        
#         return results
        
#     except Exception as e:
#         results['status'] = 'error'
#         results['error'] = str(e)
#         print(f"\n❌ Preparation failed: {e}")
#         return results


# def test_stationarity(ts):
#     """
#     Test for stationarity using Augmented Dickey-Fuller test.
    
#     Args:
#         ts (pd.Series): Time series data
        
#     Returns:
#         dict: ADF test results
#     """
#     try:
#         from statsmodels.tsa.stattools import adfuller
        
#         result = adfuller(ts.dropna())
        
#         return {
#             'test_statistic': float(result[0]),
#             'p_value': float(result[1]),
#             'n_lags': result[2],
#             'n_obs': result[3],
#             'critical_values': {
#                 '1%': float(result[4]['1%']),
#                 '5%': float(result[4]['5%']),
#                 '10%': float(result[4]['10%'])
#             },
#             'is_stationary': result[1] < 0.05,
#             'interpretation': 'Series is stationary (reject H0)' if result[1] < 0.05 else 'Series is non-stationary (fail to reject H0)'
#         }
#     except ImportError:
#         return {'error': 'statsmodels not installed'}


# def recommend_arima_parameters(ts):
#     """
#     Recommend ARIMA (p, d, q) parameters based on time series characteristics.
    
#     Args:
#         ts (pd.Series): Time series data
        
#     Returns:
#         dict: Recommended parameters with reasoning
#     """
#     # Test stationarity
#     stationarity = test_stationarity(ts)
    
#     if 'error' in stationarity:
#         return {
#             'error': stationarity['error'],
#             'default_recommendation': (1, 1, 1)
#         }
    
#     # Determine d (differencing)
#     d = 0 if stationarity['is_stationary'] else 1
    
#     # Simple heuristic for p and q
#     p = 1
#     q = 1
    
#     # Determine seasonal parameters if applicable
#     seasonal_p = 1
#     seasonal_d = 0
#     seasonal_q = 1
#     seasonal_m = 12  # For monthly aggregation
    
#     return {
#         'recommended_arima': (p, d, q),
#         'reasoning': {
#             'p': f"AR order {p}",
#             'd': f"Differencing order {d}",
#             'q': f"MA order {q}"
#         },
#         'recommended_sarima': (p, d, q, seasonal_m),
#         'seasonal_parameters': {
#             'P': seasonal_p,
#             'D': seasonal_d,
#             'Q': seasonal_q,
#             'M': seasonal_m
#         },
#         'test_statistic': stationarity.get('test_statistic'),
#         'p_value': stationarity.get('p_value')
#     }


# if __name__ == '__main__':
#     # Example usage
#     print("Time-Series Preparation Module (ARIMA)")
#     print("="*50)
#     print("\nUsage Examples:")
#     print("\n1. Load data and prepare for ARIMA:")
#     print("   from data_loader import load_data")
#     print("   from data_preprocessor import preprocess_data")
#     print("   from models.arima_model import prepare_for_arima")
#     print("")
#     print("   df, _ = load_data()")
#     print("   df, _ = preprocess_data(df)")
#     print("   result = prepare_for_arima(df, aggregation='monthly')")
#     print("   ts = result['time_series']")
#     print("")
#     print("2. Validate specific time series:")
#     print("   from models.arima_model import validate_time_series")
#     print("   validation = validate_time_series(ts)")
#     print("")
#     print("3. Get ARIMA parameter recommendations:")
#     print("   from models.arima_model import recommend_arima_parameters")
#     print("   params = recommend_arima_parameters(ts)")
#     print("   print(params['recommended_arima'])")

# # ============================================================================
# # STEP 13: ARIMA MODEL TRAINING (Added to STEP 11 Module)
# # ============================================================================

# def fit_arima(ts, order=(1, 0, 1)):
#     """
#     Fit ARIMA model to time series.
    
#     Args:
#         ts (pd.Series): Stationary time series with DatetimeIndex
#         order (tuple): (p, d, q) parameters for ARIMA
#                        p: AR order
#                        d: differencing order (should be 0 for stationary series)
#                        q: MA order
        
#     Returns:
#         dict: Model results with keys:
#             - model: Fitted ARIMA model object
#             - results: Summary results
#             - aic: Akaike Information Criterion
#             - bic: Bayesian Information Criterion
#             - order: (p, d, q) used
#             - success: Boolean indicating successful fit
#     """
#     try:
#         from statsmodels.tsa.arima.model import ARIMA
#     except ImportError:
#         raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
#     try:
#         # Fit ARIMA model
#         model = ARIMA(ts, order=order)
#         results = model.fit()
        
#         return {
#             'model': model,
#             'results': results,
#             'aic': results.aic,
#             'bic': results.bic,
#             'order': order,
#             'success': True,
#             'summary': results.summary().as_text()
#         }
#     except Exception as e:
#         return {
#             'model': None,
#             'results': None,
#             'aic': float('inf'),
#             'bic': float('inf'),
#             'order': order,
#             'success': False,
#             'error': str(e)
#         }


# def generate_forecast(model_result, steps=12, confidence=0.95):
#     """
#     Generate forecast from fitted ARIMA model.
    
#     Args:
#         model_result (dict): Result from fit_arima()
#         steps (int): Number of periods to forecast (default 12 months)
#         confidence (float): Confidence level for intervals (default 0.95 = 95%)
        
#     Returns:
#         dict: Forecast results with keys:
#             - forecast: Forecasted values
#             - lower_ci: Lower confidence interval
#             - upper_ci: Upper confidence interval
#             - forecast_df: DataFrame with all forecast data
#             - steps: Number of forecast periods
#             - success: Boolean indicating success
#     """
#     if not model_result['success']:
#         return {
#             'success': False,
#             'error': 'Model fitting failed'
#         }
    
#     try:
#         results = model_result['results']
        
#         # Get forecast
#         forecast_result = results.get_forecast(steps=steps)
#         forecast_df = forecast_result.conf_int(alpha=1-confidence)
#         forecast_df.columns = ['lower_ci', 'upper_ci']
#         forecast_df['forecast'] = forecast_result.predicted_mean
        
#         return {
#             'forecast': forecast_result.predicted_mean,
#             'lower_ci': forecast_df['lower_ci'],
#             'upper_ci': forecast_df['upper_ci'],
#             'forecast_df': forecast_df,
#             'steps': steps,
#             'confidence': confidence,
#             'success': True
#         }
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# def calculate_metrics(actual, fitted, forecast=None):
#     """
#     Calculate forecast accuracy metrics.
    
#     Args:
#         actual (pd.Series): Actual values
#         fitted (pd.Series): Fitted values from model
#         forecast (pd.Series): Optional forecast values
        
#     Returns:
#         dict: Metrics including:
#             - rmse: Root Mean Square Error
#             - mae: Mean Absolute Error
#             - mape: Mean Absolute Percentage Error
#             - mpe: Mean Percentage Error
#             - correlation: Correlation between actual and fitted
#     """
#     from sklearn.metrics import mean_squared_error, mean_absolute_error
    
#     # Ensure same length
#     min_len = min(len(actual), len(fitted))
#     actual_trim = actual.iloc[:min_len]
#     fitted_trim = fitted.iloc[:min_len]
    
#     # Calculate metrics
#     rmse = np.sqrt(mean_squared_error(actual_trim, fitted_trim))
#     mae = mean_absolute_error(actual_trim, fitted_trim)
    
#     # MAPE and MPE (avoid division by zero)
#     mape_values = np.abs((actual_trim - fitted_trim) / actual_trim) * 100
#     mape = np.mean(mape_values[np.isfinite(mape_values)])
    
#     mpe_values = (actual_trim - fitted_trim) / actual_trim * 100
#     mpe = np.mean(mpe_values[np.isfinite(mpe_values)])
    
#     # Correlation
#     correlation = np.corrcoef(actual_trim, fitted_trim)[0, 1]
    
#     metrics = {
#         'rmse': rmse,
#         'mae': mae,
#         'mape': mape,
#         'mpe': mpe,
#         'correlation': correlation,
#         'n_observations': len(actual_trim)
#     }
    
#     return metrics


# def validate_model(model_result):
#     """
#     Perform model diagnostic checks.
    
#     Args:
#         model_result (dict): Result from fit_arima()
        
#     Returns:
#         dict: Validation results with checks:
#             - residuals_mean_zero: Mean of residuals near zero
#             - residuals_white_noise: Residuals appear random
#             - ljungbox_test: Ljung-Box test for autocorrelation
#             - normality_test: Shapiro-Wilk test for normality
#     """
#     if not model_result['success']:
#         return {'success': False, 'error': 'Model fitting failed'}
    
#     try:
#         from scipy import stats
#         results = model_result['results']
#         residuals = results.resid
        
#         # Check 1: Mean of residuals
#         residuals_mean = np.abs(residuals.mean())
#         mean_near_zero = residuals_mean < np.std(residuals) * 0.1
        
#         # Check 2: Ljung-Box test for autocorrelation
#         try:
#             from statsmodels.stats.diagnostic import acorr_ljungbox
#             lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
#             white_noise = all(lb_test['lb_pvalue'] > 0.05)
#         except:
#             white_noise = None
        
#         # Check 3: Normality test (Shapiro-Wilk)
#         try:
#             shapiro_stat, shapiro_p = stats.shapiro(residuals)
#             normality = shapiro_p > 0.05
#         except:
#             normality = None
        
#         return {
#             'success': True,
#             'residuals_mean': residuals_mean,
#             'mean_near_zero': mean_near_zero,
#             'white_noise': white_noise,
#             'normality': normality,
#             'residuals_std': np.std(residuals),
#             'diagnostics_pass': mean_near_zero and (white_noise is not False)
#         }
#     except Exception as e:
#         return {'success': False, 'error': str(e)}


# def compare_models(ts, orders=None):
#     """
#     Compare multiple ARIMA configurations.
    
#     Args:
#         ts (pd.Series): Time series data
#         orders (list): List of (p, d, q) tuples to compare
#                        Default: [(0,0,1), (1,0,0), (1,0,1), (1,0,2), (2,0,1)]
        
#     Returns:
#         dict: Comparison results with:
#             - models: DataFrame of model results sorted by AIC
#             - best_model: Best model result
#             - best_order: Best (p, d, q) tuple
#     """
#     if orders is None:
#         orders = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 0, 2), (2, 0, 1)]
    
#     results = []
    
#     for order in orders:
#         model_result = fit_arima(ts, order=order)
#         if model_result['success']:
#             results.append({
#                 'order': str(order),
#                 'aic': model_result['aic'],
#                 'bic': model_result['bic'],
#                 'success': True
#             })
#         else:
#             results.append({
#                 'order': str(order),
#                 'aic': float('inf'),
#                 'bic': float('inf'),
#                 'success': False
#             })
    
#     results_df = pd.DataFrame(results).sort_values('aic')
    
#     # Fit best model
#     best_order_str = results_df.iloc[0]['order']
#     best_order = eval(best_order_str)
#     best_model = fit_arima(ts, order=best_order)
    
#     return {
#         'models': results_df,
#         'best_model': best_model,
#         'best_order': best_order,
#         'comparison_df': results_df
#     }


# def train_complete_arima_pipeline(df, aggregation='monthly', order=(1, 0, 1), 
#                                    forecast_periods=12, train_test_split=0.8):
#     """
#     Complete ARIMA training pipeline from raw data to forecast.
    
#     Args:
#         df (pd.DataFrame): Sales data with Date column
#         aggregation (str): 'monthly', 'weekly', or 'daily'
#         order (tuple): (p, d, q) for ARIMA
#         forecast_periods (int): Number of periods to forecast
#         train_test_split (float): Fraction for training (0.8 = 80% train, 20% test)
        
#     Returns:
#         dict: Complete results with:
#             - time_series: Full prepared time series
#             - train_series: Training data
#             - test_series: Test data
#             - model_result: Fitted model
#             - forecast: Forecast results
#             - metrics: Model performance metrics
#             - diagnostics: Model diagnostics
#             - ready: Boolean indicating success
#     """
#     try:
#         # Step 1: Prepare time series
#         df = set_date_as_index(df)
        
#         if aggregation == 'monthly':
#             ts, _ = aggregate_sales_monthly(df)
#         elif aggregation == 'weekly':
#             ts, _ = aggregate_sales_weekly(df)
#         else:
#             ts, _ = aggregate_sales_daily(df)
        
#         # Step 2: Train-test split
#         split_idx = int(len(ts) * train_test_split)
#         train_ts = ts.iloc[:split_idx]
#         test_ts = ts.iloc[split_idx:]
        
#         # Step 3: Fit model on training data
#         model_result = fit_arima(train_ts, order=order)
        
#         if not model_result['success']:
#             return {
#                 'success': False,
#                 'error': f"Model fitting failed: {model_result.get('error', 'Unknown error')}"
#             }
        
#         # Step 4: Evaluate on test data
#         test_forecast = model_result['results'].get_forecast(steps=len(test_ts))
#         test_fitted = test_forecast.predicted_mean
        
#         metrics = calculate_metrics(test_ts, test_fitted)
        
#         # Step 5: Refit on full data for final forecast
#         final_model = fit_arima(ts, order=order)
#         forecast_result = generate_forecast(final_model, steps=forecast_periods)
        
#         # Step 6: Validate model
#         diagnostics = validate_model(final_model)
        
#         return {
#             'status': 'success',
#             'time_series': ts,
#             'train_series': train_ts,
#             'test_series': test_ts,
#             'train_size': len(train_ts),
#             'test_size': len(test_ts),
#             'model_result': final_model,
#             'forecast': forecast_result,
#             'metrics': metrics,
#             'diagnostics': diagnostics,
#             'order': order,
#             'aggregation': aggregation,
#             'forecast_periods': forecast_periods,
#             'ready': diagnostics.get('diagnostics_pass', False)
#         }
        
#     except Exception as e:
#         return {
#             'status': 'error',
#             'error': str(e),
#             'ready': False
#         }


# # ═══════════════════════════════════════════════════════════════════
# # STEP 14: Sales Forecasting Module
# # Functions to forecast future sales with flexible time horizons
# # ═══════════════════════════════════════════════════════════════════

# def forecast_n_months(ts, model_result, n_months=12, confidence=0.95):
#     """
#     Forecast sales for the next N months.
    
#     Args:
#         ts (pd.Series): Time series data with DatetimeIndex
#         model_result (dict): Fitted ARIMA model result from fit_arima()
#         n_months (int): Number of months to forecast (default 12)
#         confidence (float): Confidence level for intervals (default 0.95)
        
#     Returns:
#         dict: Forecast data with predictions, CI, and metrics
#               {
#                 'success': True/False,
#                 'n_months': Number of months forecasted,
#                 'forecast_dates': List of forecast dates,
#                 'forecast_values': List of predicted values,
#                 'lower_ci': Lower confidence interval,
#                 'upper_ci': Upper confidence interval,
#                 'forecast_df': DataFrame with all forecast data,
#                 'mean_forecast': Mean of forecast values,
#                 'std_forecast': Std dev of forecast values,
#                 'forecast_range': (min, max) of forecast values
#               }
#     """
#     try:
#         if not model_result.get('success', False):
#             return {'success': False, 'error': 'Invalid model result'}
        
#         # Generate forecast for N months
#         forecast_result = generate_forecast(
#             model_result,
#             steps=n_months,
#             confidence=confidence
#         )
        
#         if not forecast_result.get('success', False):
#             return {
#                 'success': False,
#                 'error': forecast_result.get('error', 'Forecast generation failed')
#             }
        
#         # Extract forecast values
#         forecast_values = forecast_result['forecast'].values
#         lower_ci = forecast_result['lower_ci'].values
#         upper_ci = forecast_result['upper_ci'].values
#         forecast_dates = forecast_result['forecast'].index.tolist()
        
#         # Calculate statistics
#         mean_forecast = float(np.mean(forecast_values))
#         std_forecast = float(np.std(forecast_values))
#         forecast_range = (float(np.min(forecast_values)), float(np.max(forecast_values)))
        
#         # Create forecast dataframe
#         forecast_df = pd.DataFrame({
#             'date': forecast_dates,
#             'forecast': forecast_values,
#             'lower_ci': lower_ci,
#             'upper_ci': upper_ci,
#             'ci_width': upper_ci - lower_ci
#         })
        
#         return {
#             'success': True,
#             'n_months': n_months,
#             'forecast_dates': forecast_dates,
#             'forecast_values': forecast_values.tolist(),
#             'lower_ci': lower_ci.tolist(),
#             'upper_ci': upper_ci.tolist(),
#             'forecast_df': forecast_df,
#             'mean_forecast': mean_forecast,
#             'std_forecast': std_forecast,
#             'forecast_range': forecast_range,
#             'confidence': confidence
#         }
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# def forecast_n_weeks(ts, model_result, n_weeks=52, confidence=0.95):
#     """
#     Forecast sales for the next N weeks.
    
#     Args:
#         ts (pd.Series): Time series data with DatetimeIndex
#         model_result (dict): Fitted ARIMA model result
#         n_weeks (int): Number of weeks to forecast (default 52)
#         confidence (float): Confidence level for intervals (default 0.95)
        
#     Returns:
#         dict: Forecast data with weekly predictions
#     """
#     try:
#         if not model_result.get('success', False):
#             return {'success': False, 'error': 'Invalid model result'}
        
#         # Generate forecast for N weeks
#         forecast_result = generate_forecast(
#             model_result,
#             steps=n_weeks,
#             confidence=confidence
#         )
        
#         if not forecast_result.get('success', False):
#             return {
#                 'success': False,
#                 'error': forecast_result.get('error', 'Forecast generation failed')
#             }
        
#         # Extract forecast values
#         forecast_values = forecast_result['forecast'].values
#         lower_ci = forecast_result['lower_ci'].values
#         upper_ci = forecast_result['upper_ci'].values
#         forecast_dates = forecast_result['forecast'].index.tolist()
        
#         # Calculate statistics
#         mean_forecast = float(np.mean(forecast_values))
#         std_forecast = float(np.std(forecast_values))
#         forecast_range = (float(np.min(forecast_values)), float(np.max(forecast_values)))
        
#         # Create forecast dataframe
#         forecast_df = pd.DataFrame({
#             'date': forecast_dates,
#             'forecast': forecast_values,
#             'lower_ci': lower_ci,
#             'upper_ci': upper_ci,
#             'ci_width': upper_ci - lower_ci
#         })
        
#         return {
#             'success': True,
#             'n_weeks': n_weeks,
#             'forecast_dates': forecast_dates,
#             'forecast_values': forecast_values.tolist(),
#             'lower_ci': lower_ci.tolist(),
#             'upper_ci': upper_ci.tolist(),
#             'forecast_df': forecast_df,
#             'mean_forecast': mean_forecast,
#             'std_forecast': std_forecast,
#             'forecast_range': forecast_range,
#             'confidence': confidence
#         }
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# def forecast_custom_horizon(ts, model_result, steps, horizon_type='months', 
#                             confidence=0.95, return_format='full'):
#     """
#     Forecast sales for custom time horizons with flexible output format.
    
#     Args:
#         ts (pd.Series): Time series data with DatetimeIndex
#         model_result (dict): Fitted ARIMA model result
#         steps (int): Number of periods to forecast
#         horizon_type (str): 'months', 'weeks', 'days', or 'periods'
#         confidence (float): Confidence level for intervals (default 0.95)
#         return_format (str): 'full' (complete data) or 'summary' (summary stats)
        
#     Returns:
#         dict: Forecast with custom horizon and format
#     """
#     try:
#         if not model_result.get('success', False):
#             return {'success': False, 'error': 'Invalid model result'}
        
#         if horizon_type not in ['months', 'weeks', 'days', 'periods']:
#             return {'success': False, 'error': f'Invalid horizon_type: {horizon_type}'}
        
#         # Generate forecast
#         forecast_result = generate_forecast(
#             model_result,
#             steps=steps,
#             confidence=confidence
#         )
        
#         if not forecast_result.get('success', False):
#             return {
#                 'success': False,
#                 'error': forecast_result.get('error', 'Forecast generation failed')
#             }
        
#         # Extract forecast values
#         forecast_values = forecast_result['forecast'].values
#         lower_ci = forecast_result['lower_ci'].values
#         upper_ci = forecast_result['upper_ci'].values
#         forecast_dates = forecast_result['forecast'].index.tolist()
        
#         # Calculate statistics
#         mean_forecast = float(np.mean(forecast_values))
#         std_forecast = float(np.std(forecast_values))
#         median_forecast = float(np.median(forecast_values))
#         min_forecast = float(np.min(forecast_values))
#         max_forecast = float(np.max(forecast_values))
        
#         if return_format == 'summary':
#             # Return summary statistics only
#             return {
#                 'success': True,
#                 'horizon_type': horizon_type,
#                 'steps': steps,
#                 'confidence': confidence,
#                 'summary': {
#                     'mean': mean_forecast,
#                     'median': median_forecast,
#                     'std': std_forecast,
#                     'min': min_forecast,
#                     'max': max_forecast,
#                     'range': (min_forecast, max_forecast),
#                     'forecast_count': len(forecast_values)
#                 }
#             }
#         else:
#             # Return full forecast data
#             forecast_df = pd.DataFrame({
#                 'date': forecast_dates,
#                 'forecast': forecast_values,
#                 'lower_ci': lower_ci,
#                 'upper_ci': upper_ci,
#                 'ci_width': upper_ci - lower_ci
#             })
            
#             return {
#                 'success': True,
#                 'horizon_type': horizon_type,
#                 'steps': steps,
#                 'confidence': confidence,
#                 'forecast_dates': forecast_dates,
#                 'forecast_values': forecast_values.tolist(),
#                 'lower_ci': lower_ci.tolist(),
#                 'upper_ci': upper_ci.tolist(),
#                 'forecast_df': forecast_df,
#                 'statistics': {
#                     'mean': mean_forecast,
#                     'median': median_forecast,
#                     'std': std_forecast,
#                     'min': min_forecast,
#                     'max': max_forecast,
#                     'range': (min_forecast, max_forecast)
#                 }
#             }
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# def forecast_with_scenario(ts, model_result, n_periods, scenario='baseline',
#                            confidence=0.95, growth_rate=0.0):
#     """
#     Generate forecast with different scenarios (baseline, pessimistic, optimistic).
    
#     Args:
#         ts (pd.Series): Time series data with DatetimeIndex
#         model_result (dict): Fitted ARIMA model result
#         n_periods (int): Number of periods to forecast
#         scenario (str): 'baseline', 'pessimistic', or 'optimistic'
#         confidence (float): Confidence level
#         growth_rate (float): Additional growth rate to apply (e.g., 0.05 for +5%)
        
#     Returns:
#         dict: Scenario-based forecast
#     """
#     try:
#         if not model_result.get('success', False):
#             return {'success': False, 'error': 'Invalid model result'}
        
#         if scenario not in ['baseline', 'pessimistic', 'optimistic']:
#             return {'success': False, 'error': f'Invalid scenario: {scenario}'}
        
#         # Generate baseline forecast
#         forecast_result = generate_forecast(
#             model_result,
#             steps=n_periods,
#             confidence=confidence
#         )
        
#         if not forecast_result.get('success', False):
#             return {
#                 'success': False,
#                 'error': forecast_result.get('error', 'Forecast generation failed')
#             }
        
#         # Extract values
#         forecast_values = forecast_result['forecast'].values.copy()
#         lower_ci = forecast_result['lower_ci'].values.copy()
#         upper_ci = forecast_result['upper_ci'].values.copy()
#         forecast_dates = forecast_result['forecast'].index.tolist()
        
#         # Apply scenario adjustments
#         if scenario == 'pessimistic':
#             # Use lower confidence interval and apply negative growth
#             forecast_values = lower_ci.copy()
#             if growth_rate < 0:
#                 forecast_values = forecast_values * (1 + growth_rate)
#             adjustment_factor = 0.95
            
#         elif scenario == 'optimistic':
#             # Use upper confidence interval and apply positive growth
#             forecast_values = upper_ci.copy()
#             if growth_rate > 0:
#                 forecast_values = forecast_values * (1 + growth_rate)
#             adjustment_factor = 1.05
            
#         else:  # baseline
#             # Apply growth rate if provided
#             if growth_rate != 0:
#                 forecast_values = forecast_values * (1 + growth_rate)
#             adjustment_factor = 1.0
        
#         # Calculate adjusted CIs
#         if scenario != 'baseline':
#             ci_width = upper_ci - lower_ci
#             lower_ci = forecast_values - ci_width * 0.5
#             upper_ci = forecast_values + ci_width * 0.5
        
#         # Create scenario dataframe
#         scenario_df = pd.DataFrame({
#             'date': forecast_dates,
#             'forecast': forecast_values,
#             'lower_ci': lower_ci,
#             'upper_ci': upper_ci,
#             'scenario': scenario,
#             'growth_applied': growth_rate
#         })
        
#         return {
#             'success': True,
#             'scenario': scenario,
#             'n_periods': n_periods,
#             'growth_rate': growth_rate,
#             'forecast_dates': forecast_dates,
#             'forecast_values': forecast_values.tolist(),
#             'lower_ci': lower_ci.tolist(),
#             'upper_ci': upper_ci.tolist(),
#             'scenario_df': scenario_df,
#             'mean_forecast': float(np.mean(forecast_values)),
#             'std_forecast': float(np.std(forecast_values))
#         }
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# def generate_forecast_summary(ts, model_result, n_periods=12, 
#                              include_scenarios=False, confidence=0.95):
#     """
#     Generate comprehensive forecast summary with all key metrics and scenarios.
    
#     Args:
#         ts (pd.Series): Time series data with DatetimeIndex
#         model_result (dict): Fitted ARIMA model result
#         n_periods (int): Number of periods to forecast
#         include_scenarios (bool): Include pessimistic/optimistic scenarios
#         confidence (float): Confidence level
        
#     Returns:
#         dict: Complete forecast summary with baseline and optional scenarios
#     """
#     try:
#         if not model_result.get('success', False):
#             return {'success': False, 'error': 'Invalid model result'}
        
#         # Get baseline forecast
#         baseline = forecast_custom_horizon(
#             ts,
#             model_result,
#             steps=n_periods,
#             horizon_type='periods',
#             confidence=confidence,
#             return_format='full'
#         )
        
#         if not baseline.get('success', False):
#             return {
#                 'success': False,
#                 'error': baseline.get('error', 'Baseline forecast failed')
#             }
        
#         # Prepare summary
#         summary = {
#             'success': True,
#             'n_periods': n_periods,
#             'confidence': confidence,
#             'baseline': baseline,
#             'scenarios': {}
#         }
        
#         # Add scenarios if requested
#         if include_scenarios:
#             pessimistic = forecast_with_scenario(
#                 ts,
#                 model_result,
#                 n_periods=n_periods,
#                 scenario='pessimistic',
#                 confidence=confidence,
#                 growth_rate=-0.05
#             )
            
#             optimistic = forecast_with_scenario(
#                 ts,
#                 model_result,
#                 n_periods=n_periods,
#                 scenario='optimistic',
#                 confidence=confidence,
#                 growth_rate=0.05
#             )
            
#             if pessimistic.get('success'):
#                 summary['scenarios']['pessimistic'] = pessimistic
            
#             if optimistic.get('success'):
#                 summary['scenarios']['optimistic'] = optimistic
        
#         return summary
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e)
#         }


# if __name__ == '__main__':
#     # Example usage
#     print("ARIMA Forecasting Functions - Ready for import")
#     print("\nAvailable Functions (STEP 13):")
#     print("  - fit_arima(ts, order)")
#     print("  - generate_forecast(model_result, steps)")
#     print("  - calculate_metrics(actual, fitted)")
#     print("  - validate_model(model_result)")
#     print("  - compare_models(ts, orders)")
#     print("  - train_complete_arima_pipeline(df, aggregation, order)")
#     print("\nAvailable Functions (STEP 14 - Sales Forecasting):")
#     print("  - forecast_n_months(ts, model_result, n_months)")
#     print("  - forecast_n_weeks(ts, model_result, n_weeks)")
#     print("  - forecast_custom_horizon(ts, model_result, steps, horizon_type)")
#     print("  - forecast_with_scenario(ts, model_result, n_periods, scenario)")
#     print("  - generate_forecast_summary(ts, model_result, n_periods)")
    
    
    
#     def sarima_forecast(train_data, steps=12):
#         model = SARIMAX(
#             train_data,
#             order=(1,1,1),
#             seasonal_order=(1,1,1,12)
#         )

#         results =   model.fit(disp=False)
#         forecast = results.forecast(steps=steps)

#         return forecast, results

# def evaluate_model(actual, predicted):
#     rmse = np.sqrt(mean_squared_error(actual, predicted))
#     mae = mean_absolute_error(actual, predicted)
#     mape = np.mean(np.abs((actual - predicted) / actual)) * 100

#     return rmse, mae, mape
