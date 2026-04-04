
# --- Standard and Flask imports ---
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
from data_loader import load_data, get_store_summary, get_store_data, get_data_stats
from data_preprocessor import preprocess_data
from csv_validator import validate_csv_structure, allowed_file, process_and_save_upload
from data_storage import init_sales_db, store_sales_data, get_user_sales_data, delete_user_sales_data, get_user_sales_summary
from data_analysis import perform_eda, get_monthly_trends, get_seasonal_patterns, get_peak_and_low_periods, get_store_analysis
from data_visualization import create_all_visualizations, create_sales_over_time_plot, create_seasonality_plot, create_seasonal_breakdown_plot, create_store_performance_plot, create_yearly_comparison_plot, create_distribution_plot
# from models.arima_model import arima_forecast, sarima_forecast, evaluate_model
# from models.arima_model import fit_arima, forecast_n_months, generate_forecast_summary, arima_forecast, sarima_forecast, evaluate_model, random_forest_forecast
from models.sales_predictor import SalesPredictor, train_sales_model_from_csv, predict_sales_from_csv
from models.random_forest_model import clean_data, feature_engineering, train_model, generate_future_dates, predict_future, create_forecast_plot, get_insights
from forecast_visualization import ForecastVisualizer, create_forecast_json_chart
from recommendation_engine import RecommendationEngine, create_recommendations_json
from error_handlers import (
    app_logger, SalesForecastingException, DataValidationError, DataLoadError,
    FileUploadError, InsufficientDataError, ModelTrainingError, DatabaseError,
    AuthenticationError, AuthorizationError, ResourceNotFoundError, 
    handle_errors, validate_request_data, require_auth, require_json, safe_db_operation,
    validate_csv_file, validate_data_available, ErrorRecoveryStrategy, DatabaseOperation,
    format_error_response
)
from io import StringIO, BytesIO
import json
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from reportlab.lib.pagesizes import letter, A4
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
# from reportlab.lib import colors
# from openpyxl import Workbook
# from openpyxl.styles import Font, PatternFill, Alignment, Border, Side


def convert_to_json_serializable(obj):
    """Convert numpy/pandas data types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def generate_future_predictions(df, model_path=None):
    """Generate predictions for next 365 days using Random Forest model with enhanced insight columns"""
    try:
        app_logger.info(f"[PREDICT] Starting prediction generation with {len(df)} records")
        
        # Extract store information from original data
        store_list = []
        store_name = "All Stores"
        if 'Store' in df.columns:
            store_list = df['Store'].unique()
            if len(store_list) == 1:
                store_name = f"Store {store_list[0]}"
            else:
                store_name = f"Combined ({len(store_list)} stores)"
        
        # Handle complex data format - aggregate sales by date if needed
        if 'Weekly_Sales' in df.columns and 'Date' in df.columns:
            # Aggregate sales by date for time series forecasting
            df_simple = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            df_simple = df_simple.rename(columns={'Weekly_Sales': 'Sales'})
            app_logger.info(f"[PREDICT] Found Weekly_Sales column, aggregated to {len(df_simple)} records")
        else:
            # Try to map common column names
            date_col = None
            sales_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                if 'sales' in col.lower() or 'weekly' in col.lower():
                    sales_col = col

            if date_col and sales_col:
                df_simple = df[[date_col, sales_col]].copy()
                df_simple = df_simple.rename(columns={date_col: 'Date', sales_col: 'Sales'})
                # Aggregate if there are multiple sales per date
                if df_simple['Date'].duplicated().any():
                    df_simple = df_simple.groupby('Date')['Sales'].sum().reset_index()
                app_logger.info(f"[PREDICT] Found Date/Sales columns: {date_col}/{sales_col}, {len(df_simple)} records")
            else:
                raise ValueError(f"Dataset must contain Date and Sales columns. Found: {df.columns.tolist()}")

        # Calculate historical statistics for context
        df_simple['Date'] = pd.to_datetime(df_simple['Date'])
        df_simple = df_simple.sort_values('Date')
        
        hist_mean = df_simple['Sales'].mean()
        hist_median = df_simple['Sales'].median()
        hist_std = df_simple['Sales'].std()
        hist_min = df_simple['Sales'].min()
        hist_max = df_simple['Sales'].max()
        hist_q75 = df_simple['Sales'].quantile(0.75)
        hist_q25 = df_simple['Sales'].quantile(0.25)
        
        app_logger.info(f"[PREDICT] Historical stats: mean={hist_mean:.0f}, min={hist_min:.0f}, max={hist_max:.0f}")
        
        # Calculate seasonal pattern per month/day from historical data
        df_simple['Month'] = df_simple['Date'].dt.month
        df_simple['DayOfYear'] = df_simple['Date'].dt.dayofyear
        monthly_avg = df_simple.groupby('Month')['Sales'].mean().to_dict()
        
        # ✅ Save the date and last date BEFORE cleaning (clean_data removes Date column)
        last_date = pd.to_datetime(df_simple['Date'].max())
        
        # ✅ Train model with raw data (train_model handles cleaning internally)
        app_logger.info(f"[PREDICT] Training model with raw data shape {df_simple.shape}")
        trained_model = train_model(df_simple)
        app_logger.info(f"[PREDICT] Model training returned: {type(trained_model)}, keys: {trained_model.keys() if isinstance(trained_model, dict) else 'N/A'}")

        # Generate future dates
        future_dates_df = generate_future_dates(365, last_date)  # Returns DataFrame with Date column
        app_logger.info(f"[PREDICT] Generated future dates dataframe: shape {future_dates_df.shape}, columns {future_dates_df.columns.tolist()}")
        future_dates_list = pd.to_datetime(future_dates_df['Date']).tolist()
        app_logger.info(f"[PREDICT] Extracted {len(future_dates_list)} future dates as list")

        # Create predictions using the trained model directly (avoid predict_future bugs)
        try:
            # Extract the actual model from the dict returned by train_model
            actual_model = trained_model['model'] if isinstance(trained_model, dict) else trained_model
            expected_features = trained_model['feature_columns'] if isinstance(trained_model, dict) and 'feature_columns' in trained_model else None
            
            app_logger.info(f"[PREDICT] Expected features from model: {expected_features}")
            
            # Create features for future dates - matching the exact features used in training
            future_data = []
            for date in future_dates_list:
                # Start with temporal features
                row = {
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'week': date.isocalendar()[1],
                    'day_of_week': date.weekday(),
                    'quarter': (date.month - 1) // 3 + 1,
                    'day_of_year': date.timetuple().tm_yday,
                }
                
                # Add other features that might have been in training data
                # Use reasonable defaults for these
                row.update({
                    'Sales': hist_mean,  # Use average as baseline
                    'Store': 1,
                    'product_id': 1,
                    'category': 1,
                    'season': ((date.month - 1) // 3),
                    'holiday': 0,
                    'promotion': 0,
                    'price': hist_mean * 0.8,
                    'inventory_level': 100,
                    'temperature': 20,
                    'rainfall': 0,
                    'quantity_sold': 1,
                })
                
                future_data.append(row)
            
            future_df = pd.DataFrame(future_data)
            app_logger.info(f"[PREDICT] Created future dataframe with shape {future_df.shape}")
            app_logger.info(f"[PREDICT] Future dataframe columns: {future_df.columns.tolist()}")
            
            # Use only the features that the model was trained with
            if expected_features:
                # Filter to only the features the model knows about
                feature_cols = [col for col in expected_features if col in future_df.columns]
                app_logger.info(f"[PREDICT] Using {len(feature_cols)} features for prediction: {feature_cols}")
            else:
                # Fallback: use all numeric columns except Date-related ones
                feature_cols = [col for col in future_df.columns if col not in ['Date', 'day_name', 'month_name']]
                app_logger.info(f"[PREDICT] Using fallback feature list: {feature_cols}")
            
            # Make predictions
            X_future = future_df[feature_cols]
            app_logger.info(f"[PREDICT] X_future shape: {X_future.shape}, dtypes: {X_future.dtypes.to_dict()}")
            
            predictions = actual_model.predict(X_future)
            app_logger.info(f"[PREDICT] predictions shape: {predictions.shape}")
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Date': future_dates_list,
                'Predicted_Sales': predictions
            })
            
            app_logger.info(f"[PREDICT] Generated {len(predictions_df)} predictions successfully")
            
        except Exception as pred_error:
            app_logger.error(f"[PREDICT] Direct prediction failed: {str(pred_error)}", exc_info=True)
            raise pred_error

        # Convert to the format expected by the template with enhanced columns
        predictions = []
        prev_pred = hist_mean  # For trend calculation
        
        for idx, row in predictions_df.iterrows():
            pred_date = row['Date']
            pred_sales = float(row['Predicted_Sales'])
            
            # Calculate useful metrics for user understanding
            month_num = pred_date.month
            week_num = pred_date.isocalendar()[1]
            month_name = pred_date.strftime('%B')
            
            # Compare to historical average
            percent_vs_average = ((pred_sales - hist_mean) / hist_mean * 100) if hist_mean > 0 else 0
            percent_vs_median = ((pred_sales - hist_median) / hist_median * 100) if hist_median > 0 else 0
            
            # Seasonal context
            monthly_hist_avg = monthly_avg.get(month_num, hist_mean)
            percent_vs_monthly = ((pred_sales - monthly_hist_avg) / monthly_hist_avg * 100) if monthly_hist_avg > 0 else 0
            
            # Trend direction (comparing to previous prediction)
            trend_pct = ((pred_sales - prev_pred) / prev_pred * 100) if prev_pred > 0 else 0
            trend_direction = "📈 Up" if trend_pct > 2 else "📉 Down" if trend_pct < -2 else "→ Stable"
            
            # Risk assessment based on confidence interval
            confidence_lower = float(row['Predicted_Sales'] * 0.85)
            confidence_upper = float(row['Predicted_Sales'] * 1.15)
            confidence_range = confidence_upper - confidence_lower
            confidence_pct = ((confidence_range / pred_sales) * 100) if pred_sales > 0 else 0
            
            # Risk level
            if confidence_pct > 30:
                risk_level = "⚠️ High"
                risk_color = "danger"
            elif confidence_pct > 20:
                risk_level = "🟡 Medium"
                risk_color = "warning"
            else:
                risk_level = "✅ Low"
                risk_color = "success"
            
            # Performance indicator vs historical
            if pred_sales > hist_q75:
                performance = "🌟 Excellent"
                perf_color = "success"
            elif pred_sales > hist_median:
                performance = "👍 Good"
                perf_color = "info"
            else:
                performance = "⚠️ Below Avg"
                perf_color = "warning"
            
            # Seasonal indicator
            if percent_vs_monthly > 10:
                seasonal_note = f"Peak Season +{percent_vs_monthly:.1f}%"
            elif percent_vs_monthly < -10:
                seasonal_note = f"Low Season {percent_vs_monthly:.1f}%"
            else:
                seasonal_note = "Normal Seasonal"
            
            predictions.append({
                # Basic info
                'date': pred_date.strftime('%Y-%m-%d'),
                'day_name': pred_date.strftime('%A'),
                'predicted_sales': float(pred_sales),
                'month': pred_date.strftime('%B %Y'),
                'week': f"Week {int(week_num)}",
                'store_name': store_name,
                'stores': [int(x) for x in store_list] if len(store_list) > 0 else [],
                
                # Confidence & Risk
                'confidence_lower': float(confidence_lower),
                'confidence_upper': float(confidence_upper),
                'confidence_range': float(confidence_range),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'risk_percentage': f"{confidence_pct:.1f}%",
                
                # Comparisons
                'percent_vs_average': f"{percent_vs_average:+.1f}%",
                'percent_vs_median': f"{percent_vs_median:+.1f}%",
                'percent_vs_monthly': f"{percent_vs_monthly:+.1f}%",
                'historical_average': float(hist_mean),
                'monthly_average': float(monthly_hist_avg),
                
                # Trends
                'trend_direction': trend_direction,
                'trend_percentage': f"{trend_pct:+.1f}%",
                'performance': performance,
                'performance_color': perf_color,
                'seasonal_note': seasonal_note,
                
                # Additional context
                'vs_max': f"{((pred_sales/hist_max)*100):.1f}% of max",
                'vs_min': f"{((pred_sales/hist_min)*100):.1f}% of min",
            })
            
            prev_pred = pred_sales

        # ✅ GENERATE MONTHLY PREDICTIONS
        monthly_predictions = []
        predictions_df['Month'] = pd.to_datetime(predictions_df['Date']).dt.strftime('%B %Y')
        monthly_group = predictions_df.groupby('Month')['Predicted_Sales'].agg(['sum', 'mean', 'min', 'max', 'count']).reset_index()
        
        for idx, month_row in monthly_group.iterrows():
            month_sales = month_row['sum']
            month_avg = month_row['mean']
            month_min = month_row['min']
            month_max = month_row['max']
            days_in_month = month_row['count']
            
            # Compare to historical
            percent_vs_avg = ((month_sales - (hist_mean * days_in_month)) / (hist_mean * days_in_month) * 100) if hist_mean > 0 else 0
            
            # Risk based on variance
            variance = month_max - month_min
            confidence_range = variance
            confidence_pct = ((confidence_range / month_sales) * 100) if month_sales > 0 else 0
            
            if confidence_pct > 30:
                risk_level = "⚠️ High"
                risk_color = "danger"
            elif confidence_pct > 20:
                risk_level = "🟡 Medium"
                risk_color = "warning"
            else:
                risk_level = "✅ Low"
                risk_color = "success"
            
            monthly_predictions.append({
                'month': month_row['Month'],
                'total_sales': float(month_sales),
                'avg_daily_sales': float(month_avg),
                'min_sales': float(month_min),
                'max_sales': float(month_max),
                'days_in_month': int(days_in_month),
                'percent_vs_avg': f"{percent_vs_avg:+.1f}%",
                'risk_level': risk_level,
                'risk_color': risk_color,
            })

        # Get insights
        insights = get_insights(predictions_df)

        return predictions, trained_model, insights, monthly_predictions

    except Exception as e:
        app_logger.error(f"Error generating future predictions: {str(e)}")
        return [], None, {}, []


app = Flask(__name__)
app.config.from_object(Config)

# Register custom Jinja2 filters
@app.template_filter('average')
def average_filter(values):
    """Calculate the average of a list of values."""
    if not values or len(values) == 0:
        return 0
    try:
        return sum(values) / len(values)
    except (TypeError, ValueError):
        return 0

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Download forecast as CSV for user-uploaded data ---
# previously we offered a separate download endpoint for forecast CSV.  
# This functionality is now folded into the JSON-producing `/api/forecast-data` route,
# which checks for a ``download`` query parameter.  The old handler remains here for
# historical reference but is effectively disabled by removing its route decorator.
#
# @app.route('/api/forecast-data', methods=['GET'])
# @login_required
# def download_forecast_csv():
#     # legacy CSV download logic (migrated to get_forecast_data)
#     ...


# Configure logging for session management
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize sales database
init_sales_db()

# Load dataset at startup (with fallback)
# NOTE: we keep a flag to ensure users explicitly upload before any analysis/forecasting
# even if a default Walmart dataset is present on disk. This prevents the app from
# showing results automatically when the server restarts.
df = None
store_summary = None
data_source = None
preprocessing_stats = {}

# set to True only once a new file has been uploaded during the current session
upload_completed = False

try:
    df, data_source = load_data(prefer_uploaded=True)
    if df is not None:
        # Apply preprocessing
        df, preprocessing_stats = preprocess_data(df)
        store_summary = get_store_summary(df)
        print(f"✅ Data loaded successfully from {data_source} dataset ({len(df)} records)")
        print(f"📊 Preprocessing: {preprocessing_stats['records_removed']} records removed, {preprocessing_stats['missing_values_after']} missing values remain")
    else:
        print("⚠️ Warning: No dataset available")
except Exception as e:
    print(f"⚠️ Warning: Could not load dataset: {e}")


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db


def init_db():
    db = sqlite3.connect(app.config['DATABASE'])
    cur = db.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    db.commit()
    db.close()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def before_request():
    """Check session validity before each request with proper error handling."""
    try:
        if 'user_id' in session:
            # Session exists - set as permanent and update activity
            session.permanent = True
            app.permanent_session_lifetime = app.config['PERMANENT_SESSION_LIFETIME']
            
            # Log activity with proper logging
            username = session.get('username', 'unknown')
            app_logger.info(f"Request: {request.method} {request.path} by user '{username}'")
    except Exception as e:
        # Log the error but don't fail the request
        app_logger.error(f"Error in before_request hook: {str(e)}")
        # Don't raise - allow the request to continue


@app.errorhandler(SalesForecastingException)
def handle_sales_forecasting_exception(e):
    """Handle custom application exceptions."""
    app_logger.warning(f"Application exception: {e.error_code} - {e.message}")
    response, status_code = format_error_response(e, include_traceback=False)
    return jsonify(response), status_code


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler for uncaught exceptions."""
    app_logger.error(f"Unhandled exception: {type(e).__name__} - {str(e)}", exc_info=True)
    response, status_code = format_error_response(e, include_traceback=False)
    return jsonify(response), status_code


@app.errorhandler(404)
def page_not_found(e):
    """Handle page not found errors."""
    return jsonify({
        'status': 'error',
        'error_code': 'RESOURCE_NOT_FOUND',
        'message': 'The requested resource was not found.',
        'error_type': '404'
    }), 404


@app.errorhandler(403)
def access_denied(e):
    """Handle access denied errors."""
    return jsonify({
        'status': 'error',
        'error_code': 'AUTHORIZATION_ERROR',
        'message': 'You do not have permission to access this resource.',
        'error_type': '403'
    }), 403


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    app_logger.critical(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'error_code': 'INTERNAL_ERROR',
        'message': 'An internal server error occurred. Please try again later.',
        'error_type': '500',
        'timestamp': datetime.now().isoformat()
    }), 500


@app.after_request
def after_request(response):
    """Set security headers on response"""
    # Prevent caching of sensitive responses
    if request.path in ['/logout', '/login', '/register', '/dashboard']:
        response.headers['Cache-Control'] = 'no-store, no-cache, no-transform, must-revalidate, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    return response


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        print("[DEBUG] Health endpoint called")
        result = jsonify({'status': 'ok', 'message': 'Server is running'})
        print(f"[DEBUG] Returning response: {result}")
        return result, 200
    except Exception as e:
        print(f"[DEBUG] ERROR in health endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('register'))
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                       (username, generate_password_hash(password)))
            db.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with comprehensive error checking."""
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            
            # Validate input
            if not username or not password:
                raise DataValidationError(
                    "Username and password are required.",
                    details={'missing_fields': ['username' if not username else 'password']}
                )
            
            # Check username length
            if len(username) < 3 or len(username) > 50:
                raise DataValidationError(
                    "Username must be between 3 and 50 characters.",
                    field='username'
                )
            
            if len(password) < 6:
                raise DataValidationError(
                    "Password must be at least 6 characters.",
                    field='password'
                )
            
            # Query database safely
            db = get_db()
            user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['login_time'] = datetime.now().isoformat()
                session['ip_address'] = request.remote_addr
                
                app_logger.info(f"User '{username}' logged in successfully from {request.remote_addr}")
                flash('Logged in successfully.', 'success')
                return redirect(url_for('dashboard'))
            
            # Log failed attempt
            app_logger.warning(f"Failed login attempt for username '{username}' from {request.remote_addr}")
            raise AuthenticationError(
                "Invalid username or password.",
                reason='invalid_credentials',
                details={'username': username}
            )
            
        except (DataValidationError, AuthenticationError) as e:
            app_logger.warning(f"Login validation error: {e.message}")
            flash(e.message, 'error')
            return redirect(url_for('login'))
        except Exception as e:
            app_logger.error(f"Unexpected error during login: {str(e)}")
            flash('An unexpected error occurred during login. Please try again.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """Secure logout handler with proper error handling and logging."""
    try:
        user_id = session.get('user_id')
        username = session.get('username', 'unknown')
        ip_address = request.remote_addr
        login_time = session.get('login_time', 'unknown')
        
        # Log logout event with session details
        logout_time = datetime.now().isoformat()
        app_logger.info(f"User '{username}' (ID: {user_id}) logged out from {ip_address} | "
                       f"Session: {login_time} → {logout_time}")
        
        # Clear all session data
        session.clear()
        
        flash('You have been logged out successfully.', 'success')
        response = make_response(redirect(url_for('login')))
        
        # Ensure cookies are cleared properly
        response.set_cookie('session', '', 
                          expires=0,
                          secure=app.config.get('SESSION_COOKIE_SECURE', False),
                          httponly=app.config.get('SESSION_COOKIE_HTTPONLY', True),
                          samesite=app.config.get('SESSION_COOKIE_SAMESITE', 'Lax'))
        
        return response
        
    except Exception as e:
        app_logger.error(f"Error during logout: {str(e)}", exc_info=True)
        flash('An error occurred during logout. Please try again.', 'error')
        return redirect(url_for('dashboard'))


@app.route('/dashboard')
#@login_required  # authentication optional for viewing dashboard
def dashboard():
    """Display dashboard with data statistics and visualizations."""
    try:
        # if user has not yet uploaded a dataset this session, remind them
        if not upload_completed:
            flash('Upload a dataset to enable analysis, forecasting, and recommendations.', 'info')

        stats = {}
        ml_predictions = session.get('future_predictions', [])
        
        if df is not None and not df.empty and upload_completed:
            stats = get_data_stats(df)
            stats['data_source'] = f"Data loaded from {data_source.title()} Dataset"
            user_info = session.get('user_id', 'guest')
            app_logger.info(f"Dashboard loaded for user {user_info} with {len(df)} records")
        else:
            user_info = session.get('user_id', 'guest')
            if not upload_completed:
                app_logger.info(f"Dashboard accessed by user {user_info} - no upload yet (as expected)")
            else:
                app_logger.warning(f"Dashboard accessed but no data available for user {user_info}")
        
        return render_template('dashboard_redesigned.html', 
                             stats=stats, 
                             stores=store_summary.to_dict('records') if store_summary is not None else [], 
                             data_source=data_source, 
                             preprocessing_stats=preprocessing_stats,
                             data_uploaded=upload_completed,
                             training_results=session.get('training_results', {}),
                             insights=session.get('insights', {}),
                             recommendations=session.get('insights', {}).get('recommendations', []),
                             ml_predictions=ml_predictions,  # ✅ ADD PREDICTIONS TO DASHBOARD
                             model_trained=session.get('model_trained', False))
    except Exception as e:
        # don't send user away from dashboard; show them the page with an error message
        app_logger.error(f"Error loading dashboard: {str(e)}", exc_info=True)
        flash('Error loading dashboard. Please try uploading data again.', 'error')
        # render dashboard with empty stats so template still loads
        return render_template('dashboard_redesigned.html',
                             stats={},
                             stores=[],
                             data_source=data_source,
                             preprocessing_stats=preprocessing_stats,
                             data_uploaded=upload_completed,
                             training_results=session.get('training_results', {}),
                             insights=session.get('insights', {}),
                             recommendations=session.get('insights', {}).get('recommendations', []),
                             ml_predictions=session.get('future_predictions', []),  # ✅ ADD PREDICTIONS
                             model_trained=session.get('model_trained', False))


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle CSV file upload with comprehensive error handling."""
    global df, store_summary, data_source, preprocessing_stats
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        try:
            # Validate file presence
            if 'file' not in request.files:
                raise FileUploadError('No file provided in upload request')
            
            file = request.files['file']
            
            # Validate file
            is_valid, validation_message = validate_csv_file(file)
            if not is_valid:
                raise FileUploadError(validation_message, filename=file.filename if file else 'unknown')
            
            # Secure filename
            filename = secure_filename(file.filename)
            if not filename:
                raise FileUploadError('Invalid filename provided')
            
            # Save temporarily
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
            
            try:
                file.save(temp_filepath)
                
                # Validate CSV structure
                try:
                    is_valid, message, validated_df = validate_csv_structure(temp_filepath)
                except Exception as e:
                    raise DataValidationError(
                        f"CSV validation failed: {str(e)}",
                        field='file'
                    )
                
                if not is_valid:
                    raise DataValidationError(
                        f"CSV validation failed: {message}",
                        field='file'
                    )
                
                if validated_df is None or validated_df.empty:
                    raise DataValidationError(
                        "CSV file contains no valid data",
                        field='file'
                    )
                
                # Save processed CSV
                upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_sales.csv')
                if not process_and_save_upload(validated_df, upload_filepath):
                    raise DataLoadError(
                        "Failed to save processed file",
                        filename=filename
                    )
                
                # Reload data from uploaded file
                try:
                    df, data_source = load_data(prefer_uploaded=True)
                    if df is None or df.empty:
                        raise DataLoadError(
                            "Data loaded but dataframe is empty",
                            filename=filename
                        )
                    
                    # Apply preprocessing
                    df, preprocessing_stats = preprocess_data(df)
                    store_summary = get_store_summary(df)
                    
                    # Clear previous user data and store new data
                    delete_user_sales_data(user_id)
                    store_sales_data(user_id, df)
                    
                    # mark upload success so other pages become available
                    global upload_completed
                    upload_completed = True
                    
                    # ⚡ AUTOMATIC COMPLETE ANALYSIS & PREDICTIONS ⚡
                    try:
                        app_logger.info(f"Starting automatic complete analysis for user {user_id}")
                        
                        # Step 1: Perform EDA (Exploratory Data Analysis)
                        try:
                            eda_results = perform_eda(df)
                            session['eda_results'] = convert_to_json_serializable(eda_results)
                            session.modified = True
                            app_logger.info(f"EDA completed for user {user_id}")
                        except Exception as eda_error:
                            app_logger.warning(f"EDA failed: {str(eda_error)}")
                            session['eda_results'] = {}
                        
                        # Step 2: Get monthly trends
                        try:
                            monthly_trends = get_monthly_trends(df)
                            session['monthly_trends'] = convert_to_json_serializable(monthly_trends)
                            session.modified = True
                        except Exception as trend_error:
                            app_logger.warning(f"Monthly trends failed: {str(trend_error)}")
                            session['monthly_trends'] = {}
                        
                        # Step 3: Get seasonal patterns
                        try:
                            seasonal_patterns = get_seasonal_patterns(df)
                            session['seasonal_patterns'] = convert_to_json_serializable(seasonal_patterns)
                            session.modified = True
                        except Exception as seasonal_error:
                            app_logger.warning(f"Seasonal patterns failed: {str(seasonal_error)}")
                            session['seasonal_patterns'] = {}
                        
                        # Step 4: Get peak periods
                        try:
                            peak_periods = get_peak_and_low_periods(df)
                            session['peak_periods'] = convert_to_json_serializable(peak_periods)
                            session.modified = True
                        except Exception as peak_error:
                            app_logger.warning(f"Peak periods failed: {str(peak_error)}")
                            session['peak_periods'] = {}
                        
                        # Step 5: Get store analysis
                        try:
                            if 'Store' in df.columns:
                                store_analysis = get_store_analysis(df)
                                session['store_analysis'] = convert_to_json_serializable(store_analysis)
                                session.modified = True
                        except Exception as store_error:
                            app_logger.warning(f"Store analysis failed: {str(store_error)}")
                            session['store_analysis'] = {}
                        
                        # Step 6: Train Random Forest model
                        model_save_path = os.path.join('models', f"sales_predictor_{user_id}.joblib")
                        os.makedirs('models', exist_ok=True)
                        
                        training_result = train_sales_model_from_csv(upload_filepath, model_save_path)
                        
                        if training_result['success']:
                            # Store training results in session
                            session['training_results'] = convert_to_json_serializable(training_result['results'])
                            session['model_trained'] = True
                            session['model_path'] = model_save_path
                            # ⚠️ CRITICAL: Mark session as modified to ensure it persists
                            session.modified = True
                            app_logger.info(f"Model training success for {user_id}")

                            # Step 7: Generate predictions for next 365 days
                            try:
                                future_predictions, trained_model, insights, monthly_predictions = generate_future_predictions(df)
                                if future_predictions and len(future_predictions) > 0:
                                    # ⚠️ DON'T store predictions in session - too large for cookie!
                                    # Only store a flag that predictions were generated
                                    session['insights'] = convert_to_json_serializable(insights)
                                    session['predictions_available'] = True
                                    
                                    # ⚠️ CRITICAL: Mark session as modified to ensure it persists
                                    session.modified = True

                                    app_logger.info(f"✅ Complete analysis done! Model trained and {len(future_predictions)} predictions generated for user {user_id}")
                                else:
                                    app_logger.error(f"Prediction generation returned empty predictions for {user_id}")
                                    flash('⚠️ Warning: Model trained but prediction generation returned no results. Please check your data.', 'warning')
                            except Exception as pred_error:
                                app_logger.error(f"Prediction generation failed for {user_id}: {str(pred_error)}", exc_info=True)
                                flash(f'⚠️ Warning: Model trained but prediction generation failed: {str(pred_error)}', 'warning')
                        else:
                            app_logger.error(f"Model training failed for user {user_id}: {training_result.get('error', 'Unknown error')}")
                            flash(f'⚠️ File uploaded but model training failed: {training_result.get("error", "Unknown error")}', 'error')

                    except Exception as analysis_error:
                        app_logger.error(f"Error during automatic analysis: {str(analysis_error)}", exc_info=True)
                        flash(f'⚠️ Error during analysis: {str(analysis_error)}', 'error')
                    
                    app_logger.info(f"User {user_id} uploaded file '{filename}' with {len(df)} records")
                    flash(f'✅ File uploaded and ML model trained successfully! {message}', 'success')
                    
                    # Redirect to predictions page to show immediate results
                    return redirect(url_for('predictions'))
                    
                except Exception as data_error:
                    app_logger.error(f"Error processing uploaded data: {str(data_error)}")
                    raise DataLoadError(
                        f"Error processing data: {str(data_error)}",
                        filename=filename
                    )
                
            finally:
                # Always clean up temp file
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except Exception as cleanup_error:
                        app_logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
        
        except (FileUploadError, DataValidationError, DataLoadError) as e:
            app_logger.warning(f"Upload error: {e.message}")
            flash(f'Upload failed: {e.message}', 'error')
            return redirect(url_for('upload'))
        except Exception as e:
            app_logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
            flash(f'An unexpected error occurred during upload: {str(e)}', 'error')
            return redirect(url_for('upload'))
    
    # after a GET or a failed POST we still render upload page
    # if the upload just succeeded, the flag will already have been set above
    return render_template('upload_improved.html', data_source=data_source, data_uploaded=upload_completed)


@app.route('/analysis')
@login_required
def analysis():
    if df is None or not upload_completed:
        flash('Please upload a dataset before accessing analysis.', 'error')
        return redirect(url_for('upload'))
    stores = df['Store'].unique().tolist()
    return render_template('analysis_improved.html', 
                          stores=stores, 
                          data_uploaded=upload_completed,
                          training_results=session.get('training_results', {}))


@app.route('/api/eda-summary')
@login_required
@handle_errors
def api_eda_summary():
    """Get complete EDA summary for the current dataset."""
    validate_data_available(df)
    
    try:
        eda_results = perform_eda(df)
        return jsonify(eda_results)
    except Exception as e:
        app_logger.error(f"Error generating EDA summary: {str(e)}")
        raise DataLoadError(f"Failed to generate EDA summary: {str(e)}")


@app.route('/api/monthly-trends')
@login_required
@handle_errors
def api_monthly_trends():
    """Get monthly sales trends."""
    validate_data_available(df)
    trends = get_monthly_trends(df)
    return jsonify(trends)


@app.route('/api/seasonal-patterns')
@login_required
@handle_errors
def api_seasonal_patterns():
    """Get seasonal sales patterns."""
    validate_data_available(df)
    patterns = get_seasonal_patterns(df)
    return jsonify(patterns)


@app.route('/api/peak-periods')
@login_required
@handle_errors
def api_peak_periods():
    """Get peak and low-demand periods."""
    validate_data_available(df)
    periods = get_peak_and_low_periods(df)
    return jsonify(periods)


@app.route('/api/store-performance')
@login_required
@handle_errors
def api_store_performance():
    """Get detailed store performance analysis."""
    validate_data_available(df)
    analysis = get_store_analysis(df)
    return jsonify(analysis)


@app.route('/api/store-data/<int:store_id>')
@login_required
@handle_errors
def api_store_data(store_id):
    """Get store-specific sales data."""
    validate_data_available(df)
    
    # Validate store ID
    if store_id <= 0:
        raise DataValidationError(
            "Store ID must be a positive integer",
            field='store_id',
            details={'provided': store_id}
        )
    
    store_data = get_store_data(df, store_id)
    if store_data.empty:
        raise ResourceNotFoundError(
            f"Store {store_id} not found in dataset",
            resource_type='store',
            resource_id=store_id
        )
    
    try:
        return jsonify({
            'store_id': store_id,
            'records': len(store_data),
            'avg_sales': float(store_data['Weekly_Sales'].mean()),
            'dates': store_data['Date'].astype(str).tolist(),
            'sales': store_data['Weekly_Sales'].tolist()
        })
    except Exception as e:
        app_logger.error(f"Error serializing store data for store {store_id}: {str(e)}")
        raise DataLoadError(
            f"Failed to retrieve store {store_id} data",
            details={'store_id': store_id}
        )



@app.route('/api/user-data-summary')
@login_required
def user_data_summary():
    """Get summary of user's stored sales data."""
    user_id = session.get('user_id')
    summary = get_user_sales_summary(user_id)
    return jsonify(summary)


@app.route('/api/export-data')
@login_required
def export_data():
    """Export user's sales data as CSV."""
    user_id = session.get('user_id')
    df_user = get_user_sales_data(user_id)
    
    if df_user.empty:
        return jsonify({'error': 'No data to export'}), 404
    
    # Convert to CSV string
    csv_string = df_user.to_csv(index=False)
    return jsonify({
        'data': csv_string,
        'records': len(df_user),
        'filename': f'sales_data_{user_id}.csv'
    })


# ============================================
# STEP 10: Data Visualization Routes
# ============================================

@app.route('/api/visualizations')
@login_required
def api_visualizations():
    """Get all available visualizations."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    visualizations = create_all_visualizations(df)
    
    return jsonify({
        'sales_over_time': visualizations.get('sales_over_time'),
        'seasonality': visualizations.get('seasonality'),
        'seasonal_breakdown': visualizations.get('seasonal_breakdown'),
        'store_performance': visualizations.get('store_performance'),
        'yearly_comparison': visualizations.get('yearly_comparison'),
        'distribution': visualizations.get('distribution')
    })



# helper to refresh global dataframe from disk

def refresh_data():
    """Load the latest data (uploaded or walmart) and update globals.
    Returns the dataframe or None if unavailable."""
    global df, data_source, preprocessing_stats, store_summary
    df, data_source = load_data(prefer_uploaded=True)
    if df is not None and not df.empty:
        df, preprocessing_stats = preprocess_data(df)
        store_summary = get_store_summary(df)
    return df


@app.route('/api/plot/sales-over-time')
@login_required
def plot_sales_over_time():
    """Get sales over time visualization."""
    # ensure we are always working with the latest dataset on disk
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_sales_over_time_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Sales Over Time'})
    # prevent client-side caching of old images
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/plot/seasonality')
@login_required
def plot_seasonality():
    """Get seasonality visualization."""
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_seasonality_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Monthly Seasonality Patterns'})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/plot/seasonal-breakdown')
@login_required
def plot_seasonal_breakdown():
    """Get seasonal breakdown visualization."""
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_seasonal_breakdown_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Seasonal Breakdown'})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/plot/store-performance')
@login_required
def plot_store_perf():
    """Get store performance visualization."""
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_store_performance_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Store Performance Analysis'})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/plot/yearly-comparison')
@login_required
def plot_yearly_comp():
    """Get yearly comparison visualization."""
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_yearly_comparison_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Year-over-Year Comparison'})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/plot/distribution')
@login_required
def plot_dist():
    """Get sales distribution visualization."""
    if df is None:
        refresh_data()
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_distribution_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    response = jsonify({'plot': plot, 'title': 'Sales Distribution'})
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


@app.route('/api/forecast-data')
@login_required
@handle_errors
def api_forecast_data():
    """Return historical and forecast data in JSON or CSV format."""
    validate_data_available(df)

    # Normalize and sort history
    history_df = df.copy()
    if 'Date' in history_df.columns:
        history_df['Date'] = pd.to_datetime(history_df['Date'], errors='coerce')
        history_df = history_df.dropna(subset=['Date']).sort_values('Date')

    sales_column = 'Weekly_Sales' if 'Weekly_Sales' in history_df.columns else 'Sales' if 'Sales' in history_df.columns else None
    if sales_column is None:
        raise DataLoadError('No Sales column found in dataset for forecast output.')

    historical_payload = {
        'dates': history_df['Date'].astype(str).tolist(),
        'values': history_df[sales_column].astype(float).tolist(),
        'records': len(history_df)
    }

    # Use cached forecast state if available
    forecast_items = session.get('future_predictions') or []
    if not forecast_items:
        try:
            forecast_items, _, _, _ = generate_future_predictions(df)
        except Exception as e:
            app_logger.error(f"Unable to generate forecast data: {e}", exc_info=True)
            raise DataLoadError(f"Unable to generate forecast data: {e}")

    forecast_payload = {
        'dates': [item.get('date') for item in forecast_items],
        'values': [item.get('predicted_sales') for item in forecast_items],
        'records': len(forecast_items)
    }

    if request.args.get('download', '0').lower() in ['1', 'true', 'yes']:
        from io import StringIO
        import csv

        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(['date', 'predicted_sales'])
        for d, v in zip(forecast_payload['dates'], forecast_payload['values']):
            writer.writerow([d, v])

        csv_response = make_response(si.getvalue())
        csv_response.headers['Content-Type'] = 'text/csv'
        csv_response.headers['Content-Disposition'] = 'attachment; filename=forecast_data.csv'
        return csv_response

    mean_sales = float(np.mean(historical_payload['values'])) if historical_payload['values'] else 0.0
    forecast_months = [pd.to_datetime(date).strftime('%B %Y') for date in forecast_payload['dates']] if forecast_payload['dates'] else []

    return jsonify({
        'success': True,
        'historical': historical_payload,
        'forecast': forecast_payload,
        'mean_sales': mean_sales,
        'forecast_months': forecast_months,
        'training_results': session.get('training_results', {}),
        'insights': session.get('insights', {})
    })


@app.route('/api/forecast')
@login_required
@handle_errors
def api_forecast():
    """Backward-compatible alias for `/api/forecast-data`."""
    return api_forecast_data()


@app.route('/forecast')
@login_required
def forecast_page():
    """Render forecast page template."""
    return render_template(
        'forecast_enhanced.html',
        data_uploaded=upload_completed,
        training_results=session.get('training_results', {}),
        insights=session.get('insights', {}),
        future_predictions=session.get('future_predictions', [])
    )


@app.route('/api/visualizations/generate')
@login_required
def generate_visualizations():
    """Generate all forecast visualizations using user-uploaded data"""
    try:
        # Get user's uploaded data
        user_id = session.get('user_id')
        from data_storage import get_user_sales_data
        user_df = get_user_sales_data(user_id)
        
        # Use user data if available, otherwise fall back to global df
        if user_df is not None and not user_df.empty:
            use_df = user_df.copy()
            use_df['Date'] = pd.to_datetime(use_df['date']) if 'date' in use_df.columns else pd.to_datetime(use_df['Date'])
            use_df['Weekly_Sales'] = use_df['sales'] if 'sales' in use_df.columns else use_df['Weekly_Sales']
            use_df = use_df[['Date', 'Weekly_Sales']]
        elif df is not None:
            use_df = df.copy()
        else:
            return jsonify({'success': False, 'error': 'Dataset not available'}), 500
        
        # Prepare time series
        df_copy = use_df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # train on the full monthly series so visualizations reflect all data
        ts_train = ts_monthly
        
        # Train Random Forest model
        rf_result = random_forest_forecast(ts_train, steps=12)
        
        # Create forecast result compatible with visualizer
        forecast_result = {
            'forecast_df': pd.DataFrame({
                'forecast': rf_result['forecast'],
                'lower_ci': rf_result['conf_int']['lower'],
                'upper_ci': rf_result['conf_int']['upper']
            })
        }
        
        # Generate forecast summary for scenarios
        forecast_values = rf_result['forecast'].values
        forecast_summary = {
            "baseline": forecast_result,
            "scenarios": {
                "pessimistic": {
                    "forecast_values": forecast_values * 0.95
                },
                "optimistic": {
                    "forecast_values": forecast_values * 1.05
                }
            }
        }
        
        # Create visualizations
        visualizer = ForecastVisualizer(output_dir='static/plots')
        viz_result = visualizer.generate_all_visualizations(ts_train, forecast_result, forecast_summary)
        
        if viz_result['success']:
            return jsonify({
                'success': True,
                'visualizations': viz_result['visualizations'],
                'total_generated': viz_result['total_generated'],
                'errors': viz_result['errors']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate visualizations',
                'errors': viz_result['errors']
            }), 500
            
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/recommendation')
@login_required
def recommendation():
    """Display business recommendations page."""
    if df is None or not upload_completed:
        flash('Please upload a dataset before viewing recommendations.', 'error')
        return redirect(url_for('upload'))
    try:
        user_id = session.get('user_id')
        from data_storage import get_user_sales_data
        user_df = get_user_sales_data(user_id)
        use_df = None
        if user_df is not None and not user_df.empty:
            # Convert date column to datetime and rename columns for compatibility
            user_df['Date'] = pd.to_datetime(user_df['date']) if 'date' in user_df.columns else pd.to_datetime(user_df['Date'])
            user_df['Weekly_Sales'] = user_df['sales'] if 'sales' in user_df.columns else user_df['Weekly_Sales']
            use_df = user_df.rename(columns={'Date': 'Date', 'Weekly_Sales': 'Weekly_Sales'})
        else:
            use_df = df

        validate_data_available(use_df)

        # Prepare monthly time series
        df_copy = use_df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()

        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()

        if len(ts_monthly) < 3:
            raise InsufficientDataError(
                'Insufficient historical data for recommendations',
                required=3,
                actual=len(ts_monthly)
            )

        # Generate recommendations and quick wins
        engine = RecommendationEngine(ts_monthly)

        recommendations = engine.generate_comprehensive_recommendations()
        quick_wins = engine.get_quick_wins()
        stock_recommendations = engine.generate_stock_recommendations()
        festival_recommendations = engine.generate_festival_promotions()
        insights = {
            'elasticity': engine.calculate_price_elasticity(),
            'volatility': {'value': f"{engine.analyze_demand_patterns().get('volatility', 0):.2f}", 'description': 'Sales volatility index'},
            'peak': {'value': engine.analyze_demand_patterns().get('peak_value', 0), 'description': 'Peak sales period'},
            'position': {'value': recommendations.get('competitive_positioning', {}).get('market_position', ''), 'description': 'Market position insight'}
        }

        # Get ML predictions from session if available
        ml_predictions = session.get('future_predictions', [])
        training_results = session.get('training_results', {})

        app_logger.info(f"Recommendations generated for user {user_id}")

        return render_template(
            'recommendation_improved.html',
            quick_wins=quick_wins,
            recommendations=recommendations,
            stock_recommendations=stock_recommendations,
            festival_recommendations=festival_recommendations,
            insights=insights,
            ml_predictions=ml_predictions,
            training_results=training_results,
            model_trained=session.get('model_trained', False),
            data_uploaded=upload_completed
        )

    except (DataLoadError, InsufficientDataError) as e:
        app_logger.warning(f"Recommendation page error: {e.message}")
        return render_template('recommendation_improved.html', error=e.message), 422
    except Exception as e:
        app_logger.error(f"Error loading recommendation page: {str(e)}", exc_info=True)
        return render_template('recommendation_improved.html', error="Failed to load recommendations"), 500


@app.route('/api/recommendations')
@login_required
@handle_errors
def get_recommendations():
    """Get recommendations as JSON."""
    validate_data_available(df)
    
    try:
        # Prepare monthly time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        if len(ts_monthly) < 3:
            raise InsufficientDataError(
                'Insufficient data for recommendations',
                required=3,
                actual=len(ts_monthly)
            )
        
        # Generate recommendations
        engine = RecommendationEngine(ts_monthly)
        recommendations = engine.generate_comprehensive_recommendations()
        
        app_logger.info(f"Generated {len(recommendations)} recommendations")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except InsufficientDataError as e:
        app_logger.warning(f"Insufficient data for recommendations: {e.message}")
        raise
    except Exception as e:
        app_logger.error(f"Error generating recommendations: {str(e)}")
        raise DataLoadError(f"Failed to generate recommendations: {str(e)}")


@app.route('/api/recommendations/generate')
@login_required
@handle_errors
def generate_recommendations():
    """Generate fresh recommendations with error recovery."""
    validate_data_available(df)
    
    try:
        # Prepare monthly time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        if len(ts_monthly) < 3:
            raise InsufficientDataError(
                'Need at least 3 months of data for recommendations',
                required=3,
                actual=len(ts_monthly)
            )
        
        # Train Random Forest model for forecast context
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        forecast_data = None
        
        try:
            forecast_result = random_forest_forecast(ts_train, n_months=12)
            forecast_data = {'forecast': forecast_result}
        except Exception as forecast_error:
            app_logger.warning(f"Forecast computation failed for recommendations: {str(forecast_error)}")
            # Continue without forecast context
        
        # Generate recommendations
        engine = RecommendationEngine(ts_monthly, forecast_data)
        recommendations = engine.generate_comprehensive_recommendations()
        
        app_logger.info(f"Generated recommendations with {len(recommendations)} items")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        })
        
    except InsufficientDataError as e:
        app_logger.warning(f"Recommendation generation failed: {e.message}")
        raise
    except Exception as e:
        app_logger.error(f"Error generating recommendations: {str(e)}")
        raise DataLoadError(f"Failed to generate recommendations: {str(e)}")


@app.route('/api/analysis')
@handle_errors
def unified_analysis_api():
    """Unified API endpoint that returns complete analysis data."""
    validate_data_available(df)
    
    try:
        # Basic stats
        total_records = len(df)
        total_sales = float(df['Weekly_Sales'].sum())
        mean_sales = float(df['Weekly_Sales'].mean())
        max_sales = float(df['Weekly_Sales'].max())
        min_sales = float(df['Weekly_Sales'].min())
        std_sales = float(df['Weekly_Sales'].std())
        
        # Monthly stats - aggregate by Date
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        
        # Group by month and aggregate
        df_copy['YearMonth'] = df_copy['Date'].dt.strftime('%B %Y')
        monthly_data = df_copy.groupby('YearMonth')['Weekly_Sales'].agg(['mean', 'max', 'min', 'count'])
        
        monthly_stats = {}
        month_list = []
        for month_key, row in monthly_data.iterrows():
            monthly_stats[month_key] = {
                'avg': float(row['mean']),
                'max': float(row['max']),
                'min': float(row['min']),
                'count': int(row['count'])
            }
            month_list.append(month_key)
        
        # Store stats  
        store_stats = {}
        for store_id in sorted(df['Store'].unique()):
            store_data = df[df['Store'] == store_id]['Weekly_Sales']
            store_stats[str(int(store_id))] = {
                'avg': float(store_data.mean()),
                'max': float(store_data.max()),
                'min': float(store_data.min()),
                'sum': float(store_data.sum())
            }
        
        # Trends with error handling
        trends = []
        for i, month in enumerate(month_list):
            try:
                avg = monthly_stats[month]['avg']
                if i == 0:
                    direction = '→'
                    change_pct = 0
                else:
                    prev_avg = monthly_stats[month_list[i-1]]['avg']
                    change_pct = ((avg - prev_avg) / prev_avg * 100) if prev_avg != 0 else 0
                    direction = '↑' if change_pct > 2 else '↓' if change_pct < -2 else '→'
                
                trends.append({
                    'period': month,
                    'avg_sales': avg,
                    'direction': direction,
                    'change_pct': round(change_pct, 2)
                })
            except Exception as trend_error:
                app_logger.warning(f"Error calculating trend for {month}: {str(trend_error)}")
                continue
        
        response_data = {
            'total_records': total_records,
            'total_sales': total_sales,
            'mean_sales': mean_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'std_sales': std_sales,
            'monthly_stats': monthly_stats,
            'store_stats': store_stats,
            'trends': trends
        }
        
        app_logger.info(f"Analysis API returning data for {total_records} records from {len(store_stats)} stores")
        return jsonify(response_data)

    except Exception as e:
        app_logger.error(f"Error in unified analysis API: {str(e)}", exc_info=True)
        raise DataLoadError(f"Failed to generate analysis: {str(e)}")


@app.route('/predictions')
@login_required
def predictions():
    """Display ML predictions page with ALL automatic analysis results"""
    user_id = session.get('user_id', 'unknown')
    model_trained = session.get('model_trained', False)
    
    # ✅ Debug logging
    app_logger.info(f"Predictions route: model_trained={model_trained}, user={user_id}")

    # ✅ Check if model is trained  
    if not model_trained:
        app_logger.warning(f"User {user_id} tried to view predictions but model_trained=False in session")
        flash('📊 Please upload a dataset first to see predictions.', 'warning')
        return redirect(url_for('upload'))
    
    # Regenerate predictions from saved model and data
    try:
        # Load user's data
        df, _ = load_data(prefer_uploaded=True)
        if df is None or df.empty:
            app_logger.error(f"User {user_id} has no data available for predictions")
            flash('No data found. Please upload a dataset.', 'error')
            return redirect(url_for('upload'))
        
        # Generate predictions fresh
        ml_predictions, _, _, monthly_predictions = generate_future_predictions(df)
        
        if not ml_predictions:
            app_logger.error(f"Failed to generate predictions for user {user_id}")
            flash('Failed to generate predictions. Please try uploading again.', 'error')
            return redirect(url_for('upload'))
        
        # Get other data from session
        training_results = session.get('training_results', {})
        insights = session.get('insights', {})
        eda_results = session.get('eda_results', {})
        seasonal_patterns = session.get('seasonal_patterns', {})
        
        # Get peak/low periods and convert lists to single values
        peak_periods_data = session.get('peak_periods', {})
        peak_periods_display = {
            'peak_period': peak_periods_data.get('peak_periods', ['N/A'])[0] if peak_periods_data.get('peak_periods') else 'N/A',
            'peak_sales': peak_periods_data.get('peak_sales', [0])[0] if peak_periods_data.get('peak_sales') else 0,
            'low_period': peak_periods_data.get('low_periods', ['N/A'])[-1] if peak_periods_data.get('low_periods') else 'N/A',
            'low_sales': peak_periods_data.get('low_sales', [0])[-1] if peak_periods_data.get('low_sales') else 0,
        }
        
        monthly_trends = session.get('monthly_trends', {})
        store_analysis = session.get('store_analysis', {})
        
        app_logger.info(f"Predictions generated successfully for user {user_id}: {len(ml_predictions)} predictions")
        
        return render_template(
            'predictions.html',
            ml_predictions=ml_predictions,
            monthly_predictions=monthly_predictions,
            training_results=training_results,
            insights=insights,
            feature_importance={},
            model_trained=True,
            data_uploaded=True,
            # ⚡ Automatic Analysis Results ⚡
            eda_results=eda_results,
            seasonal_patterns=seasonal_patterns,
            peak=peak_periods_display,
            monthly_trends=monthly_trends,
            store_analysis=store_analysis
        )
        
    except Exception as e:
        app_logger.error(f"Error loading predictions for user {user_id}: {str(e)}", exc_info=True)
        flash(f'Error loading predictions: {str(e)}', 'error')
        return redirect(url_for('upload'))


# =====================================================
# SALES PREDICTION ROUTES (Random Forest)
# =====================================================

@app.route('/sales-predictor')
@login_required
def sales_predictor():
    """Redirect to predictions page (no separate ML predictor needed)"""
    return redirect(url_for('predictions'))


@app.route('/complete-analysis-page')
@login_required
def complete_analysis_page():
    """Display complete analysis page - one step analysis"""
    return render_template('complete_analysis.html')


@app.route('/api/train-sales-model', methods=['POST'])
@login_required
@handle_errors
def train_sales_model():
    """Train Random Forest model on uploaded CSV data or stored user data"""
    try:
        user_id = session.get('user_id')
        
        # Check if file is uploaded
        if 'file' not in request.files or request.files['file'].filename == '':
            # No file uploaded, use stored user data
            app_logger.info(f"No file uploaded, using stored data for user {user_id}")
            
            # Get stored user data
            df = get_user_sales_data(user_id)
            if df.empty:
                raise FileUploadError("No stored data found. Please upload data first or provide a training file.")
            
            # Convert stored data to CSV format for training
            import io
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Save to temp file
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"stored_data_{user_id}.csv")
            with open(temp_path, 'w') as f:
                f.write(csv_content)
        else:
            # File uploaded, use it
            file = request.files['file']
            if not allowed_file(file.filename):
                raise FileUploadError("Invalid file type. Please upload a CSV file.")

            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{session['user_id']}_{filename}")
            file.save(temp_path)

        # Train model
        model_save_path = os.path.join('models', f"sales_predictor_{session['user_id']}.joblib")
        os.makedirs('models', exist_ok=True)

        result = train_sales_model_from_csv(temp_path, model_save_path)

        # Clean up temp file
        os.remove(temp_path)

        if not result['success']:
            raise ModelTrainingError(f"Model training failed: {result['error']}")

        # Store training results in session for display
        session['training_results'] = convert_to_json_serializable(result['results'])
        session['model_trained'] = True

        app_logger.info(f"Random Forest model trained successfully for user {session['user_id']}")

        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'results': result['results']
        })

    except Exception as e:
        app_logger.error(f"Error training sales model: {str(e)}")
        raise


@app.route('/api/predict-sales', methods=['POST'])
@login_required
@handle_errors
def predict_sales():
    """Make predictions using trained Random Forest model"""
    try:
        user_id = session.get('user_id')
        
        # Check if model is trained
        if not session.get('model_trained', False):
            raise ModelTrainingError("No trained model found. Please train a model first.")

        model_path = os.path.join('models', f"sales_predictor_{session['user_id']}.joblib")
        if not os.path.exists(model_path):
            raise ModelTrainingError("Model file not found. Please train a model first.")

        # Get prediction data
        if 'file' not in request.files or request.files['file'].filename == '':
            # No file uploaded, use stored user data for prediction
            app_logger.info(f"No prediction file uploaded, using stored data for user {user_id}")
            
            # Get stored user data
            df = get_user_sales_data(user_id)
            if df.empty:
                raise FileUploadError("No stored data found. Please upload data first or provide a prediction file.")
            
            # Convert stored data to CSV format for prediction
            import io
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Save to temp file
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pred_stored_data_{user_id}.csv")
            with open(temp_path, 'w') as f:
                f.write(csv_content)
        else:
            # File uploaded, use it
            file = request.files['file']
            if not allowed_file(file.filename):
                raise FileUploadError("Invalid file type. Please upload a CSV file.")

            # Save prediction file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pred_{session['user_id']}_{filename}")
            file.save(temp_path)

        # Make predictions
        result = predict_sales_from_csv(model_path, temp_path)

        # Clean up temp file
        os.remove(temp_path)

        if not result['success']:
            raise ModelTrainingError(f"Prediction failed: {result['error']}")

        app_logger.info(f"Sales predictions generated for user {session['user_id']}: {result['n_predictions']} predictions")

        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'n_predictions': result['n_predictions']
        })

    except Exception as e:
        app_logger.error(f"Error making sales predictions: {str(e)}")
        raise


@app.route('/api/sales-model-status')
@login_required
def sales_model_status():
    """Check if user has a trained sales prediction model"""
    model_path = os.path.join('models', f"sales_predictor_{session['user_id']}.joblib")
    has_model = os.path.exists(model_path)

    training_results = session.get('training_results', None)

    return jsonify({
        'has_model': has_model,
        'training_results': training_results,
        'model_trained': session.get('model_trained', False)
    })


@app.route('/api/cached-predictions')
@login_required
def cached_predictions():
    """Get cached predictions from the last trained model"""
    if not session.get('model_trained', False):
        return jsonify({
            'success': False,
            'error': 'No trained model available. Please upload data first.',
            'model_trained': False
        }), 400
    
    ml_predictions = session.get('future_predictions', [])
    training_results = session.get('training_results', {})
    insights = session.get('insights', {})
    
    return jsonify({
        'success': True,
        'model_trained': True,
        'predictions': ml_predictions[:30] if ml_predictions else [],  # First 30 days
        'total_predictions': len(ml_predictions),
        'training_metrics': training_results,
        'insights': insights
    })


@app.route('/api/enhanced-predictions')
@login_required
def api_enhanced_predictions():
    """Get all enhanced predictions with detailed insights (for API usage)"""
    if not session.get('model_trained', False):
        return jsonify({
            'success': False,
            'error': 'No trained model available. Please upload data first.',
            'model_trained': False,
            'data': {
                'predictions': [],
                'summary': {},
                'training_metrics': {}
            }
        }), 400
    
    ml_predictions = session.get('future_predictions', [])
    training_results = session.get('training_results', {})
    insights = session.get('insights', {})
    
    # Calculate summary statistics from predictions
    if ml_predictions:
        predicted_sales_values = [p.get('predicted_sales', 0) for p in ml_predictions]
        summary = {
            'total_predictions': len(ml_predictions),
            'average_sales': float(np.mean(predicted_sales_values)) if predicted_sales_values else 0,
            'max_sales': float(np.max(predicted_sales_values)) if predicted_sales_values else 0,
            'min_sales': float(np.min(predicted_sales_values)) if predicted_sales_values else 0,
            'std_sales': float(np.std(predicted_sales_values)) if predicted_sales_values else 0,
            'forecast_period': '365 days',
            'high_risk_count': sum(1 for p in ml_predictions if p.get('risk_color') == 'danger'),
            'medium_risk_count': sum(1 for p in ml_predictions if p.get('risk_color') == 'warning'),
            'low_risk_count': sum(1 for p in ml_predictions if p.get('risk_color') == 'success'),
        }
    else:
        summary = {
            'total_predictions': 0,
            'average_sales': 0,
            'max_sales': 0,
            'min_sales': 0,
            'std_sales': 0,
            'forecast_period': 'N/A'
        }
    
    return jsonify({
        'success': True,
        'model_trained': True,
        'data': {
            'predictions': ml_predictions,
            'summary': summary,
            'training_metrics': training_results,
            'insights': insights
        }
    })


@app.route('/api/download-predictions-csv')
@login_required
def download_predictions_csv():
    """Download predictions as CSV file"""
    user_id = session.get('user_id', 'unknown')
    
    if not session.get('model_trained', False):
        return jsonify({
            'success': False,
            'error': 'No trained model available.'
        }), 400
    
    try:
        # Load user's data and generate predictions on-demand
        df, _ = load_data(prefer_uploaded=True)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available.'
            }), 400
        
        # Generate predictions fresh
        ml_predictions, _, _, monthly_predictions = generate_future_predictions(df)
        
        if not ml_predictions:
            return jsonify({
                'success': False,
                'error': 'No predictions available.'
            }), 400
        
        # Create DataFrame from predictions
        predictions_list = []
        for pred in ml_predictions:
            predictions_list.append({
                'Date': pred.get('date'),
                'Day': pred.get('day_name'),
                'Week': pred.get('week'),
                'Month': pred.get('month'),
                'Predicted_Sales': pred.get('predicted_sales'),
                'vs_Average_%': pred.get('percent_vs_average'),
                'vs_Monthly_%': pred.get('percent_vs_monthly'),
                'Trend': pred.get('trend_direction'),
                'Trend_%': pred.get('trend_percentage'),
                'Performance': pred.get('performance'),
                'Seasonal_Note': pred.get('seasonal_note'),
                'Risk_Level': pred.get('risk_level'),
                'Confidence_Range': f"{pred.get('confidence_lower'):.0f} - {pred.get('confidence_upper'):.0f}",
                'Store': pred.get('store_name'),
            })
        
        df_csv = pd.DataFrame(predictions_list)
        
        # Create CSV string
        csv_string = df_csv.to_csv(index=False)
        
        response = make_response(csv_string)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=sales_predictions.csv'
        return response
        
    except Exception as e:
        app_logger.error(f"Error downloading predictions CSV for user {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to generate CSV: {str(e)}'
        }), 500


@app.route('/api/download-predictions-excel')
@login_required
def download_predictions_excel():
    """Download predictions as Excel file with formatting"""
    user_id = session.get('user_id', 'unknown')
    
    if not session.get('model_trained', False):
        return jsonify({'success': False, 'error': 'No trained model available.'}), 400
    
    try:
        # Load user's data and generate predictions on-demand
        df, _ = load_data(prefer_uploaded=True)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available.'
            }), 400
        
        # Generate predictions fresh
        ml_predictions, _, _, monthly_predictions = generate_future_predictions(df)
        
        if not ml_predictions:
            return jsonify({'success': False, 'error': 'No predictions available.'}), 400
        
        # Create workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Predictions"
        
        # Define styles
        header_fill = PatternFill(start_color="667eea", end_color="667eea", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=12)
        summary_fill = PatternFill(start_color="f0f0f0", end_color="f0f0f0", fill_type="solid")
        summary_font = Font(bold=True, size=11)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        center_alignment = Alignment(horizontal='center', vertical='center')
        
        # Add summary section
        ws['A1'] = "SALES PREDICTIONS REPORT"
        ws['A1'].font = Font(bold=True, size=14, color="667eea")
        ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A3'] = f"Total Predictions: {len(ml_predictions)}"
        
        # Add headers
        headers = ['Date', 'Day', 'Week', 'Month', 'Predicted Sales', 'vs Average %', 'Trend', 
                   'Performance', 'Seasonal Note', 'Risk Level', 'Confidence Range', 'Store']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=5, column=col)
            cell.value = header
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_alignment
            cell.border = border
        
        # Add prediction data
        for row, pred in enumerate(ml_predictions, 6):
            ws.cell(row=row, column=1).value = pred.get('date')
            ws.cell(row=row, column=2).value = pred.get('day_name')
            ws.cell(row=row, column=3).value = pred.get('week')
            ws.cell(row=row, column=4).value = pred.get('month')
            ws.cell(row=row, column=5).value = float(pred.get('predicted_sales', 0))
            ws.cell(row=row, column=6).value = pred.get('percent_vs_average')
            ws.cell(row=row, column=7).value = pred.get('trend_direction')
            ws.cell(row=row, column=8).value = pred.get('performance')
            ws.cell(row=row, column=9).value = pred.get('seasonal_note')
            ws.cell(row=row, column=10).value = pred.get('risk_level')
            ws.cell(row=row, column=11).value = f"{pred.get('confidence_lower'):.0f} - {pred.get('confidence_upper'):.0f}"
            ws.cell(row=row, column=12).value = pred.get('store_name')
            
            for col in range(1, 13):
                ws.cell(row=row, column=col).border = border
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 12
        ws.column_dimensions['B'].width = 12
        ws.column_dimensions['C'].width = 10
        ws.column_dimensions['D'].width = 10
        ws.column_dimensions['E'].width = 18
        ws.column_dimensions['F'].width = 15
        ws.column_dimensions['G'].width = 12
        ws.column_dimensions['H'].width = 15
        ws.column_dimensions['I'].width = 16
        ws.column_dimensions['J'].width = 12
        ws.column_dimensions['K'].width = 18
        ws.column_dimensions['L'].width = 15
        
        # Save to bytes
        excel_bytes = BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        
        response = make_response(excel_bytes.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = 'attachment; filename=sales_predictions.xlsx'
        return response
        
    except Exception as e:
        app_logger.error(f"Error downloading predictions Excel: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to generate Excel: {str(e)}'}), 500


@app.route('/api/download-predictions-json')
@login_required
def download_predictions_json():
    """Download predictions as JSON file"""
    user_id = session.get('user_id', 'unknown')
    
    if not session.get('model_trained', False):
        return jsonify({'success': False, 'error': 'No trained model available.'}), 400
    
    try:
        # Load user's data and generate predictions on-demand
        df, _ = load_data(prefer_uploaded=True)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available.'
            }), 400
        
        # Generate predictions fresh
        ml_predictions, _, _, monthly_predictions = generate_future_predictions(df)
        
        if not ml_predictions:
            return jsonify({'success': False, 'error': 'No predictions available.'}), 400
        
        # Get other data from session
        training_results = session.get('training_results', {})
        insights = session.get('insights', {})
        
        # Prepare summary statistics
        predicted_sales = [float(p.get('predicted_sales', 0)) for p in ml_predictions]
        summary = {
            'report_date': datetime.now().isoformat(),
            'forecast_period_days': len(ml_predictions),
            'average_sales': float(np.mean(predicted_sales)) if predicted_sales else 0,
            'max_sales': float(np.max(predicted_sales)) if predicted_sales else 0,
            'min_sales': float(np.min(predicted_sales)) if predicted_sales else 0,
            'std_sales': float(np.std(predicted_sales)) if predicted_sales else 0,
            'total_forecasted_sales': float(np.sum(predicted_sales)) if predicted_sales else 0,
        }
        
        data = {
            'metadata': {
                'title': 'Sales Predictions Report',
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'summary': summary,
            'training_metrics': training_results,
            'insights': insights,
            'predictions': ml_predictions,
            'monthly_predictions': monthly_predictions
        }
        
        json_string = json.dumps(data, indent=2, default=str)
        response = make_response(json_string)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=sales_predictions.json'
        return response
        
    except Exception as e:
        app_logger.error(f"Error downloading predictions JSON for user {user_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to generate JSON: {str(e)}'}), 500


@app.route('/api/generate-pdf-report', methods=['POST'])
@login_required
def generate_pdf_report():
    """Generate a comprehensive PDF report of predictions"""
    user_id = session.get('user_id', 'unknown')
    
    if not session.get('model_trained', False):
        return jsonify({'success': False, 'error': 'No trained model available.'}), 400
    
    try:
        # Load user's data and generate predictions on-demand
        df, _ = load_data(prefer_uploaded=True)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available.'
            }), 400
        
        # Generate predictions fresh
        ml_predictions, _, _, monthly_predictions = generate_future_predictions(df)
        
        if not ml_predictions:
            return jsonify({'success': False, 'error': 'No predictions available.'}), 400
        
        # Get other data from session
        training_results = session.get('training_results', {})
        insights = session.get('insights', {})
        eda_results = session.get('eda_results', {})
        
        # Create PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1  # center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        story = []
        
        # Title
        story.append(Paragraph("📊 Sales Predictions Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        predicted_sales = [float(p.get('predicted_sales', 0)) for p in ml_predictions]
        summary_text = f"""
        This report provides a comprehensive analysis of sales predictions for the next 365 days using an advanced Random Forest machine learning model. 
        The analysis includes {len(ml_predictions)} daily predictions with confidence intervals and risk assessments.
        <br/><br/>
        <b>Key Metrics:</b><br/>
        • Total Predictions: {len(ml_predictions)} days<br/>
        • Average Daily Sales: ${float(np.mean(predicted_sales)):.2f}<br/>
        • Maximum Predicted Sales: ${float(np.max(predicted_sales)):.2f}<br/>
        • Minimum Predicted Sales: ${float(np.min(predicted_sales)):.2f}<br/>
        • Total Forecasted Sales (12 months): ${float(np.sum(predicted_sales)):.2f}<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Model Performance
        story.append(Paragraph("Model Performance Metrics", heading_style))
        perf_text = f"""
        <b>Random Forest Model Accuracy:</b><br/>
        • R² Score: {training_results.get('r2', 0):.4f}<br/>
        • Mean Absolute Error (MAE): ${training_results.get('mae', 0):.2f}<br/>
        • Root Mean Squared Error (RMSE): ${training_results.get('rmse', 0):.2f}<br/>
        • Training Samples: {training_results.get('n_samples', 0)}<br/>
        """
        story.append(Paragraph(perf_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key Insights
        story.append(Paragraph("Key Insights & Trends", heading_style))
        insights_text = f"""
        <b>Peak Month:</b> {insights.get('peak_month', 'N/A')}<br/>
        <b>Growth Trend:</b> {insights.get('trend', 'N/A')}<br/>
        <b>Volatility:</b> {'High' if eda_results.get('std_sales', 0) > 1000 else 'Medium' if eda_results.get('std_sales', 0) > 500 else 'Low'}<br/>
        """
        story.append(Paragraph(insights_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Predictions Table (first 30 days)
        story.append(Paragraph("30-Day Forecast Preview", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        table_data = [['Date', 'Sales', 'vs Avg', 'Trend', 'Risk', 'Confidence']]
        for pred in ml_predictions[:30]:
            table_data.append([
                pred.get('date', ''),
                f"${float(pred.get('predicted_sales', 0)):.0f}",
                pred.get('percent_vs_average', ''),
                pred.get('trend_direction', ''),
                pred.get('risk_level', ''),
                f"${float(pred.get('confidence_lower', 0)):.0f} - ${float(pred.get('confidence_upper', 0)):.0f}"
            ])
        
        table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", heading_style))
        recommendations = """
        <b>Action Items:</b><br/>
        1. <b>Inventory Management:</b> Prepare inventory levels based on predicted demand peaks<br/>
        2. <b>Marketing Focus:</b> Concentrate promotional efforts during predicted peak sales periods<br/>
        3. <b>Risk Mitigation:</b> Monitor high-risk prediction days closely for potential market disruptions<br/>
        4. <b>Resource Planning:</b> Allocate staff and resources based on forecasted sales volumes<br/>
        5. <b>Performance Tracking:</b> Compare actual sales with predictions to refine the model continuously<br/>
        """
        story.append(Paragraph(recommendations, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_text = """
        <i>This report was automatically generated by the Seasonal Sales Forecasting System. 
        For questions or updates, please contact the analytics team.</i>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=sales_predictions_report.pdf'
        return response
        
    except Exception as e:
        app_logger.error(f"Error generating PDF report for user {user_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to generate PDF: {str(e)}'}), 500


@app.route('/api/save-predictions-settings', methods=['POST'])
@login_required
def save_predictions_settings():
    """Save user preferences for predictions (confidence level, forecast horizon, etc.)"""
    try:
        data = request.get_json()
        
        # Store settings in session
        if 'confidenceLevel' in data:
            session['confidence_level'] = int(data['confidenceLevel'])
        if 'forecastHorizon' in data:
            session['forecast_horizon'] = int(data['forecastHorizon'])
        if 'showConfidenceRange' in data:
            session['show_confidence_range'] = data['showConfidenceRange']
        if 'showTrends' in data:
            session['show_trends'] = data['showTrends']
        if 'showSeasonality' in data:
            session['show_seasonality'] = data['showSeasonality']
        
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': 'Settings saved successfully',
            'settings': {
                'confidenceLevel': session.get('confidence_level', 95),
                'forecastHorizon': session.get('forecast_horizon', 365),
                'showConfidenceRange': session.get('show_confidence_range', True),
                'showTrends': session.get('show_trends', True),
                'showSeasonality': session.get('show_seasonality', True),
            }
        })
    except Exception as e:
        app_logger.error(f"Error saving predictions settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/load-predictions-settings')
@login_required
def load_predictions_settings():
    """Load user preferences for predictions"""
    try:
        return jsonify({
            'success': True,
            'settings': {
                'confidenceLevel': session.get('confidence_level', 95),
                'forecastHorizon': session.get('forecast_horizon', 365),
                'showConfidenceRange': session.get('show_confidence_range', True),
                'showTrends': session.get('show_trends', True),
                'showSeasonality': session.get('show_seasonality', True),
            }
        })
    except Exception as e:
        app_logger.error(f"Error loading predictions settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/send-predictions-email', methods=['POST'])
@login_required
def send_predictions_email():
    """Send predictions via email (requires email configuration)"""
    try:
        data = request.get_json()
        recipient_email = data.get('email', '')
        include_charts = data.get('include_charts', True)
        include_table = data.get('include_table', True)
        include_insights = data.get('include_insights', True)
        message = data.get('message', '')
        
        # Validate email
        if not recipient_email or '@' not in recipient_email:
            return jsonify({'success': False, 'error': 'Invalid email address'}), 400
        
        ml_predictions = session.get('future_predictions', [])
        if not ml_predictions:
            return jsonify({'success': False, 'error': 'No predictions to send'}), 400
        
        # Note: This is a stub implementation. 
        # For production, configure SMTP settings in app.config
        app_logger.info(f"Email to {recipient_email} would be sent with predictions report")
        
        return jsonify({
            'success': True,
            'message': f'Email would be sent to {recipient_email}. (Email functionality requires SMTP configuration)',
            'note': 'To enable email functionality, configure SMTP settings in app.config or environment variables'
        })
        
    except Exception as e:
        app_logger.error(f"Error sending email: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/create-share-link', methods=['POST'])
@login_required  
def create_share_link():
    """Create a shareable link for predictions"""
    try:
        ml_predictions = session.get('future_predictions', [])
        if not ml_predictions:
            return jsonify({'success': False, 'error': 'No predictions to share'}), 400
        
        # Generate unique share ID
        share_id = str(uuid.uuid4())
        
        # Store share data in session (in production, use database)
        if 'shared_predictions' not in session:
            session['shared_predictions'] = {}
        
        session['shared_predictions'][share_id] = {
            'predictions': ml_predictions,
            'created_at': datetime.now().isoformat(),
            'user_id': session.get('user_id'),
            'expires_at': (datetime.now() + timedelta(days=7)).isoformat()  # 7 day expiry
        }
        session.modified = True
        
        share_url = f"{request.base_url.rstrip('/')}/predictions-shared/{share_id}"
        
        return jsonify({
            'success': True,
            'share_url': share_url,
            'share_id': share_id,
            'expires_in_days': 7
        })
        
    except Exception as e:
        app_logger.error(f"Error creating share link: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predictions-shared/<share_id>')
def view_shared_predictions(share_id):
    """View shared predictions"""
    try:
        # Get all users' shared predictions (in production, use database)
        # For now, check if it's in any session - simplified version
        
        return render_template(
            'predictions_shared.html',
            share_id=share_id,
            message='Shared predictions view',
            shared_predictions=[]  # Would be populated from database
        )
    except Exception as e:
        app_logger.error(f"Error viewing shared predictions: {str(e)}")
        flash('Could not load shared predictions', 'error')
        return redirect(url_for('predictions'))


@app.route('/api/future-sales-prediction', methods=['POST'])
@login_required
@handle_errors
@require_json
def future_sales_prediction():
    """Predict future sales for given dates and products"""
    try:
        # Check if model is trained
        if not session.get('model_trained', False):
            raise ModelTrainingError("No trained model found. Please train a model first.")

        model_path = os.path.join('models', f"sales_predictor_{session['user_id']}.joblib")
        if not os.path.exists(model_path):
            raise ModelTrainingError("Model file not found. Please train a model first.")

        # Get request data
        data = request.get_json()
        future_dates = data.get('dates', [])
        product_info = data.get('product_info', {})

        if not future_dates:
            raise DataValidationError("No future dates provided for prediction")

        # Load model and make predictions
        predictor = SalesPredictor()
        predictor.load_model(model_path)

        predictions_df = predictor.predict_future_sales(future_dates, product_info)

        # Convert to list of dicts for JSON response
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                'date': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                'predicted_sales': float(row['predicted_sales'])
            })

        app_logger.info(f"Future sales predictions generated for user {session['user_id']}: {len(predictions)} predictions")

        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        app_logger.error(f"Error generating future sales predictions: {str(e)}")
        raise


@app.route('/api/complete-analysis', methods=['POST'])
@login_required
@handle_errors
def complete_analysis():
    """
    Complete end-to-end analysis:
    1. Upload and validate data
    2. Perform EDA and preprocessing
    3. Train Random Forest model
    4. Generate future sales predictions
    All in one request!
    """
    try:
        user_id = session.get('user_id')
        
        # Step 1: Validate file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            raise FileUploadError('No file provided')
        
        file = request.files['file']
        
        # Validate CSV file
        is_valid, validation_message = validate_csv_file(file)
        if not is_valid:
            raise FileUploadError(validation_message, filename=file.filename if file else 'unknown')
        
        # Save temporarily
        filename = secure_filename(file.filename)
        if not filename:
            raise FileUploadError('Invalid filename provided')
        
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'complete_temp_{user_id}_{filename}')
        file.save(temp_filepath)
        
        try:
            # Step 2: Validate and preprocess CSV
            is_valid, message, validated_df = validate_csv_structure(temp_filepath)
            if not is_valid:
                raise DataValidationError(f"CSV validation failed: {message}", field='file')
            
            if validated_df is None or validated_df.empty:
                raise DataValidationError("CSV file contains no valid data", field='file')
            
            # Preprocess data
            df_processed, preprocessing_stats = preprocess_data(validated_df)
            
            # Perform EDA
            eda_results = perform_eda(df_processed)
            
            # Step 3: Store data in database
            delete_user_sales_data(user_id)
            store_sales_data(user_id, df_processed)
            
            app_logger.info(f"Data analysis completed for user {user_id}: {len(df_processed)} records processed")
            
            # Step 4: Train model
            model_save_path = os.path.join('models', f"sales_predictor_{user_id}.joblib")
            os.makedirs('models', exist_ok=True)
            
            training_result = train_sales_model_from_csv(temp_filepath, model_save_path)
            
            if not training_result['success']:
                raise ModelTrainingError(f"Model training failed: {training_result['error']}")
            
            training_results = training_result['results']
            session['model_trained'] = True
            session['training_results'] = training_results
            
            app_logger.info(f"Model trained successfully for user {user_id}")
            
            # Step 5: Generate future predictions
            # Generate predictions for next 365 days
            predictor = training_result['predictor']
            
            # Create future dates (next 365 days)
            from datetime import datetime, timedelta
            start_date = datetime.now()
            future_dates = [start_date + timedelta(days=i) for i in range(365)]
            
            # Generate predictions using existing data pattern
            predictions_df = predictor.predict_future_sales(future_dates)
            
            # Convert predictions to list of dicts
            predictions_list = []
            if isinstance(predictions_df, pd.DataFrame):
                for _, row in predictions_df.iterrows():
                    predictions_list.append({
                        'date': row.get('date', row.get('Date', '')).isoformat() if hasattr(row.get('date', row.get('Date', '')), 'isoformat') else str(row.get('date', row.get('Date', ''))),
                        'predicted_sales': float(row.get('predicted_sales', row.get('Weekly_Sales', 0)))
                    })
            elif isinstance(predictions_df, (list, np.ndarray)):
                # If it's just values, create dates
                predictions_list = [
                    {
                        'date': (start_date + timedelta(days=i)).isoformat(),
                        'predicted_sales': float(pred)
                    }
                    for i, pred in enumerate(predictions_df[:365])
                ]
            
            app_logger.info(f"Future predictions generated for user {user_id}: {len(predictions_list)} predictions")
            
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            # Return comprehensive results
            return jsonify({
                'success': True,
                'message': 'Complete analysis finished! Data analyzed, model trained, and predictions generated.',
                'data_summary': {
                    'total_records': len(df_processed),
                    'date_range': {
                        'start': str(eda_results.get('min_date', 'N/A')),
                        'end': str(eda_results.get('max_date', 'N/A'))
                    },
                    'avg_sales': float(eda_results.get('avg_sales', 0)),
                    'total_sales': float(eda_results.get('total_sales', 0))
                },
                'preprocessing': preprocessing_stats,
                'model_training': {
                    'mse': training_results.get('mse'),
                    'rmse': training_results.get('rmse'),
                    'mae': training_results.get('mae'),
                    'r2_score': training_results.get('r2_score'),
                    'train_points': training_results.get('n_train_samples'),
                    'test_points': training_results.get('n_test_samples')
                },
                'future_predictions': predictions_list[:30]  # First 30 days
            })
        
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise
    
    except Exception as e:
        app_logger.error(f"Error in complete analysis: {str(e)}")
        raise


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 8000))  # Changed from 5000 to 8000 to avoid AirTunes conflict
    # debug mode disabled to prevent reloader from interfering when run in background
    app.run(host='127.0.0.1', port=port, debug=False)