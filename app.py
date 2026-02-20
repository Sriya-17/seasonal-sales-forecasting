
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
from models.arima_model import fit_arima, forecast_n_months, generate_forecast_summary, arima_forecast, sarima_forecast, evaluate_model
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
from io import StringIO

app = Flask(__name__)
app.config.from_object(Config)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Download forecast as CSV for user-uploaded data ---
@app.route('/api/forecast-data', methods=['GET'])
@login_required
def download_forecast_csv():
    user_id = session.get('user_id')
    from data_storage import get_user_sales_data
    user_df = get_user_sales_data(user_id)
    use_df = None
    if user_df is not None and not user_df.empty:
        user_df['Date'] = pd.to_datetime(user_df['date']) if 'date' in user_df.columns else pd.to_datetime(user_df['Date'])
        user_df['Weekly_Sales'] = user_df['sales'] if 'sales' in user_df.columns else user_df['Weekly_Sales']
        use_df = user_df.rename(columns={'Date': 'Date', 'Weekly_Sales': 'Weekly_Sales'})
    else:
        use_df = df
    # Only use user data if it has at least 3 months of data
    df_copy = use_df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy = df_copy.sort_values('Date')
    ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
    ts_monthly = ts_data.resample('MS').sum()
    if len(ts_monthly) < 3:
        return 'Insufficient data for forecast', 400
    ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
    model_result = fit_arima(ts_train, order=(0, 0, 1))
    forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
    forecast_df = forecast_result['forecast_df'].copy()
    forecast_df = forecast_df.reset_index()
    forecast_df.rename(columns={'index': 'date'}, inplace=True)
    csv_buffer = StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    from flask import Response
    return Response(
        csv_buffer.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment;filename=forecast_data.csv'
        }
    )


from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify, make_response
import sqlite3
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
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
from models.arima_model import fit_arima, forecast_n_months, generate_forecast_summary, arima_forecast, sarima_forecast, evaluate_model
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

app = Flask(__name__)
app.config.from_object(Config)

# Configure logging for session management
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize sales database
init_sales_db()

# Load dataset at startup (with fallback)
df = None
store_summary = None
data_source = None
preprocessing_stats = {}

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
        stats = {}
        if df is not None and not df.empty:
            stats = get_data_stats(df)
            stats['data_source'] = f"Data loaded from {data_source.title()} Dataset"
            user_info = session.get('user_id', 'guest')
            app_logger.info(f"Dashboard loaded for user {user_info} with {len(df)} records")
        else:
            user_info = session.get('user_id', 'guest')
            app_logger.warning(f"Dashboard accessed but no data available for user {user_info}")
        
        return render_template('dashboard_improved.html', 
                             stats=stats, 
                             stores=store_summary.to_dict('records') if store_summary is not None else [], 
                             data_source=data_source, 
                             preprocessing_stats=preprocessing_stats)
    except Exception as e:
        # don't send user away from dashboard; show them the page with an error message
        app_logger.error(f"Error loading dashboard: {str(e)}", exc_info=True)
        flash('Error loading dashboard. Please try uploading data again.', 'error')
        # render dashboard with empty stats so template still loads
        return render_template('dashboard_improved.html',
                             stats={},
                             stores=[],
                             data_source=data_source,
                             preprocessing_stats=preprocessing_stats)


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
                    
                    app_logger.info(f"User {user_id} uploaded file '{filename}' with {len(df)} records")
                    flash(f'✅ File uploaded successfully! {message}', 'success')
                    return redirect(url_for('forecast'))
                    
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
    
    return render_template('upload_improved.html', data_source=data_source)


@app.route('/analysis')
@login_required
def analysis():
    if df is None:
        flash('Dataset not available.', 'error')
        return redirect(url_for('dashboard'))
    stores = df['Store'].unique().tolist()
    return render_template('analysis_improved.html', stores=stores)


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


@app.route('/api/plot/sales-over-time')
@login_required
def plot_sales_over_time():
    """Get sales over time visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_sales_over_time_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Sales Over Time'})


@app.route('/api/plot/seasonality')
@login_required
def plot_seasonality():
    """Get seasonality visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_seasonality_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Monthly Seasonality Patterns'})


@app.route('/api/plot/seasonal-breakdown')
@login_required
def plot_seasonal_breakdown():
    """Get seasonal breakdown visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_seasonal_breakdown_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Seasonal Breakdown'})


@app.route('/api/plot/store-performance')
@login_required
def plot_store_perf():
    """Get store performance visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_store_performance_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Store Performance Analysis'})


@app.route('/api/plot/yearly-comparison')
@login_required
def plot_yearly_comp():
    """Get yearly comparison visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_yearly_comparison_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Year-over-Year Comparison'})


@app.route('/api/plot/distribution')
@login_required
def plot_dist():
    """Get sales distribution visualization."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    plot = create_distribution_plot(df)
    if plot is None:
        return jsonify({'error': 'Failed to generate plot'}), 500
    
    return jsonify({'plot': plot, 'title': 'Sales Distribution'})



@app.route('/forecast')
@login_required
def forecast():
    """Display forecast visualization page"""
    if df is None:
        flash('Dataset not available. Please upload a CSV file.', 'error')
        return redirect(url_for('upload'))

    import os
    from flask import current_app
    plot_dir = os.path.join(current_app.root_path, 'static', 'plots')
    def plot_exists(filename):
        return os.path.exists(os.path.join(plot_dir, filename))

    # Check for each expected plot
    historical_vs_forecast = plot_exists('historical_vs_forecast.png')
    confidence_intervals = plot_exists('confidence_intervals.png')
    forecast_statistics = plot_exists('forecast_statistics.png')
    scenario_comparison = plot_exists('scenario_comparison.png')

    # Optionally, you can add stats if you want to show them

    # Compute forecast statistics for template using user data if available
    forecast_mean = forecast_min = forecast_max = forecast_periods = None
    ci_avg_width = None
    model_used = rmse = mae = None
    baseline_mean = pessimistic_mean = optimistic_mean = None

    try:
        user_id = session.get('user_id')
        from data_storage import get_user_sales_data
        user_df = get_user_sales_data(user_id)
        use_df = None
        if user_df is not None and not user_df.empty:
            user_df['Date'] = pd.to_datetime(user_df['date']) if 'date' in user_df.columns else pd.to_datetime(user_df['Date'])
            user_df['Weekly_Sales'] = user_df['sales'] if 'sales' in user_df.columns else user_df['Weekly_Sales']
            use_df = user_df.rename(columns={'Date': 'Date', 'Weekly_Sales': 'Weekly_Sales'})
        else:
            use_df = df

        # Only use user data if it has at least 3 months of data
        df_copy = use_df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        ts_monthly = ts_data.resample('MS').sum()
        if len(ts_monthly) < 3:
            raise Exception('Insufficient data for forecasting')
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        model_result = fit_arima(ts_train, order=(0, 0, 1))
        forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
        forecast_df = forecast_result['forecast_df']
        forecast_mean = float(forecast_df['forecast'].mean())
        forecast_min = float(forecast_df['forecast'].min())
        forecast_max = float(forecast_df['forecast'].max())
        forecast_periods = len(forecast_df)
        ci_avg_width = float((forecast_df['upper_ci'] - forecast_df['lower_ci']).mean())
        model_used = 'ARIMA(0,0,1)'
        # Evaluate model on train
        eval_metrics = evaluate_model(ts_train, model_result)
        rmse = float(eval_metrics['rmse'])
        mae = float(eval_metrics['mae'])
        # Scenario means
        baseline_mean = forecast_mean
        pessimistic_mean = float(forecast_df['forecast'].mean() * 0.95)
        optimistic_mean = float(forecast_df['forecast'].mean() * 1.05)
    except Exception as e:
        # If any error, leave stats as None
        pass

    return render_template(
        'forecast_improved.html',
        generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        historical_vs_forecast=historical_vs_forecast,
        confidence_intervals=confidence_intervals,
        forecast_statistics=forecast_statistics,
        scenario_comparison=scenario_comparison,
        forecast_mean=forecast_mean,
        forecast_min=forecast_min,
        forecast_max=forecast_max,
        forecast_periods=forecast_periods,
        ci_avg_width=ci_avg_width,
        model_used=model_used,
        rmse=rmse,
        mae=mae,
        baseline_mean=baseline_mean,
        pessimistic_mean=pessimistic_mean,
        optimistic_mean=optimistic_mean
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
        
        # Split data for training
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        
        # Train ARIMA model
        model_result = fit_arima(ts_train, order=(0, 0, 1))
        
        # Generate forecast
        forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
        
        # Generate forecast with scenarios
        forecast_summary = generate_forecast_summary(ts_train, model_result, n_periods=12, include_scenarios=True)
        
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


@app.route('/api/forecast')
@handle_errors
def api_forecast():
    """Get 12-month sales forecast with fallback strategies."""
    validate_data_available(df)
    
    try:
        # Prepare time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        
        # Aggregate daily sales
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly (sum of sales per month)
        ts_monthly = ts_data.resample('MS').sum()
        
        # Ensure sufficient data
        if len(ts_monthly) < 3:
            raise InsufficientDataError(
                'Insufficient data for forecasting (need at least 3 months)',
                required=3,
                actual=len(ts_monthly)
            )
        
        # Split data - use 75% for training
        train_size = max(3, int(len(ts_monthly) * 0.75))
        ts_train = ts_monthly.iloc[:train_size]
        
        # Try ARIMA model
        try:
            model_result = fit_arima(ts_train, order=(0, 0, 1))
            forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
            app_logger.info(f"ARIMA forecast generated successfully for {len(ts_monthly)} months of data")
        except Exception as model_error:
            app_logger.warning(f"ARIMA model failed, using fallback: {str(model_error)}")
            # Use fallback recovery
            recovery_result = ErrorRecoveryStrategy.on_model_training_failure(model_error, 'simple_average')
            
            if not recovery_result['success']:
                raise ModelTrainingError(
                    "Forecast model training failed and fallback unavailable",
                    model_type='ARIMA',
                    details={'original_error': str(model_error)}
                )
            
            # Simple average fallback
            forecast_data = []
            avg_sales = float(ts_train.mean())
            std_sales = float(ts_train.std())
            
            for i in range(12):
                forecast_data.append({
                    'month': i + 1,
                    'forecast': avg_sales,
                    'conf_low': max(0, avg_sales - (2 * std_sales)),
                    'conf_high': avg_sales + (2 * std_sales)
                })
            
            return jsonify({
                'forecast': forecast_data,
                'mean_sales': float(df['Weekly_Sales'].mean()),
                'model_summary': 'Fallback Model (Simple Average)',
                'success': True,
                'note': 'Using fallback model due to insufficient data for ARIMA'
            })
        
        # Format forecast data
        forecast_data = []
        forecast_df = forecast_result['forecast_df']
        for i in range(len(forecast_df)):
            row = forecast_df.iloc[i]
            forecast_data.append({
                'month': i + 1,
                'forecast': float(max(0, row['forecast'])),
                'conf_low': float(max(0, row['lower_ci'])),
                'conf_high': float(max(0, row['upper_ci']))
            })
        
        return jsonify({
            'forecast': forecast_data,
            'mean_sales': float(df['Weekly_Sales'].mean()),
            'model_summary': 'ARIMA(0,0,1)',
            'success': True
        })
        
    except (InsufficientDataError, ModelTrainingError) as e:
        app_logger.error(f"Forecast error: {e.message}")
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error in forecast API: {str(e)}", exc_info=True)
        raise DataLoadError(
            f"Failed to generate forecast: {str(e)}",
            details={'original_error': str(e)}
        )


@app.route('/api/forecast-data')
@login_required
@handle_errors
def get_forecast_data():
    """Get forecast data as JSON for charting libraries."""
    validate_data_available(df)
    
    try:
        # Prepare time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        if len(ts_monthly) < 3:
            raise InsufficientDataError(
                'Insufficient data for forecast visualization',
                required=3,
                actual=len(ts_monthly)
            )
        
        # Split data
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        
        # Train and forecast
        model_result = fit_arima(ts_train, order=(0, 0, 1))
        forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
        
        # Create JSON chart data
        chart_data = create_forecast_json_chart(ts_train, forecast_result)
        
        return jsonify(chart_data)
        
    except InsufficientDataError as e:
        app_logger.warning(f"Insufficient data for forecast: {e.message}")
        raise
    except Exception as e:
        app_logger.error(f"Error generating forecast data: {str(e)}")
        raise DataLoadError(
            f"Failed to generate forecast data: {str(e)}"
        )



@app.route('/recommendation')
@login_required
def recommendation():
    """Display business recommendations page."""
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

        app_logger.info(f"Recommendations generated for user {user_id}")

        return render_template(
            'recommendation_improved.html',
            quick_wins=quick_wins,
            recommendations=recommendations,
            stock_recommendations=stock_recommendations,
            festival_recommendations=festival_recommendations,
            insights=insights
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
        
        # Train ARIMA model for forecast context
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        forecast_data = None
        
        try:
            model_result = fit_arima(ts_train, order=(0, 0, 1))
            forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
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

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)