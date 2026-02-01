from flask import Flask, render_template, request, redirect, url_for, session, g, flash, jsonify, make_response
import sqlite3
import os
import logging
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
from models.arima_model import fit_arima, forecast_n_months, generate_forecast_summary
from forecast_visualization import ForecastVisualizer, create_forecast_json_chart
from recommendation_engine import RecommendationEngine, create_recommendations_json

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
    """Check session validity before each request"""
    try:
        print(f"[DEBUG] Before request: {request.method} {request.path}")
        if 'user_id' in session:
            # Session exists - set as permanent and update activity
            session.permanent = True
            app.permanent_session_lifetime = app.config['PERMANENT_SESSION_LIFETIME']
            
            # Log activity
            logger.info(f"User {session.get('username')} - {request.method} {request.path}")
        print(f"[DEBUG] Before request completed")
    except Exception as e:
        print(f"[DEBUG] ERROR in before_request: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler for all exceptions"""
    print(f"[DEBUG] GLOBAL ERROR HANDLER caught: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'status': 'error', 'message': str(e)}), 500


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
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.permanent = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['login_time'] = datetime.now().isoformat()
            session['ip_address'] = request.remote_addr
            
            # Log successful login
            logger.info(f"✅ User '{username}' logged in successfully from {request.remote_addr}")
            
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        
        logger.warning(f"❌ Failed login attempt for username '{username}' from {request.remote_addr}")
        flash('Invalid username or password.', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """Secure logout handler - ends user session safely"""
    try:
        user_id = session.get('user_id')
        username = session.get('username')
        ip_address = request.remote_addr
        login_time = session.get('login_time', 'unknown')
        
        # Log logout event before clearing session
        logout_time = datetime.now().isoformat()
        session_duration = "calculated at runtime"
        logger.info(f"🔐 LOGOUT EVENT: User '{username}' (ID: {user_id}) | "
                   f"Login: {login_time} | Logout: {logout_time} | IP: {ip_address}")
        
        # Get current session data before clearing
        session_data = dict(session)
        
        # Clear all session data
        session.clear()
        
        flash('You have been logged out successfully.', 'success')
        response = make_response(redirect(url_for('login')))
        
        # Ensure cookies are cleared with proper settings
        response.set_cookie('session', '', 
                          expires=0,
                          secure=app.config.get('SESSION_COOKIE_SECURE', False),
                          httponly=app.config.get('SESSION_COOKIE_HTTPONLY', True),
                          samesite=app.config.get('SESSION_COOKIE_SAMESITE', 'Lax'))
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error during logout: {str(e)}")
        flash('An error occurred during logout. Please try again.', 'error')
        return redirect(url_for('dashboard'))


@app.route('/dashboard')
@login_required
def dashboard():
    stats = {}
    if df is not None:
        stats = get_data_stats(df)
        stats['data_source'] = f"Data loaded from {data_source.title()} Dataset"
    return render_template('dashboard.html', stats=stats, stores=store_summary.to_dict('records') if store_summary is not None else [], data_source=data_source, preprocessing_stats=preprocessing_stats)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    global df, store_summary, data_source, preprocessing_stats
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        # Check if file is in request
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))
        
        # Validate file type
        if not allowed_file(file.filename):
            flash('Only CSV files are allowed.', 'error')
            return redirect(url_for('upload'))
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save temporarily to validate
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        
        try:
            file.save(temp_filepath)
            
            # Validate CSV structure
            is_valid, message, validated_df = validate_csv_structure(temp_filepath)
            
            if not is_valid:
                flash(f'CSV validation failed: {message}', 'error')
                os.remove(temp_filepath)
                return redirect(url_for('upload'))
            
            # Save processed CSV
            upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_sales.csv')
            if process_and_save_upload(validated_df, upload_filepath):
                # Reload data from uploaded file
                df, data_source = load_data(prefer_uploaded=True)
                if df is not None:
                    # Apply preprocessing
                    df, preprocessing_stats = preprocess_data(df)
                    store_summary = get_store_summary(df)
                    
                    # Clear previous user data and store new data in database
                    delete_user_sales_data(user_id)
                    store_sales_data(user_id, df)
                    
                    flash(f'✅ File uploaded successfully! {message}. Data preprocessed, stored, and ready for analysis.', 'success')
                else:
                    flash(f'✅ File uploaded successfully! {message}. But could not reload data.', 'warning')
                
                # Clean up temp file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return redirect(url_for('upload'))
            else:
                flash('Failed to save the processed file.', 'error')
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return redirect(url_for('upload'))
                
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return redirect(url_for('upload'))
    
    return render_template('upload.html', data_source=data_source)


@app.route('/analysis')
@login_required
def analysis():
    if df is None:
        flash('Dataset not available.', 'error')
        return redirect(url_for('dashboard'))
    stores = df['Store'].unique().tolist()
    return render_template('analysis_enhanced.html', stores=stores)


@app.route('/api/eda-summary')
@login_required
def api_eda_summary():
    """Get complete EDA summary for the current dataset."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    try:
        eda_results = perform_eda(df)
        return jsonify(eda_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/monthly-trends')
@login_required
def api_monthly_trends():
    """Get monthly sales trends."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    try:
        trends = get_monthly_trends(df)
        return jsonify(trends)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/seasonal-patterns')
@login_required
def api_seasonal_patterns():
    """Get seasonal sales patterns."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    try:
        patterns = get_seasonal_patterns(df)
        return jsonify(patterns)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/peak-periods')
@login_required
def api_peak_periods():
    """Get peak and low-demand periods."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    try:
        periods = get_peak_and_low_periods(df)
        return jsonify(periods)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/store-performance')
@login_required
def api_store_performance():
    """Get detailed store performance analysis."""
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    
    try:
        analysis = get_store_analysis(df)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/store-data/<int:store_id>')
@login_required
def api_store_data(store_id):
    if df is None:
        return jsonify({'error': 'Dataset not available'}), 500
    store_data = get_store_data(df, store_id)
    if store_data.empty:
        return jsonify({'error': 'Store not found'}), 404
    return jsonify({
        'store_id': store_id,
        'records': len(store_data),
        'avg_sales': float(store_data['Weekly_Sales'].mean()),
        'dates': store_data['Date'].astype(str).tolist(),
        'sales': store_data['Weekly_Sales'].tolist()
    })



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
    try:
        # Prepare time series from dataset
        if df is None:
            return render_template('forecast.html', 
                                 error='Dataset not available',
                                 generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Prepare time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Check if visualization plots exist
        import pandas as pd
        historical_vs_forecast = os.path.exists('static/plots/historical_vs_forecast.png')
        confidence_intervals = os.path.exists('static/plots/confidence_intervals.png')
        forecast_statistics = os.path.exists('static/plots/forecast_statistics.png')
        scenario_comparison = os.path.exists('static/plots/scenario_comparison.png')
        
        # Initialize template variables
        template_vars = {
            'historical_vs_forecast': historical_vs_forecast,
            'confidence_intervals': confidence_intervals,
            'forecast_statistics': forecast_statistics,
            'scenario_comparison': scenario_comparison,
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'now': int(datetime.now().timestamp())  # Cache buster
        }
        
        # If visualizations exist, add stats
        if historical_vs_forecast or scenario_comparison:
            try:
                # Generate forecast for stats
                ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
                model_result = fit_arima(ts_train, order=(0, 0, 1))
                forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
                
                forecast_df = forecast_result.get('forecast_df')
                if forecast_df is not None:
                    template_vars['forecast_mean'] = float(forecast_df['forecast'].mean())
                    template_vars['forecast_min'] = float(forecast_df['forecast'].min())
                    template_vars['forecast_max'] = float(forecast_df['forecast'].max())
                    template_vars['forecast_periods'] = len(forecast_df)
                    template_vars['ci_avg_width'] = float((forecast_df['upper_ci'] - forecast_df['lower_ci']).mean())
                
                # Get scenario stats if available
                if scenario_comparison:
                    forecast_summary = generate_forecast_summary(ts_train, model_result, n_periods=12, include_scenarios=True)
                    baseline_df = forecast_summary.get('baseline', {}).get('forecast_df')
                    if baseline_df is not None:
                        template_vars['baseline_mean'] = float(baseline_df['forecast'].mean())
                    
                    scenarios = forecast_summary.get('scenarios', {})
                    if 'pessimistic' in scenarios:
                        pess_vals = scenarios['pessimistic'].get('forecast_values', [])
                        if pess_vals:
                            template_vars['pessimistic_mean'] = float(np.mean(pess_vals))
                    if 'optimistic' in scenarios:
                        opt_vals = scenarios['optimistic'].get('forecast_values', [])
                        if opt_vals:
                            template_vars['optimistic_mean'] = float(np.mean(opt_vals))
            except Exception as e:
                print(f"Warning: Could not generate forecast stats: {e}")
        
        return render_template('forecast.html', **template_vars)
        
    except Exception as e:
        print(f"Error in forecast route: {e}")
        return render_template('forecast.html', 
                             error=str(e),
                             generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 500


@app.route('/api/visualizations/generate')
@login_required
def generate_visualizations():
    """Generate all forecast visualizations"""
    try:
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not available'}), 500
        
        import pandas as pd
        import numpy as np
        
        # Prepare time series
        df_copy = df.copy()
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


@app.route('/api/forecast-data')
@login_required
def get_forecast_data():
    """Get forecast data as JSON for charting libraries"""
    try:
        if df is None:
            return jsonify({'success': False, 'error': 'Dataset not available'}), 500
        
        import pandas as pd
        import numpy as np
        
        # Prepare time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Split data
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        
        # Train and forecast
        model_result = fit_arima(ts_train, order=(0, 0, 1))
        forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
        
        # Create JSON chart data
        chart_data = create_forecast_json_chart(ts_train, forecast_result)
        
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"Error getting forecast data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/recommendation')
@login_required
def recommendation():
    """Display business recommendations page"""
    try:
        if df is None:
            return render_template('recommendation.html', error='Dataset not available'), 500
        
        import pandas as pd
        import numpy as np
        
        # Prepare monthly time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Generate demand analysis for template
        engine = RecommendationEngine(ts_monthly)
        analysis = engine.analyze_demand_patterns()
        summary = engine._generate_executive_summary()
        
        return render_template('recommendation.html',
                              analysis=analysis if analysis.get('status') == 'success' else None,
                              summary=summary)
        
    except Exception as e:
        print(f"Error in recommendation page: {e}")
        return render_template('recommendation.html', error=str(e)), 500


@app.route('/api/recommendations')
@login_required
def get_recommendations():
    """Get recommendations as JSON"""
    try:
        if df is None:
            return jsonify({
                'success': False,
                'message': 'Dataset not available'
            }), 500
        
        import pandas as pd
        import numpy as np
        
        # Prepare monthly time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Generate recommendations
        engine = RecommendationEngine(ts_monthly)
        recommendations = engine.generate_comprehensive_recommendations()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/recommendations/generate')
@login_required
def generate_recommendations():
    """Generate fresh recommendations"""
    try:
        if df is None:
            return jsonify({
                'success': False,
                'message': 'Dataset not available'
            }), 500
        
        import pandas as pd
        import numpy as np
        
        # Prepare monthly time series
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy = df_copy.sort_values('Date')
        ts_data = df_copy.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Train ARIMA model and get forecast for context
        ts_train = ts_monthly.iloc[:int(len(ts_monthly)*0.75)]
        
        try:
            from models.arima_model import fit_arima, forecast_n_months
            model_result = fit_arima(ts_train, order=(0, 0, 1))
            forecast_result = forecast_n_months(ts_train, model_result, n_months=12)
            forecast_data = {'forecast': forecast_result}
        except:
            forecast_data = None
        
        # Generate recommendations with forecast context
        engine = RecommendationEngine(ts_monthly, forecast_data)
        recommendations = engine.generate_comprehensive_recommendations()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)