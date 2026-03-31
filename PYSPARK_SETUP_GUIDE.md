# PySpark Integration Setup Guide
## Complete Steps to Add Apache Spark to Your Flask Application

---

## Table of Contents
1. [Prerequisites & Installation](#prerequisites--installation)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Files](#configuration-files)
4. [Integration with Flask](#integration-with-flask)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Migration from Pandas to PySpark](#migration-from-pandas-to-pyspark)

---

## Prerequisites & Installation

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2 CPU cores
- 2GB disk space for Spark

**Recommended for Big Data:**
- Python 3.9+
- 8GB+ RAM
- 4+ CPU cores
- SSD for better performance

### Step 1: Install Java (Required by Spark)

Spark requires Java 8 or higher. Check if Java is installed:

```bash
java -version
```

**If not installed:**

```bash
# macOS
brew install openjdk@11

# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# Windows (via chocolatey)
choco install openjdk11
```

### Step 2: Install PySpark

Add to your `requirements.txt`:

```txt
pyspark==3.4.1
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
```

Then install:

```bash
pip install -r requirements.txt
```

Or direct install:

```bash
pip install pyspark==3.4.1
```

### Step 3: Verify Installation

```python
# Test Spark installation
python -c "from pyspark.sql import SparkSession; print('✅ PySpark installed successfully')"
```

---

## Architecture Overview

### Component Diagram

```
Flask App (app.py)
    ↓
CSV Upload Handler
    ↓
PySparkDataProcessor (pyspark_data_processor.py)
    ├─ Load CSV (Spark)
    ├─ Clean Data
    ├─ Aggregate by Date
    ├─ Engineer Features (temporal)
    └─ Convert to Pandas (if needed)
    ↓
ML Model Layer (pyspark_ml_model.py)
    ├─ Option 1: Spark MLlib (distributed)
    └─ Option 2: scikit-learn (in-memory)
    ↓
Results Storage
    ├─ Session storage
    ├─ Database
    └─ Visualization JSONs
    ↓
Template Rendering (Flask)
    └─ Display predictions, metrics, graphs
```

### Data Flow

```
User uploads CSV
    ↓ (10-100MB)
Spark reads distributed
    ↓ (lazy evaluation)
Data aggregation & cleaning
    ↓ (in parallel)
Feature engineering
    ↓ (7 temporal features)
Convert to Pandas (if < 4GB)
    ↓
Train RandomForest
    ↓
Calculate metrics (R², MAE, RMSE)
    ↓
Generate 365-day forecast
    ↓
Store in Flask session
    ↓
Render dashboard with results
```

---

## Configuration Files

### 1. `spark_config.py` (Already Created)

Manages Spark session lifecycle. Key functions:

```python
from spark_config import SparkConfig, initialize_spark

# Get Spark session (singleton pattern)
spark = SparkConfig.get_spark_session()

# Or use config presets
spark = initialize_spark('local_development')
```

**Available Configs:**
- `local_development`: Local mode, 2GB driver, 4GB executor (default)
- `local_heavy`: Local mode, 4GB driver, 8GB executor
- `yarn_cluster`: YARN cluster deployment
- `standalone_cluster`: Spark standalone cluster

### 2. Environment Variables (Optional)

Create `.env` file:

```bash
# Spark Configuration
SPARK_HOME=/usr/local/spark
SPARK_MASTER=local[*]
SPARK_DRIVER_MEMORY=2g
SPARK_EXECUTOR_MEMORY=4g

# Flask App
FLASK_ENV=development
FLASK_DEBUG=1
```

Load in Flask app:

```python
from dotenv import load_dotenv
import os

load_dotenv()
spark_master = os.getenv('SPARK_MASTER', 'local[*]')
```

---

## Integration with Flask

### Step 1: Update `app.py`

Add PySpark imports at the top:

```python
from pyspark_data_processor import PySparkDataProcessor
from pyspark_ml_model import HybridMLPipeline, SklearnMLModel
from spark_config import SparkConfig
```

### Step 2: Initialize Spark in App Factory

```python
def create_app():
    app = Flask(__name__)
    
    # Initialize Spark
    app.spark = SparkConfig.get_spark_session()
    
    # Initialize data processor
    app.spark_processor = PySparkDataProcessor(app.spark)
    
    # Initialize ML pipeline (use sklearn for better Flask integration)
    app.ml_pipeline = SklearnMLModel()
    
    return app

app = create_app()
```

### Step 3: Update Upload Route

Replace or update your `/upload` route:

```python
@app.route('/upload', methods=['POST'])
@require_auth
def upload_file():
    """Upload and process sales data using PySpark"""
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use CSV'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        
        # Process with PySpark
        logger.info(f"Processing {file_path} with PySpark")
        
        try:
            # Step 1: Load and process with Spark
            df_spark, df_pandas = app.spark_processor.full_pipeline(
                file_path,
                date_column='Date',
                sales_column='Sales',
                missing_strategy='drop',
                convert_to_pandas_flag=True  # Get pandas for sklearn
            )
            
            if df_pandas is None or df_pandas.empty:
                return jsonify({'error': 'Processed data is empty'}), 400
            
            # Step 2: Engineer features (already done in pipeline)
            # Verify features exist
            required_features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
            missing_features = [f for f in required_features if f not in df_pandas.columns]
            
            if missing_features:
                return jsonify({'error': f'Missing features: {missing_features}'}), 400
            
            # Step 3: Train model using scikit-learn (for Flask integration)
            logger.info(f"Training ML model on {len(df_pandas)} records")
            
            metrics = app.ml_pipeline.train(
                df_pandas,
                feature_columns=required_features,
                label_column='Sales'
            )
            
            # Step 4: Generate future predictions
            from models.random_forest_model import generate_future_dates
            
            last_date = pd.to_datetime(df_pandas['Date'].max())
            future_dates = generate_future_dates(365, last_date)
            
            # Merge features for future dates
            future_dates_with_features = future_dates.copy()
            for feat in required_features:
                if feat == 'year':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.year
                elif feat == 'month':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.month
                elif feat == 'day':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.day
                elif feat == 'week':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.isocalendar().week
                elif feat == 'day_of_week':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.dayofweek
                elif feat == 'quarter':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.quarter
                elif feat == 'day_of_year':
                    future_dates_with_features[feat] = future_dates_with_features['Date'].dt.dayofyear
            
            # Make predictions
            future_predictions_df = app.ml_pipeline.predict_future(future_dates_with_features)
            
            # Step 5: Store results in session
            session['upload_completed'] = True
            session['training_results'] = {
                'test_r2': metrics['test_r2'],
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'feature_importance': app.ml_pipeline.get_feature_importance(),
            }
            
            # Convert predictions to dictionary format
            predictions_list = []
            for _, row in future_predictions_df.iterrows():
                predictions_list.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'predicted_sales': float(row['Predicted_Sales'])
                })
            
            session['future_predictions'] = predictions_list
            
            # Save model
            model_path = 'models/pyspark_model.joblib'
            app.ml_pipeline.save_model(model_path)
            
            # Clean up temp file
            os.remove(file_path)
            
            logger.info("✅ Upload and training completed successfully")
            
            return jsonify({
                'message': 'File processed successfully',
                'metrics': metrics,
                'predictions_count': len(predictions_list)
            }), 200
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
```

### Step 4: Shutdown Spark on App Shutdown

```python
@app.teardown_appcontext
def shutdown_spark(exception=None):
    """Clean up Spark session on app shutdown"""
    SparkConfig.stop_spark_session()

# Or for development
if __name__ == '__main__':
    try:
        app.run(debug=True, port=8000)
    finally:
        SparkConfig.stop_spark_session()
```

---

## Usage Examples

### Example 1: Basic Data Processing

```python
from pyspark_data_processor import PySparkDataProcessor
from spark_config import SparkConfig

# Initialize
spark = SparkConfig.get_spark_session()
processor = PySparkDataProcessor(spark)

# Load CSV
df_spark = processor.load_csv('data/sales_data.csv')

# Clean and engineer
df_clean = processor.handle_missing_values(df_spark, strategy='drop')
df_features = processor.engineer_features(df_clean)

# Convert to Pandas
df_pandas = processor.convert_to_pandas(df_features)

print(df_pandas.head())
```

### Example 2: Complete Pipeline

```python
# One-line pipeline
df_spark, df_pandas = processor.full_pipeline(
    'data/large_sales.csv',
    date_column='Date',
    sales_column='Sales',
    missing_strategy='drop',
    convert_to_pandas_flag=True
)

print(f"Shape: {df_pandas.shape}")
print(f"Columns: {df_pandas.columns.tolist()}")
```

### Example 3: Model Training (scikit-learn)

```python
from pyspark_ml_model import SklearnMLModel

# Train
model = SklearnMLModel(n_estimators=100, max_depth=20)

features = ['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year']
metrics = model.train(df_pandas, features, label_column='Sales')

print(f"R² Score: {metrics['test_r2']:.4f}")
print(f"MAE: ${metrics['test_mae']:.2f}")
print(f"RMSE: ${metrics['test_rmse']:.2f}")

# Get feature importance
importance = model.get_feature_importance()
print(importance)

# Predict
predictions = model.predict_future(future_df)
print(predictions[['Date', 'Predicted_Sales']])

# Save model
model.save_model('models/pyspark_model.joblib')
```

### Example 4: Spark MLlib Training

```python
from pyspark_ml_model import PySparkMLModel

# This is for very large datasets that don't fit in memory
model = PySparkMLModel(num_trees=100, max_depth=20)

# Prepare features
df_prepared = model.prepare_features(
    df_spark,
    feature_columns=['year', 'month', 'day', 'week', 'day_of_week', 'quarter', 'day_of_year'],
    label_column='Sales'
)

# Train
metrics = model.train(df_prepared)

# Predict
predictions = model.predict(df_prepared)

# Save
model.save_model('models/spark_rf_model')
```

---

## Troubleshooting

### Issue 1: Java Not Found

```
Error: JAVA_HOME is not set
```

**Solution:**
```bash
# Find Java installation
which java

# Set JAVA_HOME (macOS)
export JAVA_HOME=$(/usr/libexec/java_home -v 11)

# Add to ~/.zshrc or ~/.bashrc for persistence
echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
```

### Issue 2: "illegal reflective access" Warning

This is normal in Python 3.9+ with older Spark versions. Ignore or upgrade Spark to 3.3+.

```
WARNING: An illegal reflective access operation has occurred...
```

**Solution:** Use PySpark 3.4.1+

### Issue 3: Out of Memory Error

```
java.lang.OutOfMemoryError: Java heap space
```

**Solution:** Increase memory in config:
```python
spark = initialize_spark('local_heavy')  # 4GB + 8GB
```

### Issue 4: Slow Data Processing

**Issue:** PySpark is slower than expected

**Solutions:**
- Increase executor memory: `SPARK_EXECUTOR_MEMORY=8g`
- Use more CPU cores: `master='local[*]'`
- Enable adaptive query execution (already done in config)
- For small files (< 1GB), use Pandas directly

### Issue 5: Column Not Found

```
AnalysisException: 'Date'
```

**Solution:** Check CSV structure:
```python
processor = PySparkDataProcessor(spark)
df = processor.load_csv('data.csv')
df.printSchema()  # See actual columns
```

---

## Performance Tuning

### 1. Data Size Thresholds

```python
import os

file_size_mb = os.path.getsize('data.csv') / (1024*1024)

if file_size_mb < 100:
    # Small file - use Pandas for speed
    use_spark = False
elif file_size_mb < 1000:
    # Medium file - use Spark with single-machine optimization
    use_spark = True
    master = 'local[*]'
else:
    # Large file - use distributed Spark
    use_spark = True
    master = 'yarn'  # or Spark cluster
```

### 2. Memory Allocation Strategy

```
Dataset Size          Driver Memory    Executor Memory    Partition Count
< 1GB                 1GB             2GB                 4
1GB - 10GB           2GB             4GB                 8
10GB - 100GB         4GB             8GB                 16
> 100GB              8GB+            16GB+               32+
```

### 3. Optimization Tips

```python
# 1. Use inferSchema=False for known schemas (faster)
df = spark.read.csv('data.csv', header=True, inferSchema=False)

# 2. Partition data for large files
df = spark.read.csv('data.csv', header=True).repartition(16)

# 3. Cache frequently accessed dataframes
df_cached = df.cache()
df_cached.count()  # Force caching

# 4. Use Parquet instead of CSV (faster for repeated reads)
df.write.mode('overwrite').parquet('data.parquet')
df_parquet = spark.read.parquet('data.parquet')

# 5. Drop unused columns early
df = df.select('Date', 'Sales')  # Only needed columns
```

---

## Migration from Pandas to PySpark

### Step 1: Identify Bottlenecks

Use `cProfile` to find slow parts:

```python
import cProfile

cProfile.run('your_function()')
```

### Step 2: PySpark vs Pandas Equivalents

| Pandas Operation | PySpark Equivalent |
|------------------|-------------------|
| `df.load_csv()` | `spark.read.csv()` |
| `df.dropna()` | `df.dropna()` |
| `df.groupby().sum()` | `df.groupBy().agg()` |
| `df['col'].mean()` | `df.agg(F.mean('col'))` |
| `df.merge()` | `df.join()` |
| `df.to_csv()` | `df.write.csv()` |

### Step 3: Minimal Changes Approach

**Option A: Keep Spark for data → Convert to Pandas for ML**

```python
# Use Spark for big data
df_spark = processor.load_csv('huge_file.csv')
df_spark = processor.engineer_features(df_spark)

# Convert to Pandas for faster ML
df_pandas = processor.convert_to_pandas(df_spark)

# Train with sklearn (fast + mature)
model = SklearnMLModel()
model.train(df_pandas, features)
```

**Option B: Full Spark pipeline** (for truly massive data)

```python
# Everything in Spark
model = PySparkMLModel()
df_prepared = model.prepare_features(df_spark, features)
metrics = model.train(df_prepared)
```

---

## Best Practices

### ✅ Do

```python
# 1. Reuse Spark session
spark = SparkConfig.get_spark_session()  # Singleton pattern

# 2. Use lazy evaluation
df = spark.read.csv('data.csv')  # No computation yet
df_filtered = df.filter(df.Sales > 100)  # Still no computation
result = df_filtered.count()  # NOW it computes

# 3. Convert to Pandas only when needed
df_pandas = df.toPandas()  # Only for sklearn models

# 4. Drop unused columns early
df = df.select('Date', 'Sales')  # Better than selecting later

# 5. Handle errors gracefully
try:
    df = processor.load_csv(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
```

### ❌ Don't

```python
# 1. Create multiple Spark sessions
spark1 = SparkSession.builder.appName('app1').getOrCreate()
spark2 = SparkSession.builder.appName('app2').getOrCreate()  # BAD

# 2. Convert large DataFrames to Pandas unnecessarily
df_pandas = huge_df.toPandas()  # If 100GB → OutOfMemory

# 3. Use DataFrame operations you don't understand
df.collect()  # Brings ALL data to driver (dangerous on big data)

# 4. Ignore partitioning
small_df.repartition(1)  # Then trying to process 100GB (slow)

# 5. Leave sessions open
# DON'T forget to call SparkConfig.stop_spark_session()
```

---

## Summary

| Feature | Pandas | PySpark | Hybrid (Recommended) |
|---------|--------|---------|----------------------|
| **Data Size** | < 4GB | > 4GB | Automatic selection |
| **Speed (small)** | ⚡⚡⚡ | ⚡ | Pandas (small) |
| **Speed (large)** | ❌ | ⚡⚡ | Spark (large) |
| **Learning Curve** | Easy | Medium | Easy for user |
| **Memory Usage** | High | Low | Optimized |
| **ML Integration** | Excellent | Good | Best of both |

**Recommendation:** Use the HybridMLPipeline with PySpark for data processing and sklearn for ML training - best of both worlds! 🚀

---

**Last Updated:** March 25, 2026
**Version:** 1.0 - Production Ready
