# Developer's Quick Reference - Single Upload Workflow

## 🎯 The Problem We Solved

❌ **Old:** Users uploaded dataset twice (Upload page + ML Predictor page)  
✅ **New:** Users upload dataset once, model trains automatically

---

## 📋 Quick Routes Reference

### User-Facing Routes

| Route | Method | Purpose | Login | Redirect |
|-------|--------|---------|-------|----------|
| `/` | GET | Dashboard | Required | - |
| `/upload` | GET/POST | Upload & auto-train | Required | → `/predictions` |
| `/predictions` | GET | View results | Required | → `/upload` if no model |
| `/sales-predictor` | GET | View cached predictions | Required | → `/upload` if no model |

### API Routes

| Endpoint | Method | Purpose | Returns | Auth |
|----------|--------|---------|---------|------|
| `/api/cached-predictions` | GET | Get cached predictions | JSON | Required |
| `/api/sales-model-status` | GET | Check if model trained | JSON | Required |
| `/api/eda-summary` | GET | Get data analysis | JSON | Required |

---

## 🔧 The Model Training Pipeline

```python
# Location: app.py, inside upload() function

# 1. User uploads file
file = request.files['file']

# 2. Validate
is_valid, msg, validated_df = validate_csv_structure(temp_filepath)

# 3. Preprocess
df_processed, stats = preprocess_data(validated_df)

# 4. 🎯 AUTOMATIC TRAINING (new!)
model_save_path = f'models/sales_predictor_{user_id}.joblib'
training_result = train_sales_model_from_csv(upload_filepath, model_save_path)

# 5. 🎯 AUTOMATIC PREDICTIONS (new!)
predictions, trained_model, insights = generate_future_predictions(df_processed)

# 6. Store in session
session['model_trained'] = True
session['future_predictions'] = predictions
session['training_results'] = training_result['results']

# 7. Redirect
return redirect(url_for('predictions'))
```

---

## 💾 Session Variables

```python
# After successful upload, session contains:

session['model_trained']         # Boolean: True = model ready
session['future_predictions']    # List[Dict]: 365 predictions
session['training_results']      # Dict: {r2_score, rmse, mae, ...}
session['trained_model']         # Dict: Model metadata
session['insights']              # Dict: Sales insights
session['model_path']            # String: Path to .joblib file
```

---

## 🔀 Control Flow

### **Upload Workflow**
```
POST /upload
    ↓
Validate CSV → Preprocess → Store DB → Train Model → Generate Predictions
    ↓
Store in session → Flash success message → Redirect to /predictions
    ↓
GET /predictions (GET request, no upload needed)
    ↓
Load session data → Render template with predictions
```

### **ML Predictor Access**
```
GET /sales-predictor
    ↓
Check: session['model_trained'] == True?
    ├─ YES → Load cached predictions → Render
    └─ NO  → Redirect to /upload with message
```

### **API Access**
```
GET /api/cached-predictions
    ↓
Check: session['model_trained'] == True?
    ├─ YES → Return JSON with predictions
    └─ NO  → Return 400 error JSON
```

---

## 🛠️ Key Functions Called During Upload

```python
# Validation
validate_csv_file(file)              # Check file type/size
validate_csv_structure(path)         # Check columns

# Preprocessing
preprocess_data(df)                  # Clean data
store_sales_data(user_id, df)        # Save to DB

# Model Training (NEW - Automatic)
train_sales_model_from_csv(path)     # Train Random Forest
generate_future_predictions(df)      # Generate 365 predictions

# Utilities
delete_user_sales_data(user_id)      # Clear old data
secure_filename(filename)            # Prevent attacks
get_store_summary(df)                # Get data summary
```

---

## 📊 Model Training Details

```python
# Location: models/sales_predictor.py → train_sales_model_from_csv()

# Input:
csv_path = 'data/uploaded_sales.csv'
model_save_path = 'models/sales_predictor_2.joblib'

# Process:
1. Load CSV
2. Preprocess: Extract date features (year, month, day, etc.)
3. Split: 80% train, 20% test
4. Train: RandomForestRegressor(n_estimators=100)
5. Evaluate: Calculate R², RMSE, MAE
6. Save: Pickle model to .joblib file
7. Return: {success: True, results: {metrics}, predictor: obj}

# Output:
{
    'success': True,
    'results': {
        'r2_score': 0.956,
        'rmse': 829.68,
        'mae': 444.68,
        'n_train_samples': 296,
        'n_test_samples': 75
    },
    'predictor': <SalesPredictor object>
}
```

---

## 🎨 Templates Used

### Route → Template Mapping

```
/upload       → upload_improved.html
                (Simple upload form, directs to /predictions after upload)

/predictions  → predictions.html
                (Display model results, shows metrics & predictions)

/sales-predictor → sales_predictor.html
                   (Same as /predictions, but auto-redirects if no model)
```

---

## 🔍 Common Developer Tasks

### Task 1: Check if User Has Trained Model

```python
@app.route('/my-route')
@login_required
def my_route():
    if session.get('model_trained', False):
        predictions = session['future_predictions']
        # Use predictions...
    else:
        flash('Please upload data first', 'warning')
        return redirect(url_for('upload'))
```

### Task 2: Get Predictions as JSON

```python
@app.route('/api/my-endpoint')
@login_required
def my_api():
    if not session.get('model_trained', False):
        return jsonify({'error': 'No model trained'}), 400
    
    predictions = session.get('future_predictions', [])
    return jsonify({'predictions': predictions})
```

### Task 3: Access Training Metrics

```python
@app.route('/metrics')
@login_required
def show_metrics():
    metrics = session.get('training_results', {})
    return jsonify({
        'r2': metrics.get('r2_score'),
        'rmse': metrics.get('rmse'),
        'mae': metrics.get('mae')
    })
```

### Task 4: Retrain Model (for new file)

```python
@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    # User uploads new file
    file = request.files['file']
    
    # Delete old data
    delete_user_sales_data(session['user_id'])
    
    # Process new file (same as /upload)
    # This automatically retrains the model
    
    return redirect(url_for('predictions'))
```

---

## 🚨 Error Handling

### Check Before Using Predictions

```python
# Always verify model exists before using predictions
if not session.get('model_trained', False):
    raise ModelTrainingError("No trained model available")

predictions = session.get('future_predictions', [])
if not predictions:
    raise DataValidationError("No predictions generated")

# Now safe to use predictions
```

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No trained model" | User hasn't uploaded | Redirect to `/upload` |
| "Empty predictions" | Model training failed | Check logs, retry upload |
| "CSV has no Date column" | Wrong file format | Show error, ask for correct file |
| "Model file not found" | Session cleared | Ask user to re-upload |

---

## 📂 File Structure

```
seasonal_sales_forecasting/
├── app.py                              # Main Flask app
├── models/
│   ├── sales_predictor.py              # ML training code
│   ├── sales_predictor_1.joblib        # User 1's trained model
│   ├── sales_predictor_2.joblib        # User 2's trained model
│   └── ...
├── data/
│   ├── uploaded_sales.csv              # Last uploaded CSV
│   └── seasonal_sales_train_dataset_1.csv
├── templates/
│   ├── upload_improved.html            # Upload page
│   ├── predictions.html                # Results page
│   └── sales_predictor.html            # ML page
├── database/
│   └── sales.db                        # SQLite database
├── SYSTEM_REDESIGN_SUMMARY.md          # Full documentation
└── UNIFIED_WORKFLOW_GUIDE.md           # Workflow guide
```

---

## 🧪 Testing the New Workflow

### Manual Test: Complete Flow

```
1. Start app: python app.py
2. Login: http://localhost:5000/login
3. Upload: http://localhost:5000/upload
4. Select: A CSV with Date + Sales columns
5. Wait: Model trains (2-3 seconds)
6. Auto-redirect: Should go to http://localhost:5000/predictions
7. Verify: See model metrics (R², RMSE, MAE)
8. Verify: See 365 future predictions
9. Access: http://localhost:5000/sales-predictor (cached results)
10. API: curl http://localhost:5000/api/cached-predictions
```

---

## 🔧 Debugging Tips

### Check Session State

```python
# In Flask shell:
from flask import session

# Print session contents
print(session.get('model_trained'))       # Should be True
print(len(session.get('future_predictions', [])))  # Should be ~365
print(session.get('training_results'))   # Should have r2_score, etc.
```

### Check Model File

```bash
# Verify model was saved
ls -lh models/sales_predictor_*.joblib

# Load model in Python
import joblib
model = joblib.load('models/sales_predictor_2.joblib')
print(model)  # Should show model details
```

### Check Database

```bash
# View database content
sqlite3 database/sales.db
SELECT COUNT(*) FROM sales_data WHERE user_id = 2;
```

---

## ⚡ Performance Notes

| Operation | Time | Notes |
|-----------|------|-------|
| CSV upload | ~500ms | Depends on file size |
| Data preprocessing | ~200ms | Handles duplicates, dates |
| Model training | ~2-3s | 80/20 split, Random Forest |
| Prediction generation | ~500ms | For 365 days |
| Page load (cached) | ~50ms | Session-based, very fast |
| API response | ~100ms | JSON serialization |

---

## 📝 Code Comments Locations

```python
# Look for these sections in app.py:

# 1. def generate_future_predictions(df, model_path=None):
#    ↓ Core prediction logic

# 2. @app.route('/upload', methods=['GET', 'POST'])
#    ↓ NEW: Automatic model training after upload

# 3. @app.route('/sales-predictor')
#    ↓ MODIFIED: Smart redirect logic

# 4. @app.route('/api/cached-predictions')
#    ↓ NEW: JSON API for predictions

# 5. def login_required(f):
#    ↓ Authentication wrapper
```

---

## 🎓 Learning Path

**For Beginners:**
1. Read `UNIFIED_WORKFLOW_GUIDE.md` for overview
2. Trace the `/upload` route in `app.py`
3. See how session stores predictions
4. Check `templates/predictions.html` for display

**For Intermediate:**
1. Update route logic (e.g., add new fields)
2. Modify model training (hyperparameters)
3. Add new API endpoints

**For Advanced:**
1. Implement database persistence (vs session)
2. Add model versioning
3. Implement async training (Celery)
4. Add ML model comparison

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start app
cd seasonal_sales_forecasting
python app.py

# 3. Access browser
http://127.0.0.1:5000

# 4. Upload test CSV
# From data/ folder: seasonal_sales_test_dataset_1.csv

# 5. Check predictions
# Should auto-redirect to /predictions
# Then visit /sales-predictor for cached view
```

---

## 📞 Questions & Answers

**Q: Where are predictions cached?**  
A: In Flask `session` variable (in-memory during user session)

**Q: Can users access predictions from different devices?**  
A: Currently no (session-based). To enable: Store in DB instead

**Q: How long does training take?**  
A: ~2-3 seconds for typical datasets (100-1000 rows)

**Q: Can model be retrained with new data?**  
A: Yes! User re-uploads file, system automatically retrains

**Q: What file format is required?**  
A: CSV with `Date` and `Sales` (or `Weekly_Sales`) columns

---

**Last Updated:** 2026-03-26  
**For Questions:** Refer to SYSTEM_REDESIGN_SUMMARY.md or UNIFIED_WORKFLOW_GUIDE.md
