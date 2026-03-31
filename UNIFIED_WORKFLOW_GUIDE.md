# Seasonal Sales Forecasting - Complete Workflow Guide

## 🎯 System Overview

The application now uses a **streamlined single-upload workflow** where users:
1. Upload dataset **ONCE**
2. System automatically trains the ML model
3. User views predictions immediately

---

## 📋 Complete Workflow

### **Step 1: Upload Dataset (Single Time)**
- **Route:** `/upload`
- **User Action:**
  - Navigate to "Upload" page
  - Choose CSV file with sales data
  - Click "Upload"
  
- **System Actions (Automatic):**
  - Validates CSV structure (checks for Date, Sales columns)
  - Preprocesses data:
    - Handles missing values
    - Converts Date to datetime format
    - Sorts records chronologically
    - Removes duplicates
  - Stores data in database (session + backend storage)
  - **Automatically trains Random Forest model**
  - Generates 365-day future sales predictions
  - Stores trained model in `/models/sales_predictor_{user_id}.joblib`

- **User Sees:**
  - ✅ "File uploaded and ML model trained successfully!"
  - Page auto-redirects to **Predictions** view

---

### **Step 2: View Predictions (No Re-upload Needed)**
- **Route:** `/predictions` (automatic redirect after upload)
- **What's Displayed:**
  - 📊 Model performance metrics:
    - R² Score
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Training/Test sample counts
  - 📈 Future predictions table (365 days)
  - 📋 Sales insights and trends
  - 🎯 Feature importance chart

- **User Can:**
  - View next 30/60/90/365 days predictions
  - Download prediction CSV
  - See confidence intervals
  - Analyze model performance

---

### **Step 3: Access ML Predictor Anytime**
- **Route:** `/sales-predictor`
- **Functionality:**
  - Shows the **same predictions** from trained model
  - No file upload needed
  - Direct access to cached predictions
  - Auto-redirects to upload if no model trained

---

## 🛠️ Technical Architecture

### **Data Flow**

```
User Upload File
    ↓
CSV Validation (/upload)
    ↓
Data Preprocessing
├─ Handle missing values
├─ Date → DateTime conversion
├─ Sort chronologically
└─ Remove duplicates
    ↓
Store in Database & Session
    ↓
Automatic Model Training (No user action)
├─ Random Forest Regressor
├─ Date features: year, month, day, day_of_week
├─ Target: Sales
└─ Train/Test split (80/20)
    ↓
Generate Future Predictions (365 days)
    ↓
Store in Session & Cache
    ↓
Display to User (/predictions)
    ↓
Access Anytime (/sales-predictor or /api/cached-predictions)
```

### **Key Routes**

| Route | Method | Purpose | Requires Upload |
|-------|--------|---------|-----------------|
| `/upload` | GET/POST | Upload & auto-train model | ✗ (first time) |
| `/predictions` | GET | View predictions | ✓ (auto-redirect if missing) |
| `/sales-predictor` | GET | View/download predictions | ✓ (auto-redirect if missing) |
| `/api/cached-predictions` | GET | Get predictions as JSON | ✓ |
| `/api/sales-model-status` | GET | Check if model trained | N/A |

### **Session Variables**

```python
session['model_trained']         # Boolean: Model exists?
session['training_results']      # Dict: Model metrics
session['future_predictions']    # List: Predicted sales
session['trained_model']         # Dict: Model details
session['insights']              # Dict: Data insights
session['model_path']            # Str: Path to trained model
```

---

## 🚀 How It Works - Step by Step

### **Upload Process**

```python
# 1. File uploaded
@app.route('/upload', methods=['POST'])

# 2. Validation
validate_csv_file(file)          # Check file type
validate_csv_structure(df)       # Check columns

# 3. Preprocessing (Automatic)
preprocess_data(df)              # Clean & prepare
store_sales_data(user_id, df)    # Save to DB

# 4. Model Training (Automatic - No user clicks needed!)
train_sales_model_from_csv(path)
generate_future_predictions(df)

# 5. Store & Redirect
session['model_trained'] = True
session['future_predictions'] = predictions
redirect('/predictions')
```

### **View Predictions**

```python
# Route 1: /predictions (automatic after upload)
@app.route('/predictions')
def predictions():
    # Load from session
    ml_predictions = session['future_predictions']
    training_results = session['training_results']
    return render_template('predictions.html', ...)

# Route 2: /sales-predictor (anytime access)
@app.route('/sales-predictor')
def sales_predictor():
    if session['model_trained']:
        # Show cached predictions
        return render_template('sales_predictor.html', ...)
    else:
        # Redirect to upload
        redirect('/upload')

# Route 3: /api/cached-predictions (JSON API)
@app.route('/api/cached-predictions')
def cached_predictions():
    return jsonify({
        'predictions': session['future_predictions'][:30],
        'training_metrics': session['training_results']
    })
```

---

## 📊 Key Features

✅ **Single Upload** - Upload once, use predictions everywhere
✅ **Automatic Training** - No manual model training button clicks
✅ **Persistent Storage** - Predictions cached in session & database
✅ **Error Handling** - Clear error messages if data missing
✅ **No Re-uploads** - ML Predictor uses stored data
✅ **Multiple Access Points** - View predictions from `/predictions` or `/sales-predictor`
✅ **API Access** - Get predictions as JSON via `/api/cached-predictions`

---

## 🔧 Configuration

### **Files Modified**
- `app.py` - Updated routes & automatic training logic
- `upload_improved.html` - Clean upload interface
- `predictions.html` - Show model results
- `sales_predictor.html` - Display predictions

### **Models Stored**
```
models/
├─ sales_predictor_1.joblib      # User 1's trained model
├─ sales_predictor_2.joblib      # User 2's trained model
└─ ...
```

### **Data Paths**
```
data/
├─ uploaded_sales.csv             # Last uploaded CSV
└─ (temporary files auto-cleaned)

database/
└─ sales.db                        # SQLite with sales data
```

---

## 💡 Usage Examples

### **Example 1: Complete User Journey**

```
1. User navigates to /upload
2. User selects "quarterly_sales.csv"
3. User clicks "Upload"
   
   [System automatically:]
   - Validates CSV
   - Preprocesses data
   - Trains Random Forest model
   - Generates 365-day predictions
   
4. Browser auto-redirects to /predictions
5. User sees:
   - Model performance (R²: 0.956)
   - Next 30 days predictions
   - Charts & insights
6. User can download CSV or access /sales-predictor anytime
```

### **Example 2: Accessing Predictions Later**

```
User Session 1:
- Uploads data → Model trained → Predictions stored

User Session 2 (same day):
- Navigates to /sales-predictor
- System loads cached predictions
- No re-upload needed!
```

### **Example 3: API Integration**

```bash
# Get predictions as JSON
curl -X GET http://localhost:5000/api/cached-predictions \
  -H "Authorization: Bearer YOUR_SESSION"

# Response:
{
  "success": true,
  "predictions": [
    {"date": "2024-01-01", "predicted_sales": 5234.50},
    ...
  ],
  "training_metrics": {
    "r2_score": 0.956,
    "rmse": 829.68,
    "mae": 444.68
  }
}
```

---

## ✅ Benefits Over Old System

### **Before (Separate Upload & ML Predictor)**
❌ Upload data in Upload page  
❌ Upload SAME data again in ML Predictor page  
❌ Train model manually  
❌ Click multiple times for each step  
❌ Confusing workflow with duplicate uploads  

### **After (Unified Workflow)**
✅ Upload data ONCE in Upload page  
✅ Automatic model training (no manual button clicks)  
✅ Automatic predictions generated  
✅ Access predictions from multiple pages  
✅ Clean, efficient workflow  
✅ No duplicate uploads  

---

## 🐛 Error Handling

### **Scenario 1: User accesses /sales-predictor without uploading**
```
System: Detects session['model_trained'] = False
Action: Redirects to /upload with message
Message: "Please upload a dataset first to see AI predictions."
```

### **Scenario 2: User accesses /predictions without uploading**
```
System: Detects no training data
Action: Redirects to /upload with message
Message: "Please upload a dataset before accessing analysis."
```

### **Scenario 3: API call without trained model**
```
Request: GET /api/cached-predictions
Response: 
{
  "success": false,
  "error": "No trained model available. Please upload data first.",
  "model_trained": false
}
Status: 400 Bad Request
```

---

## 📈 Next Improvements

- [ ] Database persistence across sessions (vs. session-only)
- [ ] Model versioning (keep multiple trained models)
- [ ] Compare predictions across different datasets
- [ ] Export training reports to PDF
- [ ] Real-time model retraining on new data
- [ ] Ensemble predictions (multiple models)

---

## 📞 Support

For issues or questions about the workflow:
1. Check browser console for error messages
2. Review application logs in `logs/` directory
3. Verify CSV format matches expected structure
4. Ensure Date and Sales columns exist

---

**Last Updated:** 2026-03-26  
**Application:** Seasonal Sales Forecasting System  
**Version:** 2.0 (Unified Workflow)
