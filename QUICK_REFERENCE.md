# Quick Reference Guide - Seasonal Sales Forecasting App
## For Developers & Users

---

## 🎯 QUICK START

### Start the Application
```bash
cd seasonal_sales_forecasting
python app.py
# Navigate to http://localhost:8000
```

### Login
- Create account or use existing credentials
- Dashboard shows empty state: "No Data Uploaded Yet"

### Upload Sales Data
1. Click "Upload Sales Data" button
2. Select CSV file with columns: Date, Sales (or Weekly_Sales)
3. Wait for automatic ML training (~10-30 seconds)
4. Redirected to predictions page showing results

### View Results
- **Dashboard**: Shows data summary + ML metrics
- **Analysis**: Full EDA with trends, seasonal patterns
- **Predictions**: 365-day forecast with confidence intervals

---

## 📊 ML PREDICTION PIPELINE

### What Happens After Upload:

```
CSV Upload
    ↓
Data Validation (check columns, format)
    ↓
Data Cleaning (remove NaN, aggregation)
    ↓
Feature Engineering (extract temporal features)
    ↓
Random Forest Training (80/20 split)
    ↓
Model Evaluation (R², MAE, RMSE)
    ↓
Future Forecasting (365 days)
    ↓
Display Results on Predictions Page
```

### Time Required:
- Data with 100 records: ~5 seconds
- Data with 500 records: ~15 seconds  
- Data with 1000+ records: ~30 seconds

---

## 🔍 UNDERSTANDING THE METRICS

### R² Score (0.0 to 1.0)
- **0.9+**: Excellent - Model explains 90%+ of sales variance
- **0.7-0.9**: Good - Model is reliable for predictions
- **0.5-0.7**: Fair - Some predictive value but with uncertainty
- **<0.5**: Poor - Model needs more data or features

**Example:** R² = 0.82 means model explains 82% of why sales go up/down

### MAE (Mean Absolute Error) - in dollars
- **Low MAE**: Predictions are close to actual sales
- **High MAE**: Predictions have large errors
- **Example:** MAE = $1,234 means average prediction is off by $1,234

### RMSE (Root Mean Squared Error) - in dollars
- Similar to MAE but penalizes bigger errors more
- Usually slightly higher than MAE
- **Example:** RMSE = $1,567 indicates error distribution is wider

---

## 🗂 CONDITIONAL RENDERING RULES

### What Shows on Dashboard:

**BEFORE Upload:**
- ✅ Welcome message
- ✅ How it works steps
- ✅ Quick actions (Upload, Analysis, etc.)
- ✅ Feature overview
- ❌ Data statistics (hidden)
- ❌ ML metrics (hidden)
- ❌ Message: "No Data Uploaded Yet"

**AFTER Upload:**
- ✅ Data summary (Total Sales, Avg, Records, etc.)
- ✅ ML metrics (R², MAE, RMSE)
- ✅ Feature importance
- ✅ Predictions section

---

## 💾 SESSION DATA MANAGEMENT

### What's Stored in Session:

```python
session['upload_completed'] = True/False
session['training_results'] = {
    'test_r2': 0.82,
    'test_mae': 1234.56,
    'test_rmse': 1567.89,
    # ... other metrics
}
session['future_predictions'] = [
    {'date': '2025-03-26', 'predicted_sales': 5234.56},
    # ... 365 days
]
session['insights'] = {
    'average_forecast': 5200.00,
    'peak_month': 'December',
    'trend': 'increasing'
}
```

### Duration:
- Session persists for entire browser session
- Clears on logout
- Survives page navigation

---

## 🧬 FEATURE ENGINEERING EXPLAINED

### Features Created from Date:

| Feature | Range | Purpose |
|---------|-------|---------|
| year | 2020-2030 | Yearly trend |
| month | 1-12 | Monthly seasonality |
| day | 1-31 | Day of month effects |
| week | 1-53 | Weekly patterns |
| day_of_week | 0-6 (Mon-Sun) | Weekend vs weekday |
| quarter | 1-4 | Quarterly performance |
| day_of_year | 1-365 | Full year seasonality |

### Example:
```
2025-03-26 (Wednesday) →
  year: 2025
  month: 3 (March)
  day: 26
  week: 13
  day_of_week: 2 (Wednesday is 2)
  quarter: 1 (Q1)
  day_of_year: 85
```

---

## 🐛 DEBUG/TROUBLESHOOTING

### Check Model Status:
```python
# In browser console
fetch('/api/sales-model-status')
    .then(r => r.json())
    .then(d => console.log(d))
```

### View Session Data:
```python
# In Flask app context
from flask import session
print(session.get('training_results'))
print(session.get('future_predictions'))
```

### Common Issues:

| Problem | Cause | Fix |
|---------|-------|-----|
| Stats visible before upload | `data_uploaded=False` not passed | Check app.py route |
| ML metrics not showing | `training_results` empty | Verify model trained |
| Predictions all same value | Feature scaling issue | Check engineering logic |
| Training takes too long | Dataset too large | Consider sampling |
| Forecast accuracy low | Too little historical data | Use 6+ months data |

---

## 📈 SAMPLE METRICS INTERPRETATION

### Good Results (target)
```
R² Score: 0.75-0.85
  → Model is reliable for decision-making

MAE: $500-1000 (if avg sales $5000)
  → Prediction error is 10-20% of average

RMSE: ~1.2x MAE
  → Error distribution is reasonable
```

### Fair Results (acceptable)
```
R² Score: 0.60-0.75
  → Model has predictive value but higher uncertainty

MAE: 20-30% of average sales
  → Use with caution for decisions

Next step: Add more historical data
```

### Poor Results (needs improvement)
```
R² Score: <0.60
  → Model lacks predictive power

Causes:
  - Too little data (<3 months)
  - No seasonal patterns detected
  - High variance in sales
  - Missing important features

Solution: Use 12+ months of data
```

---

## 🔐 DATA HANDLING

### What Happens to Uploaded Data:
1. ✅ Validated for format
2. ✅ Stored in `data/uploaded_sales.csv`
3. ✅ Used for model training
4. ✅ Not shared or transmitted externally

### What Happens to ML Models:
1. ✅ Trained on server
2. ✅ Saved to `models/` directory
3. ✅ Accessible only to logged-in users
4. ✅ Deleted on logout or new upload

---

## 📝 EXAMPLE WORKFLOW

### Scenario: Pizza Shop Predicting Sales

**Step 1: Data Preparation**
- Collect 12 months of daily sales
- Format: `2024-01-01, 750.00, Shore Street Store`
- File: `pizza_sales.csv`

**Step 2: Upload & Train**
- Upload file to dashboard
- App automatically trains Random Forest on sales data
- Training takes ~10 seconds

**Step 3: Results**
- R² = 0.78 → "Model is 78% accurate"
- MAE = $45 → "Average prediction error is $45"
- Trend = "increasing" → "Sales trending upward"

**Step 4: Use Forecast**
- Check predicted sales for next 6 months
- Plan inventory based on peak periods
- Adjust staffing for high-sales days
- Monitor actual vs predicted accuracy

**Step 5: Improve**
- After 3 months, upload new data
- Retrain model with latest data
- Check if accuracy improved
- Iterate for better predictions

---

## 🎓 LEARNING RESOURCES

### Random Forest Concept:
- **Ensemble method**: 100 decision trees voting on predictions
- **Robust**: Works well with noisy data
- **Fast**: Makes predictions instantly
- **Feature importance**: Shows which features matter most

### Temporal Features:
- **Day of week**: Monday sales ≠ Sunday sales
- **Seasonality**: Some months always higher (holidays)
- **Trends**: Long-term increase/decrease over months/years

---

## 📞 SUPPORT

### For Errors:
1. Check browser console for JavaScript errors
2. Check Flask logs for Python errors
3. Verify CSV format matches requirements
4. Ensure data has Date and Sales columns

### For Low Accuracy:
1. Use more historical data (minimum 6+ months)
2. Ensure consistent data quality
3. Check for data entry errors
4. Consider other factors affecting sales

---

**Last Updated:** March 25, 2026
**Version:** 1.0 - Production Ready
**Status:** ✅ All components tested and working
