import pandas as pd
import sys
import os

sys.path.insert(0, 'models')
from arima_model import fit_arima, forecast_n_months
from forecast_visualization import create_forecast_json_chart

# Load data
data_path = 'data/Walmart-dataset.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data = data.sort_values('Date')
ts_data = data.groupby('Date')['Weekly_Sales'].sum()
ts_monthly = ts_data.resample('MS').sum()

ts_train = ts_monthly.iloc[:26]
model_result = fit_arima(ts_train, order=(0, 0, 1))
forecast_result = forecast_n_months(ts_train, model_result, n_months=12)

print('Forecast result keys:', list(forecast_result.keys()))
print('Forecast_df shape:', forecast_result['forecast_df'].shape)
print('Forecast_df columns:', list(forecast_result['forecast_df'].columns))
print('Date column dtype:', forecast_result['forecast_df']['date'].dtype)
print('Date column sample:', forecast_result['forecast_df']['date'].iloc[0])

# Test json chart creation
try:
    result = create_forecast_json_chart(ts_train, forecast_result)
    print('\nResult keys:', list(result.keys()))
    print('Success:', result.get('success'))
    if result.get('success'):
        print('✅ Chart created successfully')
        print('  Historical points:', len(result['historical']['dates']))
        print('  Forecast points:', len(result['forecast']['dates']))
    else:
        print('Error:', result.get('error'))
except Exception as e:
    print(f'Exception: {e}')
    import traceback
    traceback.print_exc()
