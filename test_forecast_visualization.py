"""
STEP 15: Forecast Visualization Tests
Tests for all visualization functions and routes
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.dirname(__file__))

from models.arima_model import fit_arima, forecast_n_months, forecast_with_scenario, generate_forecast_summary
from forecast_visualization import ForecastVisualizer, create_forecast_json_chart


class TestForecastVisualization(unittest.TestCase):
    """Test ForecastVisualizer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Load Walmart data
        data_path = 'Walmart-dataset.csv'
        if not os.path.exists(data_path):
            data_path = 'data/Walmart-dataset.csv'
        if not os.path.exists(data_path):
            data_path = '../Walmart-dataset.csv'
        
        data = pd.read_csv(data_path)
        
        # Prepare time series
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data = data.sort_values('Date')
        ts_data = data.groupby('Date')['Weekly_Sales'].sum()
        
        # Resample to monthly
        ts_monthly = ts_data.resample('MS').sum()
        
        # Use 26 months for training, 7 for testing
        cls.ts_train = ts_monthly.iloc[:26]
        cls.ts_test = ts_monthly.iloc[26:33]
        cls.ts_full = ts_monthly
        
        # Fit ARIMA model
        cls.model_result = fit_arima(cls.ts_train, order=(0, 0, 1))
        
        # Generate forecast
        cls.forecast_result = forecast_n_months(cls.ts_train, cls.model_result, n_months=12)
        
        # Generate forecast with scenarios
        cls.forecast_summary = generate_forecast_summary(
            cls.ts_train, 
            cls.model_result, 
            n_periods=12, 
            include_scenarios=True
        )
        
        # Initialize visualizer
        cls.visualizer = ForecastVisualizer(output_dir='static/plots')
    
    
    def test_1_visualizer_initialization(self):
        """Test ForecastVisualizer initialization"""
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(self.visualizer.output_dir, 'static/plots')
        self.assertTrue(os.path.exists('static/plots'))
        print("✅ TEST 1: Visualizer Initialization - PASSED")
    
    
    def test_2_plot_historical_vs_forecast(self):
        """Test historical vs forecast plot generation"""
        result = self.visualizer.plot_historical_vs_forecast(
            self.ts_train, 
            self.forecast_result,
            title='Test: Historical vs Forecast'
        )
        
        self.assertTrue(result['success'], f"Failed: {result.get('error')}")
        self.assertIn('filepath', result)
        self.assertIn('filename', result)
        self.assertTrue(os.path.exists(result['filepath']))
        self.assertEqual(result['filename'], 'historical_vs_forecast.png')
        
        # Check chart data
        self.assertIn('chart_data', result)
        self.assertIn('historical_dates', result['chart_data'])
        self.assertIn('forecast_values', result['chart_data'])
        
        print(f"✅ TEST 2: Historical vs Forecast Plot")
        print(f"   Mean Forecast: ${result['mean_forecast']:,.0f}")
        print(f"   Forecast Range: ${result['forecast_range'][0]:,.0f} - ${result['forecast_range'][1]:,.0f}")
        print(f"   - PASSED")
    
    
    def test_3_plot_confidence_intervals(self):
        """Test confidence interval plot generation"""
        result = self.visualizer.plot_confidence_intervals(
            self.forecast_result,
            title='Test: Confidence Intervals'
        )
        
        self.assertTrue(result['success'], f"Failed: {result.get('error')}")
        self.assertIn('filepath', result)
        self.assertTrue(os.path.exists(result['filepath']))
        self.assertEqual(result['filename'], 'confidence_intervals.png')
        
        print("✅ TEST 3: Confidence Intervals Plot - PASSED")
    
    
    def test_4_plot_forecast_statistics(self):
        """Test forecast statistics plot generation"""
        result = self.visualizer.plot_forecast_statistics(
            self.forecast_result,
            title='Test: Forecast Statistics'
        )
        
        self.assertTrue(result['success'], f"Failed: {result.get('error')}")
        self.assertIn('filepath', result)
        self.assertTrue(os.path.exists(result['filepath']))
        self.assertEqual(result['filename'], 'forecast_statistics.png')
        
        print("✅ TEST 4: Forecast Statistics Plot - PASSED")
    
    
    def test_5_plot_scenario_comparison(self):
        """Test scenario comparison plot generation"""
        result = self.visualizer.plot_scenario_comparison(
            self.forecast_summary,
            title='Test: Scenario Comparison'
        )
        
        self.assertTrue(result['success'], f"Failed: {result.get('error')}")
        self.assertIn('filepath', result)
        self.assertTrue(os.path.exists(result['filepath']))
        self.assertEqual(result['filename'], 'scenario_comparison.png')
        
        print("✅ TEST 5: Scenario Comparison Plot - PASSED")
    
    
    def test_6_generate_all_visualizations(self):
        """Test generating all visualizations at once"""
        result = self.visualizer.generate_all_visualizations(
            self.ts_train,
            self.forecast_result,
            self.forecast_summary
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(len(result['visualizations']), 0)
        self.assertEqual(result['total_errors'], 0)
        
        print(f"✅ TEST 6: Generate All Visualizations")
        print(f"   Total Generated: {result['total_generated']}")
        print(f"   Visualizations:")
        for viz in result['visualizations']:
            print(f"     - {viz['title']}")
        print(f"   - PASSED")
    
    
    def test_7_create_forecast_json_chart(self):
        """Test JSON chart data creation"""
        result = create_forecast_json_chart(self.ts_train, self.forecast_result)
        
        self.assertTrue(result['success'])
        self.assertIn('historical', result)
        self.assertIn('forecast', result)
        self.assertIn('statistics', result)
        
        # Check historical data
        self.assertEqual(len(result['historical']['dates']), len(self.ts_train))
        self.assertEqual(len(result['historical']['values']), len(self.ts_train))
        
        # Check forecast data
        self.assertGreater(len(result['forecast']['dates']), 0)
        self.assertEqual(len(result['forecast']['values']), len(result['forecast']['dates']))
        self.assertEqual(len(result['forecast']['lower_ci']), len(result['forecast']['dates']))
        self.assertEqual(len(result['forecast']['upper_ci']), len(result['forecast']['dates']))
        
        # Check statistics
        self.assertIn('forecast_mean', result['statistics'])
        self.assertIn('forecast_std', result['statistics'])
        self.assertGreater(result['statistics']['forecast_mean'], 0)
        
        print(f"✅ TEST 7: JSON Chart Data Creation")
        print(f"   Historical points: {len(result['historical']['dates'])}")
        print(f"   Forecast points: {len(result['forecast']['dates'])}")
        print(f"   Forecast Mean: ${result['statistics']['forecast_mean']:,.0f}")
        print(f"   - PASSED")
    
    
    def test_8_plot_with_invalid_data(self):
        """Test error handling with invalid data"""
        invalid_result = {'no_forecast_df': True}
        
        result = self.visualizer.plot_historical_vs_forecast(
            self.ts_train,
            invalid_result
        )
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        
        print("✅ TEST 8: Invalid Data Error Handling - PASSED")
    
    
    def test_9_plot_file_creation(self):
        """Test that all plot files are created correctly"""
        result = self.visualizer.generate_all_visualizations(
            self.ts_train,
            self.forecast_result,
            self.forecast_summary
        )
        
        self.assertTrue(result['success'])
        
        # Count files in output directory
        plot_files = [f for f in os.listdir('static/plots') if f.endswith('.png')]
        self.assertGreater(len(plot_files), 0)
        
        # Verify each file size > 0 (non-empty)
        for plot_file in plot_files:
            filepath = os.path.join('static/plots', plot_file)
            file_size = os.path.getsize(filepath)
            self.assertGreater(file_size, 1000, f"Plot file {plot_file} seems empty")
        
        print(f"✅ TEST 9: Plot File Creation")
        print(f"   Files created: {len(plot_files)}")
        print(f"   - PASSED")
    
    
    def test_10_visualization_consistency(self):
        """Test consistency of visualization data across different methods"""
        # Generate from different methods
        result1 = create_forecast_json_chart(self.ts_train, self.forecast_result)
        
        # Both should have same number of forecast points
        self.assertEqual(
            len(result1['forecast']['values']),
            len(self.forecast_result['forecast_df'])
        )
        
        # Forecast values should match
        forecast_vals_1 = result1['forecast']['values']
        forecast_vals_2 = self.forecast_result['forecast_df']['forecast'].tolist()
        
        # Allow small floating point differences
        for v1, v2 in zip(forecast_vals_1, forecast_vals_2):
            self.assertAlmostEqual(v1, v2, delta=1.0)
        
        print("✅ TEST 10: Visualization Data Consistency - PASSED")


class TestVisualizationIntegration(unittest.TestCase):
    """Test visualization integration with forecasting"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for integration tests"""
        # Load data
        data_path = 'Walmart-dataset.csv'
        if not os.path.exists(data_path):
            data_path = 'data/Walmart-dataset.csv'
        if not os.path.exists(data_path):
            data_path = '../Walmart-dataset.csv'
        
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
        data = data.sort_values('Date')
        ts_data = data.groupby('Date')['Weekly_Sales'].sum()
        ts_monthly = ts_data.resample('MS').sum()
        
        cls.ts_train = ts_monthly.iloc[:26]
        cls.model_result = fit_arima(cls.ts_train, order=(0, 0, 1))
        cls.forecast_result = forecast_n_months(cls.ts_train, cls.model_result, n_months=12)
        cls.visualizer = ForecastVisualizer(output_dir='static/plots')
    
    
    def test_1_full_visualization_pipeline(self):
        """Test complete visualization pipeline from data to plots"""
        # Step 1: Generate forecast
        self.assertIsNotNone(self.forecast_result)
        self.assertIn('forecast_df', self.forecast_result)
        
        # Step 2: Create visualizations
        result = self.visualizer.plot_historical_vs_forecast(
            self.ts_train,
            self.forecast_result
        )
        
        self.assertTrue(result['success'])
        self.assertTrue(os.path.exists(result['filepath']))
        
        # Step 3: Verify chart data
        self.assertIn('chart_data', result)
        chart_data = result['chart_data']
        
        self.assertEqual(
            len(chart_data['historical_dates']),
            len(self.ts_train)
        )
        
        print("✅ TEST 1: Full Visualization Pipeline - PASSED")
    
    
    def test_2_scenarios_to_visualization(self):
        """Test converting scenario forecasts to visualizations"""
        # Generate scenarios
        summary = generate_forecast_summary(
            self.ts_train,
            self.model_result,
            n_periods=12,
            include_scenarios=True
        )
        
        # Create visualization
        result = self.visualizer.plot_scenario_comparison(summary)
        
        self.assertTrue(result['success'])
        self.assertTrue(os.path.exists(result['filepath']))
        
        print("✅ TEST 2: Scenarios to Visualization - PASSED")
    
    
    def test_3_multi_horizon_visualizations(self):
        """Test visualizations with different forecast horizons"""
        horizons = [6, 12, 24]
        
        for horizon in horizons:
            result = forecast_n_months(self.ts_train, self.model_result, n_months=horizon)
            
            viz_result = self.visualizer.plot_historical_vs_forecast(
                self.ts_train,
                result,
                title=f'Forecast: {horizon} Months'
            )
            
            self.assertTrue(viz_result['success'])
            self.assertGreater(len(result['forecast_df']), 0)
        
        print("✅ TEST 3: Multi-Horizon Visualizations - PASSED")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestForecastVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationIntegration))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        print(f"Pass Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
