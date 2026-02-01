"""
STEP 19: Test Validation Suite
Automated tests for data isolation, forecast accuracy, and graph display
"""

import unittest
import sqlite3
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_system import (
    DataIsolationValidator,
    ForecastAccuracyValidator,
    GraphDisplayValidator,
    SystemValidator
)


class TestDataIsolation(unittest.TestCase):
    """Tests for data isolation validation"""
    
    def setUp(self):
        """Create test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize test database
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create users table
        cur.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        
        # Insert test users
        cur.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                   ('user1', 'hash1'))
        cur.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                   ('user2', 'hash2'))
        
        # Create isolated sales tables for each user
        cur.execute('CREATE TABLE user_1_sales (date TEXT, value REAL)')
        cur.execute('INSERT INTO user_1_sales VALUES (?, ?)', ('2024-01-01', 100.0))
        
        cur.execute('CREATE TABLE user_2_sales (date TEXT, value REAL)')
        cur.execute('INSERT INTO user_2_sales VALUES (?, ?)', ('2024-01-01', 200.0))
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_sales_data_isolation(self):
        """Test that sales data is isolated per user"""
        validator = DataIsolationValidator(self.db_path)
        result = validator.check_sales_data_isolation()
        self.assertTrue(result, "Sales data should be isolated per user")
    
    def test_session_isolation(self):
        """Test that session data is properly isolated"""
        validator = DataIsolationValidator(self.db_path)
        result = validator.check_session_isolation()
        self.assertTrue(result, "Session data should be isolated")
    
    def test_user_id_validation(self):
        """Test that user_id validation is enforced"""
        validator = DataIsolationValidator(self.db_path)
        result = validator.check_user_id_validation()
        self.assertTrue(result, "User ID validation should be enforced")
    
    def test_data_isolation_results(self):
        """Test that isolation validator returns proper results"""
        validator = DataIsolationValidator(self.db_path)
        validator.check_sales_data_isolation()
        results = validator.get_results()
        
        self.assertIn('checks_passed', results)
        self.assertIn('checks_failed', results)
        self.assertIn('issues', results)


class TestForecastAccuracy(unittest.TestCase):
    """Tests for forecast accuracy validation"""
    
    def test_arima_parameters(self):
        """Test ARIMA model parameters validation"""
        validator = ForecastAccuracyValidator()
        result = validator.check_arima_model_parameters()
        self.assertTrue(result, "ARIMA parameters should be valid")
    
    def test_forecast_bounds(self):
        """Test forecast values are within valid bounds"""
        validator = ForecastAccuracyValidator()
        result = validator.check_forecast_bounds()
        self.assertTrue(result, "Forecast values should be within bounds")
    
    def test_confidence_intervals(self):
        """Test confidence intervals are properly calculated"""
        validator = ForecastAccuracyValidator()
        result = validator.check_confidence_intervals()
        self.assertTrue(result, "Confidence intervals should be valid")
    
    def test_seasonality_patterns(self):
        """Test seasonality is correctly captured"""
        validator = ForecastAccuracyValidator()
        result = validator.check_seasonality_patterns()
        self.assertTrue(result, "Seasonality patterns should be detected")
    
    def test_forecast_metrics(self):
        """Test that forecast metrics are properly calculated"""
        validator = ForecastAccuracyValidator()
        validator.check_arima_model_parameters()
        validator.check_forecast_bounds()
        validator.check_confidence_intervals()
        validator.check_seasonality_patterns()
        
        results = validator.get_results()
        self.assertIn('metrics', results)
        self.assertGreater(len(results['metrics']), 0)


class TestGraphDisplay(unittest.TestCase):
    """Tests for graph display validation"""
    
    def test_chart_json_format(self):
        """Test chart JSON format is valid"""
        validator = GraphDisplayValidator()
        result = validator.check_chart_json_format()
        self.assertTrue(result, "Chart JSON should be valid")
    
    def test_data_series_integrity(self):
        """Test data series are complete and correct"""
        validator = GraphDisplayValidator()
        result = validator.check_data_series_integrity()
        self.assertTrue(result, "Data series should be complete")
    
    def test_forecast_visualization(self):
        """Test forecast graphs display correctly"""
        validator = GraphDisplayValidator()
        result = validator.check_forecast_visualization()
        self.assertTrue(result, "Forecast visualization should be correct")
    
    def test_analysis_visualizations(self):
        """Test analysis graphs display correctly"""
        validator = GraphDisplayValidator()
        result = validator.check_analysis_visualizations()
        self.assertTrue(result, "Analysis visualizations should be correct")
    
    def test_responsive_design(self):
        """Test graphs are responsive on different screen sizes"""
        validator = GraphDisplayValidator()
        result = validator.check_responsive_design()
        self.assertTrue(result, "Graphs should be responsive")
    
    def test_graph_validation_count(self):
        """Test that all graphs are properly validated"""
        validator = GraphDisplayValidator()
        validator.check_chart_json_format()
        validator.check_data_series_integrity()
        validator.check_forecast_visualization()
        validator.check_analysis_visualizations()
        validator.check_responsive_design()
        
        results = validator.get_results()
        self.assertGreaterEqual(len(results['graphs_validated']), 5)


class TestSystemValidator(unittest.TestCase):
    """Tests for overall system validation"""
    
    def setUp(self):
        """Create test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize test database
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        
        cur.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                   ('testuser', 'hash'))
        
        cur.execute('CREATE TABLE user_1_sales (date TEXT, value REAL)')
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_system_validation_complete(self):
        """Test that complete system validation runs"""
        validator = SystemValidator(self.db_path)
        results = validator.run_all_validations()
        
        self.assertIn('timestamp', results)
        self.assertIn('data_isolation', results)
        self.assertIn('forecast_accuracy', results)
        self.assertIn('graph_display', results)
        self.assertIn('summary', results)
    
    def test_validation_summary(self):
        """Test that validation summary is properly calculated"""
        validator = SystemValidator(self.db_path)
        results = validator.run_all_validations()
        
        summary = results['summary']
        self.assertIn('total_checks', summary)
        self.assertIn('passed', summary)
        self.assertIn('failed', summary)
        self.assertIn('pass_rate', summary)
        
        # Verify pass_rate is between 0 and 100
        self.assertGreaterEqual(summary['pass_rate'], 0)
        self.assertLessEqual(summary['pass_rate'], 100)
    
    def test_validation_results_export(self):
        """Test that validation results can be exported"""
        validator = SystemValidator(self.db_path)
        validator.run_all_validations()
        
        # Use temporary file for export
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            export_path = f.name
        
        try:
            result = validator.export_results(export_path)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(export_path))
            
            # Verify JSON is valid
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('timestamp', data)
            self.assertIn('summary', data)
        
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestValidationScenarios(unittest.TestCase):
    """Integration tests for complete validation scenarios"""
    
    def setUp(self):
        """Setup for scenario tests"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize with multiple users
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        
        # Create multiple test users
        for i in range(1, 4):
            cur.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                       (f'user{i}', f'hash{i}'))
            cur.execute(f'CREATE TABLE user_{i}_sales (date TEXT, value REAL)')
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Cleanup"""
        os.unlink(self.db_path)
    
    def test_multi_user_isolation(self):
        """Test isolation with multiple users"""
        validator = DataIsolationValidator(self.db_path)
        
        # Each user should have isolated data
        result = validator.check_sales_data_isolation()
        self.assertTrue(result)
        
        results = validator.get_results()
        self.assertGreaterEqual(results['checks_passed'], 1)
    
    def test_complete_validation_flow(self):
        """Test complete validation flow with all validators"""
        system_validator = SystemValidator(self.db_path)
        results = system_validator.run_all_validations()
        
        # Verify all phases completed
        self.assertGreater(len(results['data_isolation']), 0)
        self.assertGreater(len(results['forecast_accuracy']), 0)
        self.assertGreater(len(results['graph_display']), 0)
        
        # Verify summary
        summary = results['summary']
        self.assertGreater(summary['total_checks'], 0)


def run_test_suite():
    """Run complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataIsolation))
    suite.addTests(loader.loadTestsFromTestCase(TestForecastAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphDisplay))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_test_suite()
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
