"""
Error Handling Test Suite
Tests the comprehensive error handling system in the Sales Forecasting Application.
"""

import unittest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_handlers import (
    DataValidationError, FileUploadError, InsufficientDataError,
    ModelTrainingError, DatabaseError, AuthenticationError,
    AuthorizationError, ResourceNotFoundError, format_error_response,
    app_logger, validate_csv_file, validate_numeric_range,
    validate_data_available, ErrorRecoveryStrategy
)


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_data_validation_error(self):
        """Test DataValidationError creation and properties."""
        exc = DataValidationError(
            "Invalid email format",
            field='email'
        )
        
        self.assertEqual(exc.message, "Invalid email format")
        self.assertEqual(exc.error_code, 'DATA_VALIDATION_ERROR')
        self.assertEqual(exc.http_status, 400)
        self.assertEqual(exc.details['field'], 'email')
    
    def test_file_upload_error(self):
        """Test FileUploadError creation."""
        exc = FileUploadError(
            "File too large",
            filename='large_file.csv',
            file_size=1024*1024*100
        )
        
        self.assertEqual(exc.error_code, 'FILE_UPLOAD_ERROR')
        self.assertEqual(exc.http_status, 400)
        self.assertEqual(exc.details['filename'], 'large_file.csv')
        self.assertEqual(exc.details['file_size'], 1024*1024*100)
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError creation."""
        exc = InsufficientDataError(
            "Not enough data for forecast",
            required=12,
            actual=3
        )
        
        self.assertEqual(exc.error_code, 'INSUFFICIENT_DATA_ERROR')
        self.assertEqual(exc.http_status, 422)
        self.assertEqual(exc.details['required'], 12)
        self.assertEqual(exc.details['actual'], 3)
    
    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        exc = AuthenticationError(
            "Invalid credentials",
            reason='wrong_password'
        )
        
        self.assertEqual(exc.error_code, 'AUTHENTICATION_ERROR')
        self.assertEqual(exc.http_status, 401)
        self.assertEqual(exc.details['reason'], 'wrong_password')
    
    def test_authentication_error_attributes(self):
        """Test that exceptions have required attributes."""
        exc = AuthenticationError("Test error")
        
        self.assertTrue(hasattr(exc, 'error_code'))
        self.assertTrue(hasattr(exc, 'http_status'))
        self.assertTrue(hasattr(exc, 'message'))
        self.assertTrue(hasattr(exc, 'timestamp'))


class TestValidationFunctions(unittest.TestCase):
    """Test validation utility functions."""
    
    def test_validate_numeric_range_success(self):
        """Test numeric validation with valid input."""
        try:
            result = validate_numeric_range(50, min_val=0, max_val=100, field_name='percentage')
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")
    
    def test_validate_numeric_range_too_low(self):
        """Test numeric validation with value below minimum."""
        with self.assertRaises(DataValidationError) as context:
            validate_numeric_range(-5, min_val=0, max_val=100, field_name='percentage')
        
        self.assertEqual(context.exception.error_code, 'DATA_VALIDATION_ERROR')
    
    def test_validate_numeric_range_too_high(self):
        """Test numeric validation with value above maximum."""
        with self.assertRaises(DataValidationError) as context:
            validate_numeric_range(150, min_val=0, max_val=100, field_name='percentage')
        
        self.assertEqual(context.exception.error_code, 'DATA_VALIDATION_ERROR')


class TestErrorResponseFormatting(unittest.TestCase):
    """Test error response formatting."""
    
    def test_format_custom_exception_response(self):
        """Test formatting of custom exception as response."""
        exc = DataValidationError(
            "Missing required field",
            field='email'
        )
        
        response, status_code = format_error_response(exc)
        
        self.assertEqual(status_code, 400)
        self.assertEqual(response['status'], 'error')
        self.assertEqual(response['error_code'], 'DATA_VALIDATION_ERROR')
        self.assertIn('timestamp', response)
    
    def test_format_generic_exception_response(self):
        """Test formatting of generic exception as response."""
        exc = ValueError("Some value error")
        
        response, status_code = format_error_response(exc)
        
        self.assertEqual(status_code, 500)
        self.assertEqual(response['status'], 'error')
        self.assertEqual(response['error_code'], 'INTERNAL_ERROR')
        self.assertIn('error_type', response)


class TestErrorRecoveryStrategy(unittest.TestCase):
    """Test error recovery strategies."""
    
    def test_data_load_failure_recovery(self):
        """Test recovery from data loading failure."""
        error = Exception("File not found")
        success, message = ErrorRecoveryStrategy.on_data_load_failure(error)
        
        self.assertFalse(success)
        self.assertIn('CSV', message.upper())
    
    def test_model_training_failure_recovery(self):
        """Test recovery from model training failure."""
        error = Exception("Model failed")
        result = ErrorRecoveryStrategy.on_model_training_failure(error, 'simple_average')
        
        self.assertTrue(result['fallback'])
        self.assertEqual(result['method'], 'simple_average')


class TestLoggerIntegration(unittest.TestCase):
    """Test logging integration."""
    
    def test_logger_methods_available(self):
        """Test that logger has all required methods."""
        self.assertTrue(hasattr(app_logger, 'debug'))
        self.assertTrue(hasattr(app_logger, 'info'))
        self.assertTrue(hasattr(app_logger, 'warning'))
        self.assertTrue(hasattr(app_logger, 'error'))
        self.assertTrue(hasattr(app_logger, 'critical'))
    
    def test_logger_can_log(self):
        """Test that logger can successfully log messages."""
        try:
            app_logger.info("Test log message")
            app_logger.warning("Test warning message")
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Logger failed: {e}")


class TestExceptionInheritance(unittest.TestCase):
    """Test exception class hierarchy."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from base exception."""
        test_exceptions = [
            DataValidationError("test"),
            FileUploadError("test"),
            InsufficientDataError("test"),
            ModelTrainingError("test"),
            DatabaseError("test"),
            AuthenticationError("test"),
            AuthorizationError("test"),
            ResourceNotFoundError("test")
        ]
        
        for exc in test_exceptions:
            self.assertIsInstance(exc, SalesForecastingException)


class TestErrorDetails(unittest.TestCase):
    """Test error detail preservation."""
    
    def test_error_preserves_timestamp(self):
        """Test that errors preserve timestamp."""
        exc = DataValidationError("test")
        self.assertIsNotNone(exc.timestamp)
        self.assertIn('T', exc.timestamp)  # ISO format check
    
    def test_error_preserves_details(self):
        """Test that errors preserve additional details."""
        details = {'custom_key': 'custom_value'}
        exc = DataValidationError("test", details=details)
        
        self.assertEqual(exc.details['custom_key'], 'custom_value')
    
    def test_error_with_no_details(self):
        """Test error with no additional details."""
        exc = AuthenticationError("test")
        self.assertIsInstance(exc.details, dict)
        self.assertEqual(len(exc.details), 1)  # Only 'reason' should be there


# ======================== Integration Tests ========================

class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across the system."""
    
    def test_validation_error_in_response_format(self):
        """Test that validation errors format properly as responses."""
        try:
            raise DataValidationError(
                "Email is required",
                field='email'
            )
        except DataValidationError as e:
            response, status = format_error_response(e)
            
            self.assertEqual(status, 400)
            self.assertEqual(response['error_code'], 'DATA_VALIDATION_ERROR')
            self.assertIn('Email', response['message'])
    
    def test_resource_not_found_in_response(self):
        """Test resource not found error formatting."""
        try:
            raise ResourceNotFoundError(
                "Store not found",
                resource_type='store',
                resource_id=999
            )
        except ResourceNotFoundError as e:
            response, status = format_error_response(e)
            
            self.assertEqual(status, 404)
            self.assertEqual(response['details']['resource_type'], 'store')
            self.assertEqual(response['details']['resource_id'], 999)


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCustomExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorResponseFormatting))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRecoveryStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionInheritance))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorDetails))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("ERROR HANDLING TEST SUITE SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
