"""
Comprehensive error handling and custom exception classes.

This module provides:
- Custom exception classes for different error scenarios
- Error logging utilities
- Error response formatting
- Validation decorators
- Error recovery strategies
"""

import logging
import traceback
from datetime import datetime
from functools import wraps
from flask import jsonify, request
from typing import Callable, Any, Tuple, Optional, Dict


# ========================== LOGGING SETUP ==========================

class ErrorLogger:
    """Centralized error logging system with multiple levels."""
    
    def __init__(self, name: str = 'sales_forecasting'):
        self.logger = logging.getLogger(name)
        
        # Set log level
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler with formatting
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(f"[DEBUG] {message}", extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(f"[INFO] {message}", extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(f"[WARNING] {message}", extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(f"[ERROR] {message}", exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        self.logger.critical(f"[CRITICAL] {message}", exc_info=exc_info, extra=kwargs)


# Global logger instance
app_logger = ErrorLogger()


# ========================== CUSTOM EXCEPTIONS ==========================

class SalesForecastingException(Exception):
    """Base exception for all sales forecasting errors."""
    
    def __init__(self, message: str, error_code: str = 'INTERNAL_ERROR', 
                 http_status: int = 500, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class DataValidationError(SalesForecastingException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='DATA_VALIDATION_ERROR',
            http_status=400,
            details={'field': field, **(details or {})}
        )


class DataLoadError(SalesForecastingException):
    """Raised when data loading/reading fails."""
    
    def __init__(self, message: str, filename: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='DATA_LOAD_ERROR',
            http_status=500,
            details={'filename': filename, **(details or {})}
        )


class FileUploadError(SalesForecastingException):
    """Raised when file upload fails."""
    
    def __init__(self, message: str, filename: str = '', file_size: int = 0, 
                 details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='FILE_UPLOAD_ERROR',
            http_status=400,
            details={'filename': filename, 'file_size': file_size, **(details or {})}
        )


class InsufficientDataError(SalesForecastingException):
    """Raised when there's insufficient data for processing."""
    
    def __init__(self, message: str, required: int = 0, actual: int = 0, 
                 details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='INSUFFICIENT_DATA_ERROR',
            http_status=422,
            details={'required': required, 'actual': actual, **(details or {})}
        )


class ModelTrainingError(SalesForecastingException):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='MODEL_TRAINING_ERROR',
            http_status=500,
            details={'model_type': model_type, **(details or {})}
        )


class DatabaseError(SalesForecastingException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='DATABASE_ERROR',
            http_status=500,
            details={'operation': operation, **(details or {})}
        )


class AuthenticationError(SalesForecastingException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, reason: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='AUTHENTICATION_ERROR',
            http_status=401,
            details={'reason': reason, **(details or {})}
        )


class AuthorizationError(SalesForecastingException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, resource: str = '', details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='AUTHORIZATION_ERROR',
            http_status=403,
            details={'resource': resource, **(details or {})}
        )


class ResourceNotFoundError(SalesForecastingException):
    """Raised when a resource is not found."""
    
    def __init__(self, message: str, resource_type: str = '', resource_id: Any = None, 
                 details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code='RESOURCE_NOT_FOUND_ERROR',
            http_status=404,
            details={'resource_type': resource_type, 'resource_id': resource_id, 
                    **(details or {})}
        )


# ========================== ERROR FORMATTERS ==========================

def format_error_response(exc: Exception, include_traceback: bool = False) -> Dict:
    """Format exception as JSON response."""
    
    if isinstance(exc, SalesForecastingException):
        response = {
            'status': 'error',
            'error_code': exc.error_code,
            'message': exc.message,
            'timestamp': exc.timestamp,
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        }
        
        # Include details if present
        if exc.details:
            response['details'] = exc.details
        
        # Include traceback if requested (only for development)
        if include_traceback:
            response['traceback'] = traceback.format_exc()
        
        return response, exc.http_status
    
    else:
        # Generic exception handling
        error_type = type(exc).__name__
        response = {
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'error_type': error_type,
            'message': str(exc),
            'timestamp': datetime.now().isoformat(),
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        }
        
        if include_traceback:
            response['traceback'] = traceback.format_exc()
        
        return response, 500


# ========================== DECORATORS ==========================

def handle_errors(func: Callable) -> Callable:
    """Decorator to catch and format errors in API endpoints."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SalesForecastingException as e:
            app_logger.warning(f"Application error in {func.__name__}: {e.message}")
            response, status_code = format_error_response(e, include_traceback=False)
            return jsonify(response), status_code
        except Exception as e:
            app_logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            response, status_code = format_error_response(e, include_traceback=False)
            return jsonify(response), status_code
    
    return wrapper


def validate_request_data(required_fields: list = None, allowed_types: dict = None):
    """Decorator to validate request data."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate presence of required fields
            if required_fields:
                data = request.get_json() or request.form.to_dict()
                missing_fields = [f for f in required_fields if f not in data]
                
                if missing_fields:
                    raise DataValidationError(
                        f"Missing required fields: {', '.join(missing_fields)}",
                        details={'missing_fields': missing_fields}
                    )
            
            # Validate field types
            if allowed_types:
                data = request.get_json() or request.form.to_dict()
                for field, expected_type in allowed_types.items():
                    if field in data and data[field] is not None:
                        if not isinstance(data[field], expected_type):
                            raise DataValidationError(
                                f"Field '{field}' must be of type {expected_type.__name__}",
                                field=field
                            )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from flask import session
        
        if 'user_id' not in session:
            raise AuthenticationError(
                'User authentication required',
                reason='No active session'
            )
        
        return func(*args, **kwargs)
    
    return wrapper


def require_json(func: Callable) -> Callable:
    """Decorator to require JSON content type."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            raise DataValidationError(
                'Request must have Content-Type: application/json',
                details={'content_type': request.content_type}
            )
        
        return func(*args, **kwargs)
    
    return wrapper


def safe_db_operation(operation_name: str = 'database operation'):
    """Decorator for safe database operations with error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                app_logger.error(f"Database error during {operation_name}: {str(e)}", exc_info=True)
                raise DatabaseError(
                    f"Failed to complete {operation_name}",
                    operation=operation_name,
                    details={'original_error': str(e)}
                )
        
        return wrapper
    
    return decorator


# ========================== VALIDATION UTILITIES ==========================

def validate_csv_file(file) -> Tuple[bool, str]:
    """Validate uploaded CSV file."""
    try:
        if not file:
            return False, "No file provided"
        
        if not file.filename:
            return False, "No filename provided"
        
        # Check file extension
        allowed_extensions = {'csv'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return False, "Only CSV files are allowed"
        
        # Check file size (max 50MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False, f"File size exceeds maximum ({max_size / (1024*1024):.0f}MB)"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, f"File valid ({file_size / 1024:.1f}KB)"
        
    except Exception as e:
        app_logger.error(f"Error validating CSV file: {str(e)}")
        return False, f"Error validating file: {str(e)}"


def validate_numeric_range(value: float, min_val: float = None, max_val: float = None, 
                          field_name: str = 'value') -> bool:
    """Validate numeric value is within range."""
    if min_val is not None and value < min_val:
        raise DataValidationError(
            f"{field_name} must be >= {min_val}",
            field=field_name,
            details={'min': min_val, 'actual': value}
        )
    
    if max_val is not None and value > max_val:
        raise DataValidationError(
            f"{field_name} must be <= {max_val}",
            field=field_name,
            details={'max': max_val, 'actual': value}
        )
    
    return True


def validate_data_available(df) -> bool:
    """Validate that dataframe with data is available."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        raise DataLoadError(
            "No dataset available. Please upload a CSV file.",
            details={'reason': 'dataframe_is_none_or_empty'}
        )
    return True


# ========================== ERROR RECOVERY ==========================

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    @staticmethod
    def on_data_load_failure(error: Exception) -> Tuple[bool, str]:
        """Attempt recovery when data loading fails."""
        app_logger.warning(f"Data load failed: {str(error)}")
        # Return fallback data or graceful failure
        return False, "Could not load data. Please upload a CSV file."
    
    @staticmethod
    def on_model_training_failure(error: Exception, fallback_method: str = 'simple_average') -> Dict:
        """Attempt recovery when model training fails."""
        app_logger.warning(f"Model training failed, using {fallback_method}: {str(error)}")
        
        if fallback_method == 'simple_average':
            # Return simple forecast using average
            return {
                'success': True,
                'fallback': True,
                'method': 'simple_average'
            }
        
        return {
            'success': False,
            'fallback': False,
            'message': f"Model training failed and no fallback available"
        }
    
    @staticmethod
    def on_database_operation_failure(error: Exception, operation: str) -> bool:
        """Handle database operation failures."""
        app_logger.error(f"Database operation '{operation}' failed: {str(error)}")
        # In production, might retry with exponential backoff
        return False


# ========================== CONTEXT MANAGERS ==========================

class DatabaseOperation:
    """Context manager for safe database operations."""
    
    def __init__(self, connection, operation_name: str = 'database operation'):
        self.connection = connection
        self.operation_name = operation_name
        self.cursor = None
    
    def __enter__(self):
        try:
            self.cursor = self.connection.cursor()
            return self.cursor
        except Exception as e:
            raise DatabaseError(
                f"Failed to establish database connection for {self.operation_name}",
                operation=self.operation_name
            )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            app_logger.error(f"Error during {self.operation_name}: {exc_val}")
            self.connection.rollback()
            return False
        else:
            try:
                self.connection.commit()
            except Exception as e:
                app_logger.error(f"Failed to commit {self.operation_name}: {str(e)}")
                self.connection.rollback()
                return False
        return True
