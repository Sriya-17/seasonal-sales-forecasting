import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'sales.db')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'change-me-in-production')
    DATABASE = DATABASE_PATH
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Session Configuration for Security
    # Allow HTTP in development (localhost), require HTTPS in production
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access
    SESSION_COOKIE_SAMESITE = 'Lax'  # CSRF protection
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)  # 24-hour session timeout
    SESSION_REFRESH_EACH_REQUEST = True  # Reset timeout on each request
